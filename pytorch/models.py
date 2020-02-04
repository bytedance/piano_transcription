import os
import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.nn.parameter import Parameter

from stft import Spectrogram, LogmelFilterBank
from augmentation import SpecAugmentation
 

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


def init_gru(rnn):
    """Initialize a GRU layer. """
    
    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)
    
        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])
        
    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
    
    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)


class GoogleAcousticModel(nn.Module):
    def __init__(self, classes_num):
        super(GoogleAcousticModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=48, 
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(48)

        self.conv2 = nn.Conv2d(in_channels=48, out_channels=48, 
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.conv3 = nn.Conv2d(in_channels=48, out_channels=96, 
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(96)

        self.fc4 = nn.Linear(5472, 768, bias=False)
        self.bn4 = nn.BatchNorm1d(768)

        self.gru = nn.GRU(input_size=768, hidden_size=256, num_layers=1, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)

        self.fc = nn.Linear(512, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.fc4)
        init_gru(self.gru)
        init_layer(self.fc)

    def forward(self, input):
        
        x = F.relu_(self.bn1(self.conv1(input)))
        x = F.max_pool2d(x, kernel_size=(1, 1))

        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=(1, 2))
        x = F.dropout(x, p=0.25, training=self.training, inplace=True)

        x = F.relu_(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=(1, 2))
        x = F.dropout(x, p=0.25, training=self.training, inplace=True)

        x = x.transpose(1, 2).flatten(2)
        x = F.relu(self.bn4(self.fc4(x).transpose(1, 2)).transpose(1, 2))
        x = F.dropout(x, p=0.5, training=self.training, inplace=True)
        
        (x, _) = self.gru(x)
        output = torch.sigmoid(self.fc(x))
        return output


class Google_onset_frame(nn.Module):
    def __init__(self, sample_rate, frames_per_second, classes_num):
        """This class is a reimplementation of [1] with a few difference. The
        difference includes GRU is used instead of LSTM. A sampling rate of 
        32 kHz is used instead of 16 kHz. A hop size of 10 ms is used instead 
        of 31.25 ms. 
        
        [1] Hawthorne, C., Elsen, E., Song, J., Roberts, A., Simon, 
        I., Raffel, C., Engel, J., Oore, S. and Eck, D., Onsets and frames: 
        Dual-objective piano transcription. ISMIR 2018.
        """
        super(Google_onset_frame, self).__init__()

        window_size = 2048
        hop_size = sample_rate // frames_per_second
        mel_bins = 229
        fmin = 50
        fmax = sample_rate / 2

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, 
            hop_length=hop_size, win_length=window_size, window=window, 
            center=center, pad_mode=pad_mode, freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, 
            n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, 
            amin=amin, top_db=top_db, freeze_parameters=True)

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.frame_model = GoogleAcousticModel(classes_num)
        self.onset_model = GoogleAcousticModel(classes_num)

        self.final_gru = nn.GRU(input_size=88 * 2, hidden_size=256, num_layers=1, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)
        
        self.final_fc = nn.Linear(512, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_gru(self.final_gru)
        init_layer(self.final_fc)
 
    def forward(self, input):
        """
        Input: (batch_size, data_length)"""
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        frame_output = self.frame_model(x)  # (batch_size, time_steps, classes_num)
        onset_output = self.onset_model(x)  # (batch_size, time_steps, classes_num)

        x = torch.cat((frame_output, onset_output.detach()), dim=2)
        (x, _) = self.final_gru(x)
        frame_output = torch.sigmoid(self.final_fc(x))  # (batch_size, time_steps, classes_num)

        output_dict = {
            'frame_output': frame_output, 
            'onset_output': onset_output}

        return output_dict