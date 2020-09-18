import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import torch
import h5py
import time
import mir_eval
import librosa
import logging
from sklearn import metrics

from pytorch_utils import forward_dataloader


def mae(target, output, mask):
    if mask is None:
        return np.mean(np.abs(target - output))
    else:
        target *= mask
        output *= mask
        return np.sum(np.abs(target - output)) / np.clip(np.sum(mask), 1e-8, np.inf)


class SegmentEvaluator(object):
    def __init__(self, model, batch_size):
        """Evaluate segment-wise metrics.

        Args:
          model: object
          batch_size: int
        """
        self.model = model
        self.batch_size = batch_size

    def evaluate(self, dataloader):
        """Evaluate over a few mini-batches.

        Args:
          dataloader: object, used to generate mini-batches for evaluation.

        Returns:
          statistics: dict, e.g. {
            'frame_f1': 0.800, 
            (if exist) 'onset_f1': 0.500, 
            (if exist) 'offset_f1': 0.300, 
            ...}
        """

        statistics = {}
        output_dict = forward_dataloader(self.model, dataloader, self.batch_size)
        
        # Frame and onset evaluation
        if 'frame_output' in output_dict.keys():
            statistics['frame_ap'] = metrics.average_precision_score(
                output_dict['frame_roll'].flatten(), 
                output_dict['frame_output'].flatten(), average='macro')
        
        if 'onset_output' in output_dict.keys():
            statistics['onset_macro_ap'] = metrics.average_precision_score(
                output_dict['onset_roll'].flatten(), 
                output_dict['onset_output'].flatten(), average='macro')

        if 'offset_output' in output_dict.keys():
            statistics['offset_ap'] = metrics.average_precision_score(
                output_dict['offset_roll'].flatten(), 
                output_dict['offset_output'].flatten(), average='macro')

        if 'reg_onset_output' in output_dict.keys():
            """Mask indictes only evaluate where either prediction or ground truth exists"""
            mask = (np.sign(output_dict['reg_onset_output'] + output_dict['reg_onset_roll'] - 0.01) + 1) / 2
            statistics['reg_onset_mae'] = mae(output_dict['reg_onset_output'], 
                output_dict['reg_onset_roll'], mask)

        if 'reg_offset_output' in output_dict.keys():
            """Mask indictes only evaluate where either prediction or ground truth exists"""
            mask = (np.sign(output_dict['reg_offset_output'] + output_dict['reg_offset_roll'] - 0.01) + 1) / 2
            statistics['reg_offset_mae'] = mae(output_dict['reg_offset_output'], 
                output_dict['reg_offset_roll'], mask)

        if 'velocity_output' in output_dict.keys():
            """Mask indictes only evaluate where onset exists"""
            statistics['velocity_mae'] = mae(output_dict['velocity_output'], 
                output_dict['velocity_roll'] / 128, output_dict['onset_roll'])

        if 'reg_pedal_onset_output' in output_dict.keys():
            statistics['reg_pedal_onset_mae'] = mae(
                output_dict['reg_pedal_onset_roll'].flatten(), 
                output_dict['reg_pedal_onset_output'].flatten(), 
                mask=None)

        if 'reg_pedal_offset_output' in output_dict.keys():
            statistics['reg_pedal_offset_mae'] = mae(
                output_dict['reg_pedal_offset_output'].flatten(), 
                output_dict['reg_pedal_offset_roll'].flatten(), 
                mask=None)

        if 'pedal_frame_output' in output_dict.keys():
            statistics['pedal_frame_mae'] = mae(
                output_dict['pedal_frame_output'].flatten(), 
                output_dict['pedal_frame_roll'].flatten(), 
                mask=None)

        for key in statistics.keys():
            statistics[key] = np.around(statistics[key], decimals=4)

        return statistics