import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import librosa
import logging
import matplotlib.pyplot as plt

import torch
 
from utilities import (create_folder, get_filename, RegressionPostProcessor, 
    OnsetsFramesPostProcessor, write_events_to_midi, load_audio)
from models import Note_pedal
from pytorch_utils import move_data_to_device, forward
import config


class PianoTranscription(object):
    def __init__(self, model_type, checkpoint_path=None, 
        segment_samples=16000*10, device=torch.device('cuda'), 
        post_processor_type='regression'):
        """Class for transcribing piano solo recording.

        Args:
          model_type: str
          checkpoint_path: str
          segment_samples: int
          device: 'cuda' | 'cpu'
        """

        if 'cuda' in str(device) and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.segment_samples = segment_samples
        self.post_processor_type = post_processor_type
        self.frames_per_second = config.frames_per_second
        self.classes_num = config.classes_num
        self.onset_threshold = 0.3
        self.offset_threshod = 0.3
        self.frame_threshold = 0.1
        self.pedal_offset_threshold = 0.2

        # Build model
        Model = eval(model_type)
        self.model = Model(frames_per_second=self.frames_per_second, 
            classes_num=self.classes_num)

        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'], strict=False)

        # Parallel
        if 'cuda' in str(self.device):
            self.model.to(self.device)
            print('GPU number: {}'.format(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model)
        else:
            print('Using CPU.')

    def transcribe(self, audio, midi_path):
        """Transcribe an audio recording.

        Args:
          audio: (audio_samples,)
          midi_path: str, path to write out the transcribed MIDI.

        Returns:
          transcribed_dict, dict: {'output_dict':, ..., 'est_note_events': ..., 
            'est_pedal_events': ...}
        """

        audio = audio[None, :]  # (1, audio_samples)

        # Pad audio to be evenly divided by segment_samples
        audio_len = audio.shape[1]
        pad_len = int(np.ceil(audio_len / self.segment_samples)) \
            * self.segment_samples - audio_len

        audio = np.concatenate((audio, np.zeros((1, pad_len))), axis=1)

        # Enframe to segments
        segments = self.enframe(audio, self.segment_samples)
        """(N, segment_samples)"""

        # Forward
        output_dict = forward(self.model, segments, batch_size=1)
        """{'reg_onset_output': (N, segment_frames, classes_num), ...}"""

        # Deframe to original length
        for key in output_dict.keys():
            output_dict[key] = self.deframe(output_dict[key])[0 : audio_len]
        """output_dict: {
          'reg_onset_output': (segment_frames, classes_num), 
          'reg_offset_output': (segment_frames, classes_num), 
          'frame_output': (segment_frames, classes_num), 
          'velocity_output': (segment_frames, classes_num), 
          'reg_pedal_onset_output': (segment_frames, 1), 
          'reg_pedal_offset_output': (segment_frames, 1), 
          'pedal_frame_output': (segment_frames, 1)}"""

        # Post processor
        if self.post_processor_type == 'regression':
            """Proposed high-resolution regression post processing algorithm."""
            post_processor = RegressionPostProcessor(self.frames_per_second, 
                classes_num=self.classes_num, onset_threshold=self.onset_threshold, 
                offset_threshold=self.offset_threshod, 
                frame_threshold=self.frame_threshold, 
                pedal_offset_threshold=self.pedal_offset_threshold)

        elif self.post_processor_type == 'onsets_frames':
            """Google's onsets and frames post processing algorithm. Only used 
            for comparison."""
            post_processor = OnsetsFramesPostProcessor(self.frames_per_second, 
                self.classes_num)

        # Post process output_dict to MIDI events
        (est_note_events, est_pedal_events) = \
            post_processor.output_dict_to_midi_events(output_dict)

        # Write MIDI events to file
        if midi_path:
            write_events_to_midi(start_time=0, note_events=est_note_events, 
                pedal_events=est_pedal_events, midi_path=midi_path)
            print('Write out to {}'.format(midi_path))

        transcribed_dict = {
            'output_dict': output_dict, 
            'est_note_events': est_note_events,
            'est_pedal_events': est_pedal_events}

        return transcribed_dict

    def enframe(self, x, segment_samples):
        """Enframe long sequence to short segments.

        Args:
          x: (1, audio_samples)
          segment_samples: int

        Returns:
          batch: (N, segment_samples)
        """
        assert x.shape[1] % segment_samples == 0
        batch = []

        pointer = 0
        while pointer + segment_samples <= x.shape[1]:
            batch.append(x[:, pointer : pointer + segment_samples])
            pointer += segment_samples // 2

        batch = np.concatenate(batch, axis=0)
        return batch

    def deframe(self, x):
        """Deframe predicted segments to original sequence.

        Args:
          x: (N, segment_frames, classes_num)

        Returns:
          y: (audio_frames, classes_num)
        """
        if x.shape[0] == 1:
            return x[0]

        else:
            x = x[:, 0 : -1, :]
            """Remove an extra frame in the end of each segment caused by the
            'center=True' argument when calculating spectrogram."""
            (N, segment_samples, classes_num) = x.shape
            assert segment_samples % 4 == 0

            y = []
            y.append(x[0, 0 : int(segment_samples * 0.75)])
            for i in range(1, N - 1):
                y.append(x[i, int(segment_samples * 0.25) : int(segment_samples * 0.75)])
            y.append(x[-1, int(segment_samples * 0.25) :])
            y = np.concatenate(y, axis=0)
            return y


def inference(args):
    """Inference template.

    Args:
      model_type: str
      checkpoint_path: str
      post_processor_type: 'regression' | 'onsets_frames'. High-resolution 
        system should use 'regression'. 'onsets_frames' is only used to compare
        with Googl's onsets and frames system.
      audio_path: str
      cuda: bool
    """

    # Arugments & parameters
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    post_processor_type = args.post_processor_type
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    audio_path = args.audio_path
    
    sample_rate = config.sample_rate
    segment_samples = sample_rate * 10  
    """Split audio to multiple 10-second segments for inference"""

    # Paths
    midi_path = 'results/{}.mid'.format(get_filename(audio_path))
    create_folder(os.path.dirname(midi_path))
 
    # Load audio
    (audio, _) = load_audio(audio_path, sr=sample_rate, mono=True)

    # Transcriptor
    transcriptor = PianoTranscription(model_type, device=device, 
        checkpoint_path=checkpoint_path, segment_samples=segment_samples, 
        post_processor_type=post_processor_type)

    # Transcribe and write out to MIDI file
    transcribe_time = time.time()
    transcribed_dict = transcriptor.transcribe(audio, midi_path)
    print('Transcribe time: {:.3f} s'.format(time.time() - transcribe_time))

    # Visualize for debug
    plot = False
    if plot:
        output_dict = transcribed_dict['output_dict']
        fig, axs = plt.subplots(5, 1, figsize=(15, 8), sharex=True)
        mel = librosa.feature.melspectrogram(audio, sr=16000, n_fft=2048, hop_length=160, n_mels=229, fmin=30, fmax=8000)
        axs[0].matshow(np.log(mel), origin='lower', aspect='auto', cmap='jet')
        axs[1].matshow(output_dict['frame_output'].T, origin='lower', aspect='auto', cmap='jet')
        axs[2].matshow(output_dict['reg_onset_output'].T, origin='lower', aspect='auto', cmap='jet')
        axs[3].matshow(output_dict['reg_offset_output'].T, origin='lower', aspect='auto', cmap='jet')
        axs[4].plot(output_dict['pedal_frame_output'])
        axs[0].set_xlim(0, len(output_dict['frame_output']))
        axs[4].set_xlabel('Frames')
        axs[0].set_title('Log mel spectrogram')
        axs[1].set_title('frame_output')
        axs[2].set_title('reg_onset_output')
        axs[3].set_title('reg_offset_output')
        axs[4].set_title('pedal_frame_output')
        plt.tight_layout(0, .05, 0)
        fig_path = '_zz.pdf'.format(get_filename(audio_path))
        plt.savefig(fig_path)
        print('Plot to {}'.format(fig_path))
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--post_processor_type', type=str, default='regression', choices=['onsets_frames', 'regression'])
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()
    inference(args)