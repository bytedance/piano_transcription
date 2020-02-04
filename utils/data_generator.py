import os
import sys
import numpy as np
import h5py
import csv
import time
import collections
import logging

from utilities import (create_folder, int16_to_float32, traverse_folder, 
    TargetProcessor, write_events_to_midi)
import config


class MaestroDataset(object):
    def __init__(self, feature_hdf5s_dir, segment_seconds, frames_per_second):
        """Maestro dataset. Will be used for DataLoader.

        Args:
          feature_hdf5s_dir: str
          segment_seconds: float
          frames_per_second: int
        """
        self.feature_hdf5s_dir = feature_hdf5s_dir
        self.segment_seconds = segment_seconds
        self.sample_rate = config.sample_rate
        begin_note = config.begin_note
        classes_num = config.classes_num

        self.target_processor = TargetProcessor(self.segment_seconds, 
            frames_per_second, begin_note, classes_num)
        """Used to process MIDI events to target."""

    def __getitem__(self, meta):
        """Prepare input and target of a segment for training.
        
        Args:
          meta: dict, e.g. {
            'year': '2004', 
            'hdf5_name': 'MIDI-Unprocessed_SMF_12_01_2004_01-05_ORIG_MID--AUDIO_12_R1_2004_10_Track10_wav.h5, 
            'start_time': 65.0}

        Returns:
          data_dict: {
            'waveform': (samples_num,)
            'frame_roll': (frames_num, classes_num), 
            'onset_roll': (frames_num, classes_num), 
            'offset_roll': (frames_num, classes_num), 
            'distance_roll': (frames_num, classes_num), 
            'velocity_roll': (frames_num, classes_num), 
            'mask_roll':  (frames_num, classes_num), 
            'pedal_roll': (frames_num,)}
        """
        [year, hdf5_name, start_time] = meta
        hdf5_path = os.path.join(self.feature_hdf5s_dir, year, hdf5_name)
        
        data_dict = {}

        # Load hdf5
        with h5py.File(hdf5_path, 'r') as hf:
            start_sample = int(start_time * self.sample_rate)
            end_sample = int((start_time + self.segment_seconds) * self.sample_rate)
            
            waveform = int16_to_float32(hf['waveform'][start_sample : end_sample])
            data_dict['waveform'] = waveform
            
            midi_events = [e.decode() for e in hf['midi_event'][:]]
            midi_events_time = hf['midi_event_time'][:]

        # Process MIDI events to target
        (target_dict, note_events) = self.target_processor.process(start_time, 
            midi_events_time, midi_events)

        # Combine input and target
        for key in target_dict.keys():
            data_dict[key] = target_dict[key]

        debugging = False
        if debugging:
            self._write_waveform_midi_fig(data_dict, start_time, note_events)
            exit()

        return data_dict


    def _write_waveform_midi_fig(self, data_dict, start_time, note_events):
        """For debugging. Write the waveform, MIDI and plot target for a segment.

        Args:
          data_dict: {
            'waveform': (samples_num,)
            'frame_roll': (frames_num, classes_num), 
            'onset_roll': (frames_num, classes_num), 
            'offset_roll': (frames_num, classes_num), 
            'distance_roll': (frames_num, classes_num), 
            'velocity_roll': (frames_num, classes_num), 
            'mask_roll':  (frames_num, classes_num), 
            'pedal_roll': (frames_num,)}
          start_time: float
          note_events: list of dict, e.g. [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
        """

        import librosa
        import matplotlib.pyplot as plt

        create_folder('debug')
        audio_path = 'debug/debug.wav'
        midi_path = 'debug/debug.mid'
        fig_path = 'debug/debug.pdf'

        librosa.output.write_wav(audio_path, data_dict['waveform'], sr=self.sample_rate)
        write_events_to_midi(start_time, note_events, midi_path)
        x = librosa.core.stft(y=data_dict['waveform'], n_fft=2048, hop_length=320, window='hann', center=True)
        x = np.abs(x) ** 2

        fig, axs = plt.subplots(8, 1, sharex=True, figsize=(30, 30))
        axs[0].matshow(np.log(x), origin='lower', aspect='auto', cmap='jet')
        axs[1].matshow(data_dict['frame_roll'].T, origin='lower', aspect='auto', cmap='jet')
        axs[2].matshow(data_dict['onset_roll'].T, origin='lower', aspect='auto', cmap='jet')
        axs[3].matshow(data_dict['offset_roll'].T, origin='lower', aspect='auto', cmap='jet')
        axs[4].matshow(data_dict['distance_roll'].T, origin='lower', aspect='auto', cmap='jet')
        axs[5].matshow(data_dict['velocity_roll'].T, origin='lower', aspect='auto', cmap='jet')
        axs[6].matshow(data_dict['mask_roll'].T, origin='lower', aspect='auto', cmap='jet')
        axs[7].matshow(data_dict['pedal_roll'][:, None].T, origin='lower', aspect='auto', cmap='jet')
        axs[0].set_title('Log spectrogram')
        axs[1].set_title('frame_roll')
        axs[2].set_title('onset_roll')
        axs[3].set_title('offset_roll')
        axs[4].set_title('distance_roll')
        axs[5].set_title('velocity_roll')
        axs[6].set_title('mask_roll')
        axs[7].set_title('pedal_roll')
        plt.show()
        plt.savefig(fig_path)

        print('Write out to {}, {}, {}!'.format(audio_path, midi_path, fig_path))


class Sampler(object):
    def __init__(self, feature_hdf5s_dir, split, segment_seconds, hop_seconds, 
            batch_size, training, mini_data, random_seed=1234):
        """Sampler is used to sample segments for training or evaluation.

        Args:
          feature_hdf5s_dir: str
          split: 'train' | 'validation' | 'test'
          segment_seconds: float
          hop_seconds: float
          batch_size: int
          training: bool, True: sample segments for training; False: sample 
            segments for evaluation
          mini_data: bool, sample from a small amount of data for debugging
        """
        assert split in ['train', 'validation', 'test']
        self.feature_hdf5s_dir = feature_hdf5s_dir
        self.training = training
        self.segment_seconds = segment_seconds
        self.hop_seconds = hop_seconds
        self.sample_rate = config.sample_rate
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)
        self.max_evaluate_iteration = 5    # Number of mini-batches to validate

        (hdf5_names, hdf5_paths) = traverse_folder(feature_hdf5s_dir)

        self.segment_list = []

        n = 0
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as hf:
                if hf.attrs['split'].decode() == split:
                    audio_name = hdf5_path.split('/')[-1]
                    year = hf.attrs['year'].decode()
                    start_time = 0
                    while (start_time + self.segment_seconds < hf.attrs['duration']):
                        self.segment_list.append([year, audio_name, start_time])
                        start_time += self.hop_seconds
                    
                    n += 1
                    if mini_data and n == 10:
                        break
        """self.segment_list looks like:
        [['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 0], 
         ['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 1.0], 
         ['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 2.0]
         ...]"""

        logging.info('{} segments: {}'.format(split, len(self.segment_list)))

        self.pointer = 0
        self.segment_indexes = np.arange(len(self.segment_list))
        self.random_state.shuffle(self.segment_indexes)

    def __iter__(self):
        batch_size = self.batch_size
        iteration = 0

        while True:
            # If evaluation, then only generate a few mini-batches
            if not self.training:
                if iteration == self.max_evaluate_iteration:
                    break

            if self.pointer >= len(self.segment_indexes):
                self.pointer = 0
                self.random_state.shuffle(self.segment_indexes)

            batch_segment_indexes = self.segment_indexes[
                self.pointer: self.pointer + self.batch_size]
                
            self.pointer += self.batch_size

            batch_segment_list = []
            for idx in batch_segment_indexes:
                batch_segment_list.append(self.segment_list[idx])

            iteration += 1

            yield batch_segment_list

    def __len__(self):
        return -1
        
    def state_dict(self):
        state = {
            'pointer': self.pointer, 
            'segment_indexes': self.segment_indexes}
        return state
            
    def load_state_dict(self, state):
        self.pointer = state['pointer']
        self.segment_indexes = state['segment_indexes']


def collate_fn(list_data_dict):
    """Collate input and target of segments to a mini-batch.

    Args:
      list_data_dict: e.g. [
        {'waveform': (segment_samples,), 'frame_roll': (segment_frames, classes_num)}, 
        {'waveform': (segment_samples,), 'frame_roll': (segment_frames, classes_num)}, 
        ...]

    Returns:
      np_data_dict: e.g. {
        'waveform': (batch_size, segment_samples)
        'frame_roll': (batch_size, segment_frames, classes_num), 
        ...}
    """
    np_data_dict = {}
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
    
    return np_data_dict