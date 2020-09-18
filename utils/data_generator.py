import os
import sys
import numpy as np
import h5py
import csv
import time
import collections
import librosa
import sox
import logging

from utilities import (create_folder, int16_to_float32, traverse_folder, 
    pad_truncate_sequence, TargetProcessor, write_events_to_midi, 
    plot_waveform_midi_targets)
import config


class MaestroDataset(object):
    def __init__(self, hdf5s_dir, segment_seconds, frames_per_second, 
        max_note_shift=0, augmentor=None):
        """This class takes the meta of an audio segment as input, and return 
        the waveform and targets of the audio segment. This class is used by 
        DataLoader. 
        
        Args:
          feature_hdf5s_dir: str
          segment_seconds: float
          frames_per_second: int
          max_note_shift: int, number of semitone for pitch augmentation
          augmentor: object
        """
        self.hdf5s_dir = hdf5s_dir
        self.segment_seconds = segment_seconds
        self.frames_per_second = frames_per_second
        self.sample_rate = config.sample_rate
        self.max_note_shift = max_note_shift
        self.begin_note = config.begin_note
        self.classes_num = config.classes_num
        self.segment_samples = int(self.sample_rate * self.segment_seconds)
        self.augmentor = augmentor

        self.random_state = np.random.RandomState(1234)

        self.target_processor = TargetProcessor(self.segment_seconds, 
            self.frames_per_second, self.begin_note, self.classes_num)
        """Used for processing MIDI events to target."""

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
            'onset_roll': (frames_num, classes_num), 
            'offset_roll': (frames_num, classes_num), 
            'reg_onset_roll': (frames_num, classes_num), 
            'reg_offset_roll': (frames_num, classes_num), 
            'frame_roll': (frames_num, classes_num), 
            'velocity_roll': (frames_num, classes_num), 
            'mask_roll':  (frames_num, classes_num), 
            'pedal_onset_roll': (frames_num,), 
            'pedal_offset_roll': (frames_num,), 
            'reg_pedal_onset_roll': (frames_num,), 
            'reg_pedal_offset_roll': (frames_num,), 
            'pedal_frame_roll': (frames_num,)}
        """
        [year, hdf5_name, start_time] = meta
        hdf5_path = os.path.join(self.hdf5s_dir, year, hdf5_name)
         
        data_dict = {}

        note_shift = self.random_state.randint(low=-self.max_note_shift, 
            high=self.max_note_shift + 1)

        # Load hdf5
        with h5py.File(hdf5_path, 'r') as hf:
            start_sample = int(start_time * self.sample_rate)
            end_sample = start_sample + self.segment_samples

            if end_sample >= hf['waveform'].shape[0]:
                start_sample -= self.segment_samples
                end_sample -= self.segment_samples

            waveform = int16_to_float32(hf['waveform'][start_sample : end_sample])

            if self.augmentor:
                waveform = self.augmentor.augment(waveform)

            if note_shift != 0:
                """Augment pitch"""
                waveform = librosa.effects.pitch_shift(waveform, self.sample_rate, 
                    note_shift, bins_per_octave=12)

            data_dict['waveform'] = waveform

            midi_events = [e.decode() for e in hf['midi_event'][:]]
            midi_events_time = hf['midi_event_time'][:]

            # Process MIDI events to target
            (target_dict, note_events, pedal_events) = \
                self.target_processor.process(start_time, midi_events_time, 
                    midi_events, extend_pedal=True, note_shift=note_shift)

        # Combine input and target
        for key in target_dict.keys():
            data_dict[key] = target_dict[key]

        debugging = False
        if debugging:
            plot_waveform_midi_targets(data_dict, start_time, note_events)
            exit()

        return data_dict


class Augmentor(object):
    def __init__(self):
        """Data augmentor."""
        
        self.sample_rate = config.sample_rate
        self.random_state = np.random.RandomState(1234)

    def augment(self, x):
        clip_samples = len(x)

        logger = logging.getLogger('sox')
        logger.propagate = False

        tfm = sox.Transformer()
        tfm.set_globals(verbosity=0)

        tfm.pitch(self.random_state.uniform(-0.1, 0.1, 1)[0])
        tfm.contrast(self.random_state.uniform(0, 100, 1)[0])

        tfm.equalizer(frequency=self.loguniform(32, 4096, 1)[0], 
            width_q=self.random_state.uniform(1, 2, 1)[0], 
            gain_db=self.random_state.uniform(-30, 10, 1)[0])

        tfm.equalizer(frequency=self.loguniform(32, 4096, 1)[0], 
            width_q=self.random_state.uniform(1, 2, 1)[0], 
            gain_db=self.random_state.uniform(-30, 10, 1)[0])
        
        tfm.reverb(reverberance=self.random_state.uniform(0, 70, 1)[0])

        aug_x = tfm.build_array(input_array=x, sample_rate_in=self.sample_rate)
        aug_x = pad_truncate_sequence(aug_x, clip_samples)
        
        return aug_x

    def loguniform(self, low, high, size):
        return np.exp(self.random_state.uniform(np.log(low), np.log(high), size))


class Sampler(object):
    def __init__(self, hdf5s_dir, split, segment_seconds, hop_seconds, 
            batch_size, mini_data, random_seed=1234):
        """Sampler is used to sample segments for training or evaluation.

        Args:
          hdf5s_dir: str
          split: 'train' | 'validation' | 'test'
          segment_seconds: float
          hop_seconds: float
          batch_size: int
          mini_data: bool, sample from a small amount of data for debugging
        """
        assert split in ['train', 'validation', 'test']
        self.hdf5s_dir = hdf5s_dir
        self.segment_seconds = segment_seconds
        self.hop_seconds = hop_seconds
        self.sample_rate = config.sample_rate
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)

        (hdf5_names, hdf5_paths) = traverse_folder(hdf5s_dir)
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
        while True:
            batch_segment_list = []
            i = 0
            while i < self.batch_size:
                index = self.segment_indexes[self.pointer]
                self.pointer += 1

                if self.pointer >= len(self.segment_indexes):
                    self.pointer = 0
                    self.random_state.shuffle(self.segment_indexes)

                batch_segment_list.append(self.segment_list[index])
                i += 1

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


class TestSampler(object):
    def __init__(self, hdf5s_dir, split, segment_seconds, hop_seconds, 
            batch_size, mini_data, random_seed=1234):
        """Sampler for testing.

        Args:
          hdf5s_dir: str
          split: 'train' | 'validation' | 'test'
          segment_seconds: float
          hop_seconds: float
          batch_size: int
          mini_data: bool, sample from a small amount of data for debugging
        """
        assert split in ['train', 'validation', 'test']
        self.hdf5s_dir = hdf5s_dir
        self.segment_seconds = segment_seconds
        self.hop_seconds = hop_seconds
        self.sample_rate = config.sample_rate
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)
        self.max_evaluate_iteration = 20    # Number of mini-batches to validate

        (hdf5_names, hdf5_paths) = traverse_folder(hdf5s_dir)
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

        logging.info('Evaluate {} segments: {}'.format(split, len(self.segment_list)))

        self.segment_indexes = np.arange(len(self.segment_list))
        self.random_state.shuffle(self.segment_indexes)

    def __iter__(self):
        pointer = 0
        iteration = 0

        while True:
            if iteration == self.max_evaluate_iteration:
                break

            batch_segment_list = []
            i = 0
            while i < self.batch_size:
                index = self.segment_indexes[pointer]
                pointer += 1
                
                batch_segment_list.append(self.segment_list[index])
                i += 1

            iteration += 1

            yield batch_segment_list

    def __len__(self):
        return -1


def collate_fn(list_data_dict):
    """Collate input and target of segments to a mini-batch.

    Args:
      list_data_dict: e.g. [
        {'waveform': (segment_samples,), 'frame_roll': (segment_frames, classes_num), ...}, 
        {'waveform': (segment_samples,), 'frame_roll': (segment_frames, classes_num), ...}, 
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