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
    pad_truncate_sequence, write_events_to_midi, 
    plot_waveform_midi_targets)
import config


class RandomTargetProcessor(object):
    def __init__(self, segment_seconds, frames_per_second, begin_note, 
        classes_num, uniform_max):
        """Class for processing MIDI events to target.

        Args:
          segment_seconds: float
          frames_per_second: int
          begin_note: int, A0 MIDI note of a piano
          classes_num: int
        """
        self.segment_seconds = segment_seconds
        self.frames_per_second = frames_per_second
        self.begin_note = begin_note
        self.classes_num = classes_num
        self.max_piano_note = self.classes_num - 1
        self.random_state = np.random.RandomState(1234)
        self.uniform_max = uniform_max

    def process(self, start_time, midi_events_time, midi_events, 
        extend_pedal=True, note_shift=0):
        """Process MIDI events of an audio segment to target for training, 
        includes: 
        1. Parse MIDI events
        2. Get note targets
        3. Get pedal targets

        Args:
          start_time: float, start time of a segment
          midi_events_time: list of float, times of MIDI events of a recording, 
            e.g. [0, 3.3, 5.1, ...]
          midi_events: list of str, MIDI events of a recording, e.g.
            ['note_on channel=0 note=75 velocity=37 time=14',
             'control_change channel=0 control=64 value=54 time=20',
             ...]
          extend_pedal, bool, True: Notes will be set to ON until pedal is 
            released. False: Ignore pedal events.

        Returns:
          target_dict: {
            'onset_roll': (frames_num, classes_num), 
            'offset_roll': (frames_num, classes_num), 
            'reg_onset_roll': (frames_num, classes_num), 
            'reg_offset_roll': (frames_num, classes_num), 
            'frame_roll': (frames_num, classes_num), 
            'velocity_roll': (frames_num, classes_num), 
            'mask_roll':  (frames_num, classes_num), 
            'reg_pedal_onset_roll': (frames_num,),
            'reg_pedal_offset_roll': (frames_num,),
            'pedal_frame_roll': (frames_num,)}

          note_events: list of dict, e.g. [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]
        """

        # ------ 1. Parse MIDI events ------
        # Search the begin index of a segment
        for bgn_idx, event_time in enumerate(midi_events_time):
            if event_time > start_time:
                break
        """E.g., start_time: 709.0, bgn_idx: 18003, event_time: 709.0146"""

        # Search the end index of a segment
        for fin_idx, event_time in enumerate(midi_events_time):
            if event_time > start_time + self.segment_seconds:
                break
        """E.g., start_time: 709.0, bgn_idx: 18196, event_time: 719.0115"""

        note_events = []
        """E.g. [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]"""

        pedal_events = []
        """E.g. [
            {'onset_time': 696.46875, 'offset_time': 696.62604}, 
            {'onset_time': 696.8063, 'offset_time': 698.50836}, 
            ...]"""

        buffer_dict = {}    # Used to store onset of notes to be paired with offsets
        pedal_dict = {}     # Used to store onset of pedal to be paired with offset of pedal

        # Backtrack bgn_idx to earlier indexes: ex_bgn_idx, which is used for 
        # searching cross segment pedal and note events. E.g.: bgn_idx: 1149, 
        # ex_bgn_idx: 981
        _delta = int((fin_idx - bgn_idx) * 1.)  
        ex_bgn_idx = max(bgn_idx - _delta, 0)
        
        for i in range(ex_bgn_idx, fin_idx):
            # Parse MIDI messiage
            attribute_list = midi_events[i].split(' ')

            # Note
            if attribute_list[0] in ['note_on', 'note_off']:
                """E.g. attribute_list: ['note_on', 'channel=0', 'note=41', 'velocity=0', 'time=10']"""

                midi_note = int(attribute_list[2].split('=')[1])
                velocity = int(attribute_list[3].split('=')[1])

                # Onset
                if attribute_list[0] == 'note_on' and velocity > 0:
                    buffer_dict[midi_note] = {
                        'onset_time': midi_events_time[i], 
                        'velocity': velocity}

                # Offset
                else:
                    if midi_note in buffer_dict.keys():
                        note_events.append({
                            'midi_note': midi_note, 
                            'onset_time': buffer_dict[midi_note]['onset_time'], 
                            'offset_time': midi_events_time[i], 
                            'velocity': buffer_dict[midi_note]['velocity']})
                        del buffer_dict[midi_note]

            # Pedal
            elif attribute_list[0] == 'control_change' and attribute_list[2] == 'control=64':
                """control=64 corresponds to pedal MIDI event. E.g. 
                attribute_list: ['control_change', 'channel=0', 'control=64', 'value=45', 'time=43']"""

                ped_value = int(attribute_list[3].split('=')[1])
                if ped_value >= 64:
                    if 'onset_time' not in pedal_dict:
                        pedal_dict['onset_time'] = midi_events_time[i]
                else:
                    if 'onset_time' in pedal_dict:
                        pedal_events.append({
                            'onset_time': pedal_dict['onset_time'], 
                            'offset_time': midi_events_time[i]})
                        pedal_dict = {}

        # Add unpaired onsets to events
        for midi_note in buffer_dict.keys():
            note_events.append({
                'midi_note': midi_note, 
                'onset_time': buffer_dict[midi_note]['onset_time'], 
                'offset_time': start_time + self.segment_seconds, 
                'velocity': buffer_dict[midi_note]['velocity']})

        # Add unpaired pedal onsets to data
        if 'onset_time' in pedal_dict.keys():
            pedal_events.append({
                'onset_time': pedal_dict['onset_time'], 
                'offset_time': start_time + self.segment_seconds})

        # Set notes to ON until pedal is released
        if extend_pedal:
            note_events = self.extend_pedal(note_events, pedal_events)
        
        # Prepare targets
        frames_num = int(round(self.segment_seconds * self.frames_per_second)) + 1
        onset_roll = np.zeros((frames_num, self.classes_num))
        offset_roll = np.zeros((frames_num, self.classes_num))
        reg_onset_roll = np.ones((frames_num, self.classes_num))
        reg_offset_roll = np.ones((frames_num, self.classes_num))
        frame_roll = np.zeros((frames_num, self.classes_num))
        velocity_roll = np.zeros((frames_num, self.classes_num))
        mask_roll = np.ones((frames_num, self.classes_num))
        """mask_roll is used for masking out cross segment notes"""

        reg_pedal_onset_roll = np.ones(frames_num)
        reg_pedal_offset_roll = np.ones(frames_num)
        pedal_frame_roll = np.zeros(frames_num)

        # ------ 2. Get note targets ------
        # Process note events to target
        for note_event in note_events:
            """note_event: e.g., {'midi_note': 60, 'onset_time': 722.0719, 'offset_time': 722.47815, 'velocity': 103}"""

            piano_note = np.clip(note_event['midi_note'] - self.begin_note + note_shift, 0, self.max_piano_note) 
            """There are 88 keys on a piano"""

            if 0 <= piano_note <= self.max_piano_note:
                bgn_frame = int(round((note_event['onset_time'] - start_time) * self.frames_per_second))
                fin_frame = int(round((note_event['offset_time'] - start_time) * self.frames_per_second))

                bgn_frame += self.random_state.randint(low=-self.uniform_max, high=self.uniform_max + 1, size=1)[0]
                fin_frame += self.random_state.randint(low=-self.uniform_max, high=self.uniform_max + 1, size=1)[0]
                bgn_frame = np.clip(bgn_frame, 0, frames_num - 1)
                fin_frame = np.clip(fin_frame, 0, frames_num - 1)

                if fin_frame >= 0:
                    frame_roll[max(bgn_frame, 0) : fin_frame + 1, piano_note] = 1

                    offset_roll[fin_frame, piano_note] = 1
                    velocity_roll[max(bgn_frame, 0) : fin_frame + 1, piano_note] = note_event['velocity']

                    # Vector from the center of a frame to ground truth offset
                    reg_offset_roll[fin_frame, piano_note] = \
                        (note_event['offset_time'] - start_time) - (fin_frame / self.frames_per_second)

                    if bgn_frame >= 0:
                        onset_roll[bgn_frame, piano_note] = 1

                        # Vector from the center of a frame to ground truth onset
                        reg_onset_roll[bgn_frame, piano_note] = \
                            (note_event['onset_time'] - start_time) - (bgn_frame / self.frames_per_second)
                
                    # Mask out segment notes
                    else:
                        mask_roll[: fin_frame + 1, piano_note] = 0

        for k in range(self.classes_num):
            """Get regression targets"""
            reg_onset_roll[:, k] = self.get_regression(reg_onset_roll[:, k])
            reg_offset_roll[:, k] = self.get_regression(reg_offset_roll[:, k])

        # Process unpaired onsets to target
        for midi_note in buffer_dict.keys():
            piano_note = np.clip(midi_note - self.begin_note + note_shift, 0, self.max_piano_note)
            if 0 <= piano_note <= self.max_piano_note:
                bgn_frame = int(round((buffer_dict[midi_note]['onset_time'] - start_time) * self.frames_per_second))
                mask_roll[bgn_frame :, piano_note] = 0     

        # ------ 3. Get pedal targets ------
        # Process pedal events to target
        for pedal_event in pedal_events:
            bgn_frame = int(round((pedal_event['onset_time'] - start_time) * self.frames_per_second))
            fin_frame = int(round((pedal_event['offset_time'] - start_time) * self.frames_per_second))

            bgn_frame += self.random_state.randint(low=-self.uniform_max, high=self.uniform_max + 1, size=1)[0]
            fin_frame += self.random_state.randint(low=-self.uniform_max, high=self.uniform_max + 1, size=1)[0]
            bgn_frame = np.clip(bgn_frame, 0, frames_num - 1)
            fin_frame = np.clip(fin_frame, 0, frames_num - 1)

            if fin_frame >= 0:
                pedal_frame_roll[max(bgn_frame, 0) : fin_frame + 1] = 1

                reg_pedal_offset_roll[fin_frame] = \
                    (pedal_event['offset_time'] - start_time) - (fin_frame / self.frames_per_second)

                if bgn_frame >= 0:
                    reg_pedal_onset_roll[bgn_frame] = \
                        (pedal_event['onset_time'] - start_time) - (bgn_frame / self.frames_per_second)

        # Get regresssion padal targets
        reg_pedal_onset_roll = self.get_regression(reg_pedal_onset_roll)
        reg_pedal_offset_roll = self.get_regression(reg_pedal_offset_roll)

        target_dict = {
            'onset_roll': onset_roll, 'offset_roll': offset_roll,
            'reg_onset_roll': reg_onset_roll, 'reg_offset_roll': reg_offset_roll,
            'frame_roll': frame_roll, 'velocity_roll': velocity_roll, 
            'mask_roll': mask_roll, 'reg_pedal_onset_roll': reg_pedal_onset_roll, 
            'reg_pedal_offset_roll': reg_pedal_offset_roll, 'pedal_frame_roll': pedal_frame_roll
            }

        return target_dict, note_events, pedal_events

    def extend_pedal(self, note_events, pedal_events):
        """Update the offset of all notes until pedal is released.

        Args:
          note_events: list of dict, e.g., [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]
          pedal_events: list of dict, e.g., [
            {'onset_time': 696.46875, 'offset_time': 696.62604}, 
            {'onset_time': 696.8063, 'offset_time': 698.50836}, 
            ...]

        Returns:
          ex_note_events: list of dict, e.g., [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]
        """
        note_events = collections.deque(note_events)
        pedal_events = collections.deque(pedal_events)
        ex_note_events = []

        idx = 0     # Index of note events
        while pedal_events: # Go through all pedal events
            pedal_event = pedal_events.popleft()
            buffer_dict = {}    # keys: midi notes, value for each key: event index

            while note_events:
                note_event = note_events.popleft()

                # If a note offset is between the onset and offset of a pedal, 
                # Then set the note offset to when the pedal is released.
                if pedal_event['onset_time'] < note_event['offset_time'] < pedal_event['offset_time']:
                    
                    midi_note = note_event['midi_note']

                    if midi_note in buffer_dict.keys():
                        """Multiple same note inside a pedal"""
                        _idx = buffer_dict[midi_note]
                        del buffer_dict[midi_note]
                        ex_note_events[_idx]['offset_time'] = note_event['onset_time']

                    # Set note offset to pedal offset
                    note_event['offset_time'] = pedal_event['offset_time']
                    buffer_dict[midi_note] = idx
                
                ex_note_events.append(note_event)
                idx += 1

                # Break loop and pop next pedal
                if note_event['offset_time'] > pedal_event['offset_time']:
                    break

        while note_events:
            """Append left notes"""
            ex_note_events.append(note_events.popleft())

        return ex_note_events

    def get_regression(self, input):
        """Get regression target. See Fig. 2 of [1] for an example.
        [1] Q. Kong, et al., High resolution piano transcription by 
            regressing onset and offset time stamps, 2020.

        input:
          input: (frames_num,)

        Returns: (frames_num,), e.g., [0, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.9, 0.7, 0.5, 0.3, 0.1, 0, 0, ...]
        """
        step = 1. / self.frames_per_second
        output = np.ones_like(input)
        
        locts = np.where(input < 0.5)[0] 
        if len(locts) > 0:
            for t in range(0, locts[0]):
                output[t] = step * (t - locts[0]) - input[locts[0]]

            for i in range(0, len(locts) - 1):
                for t in range(locts[i], (locts[i] + locts[i + 1]) // 2):
                    output[t] = step * (t - locts[i]) - input[locts[i]]

                for t in range((locts[i] + locts[i + 1]) // 2, locts[i + 1]):
                    output[t] = step * (t - locts[i + 1]) - input[locts[i]]

            for t in range(locts[-1], len(input)):
                output[t] = step * (t - locts[-1]) - input[locts[-1]]

        output = np.clip(np.abs(output), 0., 0.05) * 20
        output = (1. - output)

        return output


class RandomMaestroDataset(object):
    def __init__(self, hdf5s_dir, segment_seconds, frames_per_second, 
        max_note_shift=0, augmentor=None, uniform_max=0):
        """Maestro dataset. Will be used for DataLoader.

        Args:
          feature_hdf5s_dir: str
          segment_seconds: float
          frames_per_second: int
          max_note_shift: int, number of semitone for pitch augmentation
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

        self.target_processor = RandomTargetProcessor(self.segment_seconds, 
            self.frames_per_second, self.begin_note, self.classes_num, uniform_max)
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
            'pedal_roll': (frames_num,)}
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
                    note_shift, bins_per_octave=12)#, res_type='kaiser_best')

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
        self.sample_rate = config.sample_rate
        self.random_state = np.random.RandomState(1234)

    def augment(self, x):
        clip_samples = len(x)

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
            batch_size, training, mini_data, random_seed=1234):
        """Sampler is used to sample segments for training or evaluation.

        Args:
          hdf5s_dir: str
          split: 'train' | 'validation' | 'test'
          segment_seconds: float
          hop_seconds: float
          batch_size: int
          training: bool, True: sample segments for training; False: sample 
            segments for evaluation
          mini_data: bool, sample from a small amount of data for debugging
        """
        assert split in ['train', 'validation', 'test']
        self.hdf5s_dir = hdf5s_dir
        self.training = training
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
            batch_size, training, mini_data, random_seed=1234):
        """Sampler is used to sample segments for training or evaluation.

        Args:
          hdf5s_dir: str
          split: 'train' | 'validation' | 'test'
          segment_seconds: float
          hop_seconds: float
          batch_size: int
          training: bool, True: sample segments for training; False: sample 
            segments for evaluation
          mini_data: bool, sample from a small amount of data for debugging
        """
        assert split in ['train', 'validation', 'test']
        self.hdf5s_dir = hdf5s_dir
        self.training = training
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