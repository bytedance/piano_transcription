import os
import logging
import h5py
import soundfile
import librosa
import numpy as np
import pandas as pd
from scipy import stats 
import datetime
import collections
import _pickle as cPickle

from piano_vad import note_detection_with_onset
import config


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
        
def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na


def traverse_folder(folder):
    paths = []
    names = []
    
    for root, dirs, files in os.walk(folder):
        for name in files:
            filepath = os.path.join(root, name)
            names.append(name)
            paths.append(filepath)
            
    return names, paths


def note_to_freq(piano_note):
    return 2 ** ((piano_note - 39) / 12) * 440

    
def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1
        
    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging


def float32_to_int16(x):
    assert np.max(np.abs(x)) <= 1.
    return (x * 32767.).astype(np.int16)


def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)
    

def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]


class TargetProcessor(object):
    def __init__(self, segment_seconds, frames_per_second, begin_note, 
        classes_num):
        """Class for processing MIDI events to target.

        Args:
          segment_seconds: float
          frames_per_second: int
          begin_note: int
          classes_num: int
        """
        self.segment_seconds = segment_seconds
        self.frames_per_second = frames_per_second
        self.begin_note = begin_note
        self.classes_num = classes_num

    def process(self, start_time, midi_events_time, midi_events, 
        extend_pedal=True):
        """Process midi events to target for training.

        Args:
          start_time: float, start time of a segment in an audio clip in seconds
          midi_events_time: list of float, e.g. [0, 3.3, 5.1, ...]
          midi_events: list of str, e.g.
            ['note_on channel=0 note=75 velocity=37 time=14',
             'control_change channel=0 control=64 value=54 time=20',
             ...]
          extend_pedal, bool, True: Notes will be set to ON until pedal is 
            released. False: Ignore pedal events.

        Returns:
          target_dict: {
            'frame_roll': (frames_num, classes_num), 
            'onset_roll': (frames_num, classes_num), 
            'offset_roll': (frames_num, classes_num), 
            'distance_roll': (frames_num, classes_num), 
            'velocity_roll': (frames_num, classes_num), 
            'mask_roll':  (frames_num, classes_num), 
            'pedal_roll': (frames_num,)}

          note_events: list of dict, e.g. [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]
        """

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

        _delta = int((fin_idx - bgn_idx) * 1.)  
        ex_bgn_idx = max(bgn_idx - _delta, 0)
        """Backtrack bgn_idx to earlier indexes. This is used for searching 
        cross segment pedal and note events."""

        for i in range(ex_bgn_idx, fin_idx):

            # Parse MIDI messiage
            attribute_list = midi_events[i].split(' ')

            # Note
            if attribute_list[0] == 'note_on':
                midi_note = int(attribute_list[2].split('=')[1])
                velocity = int(attribute_list[3].split('=')[1])

                # Onset
                if velocity > 0:
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
        frame_roll = np.zeros((frames_num, self.classes_num))
        onset_roll = np.zeros((frames_num, self.classes_num))
        offset_roll = np.zeros((frames_num, self.classes_num))
        distance_roll = np.zeros((frames_num, self.classes_num))
        velocity_roll = np.zeros((frames_num, self.classes_num))
        mask_roll = np.ones((frames_num, self.classes_num))
        """mask_roll is used for masking out cross segment notes"""

        pedal_roll = np.zeros(frames_num)

        # Process note events to target
        for note_event in note_events:
            pitch = note_event['midi_note'] - self.begin_note
            bgn_frame = int(round((note_event['onset_time'] - start_time) * self.frames_per_second))
            fin_frame = int(round((note_event['offset_time'] - start_time) * self.frames_per_second))

            if fin_frame >= 0:
                frame_roll[max(bgn_frame, 0) : fin_frame + 1, pitch] = 1

                offset_roll[fin_frame, pitch] = 1
                velocity_roll[max(bgn_frame, 0) : fin_frame + 1, pitch] = note_event['velocity']

                if bgn_frame >= 0:
                    onset_roll[bgn_frame, pitch] = 1
                    distance_roll[bgn_frame : fin_frame + 1, pitch] = \
                        np.exp(-0.01 * np.arange(fin_frame - bgn_frame + 1))
                # Cross segment notes
                else:
                    distance_roll[: fin_frame + 1, pitch] = \
                        np.exp(-0.01 * np.arange(fin_frame - bgn_frame + 1)[-bgn_frame :])
                    mask_roll[: fin_frame + 1, pitch] = 0

        # Process unpaired onsets to target
        for midi_note in buffer_dict.keys():
            piano_note = midi_note - self.begin_note
            bgn_frame = int(round((buffer_dict[midi_note]['onset_time'] - start_time) * self.frames_per_second))
            mask_roll[bgn_frame :, piano_note] = 0     

        # Process pedal events to target
        for pedal_event in pedal_events:
            bgn_frame = int(round((pedal_event['onset_time'] - start_time) * self.frames_per_second))
            fin_frame = int(round((pedal_event['offset_time'] - start_time) * self.frames_per_second))
            if fin_frame >= 0:
                bgn_frame = max(bgn_frame, 0)
                pedal_roll[bgn_frame : fin_frame + 1] = 1

        target_dict = {
            'frame_roll': frame_roll, 'onset_roll': onset_roll, 
            'offset_roll': offset_roll, 'distance_roll': distance_roll,
            'velocity_roll': velocity_roll, 'mask_roll': mask_roll, 
            'pedal_roll': pedal_roll}

        return target_dict, note_events

    def extend_pedal(self, note_events, pedal_events):
        """Set notes to ON until pedal is released.

        Args:
          note_events: list of dict, e.g. [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]
          pedal_events: list of dict, e.g. [
            {'onset_time': 696.46875, 'offset_time': 696.62604}, 
            {'onset_time': 696.8063, 'offset_time': 698.50836}, 
            ...]

        Returns:
          ex_note_events: list of dict
        """
        note_events = collections.deque(note_events)
        pedal_events = collections.deque(pedal_events)

        ex_note_events = []

        # Go through all pedal events
        while pedal_events:
            pedal_event = pedal_events.popleft()

            while note_events:
                note_event = note_events.popleft()

                # Set note offset to when pedal is released.
                if pedal_event['onset_time'] < note_event['offset_time'] and \
                    note_event['offset_time'] < pedal_event['offset_time']:
                    
                    note_event['offset_time'] = pedal_event['offset_time']
                
                ex_note_events.append(note_event)

                # Break loop and pop next pedal
                if note_event['offset_time'] > pedal_event['offset_time']:
                    break

        while note_events:
            ex_note_events.append(note_events.popleft())

        return ex_note_events


def write_events_to_midi(start_time, note_events, midi_path):
    """Write out note events to MIDI file.

    Args:
      start_time: float
      note_events: list of dict, e.g. [
        {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
        {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
        ...]
      midi_path: str
    """
    from mido import Message, MidiFile, MidiTrack
    
    # This configuration is the same as MIDIs in MAESTRO dataset
    ticks_per_beat = 384
    beats_per_second = 2
    ticks_per_second = ticks_per_beat * beats_per_second

    midi_file = MidiFile()
    midi_file.ticks_per_beat = ticks_per_beat
    track = MidiTrack()
    midi_file.tracks.append(track)
    
    # Message rolls of MIDI
    message_roll = []

    for note_event in note_events:
        # Onset
        message_roll.append({'midi_note': note_event['midi_note'], 
            'time': note_event['onset_time'], 'velocity': note_event['velocity']})

        # Offset
        message_roll.append({'midi_note': note_event['midi_note'], 
            'time': note_event['offset_time'], 'velocity': 0})

    # Sort MIDI messages by time
    message_roll.sort(key=lambda note_event: note_event['time'])

    previous_ticks = 0
    for message in message_roll:
        this_ticks = int((message['time'] - start_time) * ticks_per_second)
        if this_ticks >= 0:
            diff_ticks = this_ticks - previous_ticks
            previous_ticks = this_ticks
            track.append(Message('note_on', note=message['midi_note'], velocity=message['velocity'], time=diff_ticks))

    midi_file.save(midi_path)


def sharp_output(input, threshold=0.3):
    """Used for sharping onset or offset. E.g. when threshold=0.3, for a note, 
    [0, 0.1, 0.4, 0.7, 0, 0] will be sharped to [0, 0, 0, 1, 0, 0]

    Args:
      input: (frames_num, classes_num)

    Returns:
      output: (frames_num, classes_num)
    """
    (frames_num, classes_num) = input.shape
    output = np.zeros_like(input)

    for piano_note in range(classes_num):
        loct = None
        value = -1
        for i in range(frames_num):
            if input[i, piano_note] > threshold and input[i, piano_note] > value:
                loct = i
                value = input[i, piano_note]
            else:
                if loct:
                    output[loct, piano_note] = 1
                    loct = None
                    value = -1

    return output


def sharp_output3d(input, threshold):
    """Used for sharping onset or offset. E.g. when threshold=0.3, for a note, 
    [0, 0.1, 0.4, 0.7, 0, 0] will be sharped to [0, 0, 0, 1, 0, 0]

    Args:
      input: (N, frames_num, classes_num)

    Returns:
      output: (N, frames_num, classes_num)
    """
    return np.array([sharp_output(x, threshold) for x in input])


class PostProcessor(object):
    def __init__(self, frames_per_second, classes_num):
        """Postprocess the system output.

        Args:
          frames_per_second: int
          classes_num: int
        """
        self.frames_per_second = frames_per_second
        self.classes_num = classes_num
        self.begin_note = config.begin_note

    def output_dict_to_piano_notes(self, output_dict, frame_threshold):
        """Postprocess output_dict to piano notes.

        Args:
          output_dict: dict, e.g. {
            'frame_output': (frames_num, classes_num),
            'onset_output': (frames_num, classes_num),
            ...}

        Returns:
          est_pairs: (notes_num, 2), time of onsets and offsets in seconds. E.g.
            [[114.05, 114.44],
             [ 97.86,  98.1 ],
             ...]
          est_piano_notes: [8, 9, ...]
        """
        est_on_off_pairs = []
        est_piano_notes = []

        for piano_note in range(self.classes_num):
            
            bgn_fin_pairs = note_detection_with_onset(
                frame_output=output_dict['frame_output'][:, piano_note], 
                onset_output=output_dict['onset_output'][:, piano_note], 
                threshold=frame_threshold)
            est_on_off_pairs += bgn_fin_pairs
            est_piano_notes += [piano_note] * len(bgn_fin_pairs)

        est_on_off_pairs = np.array(est_on_off_pairs) / self.frames_per_second
        est_piano_notes = np.array(est_piano_notes)

        return est_on_off_pairs, est_piano_notes

    def on_off_pairs_notes_to_midi_events(self, est_on_off_pairs, est_piano_notes):
        """Combine on and off pairs and piano notes to midi events
        """
        midi_events = []
        for i in range(est_on_off_pairs.shape[0]):
            midi_events.append({'midi_note': est_piano_notes[i] + self.begin_note, 
                'onset_time': est_on_off_pairs[i][0], 
                'offset_time': est_on_off_pairs[i][1], 
                'velocity': 100})

        return midi_events
    

class StatisticsContainer(object):
    def __init__(self, statistics_path):
        self.statistics_path = statistics_path

        self.backup_statistics_path = '{}_{}.pickle'.format(
            os.path.splitext(self.statistics_path)[0], 
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        self.statistics_dict = {'train': [], 'validation': [], 'test': []}

    def append(self, iteration, statistics, data_type):
        statistics['iteration'] = iteration
        self.statistics_dict[data_type].append(statistics)
        
    def dump(self):
        cPickle.dump(self.statistics_dict, open(self.statistics_path, 'wb'))
        cPickle.dump(self.statistics_dict, open(self.backup_statistics_path, 'wb'))
        logging.info('    Dump statistics to {}'.format(self.statistics_path))
        logging.info('    Dump statistics to {}'.format(self.backup_statistics_path))
        
    def load_state_dict(self, resume_iteration):
        self.statistics_dict = cPickle.load(open(self.statistics_path, 'rb'))

        resume_statistics_dict = {'train': [], 'validation': [], 'test': []}
        
        for key in self.statistics_dict.keys():
            for statistics in self.statistics_dict[key]:
                if statistics['iteration'] <= resume_iteration:
                    resume_statistics_dict[key].append(statistics)
                
        self.statistics_dict = resume_statistics_dict


