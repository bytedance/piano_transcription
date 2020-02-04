import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import torch
import h5py
import mir_eval
import librosa
import logging
from sklearn import metrics

from utilities import (create_folder, traverse_folder, int16_to_float32, 
    pad_truncate_sequence, TargetProcessor, sharp_output, sharp_output3d, 
    PostProcessor, write_events_to_midi, note_to_freq)
from pytorch_utils import forward_dataloader, forward, WaveformTester
from piano_vad import note_detection_with_onset
import config


def f1_score(target, output, mask, threshold=0.5):
    binarized_output = (np.sign(output - threshold) + 1) / 2
    binarized_output[np.where(binarized_output == 0.5)] = 1
    target *= mask
    binarized_output *= mask
    
    return metrics.f1_score(target.flatten(), binarized_output.flatten())
    

class SegmentEvaluator(object):
    def __init__(self, model, batch_size):
        """Evaluate segment-wise metrics.

        Args:
          model: object
          batch_size: int
        """
        self.model = model
        self.batch_size = batch_size
        self.frame_threshold = 0.3
        self.onset_threshold = 0.3
        self.offset_threshold = 0.3

    def evaluate(self, dataloader):
        """Evaluate over a few mini-batches.

        Args:
          dataloader: object, used to generate mini-batches for evaluation.

        Returns:
          statistics: dict, e.g. {
            'frame_f1': 0.800, 
            (if exist) 'onset_f1': 0.500, 
            (if exist) 'offset_f1': 0.300}
        """

        statistics = {}
        output_dict = forward_dataloader(self.model, dataloader, self.batch_size)
        
        # Sharp onsets and offsets
        if 'onset_output' in output_dict.keys():
            output_dict['onset_output'] = sharp_output3d(
                output_dict['onset_output'], 
                threshold=self.onset_threshold)

        if 'offset_output' in output_dict.keys():
            output_dict['offset_output'] = sharp_output3d(
                output_dict['offset_output'], 
                threshold=self.offset_threshold)

        # Frame and onset evaluation
        statistics['frame_f1'] = f1_score(output_dict['frame_roll'], 
            output_dict['frame_output'], output_dict['mask_roll'], 
            self.frame_threshold)

        if 'onset_output' in output_dict.keys():
            statistics['onset_f1'] = f1_score(output_dict['onset_roll'], 
                output_dict['onset_output'], output_dict['mask_roll'], 
                self.onset_threshold)
        
        if 'offset_output' in output_dict.keys():
            statistics['offset_f1'] = f1_score(output_dict['offset_roll'], 
                output_dict['offset_output'], output_dict['mask_roll'], 
                self.offset_threshold)

        return statistics


class Evaluator(object):
    def __init__(self, model, feature_hdf5s_dir, segment_seconds, 
        frames_per_second, batch_size):
        """Class for evaluating transcription performance on music pieces.

        Args:
          model: object
          feature_hdf5s_dir: str
          segment_seconds: float
          frames_per_second: int
          batch_size: int
        """
        self.model = model
        self.feature_hdf5s_dir = feature_hdf5s_dir
        self.segment_seconds = segment_seconds
        self.frames_per_second = frames_per_second
        self.batch_size = batch_size
        
        self.sample_rate = config.sample_rate
        self.segment_samples = int(self.segment_seconds * self.sample_rate)
        self.hop_samples = self.segment_samples // 2
        self.begin_note = config.begin_note
        self.classes_num = config.classes_num

        self.frame_threshold = 0.3
        self.onset_threshold = 0.3
        self.offset_threshold = 0.3

        self.waveform_tester = WaveformTester(model, self.segment_samples, batch_size)
        self.post_processor = PostProcessor(frames_per_second, self.classes_num)

    def evaluate(self, split):
        """Evaluate all music pieces in a subset.

        Args:
          split: 'train' | 'validation' | 'test'
        """
        assert split in ['train', 'validation', 'test']

        (hdf5_names, hdf5_paths) = traverse_folder(self.feature_hdf5s_dir)

        statistics = {'frame_f1': [], 'onset_f1': [], 'offset_f1': [], 
            'tolerant_onset_f1': [], 'tolerant_onset_precision': [], 
            'tolerant_onset_recall': [], 'tolerant_offset_precision': [], 
            'tolerant_offset_recall': [], 'tolerant_offset_f1': []}
            
        n = 0
        for n, hdf5_path in enumerate(hdf5_paths):
            with h5py.File(hdf5_path, 'r') as hf:
                if hf.attrs['split'].decode() == split:
                    print(n, hdf5_path)
                    n += 1

                    # Load hdf5
                    year = hf.attrs['year'].decode()
                    waveform = int16_to_float32(hf['waveform'][:])
                    midi_events = [e.decode() for e in hf['midi_event'][:]]
                    midi_events_time = hf['midi_event_time'][:]
            
                    # Read MIDI file
                    segment_seconds = len(waveform) / self.sample_rate
                    self.target_processor = TargetProcessor(segment_seconds, 
                        self.frames_per_second, self.begin_note, self.classes_num)

                    start_time = 0
                    (target_dict, ref_events) = self.target_processor.process(
                        start_time, midi_events_time, midi_events)

                    ref_on_off_pairs = np.array([[event['onset_time'], 
                        event['offset_time']] for event in ref_events])

                    ref_piano_notes = np.array([
                        event['midi_note'] - self.begin_note for event in ref_events])

                    # Inference
                    output_dict = self.waveform_tester.forward(waveform)

                    # Truncate output to the same length as target                    
                    for key in output_dict.keys():
                        output_dict[key] = pad_truncate_sequence(
                            output_dict[key], len(target_dict['frame_roll']))

                    # Sharp onsets and offsets
                    if 'onset_output' in output_dict.keys():
                        output_dict['onset_output'] = sharp_output(
                            output_dict['onset_output'], 
                            threshold=self.onset_threshold)

                    if 'offset_output' in output_dict.keys():
                        output_dict['offset_output'] = sharp_output(
                            output_dict['offset_output'], 
                            threshold=self.offset_threshold)

                    # Frame and onset evaluation
                    frame_f1 = f1_score(target_dict['frame_roll'], 
                        output_dict['frame_output'], target_dict['mask_roll'], 
                        self.frame_threshold)
                    statistics['frame_f1'].append(frame_f1)

                    if 'onset_output' in output_dict.keys():
                        onset_f1 = f1_score(target_dict['onset_roll'], 
                            output_dict['onset_output'], target_dict['mask_roll'])
                        statistics['onset_f1'].append(onset_f1)
                    
                    if 'offset_output' in output_dict.keys():
                        offset_f1 = f1_score(target_dict['offset_roll'], 
                            output_dict['offset_output'], target_dict['mask_roll'])
                        ['offset_f1'].append(offset_f1)

                    # Post process output_dict to piano notes
                    (est_on_off_pairs, est_piano_notes) = self.post_processor.\
                        output_dict_to_piano_notes(output_dict, self.frame_threshold)

                    # Evaluate with mir_eval toolbox
                    note_precision, note_recall, note_f1, _ = \
                        mir_eval.transcription.precision_recall_f1_overlap(
                            ref_on_off_pairs, note_to_freq(ref_piano_notes), 
                            est_on_off_pairs, note_to_freq(est_piano_notes), 
                            offset_ratio=None)

                    statistics['tolerant_onset_f1'].append(note_f1)
                    statistics['tolerant_onset_precision'].append(note_precision)
                    statistics['tolerant_onset_recall'].append(note_recall)

                    # Evaluate with offset
                    if False:
                        note_off_precision, note_off_recall, note_off_f1, _ = \
                            mir_eval.transcription.precision_recall_f1_overlap(
                                ref_on_off_pairs, note_to_freq(ref_piano_notes), 
                                est_on_off_pairs, note_to_freq(est_piano_notes), 
                                offset_ratio=0.2)

                        statistics['tolerant_offset_f1'].append(note_off_f1)
                        statistics['tolerant_offset_precision'].append(note_off_precision)
                        statistics['tolerant_offset_recall'].append(note_off_recall)
                    
                    # Plot statistics
                    string = ''
                    for key in statistics.keys():
                        if len(statistics[key]) > 0:
                            string += '{}: {:.3f},'.format(key, statistics[key][-1])
                    logging.info(string)

                    # Debug
                    if False:
                        self._debug_write_midi(est_on_off_pairs, est_piano_notes)
                        exit()

                    if False:
                        self._debug_plot_onset(waveform, output_dict, 
                            ref_on_off_pairs, ref_piano_notes)
                        exit()

        return statistics

    def _debug_write_midi(self, est_on_off_pairs, est_piano_notes):
        """Write estimated on_off_pairs and piano_notes to MIDI.
        """
        est_events = []
        for i in range(est_on_off_pairs.shape[0]):
            est_events.append({'midi_note': est_piano_notes[i] + config.begin_note, 
                'onset_time': est_on_off_pairs[i][0], 
                'offset_time': est_on_off_pairs[i][1], 
                'velocity': 100})

        create_folder('debug')
        midi_path = 'debug/debug2.mid'
        write_events_to_midi(0, est_events, midi_path)
        print('Write to {}'.format(midi_path))

    def _debug_plot_onset(self, waveform, output_dict, ref_on_off_pairs, 
        ref_piano_notes):
        """Plot onset_roll and ref_onset_roll, write audio clip.
        """

        import matplotlib.pyplot as plt
        bgn = 6800  # Begin frame in a waveform
        L = 200     # Clip frames to be plotted
        
        create_folder('debug')
        fig_path = 'debug/debug3.pdf'
        wav_path = 'debug/debug3.wav'

        ref_onset_roll = np.zeros_like(output_dict['frame_output'])
        
        for i in range(len(ref_piano_notes)):
            ref_onset_roll[int(ref_on_off_pairs[i][0] * self.frames_per_second), ref_piano_notes[i]] = 1

        plt.figure(figsize=(10, 12))
        fig, axs = plt.subplots(3, 1, sharex=True)
        axs[0].matshow(output_dict['frame_output'][bgn : bgn + L].T, origin='lower', aspect='auto', cmap='jet', vmin=-1, vmax=1)
        axs[1].matshow(output_dict['onset_output'][bgn : bgn + L].T, origin='lower', aspect='auto', cmap='jet', vmin=-1, vmax=1)
        axs[2].matshow(ref_onset_roll[bgn : bgn + L].T, origin='lower', aspect='auto', cmap='jet', vmin=-1, vmax=1)
        axs[0].xaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.2)
        axs[1].xaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.2)
        axs[2].xaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.2)
        axs[0].xaxis.set_ticks(np.arange(0, L))
        axs[1].xaxis.set_ticks(np.arange(0, L))
        axs[2].xaxis.set_ticks(np.arange(0, L))
        axs[0].xaxis.set_ticklabels([])
        axs[1].xaxis.set_ticklabels([])
        axs[2].xaxis.set_ticklabels([])
        axs[0].set_title('frame_output')
        axs[1].set_title('onset_output')
        axs[2].set_title('ref_onset_roll')
        plt.tight_layout()
        plt.savefig(fig_path)

        bgn_sample = int(bgn / self.frames_per_second * self.sample_rate)
        fin_sample = bgn_sample + int(L / self.frames_per_second * self.sample_rate)
        clip = waveform[bgn_sample : fin_sample]
        librosa.output.write_wav(wav_path, clip, sr=self.sample_rate)
        
        print('Write out to {}, {}'.format(fig_path, wav_path)) 