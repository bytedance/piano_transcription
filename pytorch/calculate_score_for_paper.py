import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
sys.path.insert(1, os.path.join(sys.path[0], '../../autoth'))
from autoth.core import HyperParamsOptimizer
import numpy as np
import argparse
import librosa
import mir_eval
import torch
import time
import h5py
import pickle 
from sklearn import metrics
from concurrent.futures import ProcessPoolExecutor
 
from utilities import (create_folder, get_filename, traverse_folder, 
    int16_to_float32, TargetProcessor, RegressionPostProcessor, note_to_freq)
import config
from inference import PianoTranscription


def infer_prob(args):
    """Inference output all waveforms of a evaluation dataset.

    Args:
      cuda: bool
      audio_path: str
    """

    # Arugments & parameters
    workspace = args.workspace
    filename = args.filename
    split = args.split
    model_type = args.model_type
    loss_type = args.loss_type
    augmentation = args.augmentation
    max_note_shift = args.max_note_shift
    batch_size = args.batch_size
    iteration = args.iteration
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    
    dataset = 'maestro'
    # if args.maps_dataset:
    #     suffix = '_maps'
    # else:
    #     suffix = ''
    
    sample_rate = config.sample_rate
    segment_seconds = config.segment_seconds
    segment_samples = int(segment_seconds * sample_rate)
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num
    begin_note = config.begin_note

    # Paths
    checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
        model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        '{}_iterations.pth'.format(iteration))

    hdf5s_dir = os.path.join(workspace, 'hdf5s', dataset)

    probs_dir = os.path.join(workspace, 'probs', filename, dataset, 
        'split={}'.format(split), model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
        '{}_iterations'.format(iteration))
    create_folder(probs_dir)

    # Transcriptor
    transcriptor = PianoTranscription(model_type, device=device, 
        checkpoint_path=checkpoint_path, segment_samples=segment_samples)

    (hdf5_names, hdf5_paths) = traverse_folder(hdf5s_dir)

    n = 0
    for n, hdf5_path in enumerate(hdf5_paths):
        with h5py.File(hdf5_path, 'r') as hf:
            if hf.attrs['split'].decode() == split:
                print(n, hdf5_path)
                n += 1

                # Load audio                
                audio = int16_to_float32(hf['waveform'][:])
                midi_events = [e.decode() for e in hf['midi_event'][:]]
                midi_events_time = hf['midi_event_time'][:]
        
                # Get targets
                target_processor = TargetProcessor(
                    segment_seconds=len(audio) / sample_rate, 
                    frames_per_second=frames_per_second, begin_note=begin_note, 
                    classes_num=classes_num)

                (target_dict, note_events) = target_processor.process(
                    start_time=0, midi_events_time=midi_events_time, 
                    midi_events=midi_events, extend_pedal=True)

                ref_on_off_pairs = np.array([[event['onset_time'], event['offset_time']] for event in note_events])
                ref_midi_notes = np.array([event['midi_note'] for event in note_events])
                ref_velocity = np.array([event['velocity'] for event in note_events])

                # Transcribe
                transcribed_dict = transcriptor.transcribe(audio, midi_path=None)
                output_dict = transcribed_dict['output_dict']

                # Data to save out
                total_dict = {key: output_dict[key] for key in output_dict.keys()}
                total_dict['frame_roll'] = target_dict['frame_roll']
                total_dict['ref_on_off_pairs'] = ref_on_off_pairs
                total_dict['ref_midi_notes'] = ref_midi_notes
                total_dict['ref_velocity'] = ref_velocity

                prob_path = os.path.join(probs_dir, '{}.pkl'.format(get_filename(hdf5_path)))
                create_folder(os.path.dirname(prob_path))
                pickle.dump(total_dict, open(prob_path, 'wb'))


class RegressionScoreCalculator(object):
    def __init__(self, hdf5s_dir, probs_dir, split):
        """Evaluate piano transcription metrics of the post processed 
        pre-calculated system outputs.
        """
        self.split = split
        self.probs_dir = probs_dir
        self.frames_per_second = config.frames_per_second
        self.classes_num = config.classes_num
        self.velocity_scale = config.velocity_scale
        self.velocity = True

        self.evaluate_frame = True
        self.onset_tolerance = 0.05
        self.offset_ratio = None
        self.offset_min_tolerance = 0.05
        self.velocity = False
        
        (hdf5_names, self.hdf5_paths) = traverse_folder(hdf5s_dir)

    def __call__(self, params):
        """Calculate metrics of all songs.

        Args:
          params: list of float, thresholds
        """
        stats_dict = self.metrics(params)
        return np.mean(stats_dict['f1'])

    def metrics(self, params):
        """Calculate metrics of all songs.

        Args:
          params: list of float, thresholds
        """

        n = 0
        list_args = []

        for n, hdf5_path in enumerate(self.hdf5_paths):
            with h5py.File(hdf5_path, 'r') as hf:
                if hf.attrs['split'].decode() == self.split:
                    list_args.append([n, hdf5_path, params])
                    """e.g., [0, 'xx.h5', [0.3, 0.3, 0.3]]"""
           
        debug = False
        if debug:     
            for i in range(len(list_args)):
                print(i)
                self.calculate_score_per_song(list_args[i])
            import crash
            asdf

        # Calculate metrics in parallel
        with ProcessPoolExecutor() as exector:
            results = exector.map(self.calculate_score_per_song, list_args)

        stats_list = list(results)
        stats_dict = {}
        for key in stats_list[0].keys():
            stats_dict[key] = [e[key] for e in stats_list]
        
        return stats_dict

    def calculate_score_per_song(self, args):
        """Calculate score per song.

        Args:
          args: [n, hdf5_path, params]
        """
        n = args[0]
        hdf5_path = args[1]
        [onset_threshold, offset_threshod, frame_threshold] = args[2]

        return_dict = {}

        # Load pre-calculated system outputs and ground truths
        prob_path = os.path.join(self.probs_dir, '{}.pkl'.format(get_filename(hdf5_path)))
        total_dict = pickle.load(open(prob_path, 'rb'))

        ref_on_off_pairs = total_dict['ref_on_off_pairs']
        ref_midi_notes = total_dict['ref_midi_notes']
        output_dict = total_dict

        # Calculate frame metric
        if self.evaluate_frame:
            frame_threshold = frame_threshold
            y_pred = (np.sign(total_dict['frame_output'] - frame_threshold) + 1) / 2
            y_pred[np.where(y_pred==0.5)] = 0
            y_true = total_dict['frame_roll']
            y_pred = y_pred[0 : y_true.shape[0], :]
            y_true = y_true[0 : y_pred.shape[0], :]

            tmp = metrics.precision_recall_fscore_support(y_true.flatten(), y_pred.flatten())
            return_dict['frame_precision'] = tmp[0][1]
            return_dict['frame_recall'] = tmp[1][1]
            return_dict['frame_f1'] = tmp[2][1]

        # Post processor
        post_processor = RegressionPostProcessor(self.frames_per_second, 
            classes_num=self.classes_num, onset_threshold=onset_threshold, 
            offset_threshold=offset_threshod, 
            frame_threshold=frame_threshold)

        output_dict = post_processor.get_onset_roll_from_regression(output_dict)
        output_dict = post_processor.get_offset_roll_from_regression(output_dict)

        # Detect piano notes from output_dict
        (est_on_off_vels, est_midi_notes) = post_processor.output_dict_to_midi_notes(output_dict)

        # Calculate note metrics
        if self.velocity:
            (note_precision, note_recall, note_f1, _) = (
                   mir_eval.transcription_velocity.precision_recall_f1_overlap(
                       ref_intervals=ref_on_off_pairs,
                       ref_pitches=note_to_freq(ref_midi_notes),
                       ref_velocities=total_dict['ref_velocity'],
                       est_intervals=est_on_off_vels[:, 0 : 2],
                       est_pitches=note_to_freq(est_midi_notes),
                       est_velocities=est_on_off_vels[:, 2] * velocity_scale,
                       onset_tolerance=self.onset_tolerance, 
                       offset_ratio=self.offset_ratio, 
                       offset_min_tolerance=self.offset_min_tolerance))
        else:
            note_precision, note_recall, note_f1, _ = \
                mir_eval.transcription.precision_recall_f1_overlap(
                    ref_intervals=ref_on_off_pairs, 
                    ref_pitches=note_to_freq(ref_midi_notes), 
                    est_intervals=est_on_off_vels[:, 0 : 2], 
                    est_pitches=note_to_freq(est_midi_notes), 
                    onset_tolerance=self.onset_tolerance, 
                    offset_ratio=self.offset_ratio, 
                    offset_min_tolerance=self.offset_min_tolerance)

        print(note_f1)

        return_dict['note_precision'] = note_precision
        return_dict['note_recall'] = note_recall
        return_dict['note_f1'] = note_f1
        return return_dict


def calculate_metrics(args, thresholds=None):

    # Arugments & parameters
    workspace = args.workspace
    filename = args.filename
    split = args.split
    model_type = args.model_type
    loss_type = args.loss_type
    augmentation = args.augmentation
    max_note_shift = args.max_note_shift
    batch_size = args.batch_size
    iteration = args.iteration
    dataset = 'maestro'

    if args.maps_dataset:
        suffix = '_maps'
    else:
        suffix = ''
    
    # Paths
    hdf5s_dir = os.path.join(workspace, 'hdf5s', dataset)

    probs_dir = os.path.join(workspace, 'probs', filename, dataset, 
        'split={}'.format(split), model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
        '{}_iterations'.format(iteration))

    # Score calculator
    score_calculator = RegressionScoreCalculator(hdf5s_dir, probs_dir, split=split)

    if not thresholds:
        thresholds = [0.3, 0.3, 0.3]
    else:
        pass

    t1 = time.time()
    stats_dict = score_calculator.metrics(thresholds)
    print('Time: {:.3f}'.format(time.time() - t1))
    
    for key in stats_dict.keys():
        print('{}: {:.4f}'.format(key, np.mean(stats_dict[key])))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_infer_prob = subparsers.add_parser('infer_prob')
    parser_infer_prob.add_argument('--workspace', type=str, required=True)
    parser_infer_prob.add_argument('--filename', type=str, required=True)
    parser_infer_prob.add_argument('--split', type=str, required=True)
    parser_infer_prob.add_argument('--model_type', type=str, required=True)
    parser_infer_prob.add_argument('--loss_type', type=str, required=True)
    parser_infer_prob.add_argument('--augmentation', type=str, required=True)
    parser_infer_prob.add_argument('--max_note_shift', type=int, required=True)
    parser_infer_prob.add_argument('--batch_size', type=int, required=True)
    parser_infer_prob.add_argument('--iteration', type=int, required=True)
    parser_infer_prob.add_argument('--cuda', action='store_true', default=False)
    parser_infer_prob.add_argument('--maps_dataset', action='store_true', default=False)

    parser_metrics = subparsers.add_parser('calculate_metrics')
    parser_metrics.add_argument('--workspace', type=str, required=True)
    parser_metrics.add_argument('--filename', type=str, required=True)
    parser_metrics.add_argument('--split', type=str, required=True)
    parser_metrics.add_argument('--model_type', type=str, required=True)
    parser_metrics.add_argument('--loss_type', type=str, required=True)
    parser_metrics.add_argument('--augmentation', type=str, required=True)
    parser_metrics.add_argument('--max_note_shift', type=int, required=True)
    parser_metrics.add_argument('--batch_size', type=int, required=True)
    parser_metrics.add_argument('--iteration', type=int, required=True)
    parser_metrics.add_argument('--maps_dataset', action='store_true', default=False)

    parser_new = subparsers.add_parser('evaluate_with_opt_thres')
    parser_new.add_argument('--workspace', type=str, required=True)
    parser_new.add_argument('--filename', type=str, required=True)
    parser_new.add_argument('--model_type', type=str, required=True)
    parser_new.add_argument('--loss_type', type=str, required=True)
    parser_new.add_argument('--augmentation', type=str, required=True)
    parser_new.add_argument('--batch_size', type=int, required=True)
    parser_new.add_argument('--iteration', type=int, required=True)

    args = parser.parse_args()

    if args.mode == 'infer_prob':
        infer_prob(args)

    elif args.mode == 'calculate_metrics':
        calculate_metrics(args)

    elif args.mode == 'evaluate_with_opt_thres':
        evaluate_with_opt_thres(args)

    else:
        raise Exception('Incorrct argument!')