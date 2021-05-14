import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../pytorch'))
import librosa
import mir_eval
import argparse
import pickle
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt

from utilities import (get_filename, traverse_folder, int16_to_float32, note_to_freq, TargetProcessor, RegressionPostProcessor, read_midi)
import config
from inference import PianoTranscription

'''
def plot(args):

    workspace = args.workspace

    probs_dir = os.path.join(workspace, 'probs', 'model_type=Note_pedal', 
        'augmentation=random_target_none', 'dataset=maestro', 'split=test')

    prob_names = os.listdir(probs_dir)

    for prob_name in prob_names:
        prob_path = os.path.join(probs_dir, prob_name)
        total_dict = pickle.load(open(prob_path, 'rb'))


        import crash
        asdf
'''

def plot(args):
    """Inference the output probabilites of MAESTRO dataset.

    Args:
      cuda: bool
      audio_path: str
    """

    # Arugments & parameters
    workspace = args.workspace
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    dataset = args.dataset
    split = args.split
    post_processor_type = args.post_processor_type
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    
    sample_rate = config.sample_rate
    segment_seconds = config.segment_seconds
    segment_samples = int(segment_seconds * sample_rate)
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num
    begin_note = config.begin_note

    # Paths
    hdf5s_dir = os.path.join(workspace, 'hdf5s', dataset)

    # Transcriptor
    transcriptor = PianoTranscription(model_type, device=device, 
        checkpoint_path=checkpoint_path, segment_samples=segment_samples, 
        post_processor_type=post_processor_type)

    (hdf5_names, hdf5_paths) = traverse_folder(hdf5s_dir)

    n = 0
    for n, hdf5_path in enumerate(hdf5_paths):
        with h5py.File(hdf5_path, 'r') as hf:
            if hf.attrs['split'].decode() == split:
                print(n, hdf5_path)
                
                if n == 90:
                    # Load audio                
                    audio = int16_to_float32(hf['waveform'][:])
                    midi_events = [e.decode() for e in hf['midi_event'][:]]
                    midi_events_time = hf['midi_event_time'][:]
            
                    # Ground truths processor
                    target_processor = TargetProcessor(
                        segment_seconds=len(audio) / sample_rate, 
                        frames_per_second=frames_per_second, begin_note=begin_note, 
                        classes_num=classes_num)

                    # Get ground truths
                    (target_dict, note_events, pedal_events) = \
                        target_processor.process(start_time=0, 
                            midi_events_time=midi_events_time, 
                            midi_events=midi_events, extend_pedal=True)

                    ref_on_off_pairs = np.array([[event['onset_time'], event['offset_time']] for event in note_events])
                    ref_midi_notes = np.array([event['midi_note'] for event in note_events])
                    ref_velocity = np.array([event['velocity'] for event in note_events])

                    # Transcribe
                    transcribed_dict = transcriptor.transcribe(audio, midi_path=None)
                    output_dict = transcribed_dict['output_dict']

                    # Pack probabilites to dump
                    total_dict = {key: output_dict[key] for key in output_dict.keys()}
                    total_dict['frame_roll'] = target_dict['frame_roll']
                    total_dict['ref_on_off_pairs'] = ref_on_off_pairs
                    total_dict['ref_midi_notes'] = ref_midi_notes
                    total_dict['ref_velocity'] = ref_velocity

                    if 'pedal_frame_output' in output_dict.keys():
                        total_dict['ref_pedal_on_off_pairs'] = \
                            np.array([[event['onset_time'], event['offset_time']] for event in pedal_events])
                        total_dict['pedal_frame_roll'] = target_dict['pedal_frame_roll']

                    if True:
                        post_processor = RegressionPostProcessor(100, 
                        classes_num=100, onset_threshold=0.3, 
                        offset_threshold=0.3, 
                        frame_threshold=0.3, 
                        pedal_offset_threshold=0.2)

                        (est_on_off_note_vels, est_pedal_on_offs) = \
                            post_processor.output_dict_to_note_pedal_arrays(output_dict)

                        ref_on_off_pairs = total_dict['ref_on_off_pairs']
                        ref_midi_notes = total_dict['ref_midi_notes']

                        est_on_offs = est_on_off_note_vels[:, 0 : 2]
                        est_midi_notes = est_on_off_note_vels[:, 2]
                        est_vels = est_on_off_note_vels[:, 3] * 128

                        note_precision, note_recall, note_f1, _ = \
                        mir_eval.transcription.precision_recall_f1_overlap(
                            ref_intervals=ref_on_off_pairs, 
                            ref_pitches=note_to_freq(ref_midi_notes), 
                            est_intervals=est_on_offs, 
                            est_pitches=note_to_freq(est_midi_notes), 
                            onset_tolerance=0.05, 
                            offset_ratio=0.2, 
                            offset_min_tolerance=0.05)

                        print('note f1: {:.3f}'.format(note_f1))
                        
                    if True:
                        librosa.output.write_wav('_zz.wav', audio, sr=16000)
                        bgn = 15500
                        L = 500
                        fontsize = 7
                        vmin = 0
                        vmax = 1

                        
                        fig, axs = plt.subplots(7, 1, figsize=(4, 6), sharex=True)
                        mel = librosa.feature.melspectrogram(audio, sr=16000, n_fft=2048, hop_length=160, n_mels=229, fmin=30, fmax=8000).T
                        axs[0].matshow(np.log(mel[bgn : bgn + L]).T, origin='lower', aspect='auto', cmap='jet')
                        axs[1].matshow(target_dict['frame_roll'][bgn : bgn + L].T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
                        axs[2].matshow(output_dict['frame_output'][bgn : bgn + L].T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
                        axs[3].matshow(target_dict['reg_onset_roll'][bgn : bgn + L].T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
                        axs[4].matshow(output_dict['reg_onset_output'][bgn : bgn + L].T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
                        axs[5].matshow(target_dict['reg_offset_roll'][bgn : bgn + L].T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
                        axs[6].matshow(output_dict['reg_offset_output'][bgn : bgn + L].T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
                        axs[0].set_xlim(0, len(output_dict['frame_output'][bgn : bgn + L]))
                        axs[0].set_title('Log mel spectrogram', fontsize=fontsize)
                        axs[1].set_title('Frame-wise target', fontsize=fontsize)
                        axs[2].set_title('Frame-wise output', fontsize=fontsize)
                        axs[3].set_title('Regression onsets target', fontsize=fontsize)
                        axs[4].set_title('Regression onsets output', fontsize=fontsize)
                        axs[5].set_title('Regression offsets target', fontsize=fontsize)
                        axs[6].set_title('Regression offsets output', fontsize=fontsize)
                        axs[0].set_ylabel('Mel bins', fontsize=fontsize)
                        axs[0].yaxis.set_ticks(np.arange(0, 229, 228))
                        axs[0].yaxis.set_ticklabels([0, 228], fontsize=fontsize)
                        for i in range(6):
                            axs[i].xaxis.set_ticks([])
                            axs[i].xaxis.set_ticklabels([])
                        for i in range(1, 7):
                            axs[i].yaxis.set_ticks(np.arange(0, 88, 87))
                            axs[i].yaxis.set_ticklabels([0, 87], fontsize=fontsize)
                            axs[i].set_ylabel('Note', fontsize=fontsize)
                        axs[6].xaxis.set_ticks([0, 100, 200, 300, 400, 499])
                        axs[6].xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5'], fontsize=fontsize)
                        axs[6].set_xlabel('Seconds', fontsize=fontsize)
                        axs[6].xaxis.set_ticks_position('bottom')
                        plt.tight_layout(0, 0, 0)
                        fig_path = '_zz.pdf'
                        plt.savefig(fig_path)
                        print('Plot to {}'.format(fig_path))

                        fig, axs = plt.subplots(7, 1, figsize=(4, 6), sharex=True)
                        mel = librosa.feature.melspectrogram(audio, sr=16000, n_fft=2048, hop_length=160, n_mels=229, fmin=30, fmax=8000).T
                        axs[0].matshow(np.log(mel[bgn : bgn + L]).T, origin='lower', aspect='auto', cmap='jet')
                        axs[1].plot(target_dict['pedal_frame_roll'][bgn : bgn + L])
                        axs[2].plot(output_dict['pedal_frame_output'][bgn : bgn + L])
                        axs[3].plot(target_dict['reg_pedal_onset_roll'][bgn : bgn + L])
                        axs[4].plot(output_dict['reg_pedal_onset_output'][bgn : bgn + L])
                        axs[5].plot(target_dict['reg_pedal_offset_roll'][bgn : bgn + L])
                        axs[6].plot(output_dict['reg_pedal_offset_output'][bgn : bgn + L])
                        axs[0].set_xlim(0, len(output_dict['frame_output'][bgn : bgn + L]))
                        axs[0].set_title('Log mel spectrogram', fontsize=fontsize)
                        axs[1].set_title('Frame-wise target', fontsize=fontsize)
                        axs[2].set_title('Frame-wise output', fontsize=fontsize)
                        axs[3].set_title('Regression onsets target', fontsize=fontsize)
                        axs[4].set_title('Regression onsets output', fontsize=fontsize)
                        axs[5].set_title('Regression offsets target', fontsize=fontsize)
                        axs[6].set_title('Regression offsets output', fontsize=fontsize)
                        axs[0].set_ylabel('Mel bins', fontsize=fontsize)
                        axs[0].yaxis.set_ticks(np.arange(0, 229, 228))
                        axs[0].yaxis.set_ticklabels([0, 228], fontsize=fontsize)
                        for i in range(6):
                            axs[i].xaxis.set_ticks([])
                            axs[i].xaxis.set_ticklabels([])
                        for i in range(1, 7):
                            axs[i].set_ylim(0, 1.02)
                            axs[i].set_ylabel('Value', fontsize=fontsize)
                        axs[6].xaxis.set_ticks([0, 100, 200, 300, 400, 499])
                        axs[6].xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5'], fontsize=fontsize)
                        axs[6].set_xlabel('Seconds', fontsize=fontsize)
                        axs[6].xaxis.set_ticks_position('bottom')
                        plt.tight_layout(0, 0, 0)
                        fig_path = '_zz2.pdf'
                        plt.savefig(fig_path)
                        print('Plot to {}'.format(fig_path))


                        import crash
                        asdf

                n += 1


# def plot2(args):
#     # Onsets frames plot
#     """Inference the output probabilites of MAESTRO dataset.

#     Args:
#       cuda: bool
#       audio_path: str
#     """

#     # Arugments & parameters
#     workspace = args.workspace
#     model_type = args.model_type
#     checkpoint_path = args.checkpoint_path
#     dataset = args.dataset
#     split = args.split
#     post_processor_type = args.post_processor_type
#     device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    
#     sample_rate = config.sample_rate
#     segment_seconds = config.segment_seconds
#     segment_samples = int(segment_seconds * sample_rate)
#     frames_per_second = config.frames_per_second
#     classes_num = config.classes_num
#     begin_note = config.begin_note

#     # Paths
#     hdf5s_dir = os.path.join(workspace, 'hdf5s', dataset)

#     # Transcriptor
#     transcriptor = PianoTranscription(model_type, device=device, 
#         checkpoint_path=checkpoint_path, segment_samples=segment_samples, 
#         post_processor_type=post_processor_type)

#     (hdf5_names, hdf5_paths) = traverse_folder(hdf5s_dir)

#     n = 0
#     for n, hdf5_path in enumerate(hdf5_paths):
#         with h5py.File(hdf5_path, 'r') as hf:
#             if hf.attrs['split'].decode() == split:
#                 print(n, hdf5_path)
                
#                 if n == 90:
#                     # Load audio                
#                     audio = int16_to_float32(hf['waveform'][:])
#                     midi_events = [e.decode() for e in hf['midi_event'][:]]
#                     midi_events_time = hf['midi_event_time'][:]
            
#                     # Ground truths processor
#                     target_processor = TargetProcessor(
#                         segment_seconds=len(audio) / sample_rate, 
#                         frames_per_second=frames_per_second, begin_note=begin_note, 
#                         classes_num=classes_num)

#                     # Get ground truths
#                     (target_dict, note_events, pedal_events) = \
#                         target_processor.process(start_time=0, 
#                             midi_events_time=midi_events_time, 
#                             midi_events=midi_events, extend_pedal=True)

#                     ref_on_off_pairs = np.array([[event['onset_time'], event['offset_time']] for event in note_events])
#                     ref_midi_notes = np.array([event['midi_note'] for event in note_events])
#                     ref_velocity = np.array([event['velocity'] for event in note_events])

#                     # Transcribe
#                     transcribed_dict = transcriptor.transcribe(audio, midi_path=None)
#                     output_dict = transcribed_dict['output_dict']

#                     # Pack probabilites to dump
#                     total_dict = {key: output_dict[key] for key in output_dict.keys()}
#                     total_dict['frame_roll'] = target_dict['frame_roll']
#                     total_dict['ref_on_off_pairs'] = ref_on_off_pairs
#                     total_dict['ref_midi_notes'] = ref_midi_notes
#                     total_dict['ref_velocity'] = ref_velocity

#                     if 'pedal_frame_output' in output_dict.keys():
#                         total_dict['ref_pedal_on_off_pairs'] = \
#                             np.array([[event['onset_time'], event['offset_time']] for event in pedal_events])
#                         total_dict['pedal_frame_roll'] = target_dict['pedal_frame_roll']

#                     if True:
#                         post_processor = RegressionPostProcessor(100, 
#                         classes_num=100, onset_threshold=0.3, 
#                         offset_threshold=0.3, 
#                         frame_threshold=0.3, 
#                         pedal_offset_threshold=0.2)

#                         (est_on_off_note_vels, est_pedal_on_offs) = \
#                             post_processor.output_dict_to_note_pedal_arrays(output_dict)

#                         ref_on_off_pairs = total_dict['ref_on_off_pairs']
#                         ref_midi_notes = total_dict['ref_midi_notes']

#                         est_on_offs = est_on_off_note_vels[:, 0 : 2]
#                         est_midi_notes = est_on_off_note_vels[:, 2]
#                         est_vels = est_on_off_note_vels[:, 3] * 128

#                         note_precision, note_recall, note_f1, _ = \
#                         mir_eval.transcription.precision_recall_f1_overlap(
#                             ref_intervals=ref_on_off_pairs, 
#                             ref_pitches=note_to_freq(ref_midi_notes), 
#                             est_intervals=est_on_offs, 
#                             est_pitches=note_to_freq(est_midi_notes), 
#                             onset_tolerance=0.05, 
#                             offset_ratio=0.2, 
#                             offset_min_tolerance=0.05)

#                         print('note f1: {:.3f}'.format(note_f1))
                        
#                     if True:
#                         librosa.output.write_wav('_zz.wav', audio, sr=16000)
#                         bgn = 8000
#                         L = 1000

#                         import matplotlib.pyplot as plt
#                         fig, axs = plt.subplots(7, 1, figsize=(8, 8), sharex=True)
#                         mel = librosa.feature.melspectrogram(audio, sr=16000, n_fft=2048, hop_length=160, n_mels=229, fmin=30, fmax=8000).T
#                         axs[0].matshow(np.log(mel[bgn : bgn + L]).T, origin='lower', aspect='auto', cmap='jet')
#                         axs[1].matshow(target_dict['frame_roll'][bgn : bgn + L].T, origin='lower', aspect='auto', cmap='jet')
#                         axs[2].matshow(output_dict['frame_output'][bgn : bgn + L].T, origin='lower', aspect='auto', cmap='jet')
#                         axs[3].matshow(target_dict['onset_roll'][bgn : bgn + L].T, origin='lower', aspect='auto', cmap='jet')
#                         axs[4].matshow(output_dict['reg_onset_output'][bgn : bgn + L].T, origin='lower', aspect='auto', cmap='jet')
#                         axs[5].matshow(target_dict['offset_roll'][bgn : bgn + L].T, origin='lower', aspect='auto', cmap='jet')
#                         axs[6].matshow(output_dict['reg_offset_output'][bgn : bgn + L].T, origin='lower', aspect='auto', cmap='jet')
#                         axs[0].set_xlim(0, len(output_dict['frame_output'][bgn : bgn + L]))
#                         axs[6].set_xlabel('Frames')
#                         axs[0].set_title('Log mel spectrogram')
#                         axs[1].set_title('Ground truth frames')
#                         axs[2].set_title('frame_output')
#                         axs[3].set_title('Ground truth reg onset')
#                         axs[4].set_title('reg_onset_output')
#                         axs[5].set_title('Ground truth reg offset')
#                         axs[6].set_title('reg_offset_output')
#                         plt.tight_layout(0, .05, 0)
#                         fig_path = '_zz.pdf'
#                         plt.savefig(fig_path)
#                         print('Plot to {}'.format(fig_path))

#                         fig, axs = plt.subplots(7, 1, figsize=(8, 8), sharex=True)
#                         mel = librosa.feature.melspectrogram(audio, sr=16000, n_fft=2048, hop_length=160, n_mels=229, fmin=30, fmax=8000).T
#                         axs[0].matshow(np.log(mel[bgn : bgn + L]).T, origin='lower', aspect='auto', cmap='jet')
#                         axs[1].plot(target_dict['pedal_frame_roll'][bgn : bgn + L])
#                         axs[2].plot(output_dict['pedal_frame_output'][bgn : bgn + L])
#                         axs[3].plot(target_dict['pedal_onset_roll'][bgn : bgn + L])
#                         axs[4].plot(output_dict['reg_pedal_onset_output'][bgn : bgn + L])
#                         axs[5].plot(target_dict['pedal_offset_roll'][bgn : bgn + L])
#                         axs[6].plot(output_dict['reg_pedal_offset_output'][bgn : bgn + L])
#                         axs[0].set_xlim(0, len(output_dict['frame_output'][bgn : bgn + L]))
#                         axs[6].set_xlabel('Frames')
#                         axs[0].set_title('Log mel spectrogram')
#                         axs[1].set_title('Ground truth frames')
#                         axs[2].set_title('frame_output')
#                         axs[3].set_title('Ground truth reg onset')
#                         axs[4].set_title('reg_onset_output')
#                         axs[5].set_title('Ground truth reg offset')
#                         axs[6].set_title('reg_offset_output')
#                         plt.tight_layout(0, .05, 0)
#                         fig_path = '_zz2.pdf'
#                         plt.savefig(fig_path)
#                         print('Plot to {}'.format(fig_path))


#                         import crash
#                         asdf

#                 n += 1


def plot3(args):
    # 
    """Inference the output probabilites of MAESTRO dataset.

    Args:
      cuda: bool
      audio_path: str
    """

    # Arugments & parameters
    workspace = args.workspace
    model_type = args.model_type
    dataset = args.dataset
    split = args.split
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    
    sample_rate = config.sample_rate
    segment_seconds = config.segment_seconds
    segment_samples = int(segment_seconds * sample_rate)
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num
    begin_note = config.begin_note

    # Paths
    hdf5s_dir = os.path.join(workspace, 'hdf5s', dataset)

    # Transcriptor
    checkpoint_path = os.path.join(workspace, 'combined_models/google_note_pedal_random.pth')
    # checkpoint_path = os.path.join(workspace, 'combined_models/google_note_pedal.pth')
    transcriptor1 = PianoTranscription(model_type, device=device, 
        checkpoint_path=checkpoint_path, segment_samples=segment_samples, 
        post_processor_type='onsets_frames')

    checkpoint_path = os.path.join(workspace, 'combined_models/regress_random_note_pedal.pth')
    # checkpoint_path = os.path.join(workspace, 'combined_models/regress_note_pedal.pth')
    transcriptor2 = PianoTranscription(model_type, device=device, 
        checkpoint_path=checkpoint_path, segment_samples=segment_samples, 
        post_processor_type='regression')

    (hdf5_names, hdf5_paths) = traverse_folder(hdf5s_dir)

    n = 0
    for n, hdf5_path in enumerate(hdf5_paths):
        with h5py.File(hdf5_path, 'r') as hf:
            if hf.attrs['split'].decode() == split:
                print(n, hdf5_path)
                
                if n == 90:
                    # Load audio                
                    audio = int16_to_float32(hf['waveform'][:])
                    midi_events = [e.decode() for e in hf['midi_event'][:]]
                    midi_events_time = hf['midi_event_time'][:]
            
                    # Ground truths processor
                    target_processor = TargetProcessor(
                        segment_seconds=len(audio) / sample_rate, 
                        frames_per_second=frames_per_second, begin_note=begin_note, 
                        classes_num=classes_num)

                    # Get ground truths
                    (target_dict, note_events, pedal_events) = \
                        target_processor.process(start_time=0, 
                            midi_events_time=midi_events_time, 
                            midi_events=midi_events, extend_pedal=True)

                    ref_on_off_pairs = np.array([[event['onset_time'], event['offset_time']] for event in note_events])
                    ref_midi_notes = np.array([event['midi_note'] for event in note_events])
                    ref_velocity = np.array([event['velocity'] for event in note_events])

                    # Transcribe
                    transcribed_dict = transcriptor1.transcribe(audio, midi_path=None)
                    output_dict = transcribed_dict['output_dict']

                    # Pack probabilites to dump
                    total_dict1 = {key: output_dict[key] for key in output_dict.keys()}
                    total_dict1['frame_roll'] = target_dict['frame_roll']
                    total_dict1['ref_on_off_pairs'] = ref_on_off_pairs
                    total_dict1['ref_midi_notes'] = ref_midi_notes
                    total_dict1['ref_velocity'] = ref_velocity

                    if 'pedal_frame_output' in output_dict.keys():
                        total_dict1['ref_pedal_on_off_pairs'] = \
                            np.array([[event['onset_time'], event['offset_time']] for event in pedal_events])
                        total_dict1['pedal_frame_roll'] = target_dict['pedal_frame_roll']

                    # Transcribe
                    transcribed_dict = transcriptor2.transcribe(audio, midi_path=None)
                    output_dict = transcribed_dict['output_dict']

                    # Pack probabilites to dump
                    total_dict2 = {key: output_dict[key] for key in output_dict.keys()}
                    total_dict2['frame_roll'] = target_dict['frame_roll']
                    total_dict2['ref_on_off_pairs'] = ref_on_off_pairs
                    total_dict2['ref_midi_notes'] = ref_midi_notes
                    total_dict2['ref_velocity'] = ref_velocity

                    if 'pedal_frame_output' in output_dict.keys():
                        total_dict2['ref_pedal_on_off_pairs'] = \
                            np.array([[event['onset_time'], event['offset_time']] for event in pedal_events])
                        total_dict2['pedal_frame_roll'] = target_dict['pedal_frame_roll']

                    if True:
                        librosa.output.write_wav('_zz.wav', audio, sr=16000)
                        bgn = 15500
                        L = 500
                        fontsize = 7
                        vmin = None
                        vmax = None

                        import matplotlib.pyplot as plt
                        fig, axs = plt.subplots(5, 1, figsize=(4, 4.5), sharex=True)
                        mel = librosa.feature.melspectrogram(audio, sr=16000, n_fft=2048, hop_length=160, n_mels=229, fmin=30, fmax=8000).T
                        axs[0].matshow(np.log(mel[bgn : bgn + L]).T, origin='lower', aspect='auto', cmap='jet')
                        axs[1].matshow(total_dict1['reg_onset_output'][bgn : bgn + L].T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
                        axs[2].matshow(total_dict2['reg_onset_output'][bgn : bgn + L].T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
                        axs[3].matshow(total_dict1['reg_offset_output'][bgn : bgn + L].T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
                        axs[4].matshow(total_dict2['reg_offset_output'][bgn : bgn + L].T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
                        axs[0].set_xlim(0, len(total_dict1['frame_output'][bgn : bgn + L]))
                        axs[0].set_title('Log mel spectrogram', fontsize=fontsize)
                        axs[1].set_title('Google\'s onsets output', fontsize=fontsize)
                        axs[2].set_title('Regression onsets output', fontsize=fontsize)
                        axs[3].set_title('Google\'s offsets output', fontsize=fontsize)
                        axs[4].set_title('Regression offsets output', fontsize=fontsize)
                        axs[4].set_xlabel('Seconds')
                        axs[0].yaxis.set_ticks(np.arange(0, 229, 228))
                        axs[0].yaxis.set_ticklabels([0, 228], fontsize=fontsize)
                        axs[0].set_ylabel('Mel bins', fontsize=fontsize)
                        for i in range(4):
                            axs[i].xaxis.set_ticks([])
                            axs[i].xaxis.set_ticklabels([])
                        for i in range(1, 5):
                            axs[i].yaxis.set_ticks(np.arange(0, 88, 87))
                            axs[i].yaxis.set_ticklabels([0, 87], fontsize=fontsize)
                            axs[i].set_ylabel('Note', fontsize=fontsize)
                        axs[4].xaxis.set_ticks([0, 100, 200, 300, 400, 499])
                        axs[4].xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5'], fontsize=fontsize)
                        axs[4].set_xlabel('Seconds', fontsize=fontsize)
                        axs[4].xaxis.set_ticks_position('bottom')
                        plt.tight_layout(0, 0, 0)
                        fig_path = '_zz.pdf'
                        plt.savefig(fig_path)
                        print('Plot to {}'.format(fig_path))

                        import crash
                        asdf

                        fig, axs = plt.subplots(5, 1, figsize=(8, 8), sharex=True)
                        mel = librosa.feature.melspectrogram(audio, sr=16000, n_fft=2048, hop_length=160, n_mels=229, fmin=30, fmax=8000).T
                        axs[0].matshow(np.log(mel[bgn : bgn + L]).T, origin='lower', aspect='auto', cmap='jet')
                        axs[1].plot(total_dict1['reg_pedal_offset_output'][bgn : bgn + L])
                        axs[2].plot(total_dict2['reg_pedal_offset_output'][bgn : bgn + L])
                        axs[0].set_xlim(0, len(output_dict['frame_output'][bgn : bgn + L]))
                        axs[6].set_xlabel('Frames')
                        axs[0].set_title('Log mel spectrogram')
                        axs[1].set_title('Ground truth frames')
                        axs[2].set_title('frame_output')
                        axs[3].set_title('Ground truth reg onset')
                        axs[4].set_title('reg_onset_output')
                        axs[5].set_title('Ground truth reg offset')
                        axs[6].set_title('reg_offset_output')
                        plt.tight_layout(0, .05, 0)
                        fig_path = '_zz2.pdf'
                        plt.savefig(fig_path)
                        print('Plot to {}'.format(fig_path))


                        import crash
                        asdf

                n += 1


def plot_midi(args):

    audio_path = args.audio_path
    midi_path = args.midi_path
    fig_path = 'results/{}.png'.format(get_filename(audio_path))

    (audio, _) = librosa.core.load(audio_path, sr=config.sample_rate, mono=True)
    audio_seconds = audio.shape[0] / config.sample_rate

    midi_dict = read_midi(midi_path)

    target_processor = TargetProcessor(segment_seconds=audio_seconds, 
        frames_per_second=config.frames_per_second, begin_note=config.begin_note, 
        classes_num=config.classes_num)

    (target_dict, note_events, pedal_events) = target_processor.process(
        start_time=0, 
        midi_events_time=midi_dict['midi_event_time'], 
        midi_events=midi_dict['midi_event'])
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 4), sharex=True)
    logmel = np.log(librosa.feature.melspectrogram(audio, sr=16000, n_fft=2048, hop_length=160, n_mels=229, fmin=30, fmax=8000)).T
    axs[0].matshow(logmel.T, origin='lower', aspect='auto', cmap='jet')
    axs[1].matshow(target_dict['frame_roll'].T, origin='lower', aspect='auto', cmap='jet', vmin=-1, vmax=1)
    axs[2].plot(target_dict['pedal_frame_roll'])
    axs[2].set_ylim(-0.02, 1.02)
    axs[0].set_title('Log mel spectrogram')
    axs[1].set_title('Transcribed notes')
    axs[2].set_title('Transcribed pedals')
    axs[0].yaxis.set_ticks(np.arange(0, 229, 228))
    axs[0].yaxis.set_ticklabels([0, 228])
    axs[1].yaxis.set_ticks(np.arange(0, 88, 87))
    axs[1].yaxis.set_ticklabels([0, 87])
    axs[0].set_ylabel('Mel bins')
    axs[1].set_ylabel('Notes')
    axs[2].set_ylabel('Probability')
    fps = config.frames_per_second
    axs[2].xaxis.set_ticks(np.arange(0, audio_seconds * fps + 1, 5 * fps))
    axs[2].xaxis.set_ticklabels(np.arange(0, audio_seconds + 1e-6, 5))
    axs[2].set_xlabel('Seconds')
    plt.tight_layout(0, 0, 0)
    plt.savefig(fig_path)
    print('Save out to {}'.format(fig_path))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_plot = subparsers.add_parser('plot')
    parser_plot.add_argument('--workspace', type=str, required=True)
    parser_plot.add_argument('--model_type', type=str, required=True)
    parser_plot.add_argument('--checkpoint_path', type=str, required=True)
    parser_plot.add_argument('--dataset', type=str, required=True, choices=['maestro', 'maps'])
    parser_plot.add_argument('--split', type=str, required=True)
    parser_plot.add_argument('--post_processor_type', type=str, default='regression')
    parser_plot.add_argument('--cuda', action='store_true', default=False)
    
    parser_plot = subparsers.add_parser('plot2')
    parser_plot.add_argument('--workspace', type=str, required=True)
    parser_plot.add_argument('--model_type', type=str, required=True)
    parser_plot.add_argument('--checkpoint_path', type=str, required=True)
    parser_plot.add_argument('--dataset', type=str, required=True, choices=['maestro', 'maps'])
    parser_plot.add_argument('--split', type=str, required=True)
    parser_plot.add_argument('--post_processor_type', type=str, default='regression')
    parser_plot.add_argument('--cuda', action='store_true', default=False)

    parser_plot = subparsers.add_parser('plot3')
    parser_plot.add_argument('--workspace', type=str, required=True)
    parser_plot.add_argument('--model_type', type=str, required=True)
    parser_plot.add_argument('--dataset', type=str, required=True, choices=['maestro', 'maps'])
    parser_plot.add_argument('--split', type=str, required=True)
    parser_plot.add_argument('--cuda', action='store_true', default=False)

    parser_plot = subparsers.add_parser('plot_midi')
    parser_plot.add_argument('--audio_path', type=str, required=True)
    parser_plot.add_argument('--midi_path', type=str, required=True)

    args = parser.parse_args()

    if args.mode == 'plot':
        plot(args)

    elif args.mode == 'plot2':
        plot2(args)

    elif args.mode == 'plot3':
        plot3(args)

    elif args.mode == 'plot_midi':
        plot_midi(args)

    else:
        raise Exception('Incorrct argument!')
