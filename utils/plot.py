import os
import sys
# sys.path.insert(1, os.path.join(sys.path[0], '../pytorch'))
import numpy as np
import argparse
import h5py
import math
import time
import logging
import _pickle as cPickle
import librosa
import matplotlib.pyplot as plt
 
from utilities import create_folder, read_midi, TargetProcessor
import config


def visualization(args):
    """Visualize spectrogram and transcribed MIDI.
    """

    audio_name = 'cut_bach'
    midi_path = 'results/{}.mid'.format(audio_name)
    audio_path = 'examples/{}.wav'.format(audio_name)
    fig_path = 'debug/{}.png'.format(audio_name)
    create_folder(os.path.dirname(fig_path))

    # Load MIDI
    midi_dict = read_midi(midi_path)
    
    segment_seconds = 20    # Clip length
    target_processor = TargetProcessor(segment_seconds, 
        config.frames_per_second, config.begin_note, config.classes_num)

    (data_dict, note_events) = target_processor.process(0, 
        midi_dict['midi_event_time'], midi_dict['midi_event'])

    # Spectrogram
    (audio, _) = librosa.core.load(audio_path, sr=config.sample_rate, mono=True)
    x = librosa.core.stft(y=audio, n_fft=2048, hop_length=320, window='hann', center=True)
    x = np.abs(x) ** 2

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    axs[0].matshow(np.log(x), origin='lower', aspect='auto', cmap='jet')
    axs[1].matshow(data_dict['frame_roll'].T, origin='lower', aspect='auto', cmap='jet', vmin=-1, vmax=1)
    axs[0].set_title('Log spectrogram')
    axs[1].set_title('Transcribed MIDI')
    axs[0].xaxis.set_ticks([])
    axs[1].xaxis.set_ticks([])
    plt.tight_layout()
    plt.savefig(fig_path)
    print('Write fig to {}'.format(fig_path))


def plot_statistics(args):
    """Plot statistcis of different iterations.
    """
    # Arugments & parameters
    workspace = args.workspace
    select = args.select
    
    # Paths
    split = 'test'
    linewidth = 1
    save_out_path = 'results/{}_{}.png'.format(split, select)
    create_folder(os.path.dirname(save_out_path))

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    

    for metric_type, i in zip(['frame_f1', 'onset_f1'], [0, 1]):
        lines = []

        if select == '1a':
            result_dict = _load_metrics(workspace, 'main','Google_onset_frame', 'onset_frame_bce', 'none', 16)
            if metric_type in result_dict[split].keys():
                line, = axes[i].plot(result_dict[split][metric_type], 
                    label='Google_onset_frame', linewidth=linewidth, color='b')
                lines.append(line)
 
        max_plot_iteration = 200001
        iterations = np.arange(0, max_plot_iteration, 5000)
        axes[i].set_xlabel('Iterations')
        axes[i].set_ylabel('F1 score')
        axes[i].set_ylim(0, 1)
        axes[i].set_xlim(0, len(iterations))
        axes[i].xaxis.set_ticks(np.arange(0, len(iterations), 10))
        axes[i].xaxis.set_ticklabels(np.arange(0, max_plot_iteration, 50000))
        axes[i].yaxis.set_ticks(np.arange(0, 1.01, 0.1))
        axes[i].grid(color='b', linestyle='solid', linewidth=0.3)
        axes[i].legend(handles=lines, loc=4, fontsize=8)
        axes[i].set_title('{} in training \n(not evaluated with mir_eval)'.format(metric_type))

    plt.tight_layout()
    plt.savefig(save_out_path)
    print('Save figure to {}'.format(save_out_path))


def _load_metrics(workspace, filename, model_type, loss_type, augmentation, batch_size):
    """Load metrics from statistics.pkl.

    Returns:
      result_dict: {
        'valid': {'frame_f1': [0.50, 0.60, ...], 'onset_f1': [0.30, 0.40, ...]}, 
        'test': {'frame_f1': [0.50, 0.60, ...], 'onset_f1': [0.30, 0.40, ...]}
      }
    """

    statistics_path = os.path.join(workspace, 'statistics', filename, 
        model_type, 'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation), 
        'batch_size={}'.format(batch_size), 'statistics.pkl')

    statistics = cPickle.load(open(statistics_path, 'rb'))
    
    result_dict = {}
    for split in ['train', 'validation', 'test']:
        result_dict[split] = {}
        
        result_dict[split]['frame_f1'] = [stat['frame_f1'] for stat in statistics[split]]

        if 'onset_f1' in statistics[split][0].keys():
            result_dict[split]['onset_f1'] = [stat['onset_f1'] for stat in statistics[split]]

        if 'offset_f1' in statistics[split][0].keys():
            result_dict[split]['offset_f1'] = [stat['offset_f1'] for stat in statistics[split]]

    return result_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    # Plot statistics
    parser_visualization = subparsers.add_parser('visualization')

    parser_plot = subparsers.add_parser('statistics')
    parser_plot.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_plot.add_argument('--select', type=str, default='1a')
 
    # Parse arguments
    args = parser.parse_args()

    if args.mode == 'visualization':
        visualization(args)

    elif args.mode == 'statistics':
        plot_statistics(args)

    else:
        raise Exception('Error argument!')