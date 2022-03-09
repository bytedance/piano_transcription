import os
import sys
import numpy as np
import argparse
import h5py
import pickle
import matplotlib.pyplot as plt

from utilities import create_folder


def plot(args):
    
    # Arguments & parameters
    workspace = args.workspace
    select = args.select
    
    max_plot_iteration = 300001
    iterations = np.arange(0, max_plot_iteration, 5000)
    metric_types = ['frame_ap', 'reg_onset_mae', 'reg_offset_mae', 
        'velocity_mae', 'reg_pedal_onset_mae', 'reg_pedal_offset_mae']
        
    save_out_path = 'results/{}.pdf'.format(select)
    create_folder(os.path.dirname(save_out_path))
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(8, 5))
    lines = []
        
    def _load_metrics(filename, model_type, loss_type, augmentation, 
        max_note_shift, batch_size, data_type, metric_type):
        statistics_path = os.path.join(workspace, 'statistics', filename, 
            model_type, 'loss_type={}'.format(loss_type), 
            'augmentation={}'.format(augmentation), 'max_note_shift={}'.format(max_note_shift), 
            'batch_size={}'.format(batch_size), 'statistics.pkl')

        statistics_dict = pickle.load(open(statistics_path, 'rb'))

        if metric_type in statistics_dict[data_type][0].keys():
            metrics = np.array([statistics[metric_type] for statistics in statistics_dict[data_type]])
            return metrics
        else:
            return None
        
    ylims = [[0, 1], [0, 0.5], [0, 0.5], [0, 0.3], [0, 0.3], [0, 0.3]]
    legend_locs = [4, 1, 1, 1, 1, 1]
    

    if select == '1a':

        for j, metric_type in enumerate(metric_types):
            lines = []
            for data_type in ['train', 'test']:

                metrics = _load_metrics('main', 
                    'Regress_onset_offset_frame_velocity_CRNN', 
                    'regress_onset_offset_frame_velocity_bce', 'none', 0, 12, 
                    data_type, metric_type)
                
                if metrics is not None:
                    line, = axes[j // 3, j % 3].plot(metrics, label=data_type)
                    lines.append(line)

            axes[j // 3, j % 3].set_title(metric_type)
            axes[j // 3, j % 3].legend(handles=lines, loc=legend_locs[j])
            axes[j // 3, j % 3].set_ylim(ylims[j][0], ylims[j][1])
            axes[j // 3, j % 3].set_xlim(0, len(iterations))
            axes[j // 3, j % 3].xaxis.set_ticks(np.arange(0, len(iterations), 20))
            axes[j // 3, j % 3].xaxis.set_ticklabels(['0', '100k', '200k', '300k'])
            axes[j // 3, j % 3].set_xlabel('Iterations')
                
    plt.tight_layout(0, 1, 0)
    plt.savefig(save_out_path)
    print('Write out to {}'.format(save_out_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_plot = subparsers.add_parser('plot')
    parser_plot.add_argument('--workspace', type=str, required=True)
    parser_plot.add_argument('--select', type=str, required=True)
    
    args = parser.parse_args()

    if args.mode == 'plot':
        plot(args)
        
    else:
        raise Exception('Error argument!')