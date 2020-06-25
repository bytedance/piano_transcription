import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
# sys.path.insert(1, os.path.join(sys.path[0], '../../autoth'))
from autoth.core import HyperParamsOptimizer
import numpy as np
import argparse
import librosa
import mir_eval
import torch
import time
import h5py
import _pickle as cPickle 
from concurrent.futures import ProcessPoolExecutor
 
from utilities import (create_folder, get_filename, PostProcessor, traverse_folder, 
    int16_to_float32, write_events_to_midi, TargetProcessor, note_to_freq)
# from models import *
from calculate_score_for_paper import ScoreCalculatorRegress
from pytorch_utils import WaveformTester
import config


def optimizer_thres(args):
    """Inference a waveform.

    Args:
      cuda: bool
      audio_path: str
    """
 
    # Arugments & parameters
    workspace = args.workspace
    filename = args.filename
    model_type = args.model_type
    loss_type = args.loss_type
    augmentation = args.augmentation
    batch_size = args.batch_size
    iteration = args.iteration

    split = 'validation'
    
    # Paths
    optimized_params_path = os.path.join(workspace, 'opt_hyper_params', filename, 
        model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        '{}_iterations'.format(iteration), 'thres.pkl')
    create_folder(os.path.dirname(optimized_params_path))

    feature_hdf5s_dir = os.path.join(workspace, 'features')

    probs_dir = os.path.join(workspace, 'probs', filename, 
        model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 'split={}'.format(split), 
        '{}_iterations'.format(iteration))

    # Score calculator
    score_calculator = ScoreCalculatorRegress(feature_hdf5s_dir, probs_dir, velocity=True, split=split, verbose=False)

    hyper_params_opt = HyperParamsOptimizer(score_calculator, learning_rate=1e-2, epochs=50)
    (score, params) = hyper_params_opt.do_optimize(init_params=[0.3, 0.3, 0.1])

    cPickle.dump(params, open(optimized_params_path, 'wb'))
    print('Write to {}'.format(optimized_params_path))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_b = subparsers.add_parser('optimizer_thres')
    parser_b.add_argument('--workspace', type=str, required=True)
    parser_b.add_argument('--filename', type=str, required=True)
    parser_b.add_argument('--model_type', type=str, required=True)
    parser_b.add_argument('--loss_type', type=str, required=True)
    parser_b.add_argument('--augmentation', type=str, required=True)
    parser_b.add_argument('--batch_size', type=int, required=True)
    parser_b.add_argument('--iteration', type=int, required=True)

    args = parser.parse_args()

    if args.mode == 'optimizer_thres':
        optimizer_thres(args)

    else:
        raise Exception('Incorrct argument!')