import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import librosa
import logging
import matplotlib.pyplot as plt
from sklearn import metrics
import _pickle as cPickle
import mir_eval

import torch
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
 
from utilities import (create_folder, get_filename, create_logging, int16_to_float32, 
    PostProcessor, StatisticsContainer, write_events_to_midi)
from models import Google_onset_frame
from pytorch_utils import move_data_to_device, forward, WaveformTester
from data_generator import MaestroDataset, Sampler, collate_fn
from evaluate import Evaluator, SegmentEvaluator
import config
from losses import get_loss_func


def train(args):
    """Train a piano transcription system.

    Args:
      workspace: str, directory of your workspace
      model_type: str, e.g. 'Google_onset_frame'
      loss_type: str, e.g. 'onset_frame_bce'
      augmentation: str, e.g. 'none'
      batch_size: int
      learning_rate: float
      resume_iteration: int
      early_stop: int
      cuda: bool
      mini_data: bool, use a small amount of data to debug training
    """

    # Arugments & parameters
    workspace = args.workspace
    model_type = args.model_type
    loss_type = args.loss_type
    augmentation = args.augmentation
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    resume_iteration = args.resume_iteration
    early_stop = args.early_stop
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    mini_data = args.mini_data
    filename = args.filename

    sample_rate = config.sample_rate
    segment_seconds = config.segment_seconds
    hop_seconds = config.hop_seconds
    segment_samples = int(segment_seconds * sample_rate)
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num
    num_workers = 0

    # Loss function
    loss_func = get_loss_func(loss_type)

    # Paths
    feature_hdf5s_dir = os.path.join(workspace, 'features')

    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 
        model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))
    create_folder(checkpoints_dir)

    statistics_path = os.path.join(workspace, 'statistics', filename, 
        model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(workspace, 'logs', filename, 
        model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))
    create_folder(logs_dir)

    create_logging(logs_dir, filemode='w')
    logging.info(args)

    if 'cuda' in str(device):
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU.')
        device = 'cpu'
    
    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, frames_per_second=frames_per_second, 
        classes_num=classes_num)

    # Dataset
    dataset = MaestroDataset(feature_hdf5s_dir=feature_hdf5s_dir, 
        segment_seconds=segment_seconds, frames_per_second=frames_per_second)

    # Sampler for training
    train_sampler = Sampler(feature_hdf5s_dir=feature_hdf5s_dir, split='train', 
        segment_seconds=segment_seconds, hop_seconds=hop_seconds, 
        batch_size=batch_size, training=True, mini_data=mini_data)

    # Sampler for evaluation
    evaluate_train_sampler = Sampler(feature_hdf5s_dir=feature_hdf5s_dir, 
        split='train', segment_seconds=segment_seconds, hop_seconds=hop_seconds, 
        batch_size=batch_size, training=False, mini_data=mini_data)

    evaluate_validate_sampler = Sampler(feature_hdf5s_dir=feature_hdf5s_dir, 
        split='validation', segment_seconds=segment_seconds, hop_seconds=hop_seconds, 
        batch_size=batch_size, training=False, mini_data=mini_data)

    evaluate_test_sampler = Sampler(feature_hdf5s_dir=feature_hdf5s_dir, 
        split='test', segment_seconds=segment_seconds, hop_seconds=hop_seconds, 
        batch_size=batch_size, training=False, mini_data=mini_data)

    # Dataloader
    train_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=train_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    evaluate_train_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=evaluate_train_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    validate_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=evaluate_validate_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=evaluate_test_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    # Evaluator
    evaluator = SegmentEvaluator(model, batch_size)

    # Statistics
    statistics_container = StatisticsContainer(statistics_path)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    # Resume training
    if resume_iteration > 0:
        resume_checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
            model_type, 'loss_type={}'.format(loss_type), 
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
                '{}_iterations.pth'.format(resume_iteration))

        logging.info('Loading checkpoint {}'.format(resume_checkpoint_path))
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        train_sampler.load_state_dict(checkpoint['sampler'])
        statistics_container.load_state_dict(resume_iteration)
        iteration = checkpoint['iteration']

    else:
        iteration = 0
    
    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)

    train_bgn_time = time.time()

    for batch_data_dict in train_loader:
        # Evaluation
        if iteration % 5000 == 0 and iteration > 0:
            logging.info('------------------------------------')
            logging.info('Iteration: {}'.format(iteration))

            evaluate_train_statistics = evaluator.evaluate(evaluate_train_loader)
            validate_statistics = evaluator.evaluate(validate_loader)
            test_statistics = evaluator.evaluate(test_loader)

            logging.info('    Train statistics: {}'.format(evaluate_train_statistics))
            logging.info('    Validation statistics: {}'.format(validate_statistics))
            logging.info('    Test statistics: {}'.format(test_statistics))

            statistics_container.append(iteration, evaluate_train_statistics, data_type='train')
            statistics_container.append(iteration, validate_statistics, data_type='validation')
            statistics_container.append(iteration, test_statistics, data_type='test')
            statistics_container.dump()
        
        # Save model
        if iteration % 10000 == 0:
            checkpoint = {
                'iteration': iteration, 
                'model': model.module.state_dict(), 
                'optimizer': optimizer.state_dict(), 
                'sampler': train_sampler.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))
        
        # Move data to device
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
        
        # Forward
        model.train()
        batch_output_dict = model(batch_data_dict['waveform'])

        # Loss
        loss = loss_func(model, batch_output_dict, batch_data_dict)

        print(iteration, loss)

        # Backward
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        # Stop learning
        if iteration == early_stop:
            break

        iteration += 1
        

def evaluate(args):
    """Evaluate on music pieces.

    Args:
      workspace: str, directory of your workspace
      model_type: str, e.g. 'Google_onset_frame'
      loss_type: str, e.g. 'onset_frame_bce'
      augmentation: str, e.g. 'none'
      batch_size: int
      iteration: int, iteration of model to be loaded
      cuda: bool
      mini_data: bool, use a small amount of data to debug evaluation
    """

    # Arugments & parameters
    workspace = args.workspace
    model_type = args.model_type
    loss_type = args.loss_type
    augmentation = args.augmentation
    batch_size = args.batch_size
    iteration = args.iteration
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    mini_data = args.mini_data
    filename = args.filename
    
    sample_rate = config.sample_rate
    segment_seconds = config.segment_seconds
    hop_seconds = config.hop_seconds
    segment_samples = int(segment_seconds * sample_rate)
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num
    num_workers = 8

    # Paths
    feature_hdf5s_dir = os.path.join(workspace, 'features')

    checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
        model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        '{}_iterations.pth'.format(iteration))

    logs_dir = os.path.join(workspace, 'logs', filename, 
        model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))
    create_folder(logs_dir)

    create_logging(logs_dir, filemode='w')
    logging.info(args)
    
    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, frames_per_second=frames_per_second, 
        classes_num=classes_num)

    # Load model
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    if 'cuda' in str(device):
        model.to(device)

    # Evaluate on music pieces
    evaluator = Evaluator(model, feature_hdf5s_dir, segment_seconds, 
        frames_per_second, batch_size)

    statistics = evaluator.evaluate('validation')
 
    for key in statistics.keys():
        print('{}: {:.3f}'.format(key, np.mean(statistics[key])))


def inference(args):
    """Inference a waveform.

    Args:
      workspace: str, directory of your workspace
      model_type: str, e.g. 'Google_onset_frame'
      loss_type: str, e.g. 'onset_frame_bce'
      augmentation: str, e.g. 'none'
      batch_size: int
      iteration: int, iteration of model to be loaded
      cuda: bool
    """

    # Arugments & parameters
    workspace = args.workspace
    model_type = args.model_type
    loss_type = args.loss_type
    augmentation = args.augmentation
    batch_size = args.batch_size
    iteration = args.iteration
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    audio_path = args.audio_path
    filename = args.filename
    
    sample_rate = config.sample_rate
    segment_seconds = config.segment_seconds
    segment_samples = int(segment_seconds * sample_rate)
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num

    # Paths
    checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
        model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        '{}_iterations.pth'.format(iteration))
    
    midi_path = 'results/{}.mid'.format(get_filename(audio_path))
    create_folder(os.path.dirname(midi_path))

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, frames_per_second=frames_per_second, 
        classes_num=classes_num)

    # Load model
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    if 'cuda' in str(device):
        model.to(device)
 
    # Load audio
    (audio, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    # Inference
    waveform_tester = WaveformTester(model, segment_samples, batch_size)
    output_dict = waveform_tester.forward(audio)

    # Postprocess
    post_processor = PostProcessor(frames_per_second, classes_num)
    (est_on_off_pairs, est_piano_notes) = post_processor.\
        output_dict_to_piano_notes(output_dict, frame_threshold=0.3)

    est_note_events = post_processor.on_off_pairs_notes_to_midi_events(
        est_on_off_pairs, est_piano_notes)

    # Write transcribed result to MIDI
    write_events_to_midi(start_time=0, note_events=est_note_events, midi_path=midi_path)
    print('Write out to {}'.format(midi_path))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train') 
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--loss_type', type=str, required=True)
    parser_train.add_argument('--augmentation', type=str, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--resume_iteration', type=int, required=True)
    parser_train.add_argument('--early_stop', type=int, required=True)
    parser_train.add_argument('--mini_data', action='store_true', default=False)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    
    parser_evalute = subparsers.add_parser('evaluate') 
    parser_evalute.add_argument('--workspace', type=str, required=True)
    parser_evalute.add_argument('--model_type', type=str, required=True)
    parser_evalute.add_argument('--loss_type', type=str, required=True)
    parser_evalute.add_argument('--augmentation', type=str, required=True)
    parser_evalute.add_argument('--batch_size', type=int, required=True)
    parser_evalute.add_argument('--iteration', type=int, required=True)
    parser_evalute.add_argument('--cuda', action='store_true', default=False)
    parser_evalute.add_argument('--mini_data', action='store_true', default=False)

    parser_evalute = subparsers.add_parser('inference') 
    parser_evalute.add_argument('--workspace', type=str, required=True)
    parser_evalute.add_argument('--model_type', type=str, required=True)
    parser_evalute.add_argument('--loss_type', type=str, required=True)
    parser_evalute.add_argument('--augmentation', type=str, required=True)
    parser_evalute.add_argument('--batch_size', type=int, required=True)
    parser_evalute.add_argument('--iteration', type=int, required=True)
    parser_evalute.add_argument('--cuda', action='store_true', default=False)
    parser_evalute.add_argument('--audio_path', type=str, required=True)

    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'calculate_scalar':
        calculate_scalar(args)

    elif args.mode == 'train':
        train(args)

    elif args.mode == 'evaluate':
        evaluate(args)

    elif args.mode == 'inference':
        inference(args)

    else:
        raise Exception('Error argument!')