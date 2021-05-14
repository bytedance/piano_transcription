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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from utilities import (create_folder, get_filename, create_logging, 
    StatisticsContainer, RegressionPostProcessor)
from cfp import feature_extraction, create_batches
from data_generator import MaestroDataset, Augmentor, Sampler, TestSampler, collate_fn
from models import Regress_onset_offset_frame_velocity_CRNN, Regress_pedal_CRNN
from pytorch_utils import move_data_to_device
from losses import get_loss_func
from evaluate import SegmentEvaluator
import config
import wave

from IPython import display

def train(args):
    """Train a piano transcription system.

    Args:
      workspace: str, directory of your workspace
      model_type: str, e.g. 'Regressonset_regressoffset_frame_velocity_CRNN'
      loss_type: str, e.g. 'regress_onset_offset_frame_velocity_bce'
      augmentation: str, e.g. 'none'
      batch_size: int
      learning_rate: float
      reduce_iteration: int
      resume_iteration: int
      early_stop: int
      device: 'cuda' | 'cpu'
      mini_data: bool
    """
    axs=[]
    f = plt.figure(figsize=(10,10))
    axs.append(f.add_subplot(2,4,1))
    axs.append(f.add_subplot(2,4,2))
    axs.append(f.add_subplot(2,4,3))
    axs.append(f.add_subplot(2,4,4))
    axs.append(f.add_subplot(2,4,5))
    axs.append(f.add_subplot(2,4,6))
    axs.append(f.add_subplot(2,4,7))
    losses = []
    
    val_frame = []
    val_reg_onset = []
    val_reg_offset = []
    
    train_frame = []
    train_reg_onset = []
    train_reg_offset = []
    
    # Arugments & parameters
    workspace = args.workspace
    model_type = args.model_type
    loss_type = args.loss_type
    augmentation = args.augmentation
    max_note_shift = args.max_note_shift
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    reduce_iteration = args.reduce_iteration
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
    num_workers = 8

    # Loss function
    loss_func = get_loss_func(loss_type)

    # Paths
    hdf5s_dir = os.path.join(workspace, 'hdf5s', 'maestro')

    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 
        model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 
        'max_note_shift={}'.format(max_note_shift),
        'batch_size={}'.format(batch_size))
    create_folder(checkpoints_dir)

    statistics_path = os.path.join(workspace, 'statistics', filename, 
        model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 
        'max_note_shift={}'.format(max_note_shift), 
        'batch_size={}'.format(batch_size), 'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(workspace, 'logs', filename, 
        model_type, 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 
        'max_note_shift={}'.format(max_note_shift), 
        'batch_size={}'.format(batch_size))
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
    model = Model(frames_per_second=frames_per_second, classes_num=classes_num)

    if augmentation == 'none':
        augmentor = None
    elif augmentation == 'aug':
        augmentor = Augmentor()
    else:
        raise Exception('Incorrect argumentation!')
    
    # Dataset
    train_dataset = MaestroDataset(hdf5s_dir=hdf5s_dir, 
        segment_seconds=segment_seconds, frames_per_second=frames_per_second, 
        max_note_shift=max_note_shift, augmentor=augmentor)

    evaluate_dataset = MaestroDataset(hdf5s_dir=hdf5s_dir, 
        segment_seconds=segment_seconds, frames_per_second=frames_per_second, 
        max_note_shift=0)

    # Sampler for training
    train_sampler = Sampler(hdf5s_dir=hdf5s_dir, split='train', 
        segment_seconds=segment_seconds, hop_seconds=hop_seconds, 
        batch_size=batch_size, mini_data=mini_data)

    # Sampler for evaluation
    evaluate_train_sampler = TestSampler(hdf5s_dir=hdf5s_dir, 
        split='train', segment_seconds=segment_seconds, hop_seconds=hop_seconds, 
        batch_size=batch_size, mini_data=mini_data)

    evaluate_validate_sampler = TestSampler(hdf5s_dir=hdf5s_dir, 
        split='validation', segment_seconds=segment_seconds, hop_seconds=hop_seconds, 
        batch_size=batch_size, mini_data=mini_data)

    evaluate_test_sampler = TestSampler(hdf5s_dir=hdf5s_dir, 
        split='test', segment_seconds=segment_seconds, hop_seconds=hop_seconds, 
        batch_size=batch_size, mini_data=mini_data)

    # Dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
        batch_sampler=train_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    evaluate_train_loader = torch.utils.data.DataLoader(dataset=evaluate_dataset, 
        batch_sampler=evaluate_train_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    validate_loader = torch.utils.data.DataLoader(dataset=evaluate_dataset, 
        batch_sampler=evaluate_validate_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(dataset=evaluate_dataset, 
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

        # print("wave shape", batch_data_dict['feature'][0].shape)
        # print(len(batch_data_dict['feature']))
        features_batch = torch.empty(0,256,384,2)
        for i in range(len(batch_data_dict['feature'])):
            feature = batch_data_dict['feature'][i]
            # print(feature.shape)
            features = create_batches(feature[:,:,[1, 3]], b_size=1, timesteps=256, feature_num=384)
            features_batch = torch.cat((features_batch.float(), torch.from_numpy(features[0]).float()))
            # print(features_batch.shape)

        # print(features_batch.shape)
        # print(batch_data_dict['reg_onset_roll'].shape)
        # # print("frame_output ", output_dict['frame_output'].shape)
        # print(batch_data_dict['frame_roll'].shape)
        # print(batch_data_dict['mask_roll'].shape)
    
        # with wave.open("/content/sound1.wav", "w") as f:
        #     # 2 Channels.
        #     f.setnchannels(2)
        #     # 2 bytes per sample.
        #     f.setsampwidth(2)
        #     f.setframerate(44100)
        #     f.writeframes(batch_data_dict['waveform'].tobytes())
        # break
        
        # Evaluation 
        if iteration % 5000 == 0:# and iteration > 0:   5000
            logging.info('------------------------------------')
            logging.info('Iteration: {}'.format(iteration))

            train_fin_time = time.time()

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

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'Train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(train_time, validate_time))

            train_bgn_time = time.time()
            
            val_frame.append(validate_statistics['frame_ap'])
            val_reg_onset.append(validate_statistics['reg_onset_mae'])
            val_reg_offset.append(validate_statistics['reg_offset_mae'])
            
            train_frame.append(evaluate_train_statistics['frame_ap'])
            train_reg_onset.append(evaluate_train_statistics['reg_onset_mae'])
            train_reg_offset.append(evaluate_train_statistics['reg_offset_mae'])
        
        # Save model
        if iteration % 10000 == 0:
            checkpoint = {
                'iteration': iteration, 
                'model': model.module.state_dict(), 
                'sampler': train_sampler.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))
        
        # # Reduce learning rate
        if iteration % reduce_iteration == 0 and iteration > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9
        
        # Move data to device
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
        features_batch = move_data_to_device(features_batch.float(), device)
        
        model.train()
        batch_output_dict = model(features_batch)

        loss = loss_func(model, batch_output_dict, batch_data_dict)

        # print(iteration, loss)

        # Backward
        loss.backward()
        
        losses.append(loss.item())

        if iteration % 100 == 0:
            #print(loss, iteration, "LOSS")
            axs[0].plot(losses)
            axs[0].set_yscale('log')
            axs[0].set_xlabel("losses")
            # clear output window and diplay updated figure
            axs[1].plot(val_frame)
            axs[1].set_yscale('log')
            axs[1].set_xlabel("val_frame")
            
            axs[2].plot(val_reg_onset)
            axs[2].set_yscale('log')
            axs[2].set_xlabel("val_reg_onset")
            # clear output window and diplay updated figure
            axs[3].plot(val_reg_offset)
            axs[3].set_yscale('log')
            axs[3].set_xlabel("val_reg_offset")  
        
            axs[4].plot(train_frame)
            axs[4].set_yscale('log')
            axs[4].set_xlabel("train_frame") 
            # clear output window and diplay updated figure
            axs[5].plot(train_reg_onset)
            axs[5].set_yscale('log')
            axs[5].set_xlabel("train_reg_onset")   
            # clear output window and diplay updated figure
            axs[6].plot(train_reg_offset)
            axs[6].set_yscale('log')
            axs[6].set_xlabel("train_reg_offset")  
            
            plt.savefig('/local/CPSC532s_Results/Attention_Gru/Attention_Gru.png')
            #plt.show()
        
        if iteration % 100 == 0:
            print(loss, iteration, "LOSS")
        
        if iteration % 2 != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Stop learning
        # if iteration == early_stop:
        #     break

        iteration += 1
        # print("iteration", iteration)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train') 
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--loss_type', type=str, required=True)
    parser_train.add_argument('--augmentation', type=str, required=True, choices=['none', 'aug'])
    parser_train.add_argument('--max_note_shift', type=int, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--reduce_iteration', type=int, required=True)
    parser_train.add_argument('--resume_iteration', type=int, required=True)
    parser_train.add_argument('--early_stop', type=int, required=True)
    parser_train.add_argument('--mini_data', action='store_true', default=False)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    else:
        raise Exception('Error argument!')