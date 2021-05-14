import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import time
import librosa
import torch
import torch.nn as nn
from cfp import feature_extraction, create_batches

from utilities import pad_truncate_sequence


def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


def append_to_dict(dict, key, value):
    
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]

 
def forward_dataloader(model, dataloader, batch_size, return_target=True):
    """Forward data generated from dataloader to model.

    Args:
      model: object
      dataloader: object, used to generate mini-batches for evaluation.
      batch_size: int
      return_target: bool

    Returns:
      output_dict: dict, e.g. {
        'frame_output': (segments_num, frames_num, classes_num),
        'onset_output': (segments_num, frames_num, classes_num),
        'frame_roll': (segments_num, frames_num, classes_num),
        'onset_roll': (segments_num, frames_num, classes_num),
        ...}
    """

    output_dict = {}
    device = next(model.parameters()).device

    for n, batch_data_dict in enumerate(dataloader):

        features_batch = torch.empty(0,256,384,2)
        for i in range(len(batch_data_dict['feature'])):
          feature = batch_data_dict['feature'][i]
          features = create_batches(feature[:,:,[1, 3]], b_size=1, timesteps=256, feature_num=384)
          features_batch = torch.cat((features_batch, torch.from_numpy(features[0])))
        
        batch_feature = move_data_to_device(features_batch.float(), device)

        with torch.no_grad():
            model.eval()
            batch_output_dict = model(batch_feature)

        for key in batch_output_dict.keys():
            if '_list' not in key:
                append_to_dict(output_dict, key, 
                    batch_output_dict[key].data.cpu().numpy())

        if return_target:
            for target_type in batch_data_dict.keys():
                if 'roll' in target_type or 'reg_distance' in target_type or \
                    'reg_tail' in target_type:
                    append_to_dict(output_dict, target_type, 
                        batch_data_dict[target_type])

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)
    
    return output_dict


def forward(model, x, batch_size):
    """Forward data to model in mini-batch. 
    
    Args: 
      model: object
      x: (N, segment_samples)
      batch_size: int

    Returns:
      output_dict: dict, e.g. {
        'frame_output': (segments_num, frames_num, classes_num),
        'onset_output': (segments_num, frames_num, classes_num),
        ...}
    """
    
    output_dict = {}
    device = next(model.parameters()).device
    
    pointer = 0
    while True:
        if pointer >= len(x):
            break

        batch_feature = move_data_to_device(x[pointer : pointer + batch_size], device)
        pointer += batch_size

        with torch.no_grad():
            model.eval()
            batch_output_dict = model(batch_feature)

        for key in batch_output_dict.keys():
            # if '_list' not in key:
            append_to_dict(output_dict, key, batch_output_dict[key].data.cpu().numpy())

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)

    return output_dict
