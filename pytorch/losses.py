import torch
import torch.nn.functional as F


def bce(output, target, mask):
    """Binary crossentropy (BCE) with mask. The positions with mask=0 will be 
    deactivated when calculation BCE.
    """
    eps = 1e-7
    output = torch.clamp(output, eps, 1. - eps)
    matrix = - target * torch.log(output) - (1. - target) * torch.log(1. - output)
    return torch.sum(matrix * mask) / torch.sum(mask)


def onset_frame_bce(model, output_dict, target_dict):
    return bce(output_dict['frame_output'], target_dict['frame_roll'], target_dict['mask_roll']) + \
        bce(output_dict['onset_output'], target_dict['onset_roll'], target_dict['mask_roll'])


def get_loss_func(loss_type):
    if loss_type == 'onset_frame_bce':
        return onset_frame_bce

    else:
        raise Exception('Incorrect loss_type!')