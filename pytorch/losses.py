import torch
import torch.nn.functional as F


def bce(output, target, mask):
    """Binary crossentropy (BCE) with mask. The positions where mask=0 will be 
    deactivated when calculation BCE.
    """
    eps = 1e-7
    output = torch.clamp(output, eps, 1. - eps)
    matrix = - target * torch.log(output) - (1. - target) * torch.log(1. - output)
    return torch.sum(matrix * mask) / torch.sum(mask)

############ High-resolution regression loss ############
def regress_onset_offset_frame_velocity_bce(model, output_dict, target_dict):
    """High-resolution piano note regression loss, including onset regression, 
    offset regression, velocity regression and frame-wise classification losses.
    """
    onset_loss = bce(output_dict['reg_onset_output'], target_dict['reg_onset_roll'], target_dict['mask_roll'])
    offset_loss = bce(output_dict['reg_offset_output'], target_dict['reg_offset_roll'], target_dict['mask_roll'])
    frame_loss = bce(output_dict['frame_output'], target_dict['frame_roll'], target_dict['mask_roll'])
    velocity_loss = bce(output_dict['velocity_output'], target_dict['velocity_roll'] / 128, target_dict['onset_roll'])
    total_loss = onset_loss + offset_loss + frame_loss + velocity_loss
    return total_loss


def regress_pedal_bce(model, output_dict, target_dict):
    """High-resolution piano pedal regression loss, including pedal onset 
    regression, pedal offset regression and pedal frame-wise classification losses.
    """
    onset_pedal_loss = F.binary_cross_entropy(output_dict['reg_pedal_onset_output'], target_dict['reg_pedal_onset_roll'][:, :, None])
    offset_pedal_loss = F.binary_cross_entropy(output_dict['reg_pedal_offset_output'], target_dict['reg_pedal_offset_roll'][:, :, None])
    frame_pedal_loss = F.binary_cross_entropy(output_dict['pedal_frame_output'], target_dict['pedal_frame_roll'][:, :, None])
    total_loss = onset_pedal_loss + offset_pedal_loss + frame_pedal_loss
    return total_loss

############ Google's onsets and frames system loss ############
def google_onset_offset_frame_velocity_bce(model, output_dict, target_dict):
    """Google's onsets and frames system piano note loss. Only used for comparison.
    """
    onset_loss = bce(output_dict['reg_onset_output'], target_dict['onset_roll'], target_dict['mask_roll'])
    offset_loss = bce(output_dict['reg_offset_output'], target_dict['offset_roll'], target_dict['mask_roll'])
    frame_loss = bce(output_dict['frame_output'], target_dict['frame_roll'], target_dict['mask_roll'])
    velocity_loss = bce(output_dict['velocity_output'], target_dict['velocity_roll'] / 128, target_dict['onset_roll'])
    total_loss = onset_loss + offset_loss + frame_loss + velocity_loss
    return total_loss


def google_pedal_bce(model, output_dict, target_dict):
    """Google's onsets and frames system piano pedal loss. Only used for comparison.
    """
    onset_pedal_loss = F.binary_cross_entropy(output_dict['reg_pedal_onset_output'], target_dict['pedal_onset_roll'][:, :, None])
    offset_pedal_loss = F.binary_cross_entropy(output_dict['reg_pedal_offset_output'], target_dict['pedal_offset_roll'][:, :, None])
    frame_pedal_loss = F.binary_cross_entropy(output_dict['pedal_frame_output'], target_dict['pedal_frame_roll'][:, :, None])
    total_loss = onset_pedal_loss + offset_pedal_loss + frame_pedal_loss
    return total_loss


def get_loss_func(loss_type):
    if loss_type == 'regress_onset_offset_frame_velocity_bce':
        return regress_onset_offset_frame_velocity_bce

    elif loss_type == 'regress_pedal_bce':
        return regress_pedal_bce

    elif loss_type == 'google_onset_offset_frame_velocity_bce':
        return google_onset_offset_frame_velocity_bce

    elif loss_type == 'google_pedal_bce':
        return google_pedal_bce

    else:
        raise Exception('Incorrect loss_type!')