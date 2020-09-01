from losses import *


def google_onset_offset_frame_velocity_bce(model, output_dict, target_dict):
    """Total loss contains onset regression, offset regression, framewise
    classification and velocity regression.
    """
    onset_loss = bce(output_dict['reg_onset_output'], target_dict['onset_roll'], target_dict['mask_roll'])
    offset_loss = bce(output_dict['reg_offset_output'], target_dict['offset_roll'], target_dict['mask_roll'])
    frame_loss = bce(output_dict['frame_output'], target_dict['frame_roll'], target_dict['mask_roll'])
    velocity_loss = bce(output_dict['velocity_output'], target_dict['velocity_roll'] / 128, target_dict['onset_roll'])
    total_loss = onset_loss + offset_loss + frame_loss + velocity_loss
    return total_loss


def get_loss_func(loss_type):
    if loss_type == 'google_onset_offset_frame_velocity_bce':
        return google_onset_offset_frame_velocity_bce

    else:
        raise Exception('Incorrect loss_type!')