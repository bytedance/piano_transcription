import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import librosa
import torch
 
from utilities import (create_folder, get_filename, PostProcessor, 
    write_events_to_midi)
from models import Google_onset_frame
from pytorch_utils import WaveformTester
import config


def inference(args):
    """Inference a waveform.

    Args:
      cuda: bool
      audio_path: str
    """

    # Arugments & parameters
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    audio_path = args.audio_path
    
    sample_rate = config.sample_rate
    segment_seconds = config.segment_seconds
    segment_samples = int(segment_seconds * sample_rate)
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num
    batch_size = 4

    # Paths
    checkpoint_path = '/mnt/cephfs_new_wj/speechsv/kongqiuqiang/workspaces/piano_transcription/checkpoints/main/CnnGoogle_onset_frame/loss_type=onset_frame_bce/augmentation=none/batch_size=32/100000_iterations.pth'
    
    midi_path = 'results/{}.mid'.format(get_filename(audio_path))
    create_folder(os.path.dirname(midi_path))

    if 'cuda' in str(device):
        print('Using GPU.')
        device = 'cuda'
    else:
        print('Using CPU.')
        device = 'cpu'

    # Model
    Model = Google_onset_frame
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

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--audio_path', type=str, required=True)
    args = parser.parse_args()

    inference(args)