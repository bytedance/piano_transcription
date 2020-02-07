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
    checkpoint_path = args.checkpoint_path
    audio_path = args.audio_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    
    sample_rate = config.sample_rate
    segment_seconds = config.segment_seconds
    segment_samples = int(segment_seconds * sample_rate)
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num
    batch_size = 16

    # Paths
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

    # Inference probabilities
    waveform_tester = WaveformTester(model, segment_samples, batch_size)
    output_dict = waveform_tester.forward(audio)

    # Postprocess
    post_processor = PostProcessor(frames_per_second, classes_num)

    # Sharp onsets and offsets
    output_dict = post_processor.sharp_output_dict(
        output_dict, onset_threshold=0.1, offset_threshold=0.3)

    # Post process output_dict to piano notes
    (est_on_off_pairs, est_piano_notes) = post_processor.\
        output_dict_to_piano_notes(output_dict, frame_threshold=0.3)

    # Combine on and off pairs and piano notes to midi events.
    est_note_events = post_processor.on_off_pairs_notes_to_midi_events(
        est_on_off_pairs, est_piano_notes)

    # Write transcribed result to MIDI
    write_events_to_midi(start_time=0, note_events=est_note_events, midi_path=midi_path)
    print('Write out to {}'.format(midi_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()

    inference(args)