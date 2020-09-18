import os
import argparse
import torch


def combine_note_and_pedal_models(args):
    """Combine trained note transcription and pedal transcription models to a 
    unified model.
    """

    # Arguments & parameters
    note_checkpoint_path = args.note_checkpoint_path
    pedal_checkpoint_path = args.pedal_checkpoint_path
    output_checkpoint_path = args.output_checkpoint_path

    # Load models
    note_checkpoint = torch.load(note_checkpoint_path, map_location='cpu')
    pedal_checkpoint = torch.load(pedal_checkpoint_path, map_location='cpu')

    # Combine to new model
    full_checkpoint = {
        'model': {
            'note_model': note_checkpoint['model'], 
            'pedal_model': pedal_checkpoint['model']}}

    os.makedirs(os.path.dirname(output_checkpoint_path), exist_ok=True)
    torch.save(full_checkpoint, output_checkpoint_path)
    print('Model saved to {}'.format(output_checkpoint_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--note_checkpoint_path', type=str, required=True)
    parser.add_argument('--pedal_checkpoint_path', type=str, required=True)
    parser.add_argument('--output_checkpoint_path', type=str, required=True)

    args = parser.parse_args()
    
    combine_note_and_pedal_models(args)