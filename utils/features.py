import numpy as np
import argparse
import csv
import os
import time
import logging
import h5py
import librosa
import logging

from utilities import (create_folder, float32_to_int16, create_logging, 
    get_filename, read_metadata, read_midi, read_maps_midi)
import config


def pack_maestro_dataset_to_hdf5(args):
    """Load & resample MAESTRO audio files, then write to hdf5 files.

    Args:
      dataset_dir: str, directory of dataset
      workspace: str, directory of your workspace
    """

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace

    sample_rate = config.sample_rate

    # Paths
    csv_path = os.path.join(dataset_dir, 'maestro-v2.0.0.csv')
    waveform_hdf5s_dir = os.path.join(workspace, 'hdf5s', 'maestro')

    logs_dir = os.path.join(workspace, 'logs', get_filename(__file__))
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    # Read meta dict
    meta_dict = read_metadata(csv_path)

    audios_num = len(meta_dict['canonical_composer'])
    logging.info('Total audios number: {}'.format(audios_num))

    feature_time = time.time()

    # Load & resample each audio file to a hdf5 file
    for n in range(audios_num):
        logging.info('{} {}'.format(n, meta_dict['midi_filename'][n]))

        # Read midi
        midi_path = os.path.join(dataset_dir, meta_dict['midi_filename'][n])
        midi_dict = read_midi(midi_path)

        # Load audio
        audio_path = os.path.join(dataset_dir, meta_dict['audio_filename'][n])
        (audio, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

        packed_hdf5_path = os.path.join(waveform_hdf5s_dir, '{}.h5'.format(
            os.path.splitext(meta_dict['audio_filename'][n])[0]))

        create_folder(os.path.dirname(packed_hdf5_path))

        with h5py.File(packed_hdf5_path, 'w') as hf:
            hf.attrs.create('canonical_composer', data=meta_dict['canonical_composer'][n].encode(), dtype='S100')
            hf.attrs.create('canonical_title', data=meta_dict['canonical_title'][n].encode(), dtype='S100')
            hf.attrs.create('split', data=meta_dict['split'][n].encode(), dtype='S20')
            hf.attrs.create('year', data=meta_dict['year'][n].encode(), dtype='S10')
            hf.attrs.create('midi_filename', data=meta_dict['midi_filename'][n].encode(), dtype='S100')
            hf.attrs.create('audio_filename', data=meta_dict['audio_filename'][n].encode(), dtype='S100')
            hf.attrs.create('duration', data=meta_dict['duration'][n], dtype=np.float32)

            hf.create_dataset(name='midi_event', data=[e.encode() for e in midi_dict['midi_event']], dtype='S100')
            hf.create_dataset(name='midi_event_time', data=midi_dict['midi_event_time'], dtype=np.float32)
            hf.create_dataset(name='waveform', data=float32_to_int16(audio), dtype=np.int16)
        
    logging.info('Write hdf5 to {}'.format(packed_hdf5_path))
    logging.info('Time: {:.3f} s'.format(time.time() - feature_time))


def pack_maps_dataset_to_hdf5(args):
    """MAPS is a piano dataset only used for evaluating our piano transcription
    system (optional). Ref:

    [1] Emiya, Valentin. "MAPS Database A piano database for multipitch 
    estimation and automatic transcription of music. 2016

    Load & resample MAPS audio files, then write to hdf5 files.

    Args:
      dataset_dir: str, directory of dataset
      workspace: str, directory of your workspace
    """

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace

    sample_rate = config.sample_rate
    pianos = ['ENSTDkCl', 'ENSTDkAm']

    # Paths
    waveform_hdf5s_dir = os.path.join(workspace, 'hdf5s', 'maps')

    logs_dir = os.path.join(workspace, 'logs', get_filename(__file__))
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    feature_time = time.time()
    count = 0

    # Load & resample each audio file to a hdf5 file
    for piano in pianos:
        sub_dir = os.path.join(dataset_dir, piano, 'MUS')

        audio_names = [os.path.splitext(name)[0] for name in os.listdir(sub_dir) 
            if os.path.splitext(name)[-1] == '.mid']
        
        for audio_name in audio_names:
            print('{} {}'.format(count, audio_name))
            audio_path = '{}.wav'.format(os.path.join(sub_dir, audio_name))
            midi_path = '{}.mid'.format(os.path.join(sub_dir, audio_name))

            (audio, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
            midi_dict = read_maps_midi(midi_path)
            
            packed_hdf5_path = os.path.join(waveform_hdf5s_dir, '{}.h5'.format(audio_name))
            create_folder(os.path.dirname(packed_hdf5_path))

            with h5py.File(packed_hdf5_path, 'w') as hf:
                hf.attrs.create('split', data='test'.encode(), dtype='S20')
                hf.attrs.create('midi_filename', data='{}.mid'.format(audio_name).encode(), dtype='S100')
                hf.attrs.create('audio_filename', data='{}.wav'.format(audio_name).encode(), dtype='S100')
                hf.create_dataset(name='midi_event', data=[e.encode() for e in midi_dict['midi_event']], dtype='S100')
                hf.create_dataset(name='midi_event_time', data=midi_dict['midi_event_time'], dtype=np.float32)
                hf.create_dataset(name='waveform', data=float32_to_int16(audio), dtype=np.int16)
            
            count += 1

    logging.info('Write hdf5 to {}'.format(packed_hdf5_path))
    logging.info('Time: {:.3f} s'.format(time.time() - feature_time))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    parser_pack_maestro = subparsers.add_parser('pack_maestro_dataset_to_hdf5')
    parser_pack_maestro.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_pack_maestro.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')

    parser_pack_maps = subparsers.add_parser('pack_maps_dataset_to_hdf5')
    parser_pack_maps.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_pack_maps.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')

    # Parse arguments
    args = parser.parse_args()
    
    if args.mode == 'pack_maestro_dataset_to_hdf5':
        pack_maestro_dataset_to_hdf5(args)
        
    elif args.mode == 'pack_maps_dataset_to_hdf5':
        pack_maps_dataset_to_hdf5(args)

    else:
        raise Exception('Incorrect arguments!')