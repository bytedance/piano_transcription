#!bin/bash

DATASET_DIR="/local/musicnet/"

# Modify to your workspace
WORKSPACE="/local/mel_musicnet/"

python3 utils/features.py pack_maestro_dataset_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

