#!bin/bash

# MAESTRO dataset directory
DATASET_DIR="/mnt/cephfs_new_wj/speechsv/kongqiuqiang/datasets/maestro/dataset_root"

# Modify to your workspace
WORKSPACE="/mnt/cephfs_new_wj/speechsv/kongqiuqiang/workspaces/pub_piano_transcription"

# Pack audio files to hdf5 format. This will speed up training.
python3 utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# ============ Inference using pretrained model ============
python3 pytorch/main_inference.py --cuda --audio_path='examples/cut_liszt.wav'

# ============ Train ============
python3 pytorch/main.py train --workspace=$WORKSPACE --model_type='Google_onset_frame' --loss_type='onset_frame_bce' --augmentation='none' --batch_size=16 --learning_rate=1e-3 --resume_iteration=0 --early_stop=2000000 --cuda

# Plot training statistics
python3 utils/plot_statistics.py plot --workspace=$WORKSPACE --select=1a

# ============ Evaluate ============
python3 pytorch/main.py evaluate --workspace=$WORKSPACE --model_type='Google_onset_frame' --loss_type='onset_frame_bce' --augmentation='none' --batch_size=16 --iteration=80000 --cuda

# ============ Inference ============
python3 pytorch/main.py inference --workspace=$WORKSPACE --model_type='Google_onset_frame' --loss_type='onset_frame_bce' --augmentation='none' --batch_size=16 --iteration=80000 --cuda --audio_path='examples/cut_liszt.wav'