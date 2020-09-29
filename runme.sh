#!bin/bash

# ============ Inference using pretrained model ============
# Download checkpoint and inference
CHECKPOINT_PATH="CRNN_note_F1=0.9677_pedal_F1=0.9186.pth"
wget -O $CHECKPOINT_PATH "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
MODEL_TYPE="Note_pedal"
python3 pytorch/inference.py --model_type=$MODEL_TYPE --checkpoint_path=$CHECKPOINT_PATH --audio_path='resources/cut_liszt.mp3' --cuda

# ============ Train piano transcription system from scratch ============
# MAESTRO dataset directory. Users need to download MAESTRO dataset into this folder.
DATASET_DIR="./datasets/maestro/dataset_root"

# Modify to your workspace
WORKSPACE="./workspaces/piano_transcription"

# Pack audio files to hdf5 format for training
python3 utils/features.py pack_maestro_dataset_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# --- 1. Train note transcription system ---
python3 pytorch/main.py train --workspace=$WORKSPACE --model_type='Regress_onset_offset_frame_velocity_CRNN' --loss_type='regress_onset_offset_frame_velocity_bce' --augmentation='none' --max_note_shift=0 --batch_size=12 --learning_rate=5e-4 --reduce_iteration=10000 --resume_iteration=0 --early_stop=300000 --cuda

# --- 2. Train pedal transcription system ---
python3 pytorch/main.py train --workspace=$WORKSPACE --model_type='Regress_pedal_CRNN' --loss_type='regress_pedal_bce' --augmentation='none' --max_note_shift=0 --batch_size=12 --learning_rate=5e-4 --reduce_iteration=10000 --resume_iteration=0 --early_stop=300000 --cuda

# --- 3. Combine the note and pedal models ---
# Users should copy and rename the following paths to their trained model paths
NOTE_CHECKPOINT_PATH="Regress_onset_offset_frame_velocity_CRNN_onset_F1=0.9677.pth"
PEDAL_CHECKPOINT_PATH="Regress_pedal_CRNN_onset_F1=0.9186.pth"
NOTE_PEDAL_CHECKPOINT_PATH="CRNN_note_F1=0.9677_pedal_F1=0.9186.pth"
python3 pytorch/combine_note_and_pedal_models.py --note_checkpoint_path=$NOTE_CHECKPOINT_PATH --pedal_checkpoint_path=$PEDAL_CHECKPOINT_PATH --output_checkpoint_path=$NOTE_PEDAL_CHECKPOINT_PATH

# ============ Evaluate (optional) ============
# Inference probability for evaluation
python3 pytorch/calculate_score_for_paper.py infer_prob --workspace=$WORKSPACE --model_type='Note_pedal' --checkpoint_path=$NOTE_PEDAL_CHECKPOINT_PATH --augmentation='none' --dataset='maestro' --split='test' --cuda

# Calculate metrics
python3 pytorch/calculate_score_for_paper.py calculate_metrics --workspace=$WORKSPACE --model_type='Note_pedal' --augmentation='aug' --dataset='maestro' --split='test'
python3 pytorch/calculate_score_for_paper.py calculate_metrics --workspace=$WORKSPACE --model_type='Note_pedal' --augmentation='aug' --dataset='maps' --split='test'