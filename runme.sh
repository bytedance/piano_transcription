#!bin/bash

# ============ Inference using pretrained model ============
# Inference with note and pedal
MODEL_TYPE="Note_pedal"
CHECKPOINT_PATH='released_models/note_F1=0.9677_pedal_F1=0.8658.pth'
python3 pytorch/inference.py --model_type=$MODEL_TYPE --checkpoint_path=$CHECKPOINT_PATH --audio_path='resources/cut_liszt.mp3' --checkpoint_path=$CHECKPOINT_PATH --cuda

# Inference without pedal
MODEL_TYPE="Regress_onset_offset_frame_velocity_CRNN"
CHECKPOINT_PATH='released_models/Regress_onset_offset_frame_velocity_CRNN_onset_F1=0.9677.pth'
python3 pytorch/inference.py --model_type=$MODEL_TYPE --checkpoint_path=$CHECKPOINT_PATH --audio_path='resources/cut_liszt.mp3' --checkpoint_path=$CHECKPOINT_PATH --cuda

# ============ 1. Train piano transcription system from scratch ============

# MAESTRO dataset directory
DATASET_DIR="datasets/maestro/dataset_root"

# Modify to your workspace
WORKSPACE="workspaces/piano_transcription"

# Pack audio files to hdf5 format. This will speed up training.
python3 utils/features.py pack_maestro_dataset_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# Train
python3 pytorch/main.py train --workspace=$WORKSPACE --model_type='Regress_onset_offset_frame_velocity_CRNN' --loss_type='regress_onset_offset_frame_velocity_bce' --augmentation='none' --max_note_shift=0 --batch_size=12 --learning_rate=5e-4 --reduce_iteration=10000 --resume_iteration=0 --early_stop=300000 --cuda

# Plot statistics
python3 utils/plot_statistics.py plot --workspace=$WORKSPACE --select=1a

# ============ 2. Train piano pedal transcription system from scratch ============
# Train pedal transcription
python3 pytorch/main.py train --workspace=$WORKSPACE --model_type='Regress_pedal_CRNN' --loss_type='regress_pedal_bce' --augmentation='none' --max_note_shift=0 --batch_size=12 --learning_rate=5e-4 --reduce_iteration=10000 --resume_iteration=0 --early_stop=300000 --cuda

# ============ 3. Combine the note and pedal models ============
NOTE_CHECKPOINT_PATH="released_models/Regress_onset_offset_frame_velocity_CRNN_onset_F1=0.9677.pth"
PEDAL_CHECKPOINT_PATH="released_models/Regress_pedal_CRNN_onset_F1=0.8658.pth"
NOTE_PEDAL_CHECKPOINT_PATH="released_models/note_F1=0.9677_pedal_F1=0.8658.pth"
python3 pytorch/combine_note_and_pedal_models.py --note_checkpoint_path=$NOTE_CHECKPOINT_PATH --pedal_checkpoint_path=$PEDAL_CHECKPOINT_PATH --output_checkpoint_path=$NOTE_PEDAL_CHECKPOINT_PATH

# Inference probability for evaluation
# PROBS_DIR=$WORKSPACE"/probs/Note_pedal"
# python3 pytorch/calculate_score_for_paper.py infer_prob --workspace=$WORKSPACE --model_type='Note_pedal' --checkpoint_path=$NOTE_PEDAL_CHECKPOINT_PATH --probs_dir=$PROBS_DIR --split='test' --cuda
python3 pytorch/calculate_score_for_paper.py infer_prob --workspace=$WORKSPACE --model_type='Note_pedal' --checkpoint_path=$NOTE_PEDAL_CHECKPOINT_PATH --dataset='maestro' --split='test' --cuda

# Calculate metrics
python3 pytorch/calculate_score_for_paper.py calculate_metrics --workspace=$WORKSPACE --model_type='Note_pedal' --dataset='maestro' --split='test'
python3 pytorch/calculate_score_for_paper.py calculate_metrics --workspace=$WORKSPACE --model_type='Note_pedal' --dataset='maps' --split='test'