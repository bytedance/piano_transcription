
#!bin/bash

DATASET_DIR="/content/drive/MyDrive/532Project/maestro_full/maestro-v2.0.0/"

# Modify to your workspace
WORKSPACE="/content/drive/MyDrive/532Project/MelTraining/workspaceOrigMel"
NOTE_CHECKPOINT_PATH="/content/drive/MyDrive/532Project/MelTraining/workspaceOrigMel/checkpoints/main/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=none/max_note_shift=0/batch_size=6/41000_iterations.pth"
NOTE_PEDAL_CHECKPOINT_PATH="/content/drive/MyDrive/Mel/41000_combined.pth"

python3 pytorch/combine_note_and_pedal_models.py --note_checkpoint_path=$NOTE_CHECKPOINT_PATH --output_checkpoint_path=$NOTE_PEDAL_CHECKPOINT_PATH


python3 pytorch/calculate_score_for_paper.py infer_prob --workspace=$WORKSPACE --model_type='Note_pedal' --checkpoint_path=$NOTE_PEDAL_CHECKPOINT_PATH --augmentation='none' --dataset='maestro' --split='test' --cuda

# Calculate metrics
python3 pytorch/calculate_score_for_paper.py calculate_metrics --workspace=$WORKSPACE --model_type='Note_pedal' --augmentation='none' --dataset='maestro' --split='test'