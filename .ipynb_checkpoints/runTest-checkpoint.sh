
#!bin/bash

DATASET_DIR="/local/maestro-v2/"

# Modify to your workspace
WORKSPACE="/local/mel_project532s/"
NOTE_CHECKPOINT_PATH="/local/mel_project532s/checkpoints/main/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=none/max_note_shift=0/batch_size=6/250000_iterations.pth"
NOTE_PEDAL_CHECKPOINT_PATH="/local/CPSC532s_Results/MEL_RESULTS/Original_Mel/combined/250000_combined.pth"

python3 pytorch/combine_note_and_pedal_models.py --note_checkpoint_path=$NOTE_CHECKPOINT_PATH --output_checkpoint_path=$NOTE_PEDAL_CHECKPOINT_PATH


python3 pytorch/calculate_score_for_paper.py infer_prob --workspace=$WORKSPACE --model_type='Note_pedal' --checkpoint_path=$NOTE_PEDAL_CHECKPOINT_PATH --augmentation='none' --dataset='maestro' --split='test' --cuda

# Calculate metrics
python3 pytorch/calculate_score_for_paper.py calculate_metrics --workspace=$WORKSPACE --model_type='Note_pedal' --augmentation='none' --dataset='maestro' --split='test'