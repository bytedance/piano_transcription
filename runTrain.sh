#!bin/bash

DATASET_DIR="/local/maestro-v2/"

# Modify to your workspace
WORKSPACE="/content/drive/MyDrive/532Project/MelTraining/musicnetorig"

python3 pytorch/mainTest.py train --workspace=$WORKSPACE --model_type='Regress_onset_offset_frame_velocity_CRNN' --loss_type='regress_onset_offset_frame_velocity_bce' --augmentation='none' --max_note_shift=0 --batch_size=1 --learning_rate=5e-4 --reduce_iteration=10000 --resume_iteration=0 --early_stop=500000 --cuda
