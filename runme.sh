#!bin/bash

# ============ Inference using pretrained model ============

MODEL_TYPE="Regress_onset_offset_frame_velocity_CRNN"
CHECKPOINT_PATH='/mnt/cephfs_new_wj/speechsv/kongqiuqiang/released_models/pub_piano_transcription/v2.0/Regress_onset_offset_frame_velocity_CRNN_onset_F1=0.9677.pth'
python3 pytorch/inference.py --model_type=$MODEL_TYPE --checkpoint_path=$CHECKPOINT_PATH --audio_path='examples/cut_liszt.wav' --checkpoint_path=$CHECKPOINT_PATH --cuda 

# ============ Train piano transcription systems from scratch ============

# MAESTRO dataset directory
DATASET_DIR="/mnt/cephfs_new_wj/speechsv/kongqiuqiang/datasets/maestro/dataset_root"

# Modify to your workspace
WORKSPACE="/mnt/cephfs_new_wj/speechsv/kongqiuqiang/workspaces/pub_piano_transcription_v2"

# Pack audio files to hdf5 format. This will speed up training.
python3 utils/features.py pack_maestro_dataset_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# Train
python3 pytorch/main.py train --workspace=$WORKSPACE --model_type='Regress_onset_offset_frame_velocity_CRNN' --loss_type='regress_onset_offset_frame_velocity_bce' --augmentation='none' --max_note_shift=0 --batch_size=12 --learning_rate=5e-4 --reduce_iteration=10000 --resume_iteration=0 --early_stop=300000 --cuda

# Evaluate
python3 pytorch/calculate_score_for_paper.py infer_prob --workspace=$WORKSPACE --filename='main' --model_type='Regress_onset_offset_frame_velocity_CRNN' --loss_type='regress_onset_offset_frame_velocity_bce' --augmentation='none' --max_note_shift=0 --batch_size=12 --split='test' --iteration=300000 --cuda

python3 pytorch/calculate_score_for_paper.py calculate_metrics --workspace=$WORKSPACE --filename='main' --model_type='Regress_onset_offset_frame_velocity_CRNN' --loss_type='regress_onset_offset_frame_velocity_bce' --augmentation='none' --max_note_shift=0 --batch_size=12 --split='test' --iteration=300000


####### Evaluation on MAPS #################
# MAPS_DATASET_DIR="/mnt/cephfs_new_wj/speechsv/kongqiuqiang/datasets/maps/dataset_root"
# python3 utils/features.py pack_maps_dataset_to_hdf5 --dataset_dir=$MAPS_DATASET_DIR --workspace=$WORKSPACE

