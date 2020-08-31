CHECKPOINT_PATH='/mnt/cephfs_new_wj/speechsv/kongqiuqiang/released_models/pub_piano_transcription/v3.0/note_F1=0.9677_pedal_F1=0.8658.pth'

DATASET_DIR="/mnt/cephfs_new_wj/speechsv/kongqiuqiang/datasets/maestro/dataset_root"

# Modify to your workspace
WORKSPACE="/mnt/cephfs_new_wj/speechsv/kongqiuqiang/workspaces/pub_piano_transcription_v2"

MAPS_DATASET_DIR="/mnt/cephfs_new_wj/speechsv/kongqiuqiang/datasets/maps/dataset_root"
python3 utils/features.py pack_maps_dataset_to_hdf5 --dataset_dir=$MAPS_DATASET_DIR --workspace=$WORKSPACE

NOTE_PEDAL_CHECKPOINT_PATH="/mnt/cephfs_new_wj/speechsv/kongqiuqiang/released_models/pub_piano_transcription/v3.0/note_F1=0.9677_pedal_F1=0.8658.pth"

CUDA_VISIBLE_DEVICES=3 python3 pytorch/calculate_score_for_paper.py infer_prob --workspace=$WORKSPACE --model_type='Note_pedal' --checkpoint_path=$NOTE_PEDAL_CHECKPOINT_PATH --augmentation='aug' --dataset='maestro' --split='test' --cuda

CUDA_VISIBLE_DEVICES=3 python3 pytorch/calculate_score_for_paper.py infer_prob --workspace=$WORKSPACE --model_type='Note_pedal' --checkpoint_path=$NOTE_PEDAL_CHECKPOINT_PATH --augmentation='aug' --dataset='maps' --split='test' --cuda

python3 pytorch/calculate_score_for_paper.py calculate_metrics --workspace=$WORKSPACE --model_type='Note_pedal' --augmentation='aug' --dataset='maestro' --split='test'
python3 pytorch/calculate_score_for_paper.py calculate_metrics --workspace=$WORKSPACE --model_type='Note_pedal' --augmentation='aug' --dataset='maps' --split='test'

NOTE_CHECKPOINT_PATH="/mnt/cephfs_new_wj/speechsv/kongqiuqiang/workspaces/pub_piano_transcription_v2/checkpoints/main/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=0/batch_size=12/300000_iterations.pth"
PEDAL_CHECKPOINT_PATH="/mnt/cephfs_new_wj/speechsv/kongqiuqiang/workspaces/pub_piano_transcription_v2/checkpoints/main/Regress_pedal_CRNN/loss_type=regress_pedal_bce/augmentation=aug/max_note_shift=0/batch_size=12/300000_iterations.pth"
NOTE_PEDAL_CHECKPOINT_PATH=$WORKSPACE"/_models/note_pedal_300k_iterations.pth"
python3 pytorch/combine_note_and_pedal_models.py --note_checkpoint_path=$NOTE_CHECKPOINT_PATH --pedal_checkpoint_path=$PEDAL_CHECKPOINT_PATH --output_checkpoint_path=$NOTE_PEDAL_CHECKPOINT_PATH


python3 pytorch/main_random_target.py train --workspace=$WORKSPACE --model_type='Regress_onset_offset_frame_velocity_CRNN' --loss_type='regress_onset_offset_frame_velocity_bce' --augmentation='none' --max_note_shift=0 --batch_size=12 --learning_rate=5e-4 --reduce_iteration=10000 --resume_iteration=0 --early_stop=300000 --cuda

MODEL_TYPE="Note_pedal"
CHECKPOINT_PATH=$WORKSPACE"/_models/note_pedal_160k_iterations.pth"
CUDA_VISIBLE_DEVICES=0 python3 pytorch/inference.py --model_type=$MODEL_TYPE --checkpoint_path=$CHECKPOINT_PATH --audio_path=$MAPS_DATASET_DIR'/ENSTDkAm/MUS/MAPS_MUS-chpn-p4_ENSTDkAm.wav' --cuda

CUDA_VISIBLE_DEVICES=0 python3 pytorch/main2.py train --workspace=$WORKSPACE --model_type='Regress_onset_offset_frame_velocity_CRNN' --loss_type='google_onset_offset_frame_velocity_bce' --augmentation='none' --max_note_shift=0 --batch_size=12 --learning_rate=5e-4 --reduce_iteration=10000 --resume_iteration=0 --early_stop=300000 --cuda
