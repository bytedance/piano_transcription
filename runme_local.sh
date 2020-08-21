CHECKPOINT_PATH='/mnt/cephfs_new_wj/speechsv/kongqiuqiang/released_models/pub_piano_transcription/v3.0/note_F1=0.9677_pedal_F1=0.8658.pth'

DATASET_DIR="/mnt/cephfs_new_wj/speechsv/kongqiuqiang/datasets/maestro/dataset_root"

# Modify to your workspace
WORKSPACE="/mnt/cephfs_new_wj/speechsv/kongqiuqiang/workspaces/pub_piano_transcription_v2"

MAPS_DATASET_DIR="/mnt/cephfs_new_wj/speechsv/kongqiuqiang/datasets/maps/dataset_root"
python3 utils/features.py pack_maps_dataset_to_hdf5 --dataset_dir=$MAPS_DATASET_DIR --workspace=$WORKSPACE

NOTE_PEDAL_CHECKPOINT_PATH="/mnt/cephfs_new_wj/speechsv/kongqiuqiang/released_models/pub_piano_transcription/v3.0/note_F1=0.9677_pedal_F1=0.8658.pth"

python3 pytorch/calculate_score_for_paper.py infer_prob --workspace=$WORKSPACE --model_type='Note_pedal' --checkpoint_path=$NOTE_PEDAL_CHECKPOINT_PATH --dataset='maps' --split='test' --cuda
