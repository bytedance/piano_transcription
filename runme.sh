#!bin/bash

# MAESTRO dataset directory
DATASET_DIR="/mnt/cephfs_new_wj/speechsv/kongqiuqiang/datasets/maestro/dataset_root"

# Modify to your workspace
WORKSPACE="/mnt/cephfs_new_wj/speechsv/kongqiuqiang/workspaces/pub_piano_transcription"

# Pack audio files to hdf5 format. This will speed up training.
python3 utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

python3 pytorch/main.py train --workspace=$WORKSPACE --model_type='Cnn' --loss_type='frame_bce' --augmentation='none' --batch_size=32 --learning_rate=1e-3 --resume_iteration=0 --early_stop=1000000 --cuda

python3 pytorch/main.py evaluate --workspace=$WORKSPACE --model_type='CnnGoogle_onset_frame' --loss_type='onset_frame_bce' --augmentation='none' --batch_size=32 --iteration=80000 --cuda --mini_data
 
python3 pytorch/main.py inference --workspace=$WORKSPACE --model_type='CnnGoogle_onset_frame' --loss_type='onset_frame_bce' --augmentation='none' --batch_size=32 --iteration=80000 --cuda

python3 utils/plot_statistics.py plot --workspace=$WORKSPACE --select=1a

# French suite
youtube-dl -o 'french_video.%(ext)s' -f bestvideo -x https://www.youtube.com/watch?v=0sDleZkIK-w
youtube-dl -o 'french_audio.%(ext)s' -f bestaudio -x https://www.youtube.com/watch?v=0sDleZkIK-w
ffmpeg -loglevel panic -i french_video.webm -ss 01:00:00 -t 00:00:20 cut_french_video.mp4
ffmpeg -loglevel panic -i french_audio.opus -ac 1 -ar 32000 -ss 01:00:00 -t 00:00:20 cut_french_audio.wav

# Liebestraum
youtube-dl -o 'liszt_video.%(ext)s' -f bestvideo -x https://www.youtube.com/watch?v=2FqugGjOkQE
youtube-dl -o 'liszt_audio.%(ext)s' -f bestaudio -x https://www.youtube.com/watch?v=2FqugGjOkQE
ffmpeg -loglevel panic -i liszt_video.mp4 -ss 00:02:08 -t 00:00:40 cut_liszt_video.mp4
ffmpeg -loglevel panic -i liszt_audio.opus -ac 1 -ar 32000 -ss 00:02:08 -t 00:00:40 cut_liszt_audio.wav

python3 pytorch/main.py inference --workspace=$WORKSPACE --model_type='CnnGoogle_onset_frame' --loss_type='onset_frame_bce' --augmentation='none' --batch_size=32 --iteration=80000 --cuda

# Then open _zz.mid with GarageBand, change instrument, export to wav
# Then python3 pytorch/create_demo.py
# Then use imovie to edit video & audio