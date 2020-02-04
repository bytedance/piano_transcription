
# Piano transcription

Piano transcription is the task to transcribe piano recording in waveform to MIDI. This repo used MAESTRO dataset [1] to build a piano transcription system. MAESTRO dataset consists of over 200 hours of virtuosic piano performances captured with fine alignment (~3 ms) between note labels and audio waveforms.


## Download dataset from https://magenta.tensorflow.org/datasets/maestro
version: V2.0.0

After downloading, the dataset looks like:



Requirements:
mido==1.2.9
mir_eval==0.5
torch==1.0.1.post2
librosa==0.6.3