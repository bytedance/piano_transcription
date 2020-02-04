
# Piano transcription

Piano transcription is the task to transcribe piano recordings to MIDI. That is, transcribe waveform to symbolic music notes. This codebase contains PyTorch implementation of 1. Inference a piano audio recording to MIDI using pretrained model; 2. Training a piano transcription system.

## Inference using pretrained model
First, install dependencies in requirements.txt

Then, execute the following command to inference an audio recording in wav format.

`python3 pytorch/main_inference.py --cuda --audio_path='examples/cut_liszt.wav'
`

Demo: https://www.youtube.com/watch?v=easks37Q4iE

## Training a piano transcription system

### 0. Prepare data
MAESTRO dataset [1] is used for training the piano transcription system. MAESTRO consists of over 200 hours of virtuosic piano performances captured with fine alignment (~3 ms) between note labels and audio waveforms. MAESTRO dataset can be downloaded from https://magenta.tensorflow.org/datasets/maestro. This codebase used MAESTRO V2.0.0 for training.

Statistics of MAESTRO V2.0.0 [[ref]](https://magenta.tensorflow.org/datasets/maestro#v200):

| Split      | Performances | Duration (hours) | Size (GB) | Notes (millions) |
|------------|--------------|------------------|-----------|------------------|
| Train      |          967 |            161.3 |      97.7 |             5.73 |
| Validation |          137 |             19.4 |      11.8 |             0.64 |
| Test       |          178 |             20.5 |      12.4 |             0.76 |
| **Total**  |      **1282**|         **201.2**|  **121.8**|          **7.13**|

After downloading, the dataset looks like:

<pre>
dataset_root
├── 2004
│    └── (264 files)
├── 2006
│    └── (230 files)
├── 2008
│    └── (294 files)
├── 2009
│    └── (250 files) 
├── 2011
│    └── (326 files)
├── 2013
│    └── (254 files)
├── 2014
│    └── (210 files)
├── 2015
│    └── (258 files)
├── 2017
│    └── (280 files)
├── 2018
│    └── (198 files)
├── LICENSE
├── maestro-v2.0.0.csv
├── maestro-v2.0.0.json
└── README
</pre>

### 1. Train
The baseline system is built using "Onsets and frames: Dual-objective piano transcription." [2]. There are some difference in the implementation including:
1) GRU is used instead of LSTM. 
2) A sampling rate of 32 kHz is used instead of 16 kHz. 
3) A hop size of 10 ms is used instead of 31.25 ms. 
4) Notes are set to ON until pedal is released.
5) Cross segment notes are masked out in training.

An example of target for training can be [viewed](appendixes/target.png), its corresponding waveform can be [downloaded](appendixes/target.wav).

To train the systems, execute commands in runme.sh, which includes:
1) Config dataset path and your workspace.
2) Pack audio recordings to hdf5 files.
3) Train.
4) Evaluate.
5) Inference using trained model.

## Results
The training takes xxx to train using xxx. The training looks like:

The statistics during training looks like:

Evaluation results using mir_eval toolbox are:

Demos

Andras Schiff: J.S.Bach - French Suites [[wav]](examples/cut_bach.wav) [[transcribed_midi]](appendixes/cut_bach.mid)
Transcribed MIDI:

<img src="appendixes/cut_bach.png">

Lang Lang: Franz Liszt - Love Dream (Liebestraum) [[wav]](examples/cut_liszt.wav) [[transcribed_midi]](appendixes/cut_liszt.mid)
Transcribed MIDI:

<img src="appendixes/cut_liszt.png">



## Reference
[1] Curtis Hawthorne, Andriy Stasyuk, Adam Roberts, Ian Simon, Cheng-Zhi Anna Huang, Sander Dieleman, Erich Elsen, Jesse Engel, and Douglas Eck. "Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset." In International Conference on Learning Representations (ICLR), 2019.

[2] Hawthorne, Curtis, Erich Elsen, Jialin Song, Adam Roberts, Ian Simon, Colin Raffel, Jesse Engel, Sageev Oore, and Douglas Eck. "Onsets and frames: Dual-objective piano transcription.", International Society for Music Information Retrieval (ISMIR), 2018.