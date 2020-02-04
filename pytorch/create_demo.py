import numpy as np
import librosa
# import config

cut_wav_path = '/mnt/cephfs_new_wj/speechsv/kongqiuqiang/workspaces/piano_transcription/examples/liszt/cut_liszt_audio.wav'
transcribed_wav_path = '/mnt/cephfs_new_wj/speechsv/kongqiuqiang/workspaces/piano_transcription/examples/liszt/transcribed_liszt2.wav'

sample_rate = 32000
(cut_wav, fs) = librosa.core.load(cut_wav_path, sr=sample_rate, mono=True)
(transcribed_wav, fs) = librosa.core.load(transcribed_wav_path, sr=sample_rate, mono=True)
transcribed_wav = transcribed_wav[0 : len(cut_wav)]
# sample_rate = config.sample_rate

combined_wav = np.concatenate((cut_wav[0 : 8 * sample_rate], transcribed_wav[8 * sample_rate:]), axis=0)
# combined_wav /= np.max(np.abs(combined_wav))

librosa.output.write_wav('combined.wav', combined_wav, sr=fs)