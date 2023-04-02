# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import os
from pathlib import Path

import cog
import librosa

# model repo: https://github.com/bytedance/piano_transcription
# package repo: https://github.com/qiuqiangkong/piano_transcription_inference
from piano_transcription_inference import PianoTranscription, sample_rate
from synthviz import create_video

# adapted from example: https://github.com/minzwon/sota-music-tagging-models/blob/master/predict.py


class Predictor(cog.Predictor):
    transcriptor: PianoTranscription

    def setup(self):
        self.transcriptor = PianoTranscription(
            device="cuda", checkpoint_path="./model.pth"
        )

    @cog.input("audio_input", type=Path, help="Input audio file")
    def predict(self, audio_input):
        midi_intermediate_filename = "transcription.mid"
        video_filename = os.path.join(Path.cwd(), "output.mp4")
        audio, _ = librosa.core.load(str(audio_input), sr=sample_rate)
        # Transcribe audio
        self.transcriptor.transcribe(audio, midi_intermediate_filename)

        # 'Visualization' output option
        create_video(
            input_midi=midi_intermediate_filename, video_filename=video_filename
        )
        print(
            f"Created video of size {os.path.getsize(video_filename)} bytes at path {video_filename}"
        )
        # Return path to video
        return Path(video_filename)
