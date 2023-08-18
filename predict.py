# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import os
from pathlib import Path

from cog import BaseModel, BasePredictor, Path, Input
import librosa
from typing import Optional

# model repo: https://github.com/bytedance/piano_transcription
# package repo: https://github.com/qiuqiangkong/piano_transcription_inference
from piano_transcription_inference import PianoTranscription, sample_rate
from synthviz import create_video

# adapted from example: https://github.com/minzwon/sota-music-tagging-models/blob/master/predict.py

class Output(BaseModel):
    midi: Path
    video: Optional[Path]

class Predictor(BasePredictor):
    transcriptor: PianoTranscription

    def setup(self):
        self.transcriptor = PianoTranscription(
            device="cuda", checkpoint_path="./model.pth"
        )

    def predict(self, 
            audio_input: Path = Input(description="Input audio file"),
            make_video: bool = Input(default=False, description="Option to create demo video, instead of transcriber MIDI"),
        ) -> Output:
        midi_intermediate_filename = "transcription.mid"
        video_filename = os.path.join(Path.cwd(), "output.mp4")
        audio, _ = librosa.core.load(str(audio_input), sr=sample_rate)
        # Transcribe audio
        self.transcriptor.transcribe(audio, midi_intermediate_filename)

        if make_video == True:
            # 'Visualization' output option
            create_video(
                input_midi=midi_intermediate_filename, video_filename=video_filename
            )
            print(
                f"Created video of size {os.path.getsize(video_filename)} bytes at path {video_filename}"
            )
            # Return path to video
            return Output(midi=Path(midi_intermediate_filename), video=Path(video_filename))
        else:
            return Output(midi=Path(midi_intermediate_filename))
