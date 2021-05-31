import sys
sys.path.append('utils')
sys.path.append('pytorch')
from piano_transcription_inference import PianoTranscription, sample_rate, load_audio
import gradio as gr
import os
from collections import namedtuple
from plot_for_paper import plot_midi

os.makedirs("results", exist_ok=True)


plot_args = namedtuple('PlotArgs', ['audio_path', 'midi_path'])

# Transcriptor
transcriptor = PianoTranscription(device='cpu')    # 'cuda' | 'cpu'


def transcribe(aud):
  # Load audio
  (audio, _) = load_audio(aud.name, sr=sample_rate, mono=True)

  # Transcribe and write out to MIDI file
  transcribed_dict = transcriptor.transcribe(audio, './out.mid')
  audsplit = aud.name.split("p/")
  split2 = audsplit[1].split(".")
  plot_midi(plot_args(aud.name, 'out.mid'))

  return f"./out.mid", f"results/{split2[0]}.png"


inputs = gr.inputs.Audio(label="Input Audio", type="file")
outputs =  [
            gr.outputs.File(label="Output Midi"),
            gr.outputs.Image(type="file", label="Output Visualization"),
            ]


title = "piano transcription"
description = "demo for piano transcription by Bytedance. To use it, simply upload your audio, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2010.01815'>High-resolution Piano Transcription with Pedals by Regressing Onsets and Offsets Times</a> | <a href='https://github.com/bytedance/piano_transcription'>Github Repo</a></p>"
examples = [
    ["resources/cut_bach.mp3"],
    ["resources/cut_liszt.mp3"]
]

gr.Interface(transcribe, inputs, outputs, title=title, description=description, article=article, examples=examples).launch()