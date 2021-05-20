from piano_transcription_inference import PianoTranscription, sample_rate, load_audio
import gradio as gr
from visual_midi import Plotter
from visual_midi import Preset
from pretty_midi import PrettyMIDI


def transcribe(aud):
  # Load audio
  (audio, _) = load_audio(aud.name, sr=sample_rate, mono=True)

  # Transcriptor
  transcriptor = PianoTranscription(device='cpu')

  # Transcribe and write out to MIDI file
  transcribed_dict = transcriptor.transcribe(audio, './out.mid')
  pm = PrettyMIDI('./out.mid')
  plotter = Plotter()
  plotter.show(pm, "./example-01.html")

  return f"./out.mid", f"./example-01.html"


inputs = gr.inputs.Audio(label="Input Audio", type="file")
outputs =  [
            gr.outputs.File(label="Output Midi"),
            gr.outputs.File(label="Output Visualization"),
            ]


title = "piano transcription"
description = "demo for piano transcription by Bytedance. To use it, simply upload your audio, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2010.01815'>High-resolution Piano Transcription with Pedals by Regressing Onsets and Offsets Times</a> | <a href='https://github.com/bytedance/piano_transcription'>Github Repo</a></p>"
examples = [
    ["resources/cut_bach.mp3"],
    ["resources/cut_liszt.mp3"]
]

gr.Interface(transcribe, inputs, outputs, title=title, description=description, article=article, examples=examples).launch()