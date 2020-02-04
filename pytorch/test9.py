from mido import Message, MidiFile, MidiTrack
import numpy as np
notes = np.array([60, 63, 60, 63, 67, 70, 67]) + 12
midi_file = MidiFile()
midi_file.ticks_per_beat = 384
track = MidiTrack()
midi_file.tracks.append(track)
track.append(Message('program_change', program=12))
d = 640
deltas = [d//8, d //2, d // 2, d//2, d//8, d//8*3, d]
for i, note in enumerate(notes):
    track.append(Message('note_on', note=note, velocity=100, time=0))
    track.append(Message('note_off', note=note, velocity=100, time=deltas[i]))
midi_file.save('test.mid')
print(midi_file.ticks_per_beat)