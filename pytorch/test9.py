from mido import Message, MidiFile, MidiTrack, MetaMessage
notes = [60, 62, 64]
midi_file = MidiFile()
midi_file.ticks_per_beat = 384

beats_per_second = 2
ticks_per_second = midi_file.ticks_per_beat * beats_per_second
microseconds_per_beat = int(1e6 // beats_per_second)

# Track 0
track0 = MidiTrack()
track0.append(MetaMessage('set_tempo', tempo=microseconds_per_beat, time=0))
track0.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
track0.append(MetaMessage('end_of_track', time=1))
midi_file.tracks.append(track0)

track1 = MidiTrack()
for note in notes:
    # track1.append(Message('program_change', program=12, time=0))
    track1.append(Message('note_on', note=note, velocity=100, time=0))
    track1.append(Message('note_off', note=note, velocity=100, time=200))
track1.append(Message('control_change'))
track1.append(MetaMessage('end_of_track', time=1))
midi_file.tracks.append(track1)