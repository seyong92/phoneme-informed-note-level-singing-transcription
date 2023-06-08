from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo


def save_midi(path, pitches, intervals, bpm, add_start_point=False):
    """
    Save extracted notes as a MIDI file
    Parameters
    ----------
    path: the path to save the MIDI file
    pitches: np.ndarray of bin_indices
    intervals: list of (onset_index, offset_index)
    bpm: bpm for MIDI file.
    add_start_point: add a short fake note at the beginning of the MIDI file
    """
    file = MidiFile()
    track = MidiTrack()
    file.tracks.append(track)
    ticks_per_second = file.ticks_per_beat * bpm / 60

    events = []
    if add_start_point:
        events.append(dict(type='on', pitch=0, time=0, velocity=1))  # for start point
        events.append(dict(type='off', pitch=0, time=0, velocity=1))
    for i in range(len(pitches)):
        events.append(dict(type='on', pitch=pitches[i], time=intervals[i][0], velocity=100))
        events.append(dict(type='off', pitch=pitches[i], time=intervals[i][1], velocity=100))
    events.sort(key=lambda row: row['time'])

    track.append(MetaMessage('set_tempo', tempo=bpm2tempo(bpm), time=0))
    last_tick = 0
    for event in events:
        current_tick = int(event['time'] * ticks_per_second)
        velocity = int(event['velocity'])
        pitch = event['pitch']
        track.append(Message('note_' + event['type'], note=pitch, velocity=velocity, time=current_tick - last_tick))
        last_tick = current_tick

    file.save(path)
