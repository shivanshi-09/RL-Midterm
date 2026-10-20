"""Magenta RL Tuner melody representation, shared by preparation and generation.

We use note-seq's implementations, rather than approximating squash or the
quantizer. MIDI ticks are read as quarter-note beats; at QPM 60 one second is
one beat, allowing note-seq to quantize them without retaining performance tempo.
"""

import math
from importlib.metadata import version

import note_seq

MIN_NOTE = 48
MAX_NOTE = 84  # Exclusive: 36 pitches, plus two special events.
NUM_CLASSES = 38
STEPS_PER_QUARTER = 4
ENCODING = {
    "version": 1,
    "num_classes": NUM_CLASSES,
    "min_note": MIN_NOTE,
    "max_note_exclusive": MAX_NOTE,
    "transpose_to_key": 0,
    "steps_per_quarter": STEPS_PER_QUARTER,
    "no_event": 0,
    "note_off": 1,
    "pitch_token_formula": "midi_pitch - 48 + 2",
    "note_seq_version": version("note-seq"),
    "same_step_onsets": "keep highest pitch, as in Magenta",
    "short_notes": "extend zero-step durations to one step, as in Magenta",
    "long_gaps": "retain as rests within the source melody",
    "leading_silence": "trim before quantization",
}


def encode_notes(notes):
    """Return token IDs, squash transposition, and measurable conversion effects.

    Input is a cleaned monophonic list of onset/offset in quarter-note beats
    and midi_pitch. Long rests are retained; no bar-length or pitch-diversity
    filters are added to the existing six-note corpus filter.
    """
    if not notes:
        raise ValueError("Cannot encode an empty melody")
    ns = note_seq.NoteSequence()
    ns.tempos.add(qpm=60)
    ns.time_signatures.add(numerator=4, denominator=4)
    for n in notes:
        ns.notes.add(pitch=n["midi_pitch"], start_time=n["onset"],
                     end_time=n["offset"], velocity=100, instrument=0)
    ns.total_time = max(n["offset"] for n in notes)
    quantized = note_seq.quantize_note_sequence(ns, STEPS_PER_QUARTER)
    starts = [n.quantized_start_step for n in quantized.notes]
    zero_before_extension = sum(
        math.floor(n["onset"] * 4 + .5) == math.floor(n["offset"] * 4 + .5)
        for n in notes
    )
    errors = [abs(n[k] - math.floor(n[k] * 4 + .5) / 4)
              for n in notes for k in ("onset", "offset")]
    melody = note_seq.Melody()
    melody.from_quantized_sequence(
        quantized, instrument=0, gap_bars=float("inf"),
        ignore_polyphonic_notes=True, pad_end=False,
    )
    original_events = list(melody)
    transpose = melody.squash(MIN_NOTE, MAX_NOTE, transpose_to_key=0)
    codec = note_seq.MelodyOneHotEncoding(min_note=MIN_NOTE, max_note=MAX_NOTE)
    tokens = [int(codec.encode_event(e)) for e in melody]
    stats = {
        "source_notes": len(notes),
        "note_on_events": sum(e >= 0 for e in original_events),
        "notes_extended_to_one_step": zero_before_extension,
        "notes_discarded_same_step": len(starts) - len(set(starts)),
        "octave_folded_note_on_events": int(sum(
            a >= 0 and b != a + transpose
            for a, b in zip(original_events, melody)
        )),
        "timing_snap_error_sum_beats": sum(errors),
        "timing_snap_error_max_beats": max(errors),
    }
    if len(tokens) != len(melody) or any(t < 0 or t >= NUM_CLASSES for t in tokens):
        raise ValueError("Invalid Magenta event encoding")
    return tokens, int(transpose), stats


def decode_tokens(tokens, qpm=80, transpose_amount=0):
    """Decode normalized events to MIDI notes; inverse a source's global shift.

    Individual octave folds in squash cannot be inverted by the global shift.
    For new melodies use transpose_amount=0 (the normalized register/key).
    """
    if qpm <= 0 or not math.isfinite(qpm):
        raise ValueError("QPM must be finite and positive")
    codec = note_seq.MelodyOneHotEncoding(min_note=MIN_NOTE, max_note=MAX_NOTE)
    events = [codec.decode_event(int(t)) for t in tokens]
    if transpose_amount:
        events = [e - transpose_amount if e >= 0 else e for e in events]
        if any(e > 127 or (e < 0 and e not in (-2, -1)) for e in events):
            raise ValueError("Inverse transposition would leave the MIDI range")
    melody = note_seq.Melody(events, steps_per_quarter=STEPS_PER_QUARTER,
                             steps_per_bar=16)
    return melody.to_sequence(qpm=qpm, velocity=100)
