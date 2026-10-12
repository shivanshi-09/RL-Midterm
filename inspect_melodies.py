#!/usr/bin/env python3

"""
Inspect cleaned HookTheory melodies with absolute MIDI pitches.

TRAIN and TEST should have pitch_source="midi".
VALID is expected to have pitch_source="json" because no VALID MIDI
archive was supplied.

This script does NOT:
    - quantize
    - octave-fold
    - normalize pitches
    - convert to 38 classes

It reports actual MIDI note numbers directly read from the MIDI files.
"""

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path


# ----------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------

def load_json_gz(path):
    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Quantiles / summaries
# ----------------------------------------------------------------------

def quantile(values, q):
    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    pos = q * (len(values) - 1)

    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return values[lo]

    return (
        values[lo]
        + (values[hi] - values[lo])
        * (pos - lo)
    )


def summary(values):
    if not values:
        return {
            "n": 0,
            "min": None,
            "p05": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "p95": None,
            "max": None,
            "std": None,
        }

    n = len(values)

    if n > 1:
        mean_value = sum(values) / n
        variance = (
            sum(
                (x - mean_value) ** 2
                for x in values
            )
            / n
        )
        std_value = math.sqrt(variance)
    else:
        mean_value = values[0]
        std_value = 0.0

    return {
        "n": n,
        "min": min(values),
        "p05": quantile(values, 0.05),
        "p25": quantile(values, 0.25),
        "median": quantile(values, 0.50),
        "mean": mean_value,
        "p75": quantile(values, 0.75),
        "p95": quantile(values, 0.95),
        "max": max(values),
        "std": std_value,
    }


def print_summary(
    name,
    values,
    digits=4,
):
    s = summary(values)

    def fmt(x):
        if x is None:
            return "None"

        return f"{x:.{digits}f}"

    print(f"\n{name}")

    print(f"  n      = {s['n']}")
    print(f"  min    = {fmt(s['min'])}")
    print(f"  p05    = {fmt(s['p05'])}")
    print(f"  p25    = {fmt(s['p25'])}")
    print(f"  median = {fmt(s['median'])}")
    print(f"  mean   = {fmt(s['mean'])}")
    print(f"  p75    = {fmt(s['p75'])}")
    print(f"  p95    = {fmt(s['p95'])}")
    print(f"  max    = {fmt(s['max'])}")
    print(f"  std    = {fmt(s['std'])}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--report",
        default=None,
    )

    args = parser.parse_args()

    data = load_json_gz(args.input)

    # ------------------------------------------------------------
    # Counters / containers
    # ------------------------------------------------------------

    split_counts = Counter()
    source_counts = Counter()

    melody_lengths = []
    melody_spans = []

    all_midi_pitches = []

    midi_pitch_spans = []
    midi_pitch_mins = []
    midi_pitch_maxs = []

    all_onsets = []
    all_offsets = []
    all_durations = []
    all_iois = []

    candidate_steps_per_beat = [
        1,
        2,
        4,
        8,
        16,
    ]

    quantization_errors = {
        str(spb): []
        for spb in candidate_steps_per_beat
    }

    quantized_onset_exact = {
        str(spb): 0
        for spb in candidate_steps_per_beat
    }

    quantized_offset_exact = {
        str(spb): 0
        for spb in candidate_steps_per_beat
    }

    total_onsets = 0
    total_offsets = 0

    overlapping_melodies = 0
    simultaneous_onset_melodies = 0

    overlapping_note_pairs = 0
    simultaneous_onset_pairs = 0

    nonzero_first_onsets = 0
    first_onsets = []

    example_records = []

    midi_records_processed = 0

    # ------------------------------------------------------------
    # Process records
    # ------------------------------------------------------------

    for record_id, record in data.items():

        split = record.get(
            "split",
            "UNKNOWN",
        )

        source = record.get(
            "pitch_source",
            "unknown",
        )

        split_counts[split] += 1
        source_counts[source] += 1

        # Only actual MIDI-backed records are used for the
        # absolute MIDI-pitch analysis.
        if source != "midi":
            continue

        melody = (
            record
            .get("annotations", {})
            .get("melody", [])
        )

        if not melody:
            continue

        midi_records_processed += 1

        if midi_records_processed % 500 == 0:
            print(
                f"Processed "
                f"{midi_records_processed} "
                f"MIDI-backed melodies",
                flush=True,
            )

        # --------------------------------------------------------
        # The cleaner already sorted the melody.
        #
        # Do NOT sort it again here.
        # --------------------------------------------------------

        if any(
            "midi_pitch" not in note
            for note in melody
        ):
            raise ValueError(
                f"{record_id} is marked MIDI-backed "
                "but has no midi_pitch."
            )

        melody_lengths.append(
            len(melody)
        )

        # --------------------------------------------------------
        # Extract timing and pitch information
        # --------------------------------------------------------

        onsets = []
        offsets = []
        durations = []
        pitches = []

        previous_onset = None

        for note in melody:

            onset = float(
                note["onset"]
            )

            offset = float(
                note["offset"]
            )

            pitch = int(
                note["midi_pitch"]
            )

            onsets.append(onset)
            offsets.append(offset)
            durations.append(
                offset - onset
            )
            pitches.append(pitch)

            if previous_onset is not None:
                all_iois.append(
                    onset - previous_onset
                )

            previous_onset = onset

        # --------------------------------------------------------
        # Absolute MIDI pitch
        # --------------------------------------------------------

        all_midi_pitches.extend(
            pitches
        )

        pitch_min = min(pitches)
        pitch_max = max(pitches)

        midi_pitch_mins.append(
            pitch_min
        )

        midi_pitch_maxs.append(
            pitch_max
        )

        midi_pitch_spans.append(
            pitch_max - pitch_min
        )

        # --------------------------------------------------------
        # Timing
        # --------------------------------------------------------

        all_onsets.extend(onsets)
        all_offsets.extend(offsets)
        all_durations.extend(durations)

        first_onset = onsets[0]

        first_onsets.append(
            first_onset
        )

        if abs(first_onset) > 1e-9:
            nonzero_first_onsets += 1

        melody_spans.append(
            offsets[-1] - first_onset
        )

        # --------------------------------------------------------
        # Quantization diagnostics
        # --------------------------------------------------------

        for spb in candidate_steps_per_beat:

            key = str(spb)

            # Since grid = 1/spb, the nearest grid point is:
            #
            #     round(value * spb) / spb
            #
            # This avoids the extra division by grid.

            for onset in onsets:

                error = abs(
                    onset
                    - round(onset * spb)
                    / spb
                )

                quantization_errors[key].append(
                    error
                )

                if error < 1e-9:
                    quantized_onset_exact[key] += 1

            for offset in offsets:

                error = abs(
                    offset
                    - round(offset * spb)
                    / spb
                )

                quantization_errors[key].append(
                    error
                )

                if error < 1e-9:
                    quantized_offset_exact[key] += 1

        total_onsets += len(onsets)
        total_offsets += len(offsets)

        # --------------------------------------------------------
        # Overlap
        #
        # Melody is already sorted by onset.
        # For each note, only subsequent notes whose onset is
        # before the current note's offset can overlap.
        # --------------------------------------------------------

        has_overlap = False

        for i in range(
            len(melody)
        ):

            onset_i = onsets[i]
            offset_i = offsets[i]

            for j in range(
                i + 1,
                len(melody),
            ):

                onset_j = onsets[j]
                offset_j = offsets[j]

                if onset_j >= offset_i:
                    break

                if (
                    onset_j < offset_i
                    and onset_i < offset_j
                ):
                    has_overlap = True
                    overlapping_note_pairs += 1

        if has_overlap:
            overlapping_melodies += 1

        # --------------------------------------------------------
        # Simultaneous onsets
        # --------------------------------------------------------

        onset_counter = Counter(
            onsets
        )

        pairs_here = sum(
            count * (count - 1) // 2
            for count in onset_counter.values()
            if count > 1
        )

        if pairs_here:

            simultaneous_onset_melodies += 1

            simultaneous_onset_pairs += (
                pairs_here
            )

        # --------------------------------------------------------
        # Examples
        # --------------------------------------------------------

        if len(example_records) < args.examples:

            example_records.append(
                {
                    "id": record_id,
                    "split": split,
                    "notes": melody,
                }
            )

    print(
        f"Finished processing "
        f"{midi_records_processed} "
        f"MIDI-backed melodies.",
        flush=True,
    )

    # ------------------------------------------------------------
    # Quantization report
    # ------------------------------------------------------------

    quantization_report = {}

    for spb in candidate_steps_per_beat:

        key = str(spb)

        errors = quantization_errors[
            key
        ]

        quantization_report[key] = {

            "grid_beats":
                1.0 / spb,

            "error_summary_beats":
                summary(errors),

            "onsets_exact_fraction": (
                quantized_onset_exact[key]
                / total_onsets
                if total_onsets
                else None
            ),

            "offsets_exact_fraction": (
                quantized_offset_exact[key]
                / total_offsets
                if total_offsets
                else None
            ),
        }

    # ------------------------------------------------------------
    # Machine-readable report
    # ------------------------------------------------------------

    report = {

        "input_records": len(data),

        "split_counts":
            dict(split_counts),

        "pitch_source_counts":
            dict(source_counts),

        "midi_backed_records_inspected":
            len(melody_lengths),

        "absolute_midi_pitch":
            summary(all_midi_pitches),

        "midi_pitch_span_per_melody_semitones":
            summary(midi_pitch_spans),

        "midi_pitch_min_per_melody":
            summary(midi_pitch_mins),

        "midi_pitch_max_per_melody":
            summary(midi_pitch_maxs),

        "melody_length_notes":
            summary(melody_lengths),

        "melody_span_beats":
            summary(melody_spans),

        "timing": {

            "onset_beats":
                summary(all_onsets),

            "offset_beats":
                summary(all_offsets),

            "duration_beats":
                summary(all_durations),

            "inter_onset_interval_beats":
                summary(all_iois),

            "first_onset_beats":
                summary(first_onsets),

            "nonzero_first_onset_count":
                nonzero_first_onsets,
        },

        "quantization":
            quantization_report,

        "overlap": {

            "melodies_with_overlapping_notes":
                overlapping_melodies,

            "fraction_with_overlapping_notes": (
                overlapping_melodies
                / len(melody_lengths)
                if melody_lengths
                else None
            ),

            "overlapping_note_pairs":
                overlapping_note_pairs,

            "melodies_with_simultaneous_onsets":
                simultaneous_onset_melodies,

            "fraction_with_simultaneous_onsets": (
                simultaneous_onset_melodies
                / len(melody_lengths)
                if melody_lengths
                else None
            ),

            "simultaneous_onset_pairs":
                simultaneous_onset_pairs,
        },

        "examples":
            example_records,
    }

    # ------------------------------------------------------------
    # Human-readable output
    # ------------------------------------------------------------

    print("=" * 72)
    print(
        "MIDI-BACKED HOOKTHEORY MELODY INSPECTION"
    )
    print("=" * 72)

    print("\nDataset")

    print(
        f"  Records in cleaned file: "
        f"{len(data)}"
    )

    for split in (
        "TRAIN",
        "VALID",
        "TEST",
    ):

        print(
            f"  {split:5s}: "
            f"{split_counts.get(split, 0)}"
        )

    print("\nPitch sources")

    for source, count in sorted(
        source_counts.items()
    ):

        print(
            f"  {source:8s}: "
            f"{count}"
        )

    print(
        "\nAbsolute MIDI pitch statistics are "
        "computed ONLY from pitch_source='midi' "
        "records."
    )

    print_summary(
        "Absolute MIDI pitch "
        "(all MIDI-backed notes)",
        all_midi_pitches,
        2,
    )

    print_summary(
        "MIDI pitch span within melody "
        "(semitones)",
        midi_pitch_spans,
        2,
    )

    print_summary(
        "Minimum MIDI pitch per melody",
        midi_pitch_mins,
        2,
    )

    print_summary(
        "Maximum MIDI pitch per melody",
        midi_pitch_maxs,
        2,
    )

    # ------------------------------------------------------------
    # MIDI pitch histogram
    # ------------------------------------------------------------

    print("\nMIDI pitch histogram")

    pitch_counts = Counter(
        all_midi_pitches
    )

    for pitch in sorted(
        pitch_counts
    ):

        print(
            f"  MIDI {pitch:3d}: "
            f"{pitch_counts[pitch]}"
        )

    # ------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------

    print("\n" + "-" * 72)
    print("TIMING")
    print("-" * 72)

    print_summary(
        "Onset positions (beats)",
        all_onsets,
        4,
    )

    print_summary(
        "Offset positions (beats)",
        all_offsets,
        4,
    )

    print_summary(
        "Note durations (beats)",
        all_durations,
        4,
    )

    print_summary(
        "Inter-onset intervals (beats)",
        all_iois,
        4,
    )

    print_summary(
        "First onset (beats)",
        first_onsets,
        6,
    )

    print(
        "\n  MIDI-backed melodies with "
        f"nonzero first onset: "
        f"{nonzero_first_onsets}/"
        f"{len(melody_lengths)}"
    )

    # ------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------

    print("\n" + "-" * 72)
    print("QUANTIZATION DIAGNOSTICS")
    print("-" * 72)

    for spb in candidate_steps_per_beat:

        q = quantization_report[
            str(spb)
        ]

        e = q[
            "error_summary_beats"
        ]

        print(
            f"\n  {spb} steps/beat "
            f"(grid = "
            f"{q['grid_beats']:.5f} beat)"
        )

        print(
            f"    error median = "
            f"{e['median']:.6f} beat"
            f", p95 = "
            f"{e['p95']:.6f} beat"
            f", max = "
            f"{e['max']:.6f} beat"
        )

        print(
            f"    exact onsets  = "
            f"{100 * q['onsets_exact_fraction']:.2f}%"
        )

        print(
            f"    exact offsets = "
            f"{100 * q['offsets_exact_fraction']:.2f}%"
        )

    # ------------------------------------------------------------
    # Overlap
    # ------------------------------------------------------------

    print("\n" + "-" * 72)
    print("OVERLAP / POLYPHONY")
    print("-" * 72)

    overlap = report["overlap"]

    print(
        f"  Melodies with overlapping notes: "
        f"{overlap['melodies_with_overlapping_notes']} "
        f"("
        f"{100 * overlap['fraction_with_overlapping_notes']:.2f}%"
        f")"
    )

    print(
        f"  Overlapping note pairs: "
        f"{overlap['overlapping_note_pairs']}"
    )

    print(
        f"  Melodies with simultaneous onsets: "
        f"{overlap['melodies_with_simultaneous_onsets']} "
        f"("
        f"{100 * overlap['fraction_with_simultaneous_onsets']:.2f}%"
        f")"
    )

    print(
        f"  Simultaneous-onset pairs: "
        f"{overlap['simultaneous_onset_pairs']}"
    )

    # ------------------------------------------------------------
    # Examples
    # ------------------------------------------------------------

    print("\n" + "-" * 72)
    print(
        f"FIRST {len(example_records)} "
        "MIDI-BACKED EXAMPLES"
    )
    print("-" * 72)

    for example in example_records:

        print(
            f"\n{example['id']} "
            f"[{example['split']}] "
            f"({len(example['notes'])} notes)"
        )

        for i, note in enumerate(
            example["notes"]
        ):

            print(
                f"  {i:3d}: "
                f"on={note['onset']:.4f}, "
                f"off={note['offset']:.4f}, "
                f"dur="
                f"{note['offset'] - note['onset']:.4f}, "
                f"MIDI="
                f"{note['midi_pitch']}"
            )

    # ------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------

    if args.report:

        report_path = Path(
            args.report
        )

        report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with report_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                report,
                f,
                indent=2,
            )

        print(
            f"\nFull JSON report written to: "
            f"{report_path}"
        )


if __name__ == "__main__":
    main()