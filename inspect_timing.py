#!/usr/bin/env python3

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


# Magenta Melody / RL Tuner timing resolution:
# 4 steps per quarter note = 16th-note resolution.
STEPS_PER_QUARTER = 4
STEP_BEATS = 1.0 / STEPS_PER_QUARTER


def load_json_gz(path):
    """Load the MIDI-backed cleaned dataset."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def quantize_beat(beat):
    """
    Reproduce Magenta's timing quantization.

    Input:
        beat = time measured in quarter-note beats.

    Output:
        integer timestep at 4 steps per quarter note.

    Equivalent to:
        floor(beat * 4 + 0.5)
    """
    return math.floor(
        float(beat) * STEPS_PER_QUARTER + 0.5
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Path to cleaned MIDI-backed JSON.gz file",
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=10,
        help="Number of melodies to print in detail",
    )

    args = parser.parse_args()

    data = load_json_gz(args.input)

    print("=" * 70)
    print("MAGENTA TIMING INSPECTION")
    print("=" * 70)

    print(f"Input records: {len(data)}")
    print(f"Steps per quarter: {STEPS_PER_QUARTER}")
    print(f"Step duration: {STEP_BEATS} beats")
    print("Resolution: 16th note")
    print()

    total_notes = 0
    records_used = 0

    exact_onsets = 0
    exact_offsets = 0

    onset_errors = []
    offset_errors = []

    duration_steps = Counter()
    ioi_steps = Counter()

    examples = []

    for record_id, record in data.items():

        # ------------------------------------------------------------
        # The MIDI-backed cleaner stores the melody here.
        # ------------------------------------------------------------
        annotations = record.get("annotations", {})
        notes = annotations.get("melody")

        if not notes:
            continue

        # We only want MIDI-derived melodies.
        # The MIDI-backed cleaner marks these with pitch_source="midi".
        if record.get("pitch_source") not in (None, "midi"):
            continue

        notes = sorted(
            notes,
            key=lambda n: (
                float(n["onset"]),
                float(n["offset"]),
            ),
        )

        records_used += 1
        total_notes += len(notes)

        previous_start = None

        for note in notes:

            onset = float(note["onset"])
            offset = float(note["offset"])

            # --------------------------------------------------------
            # Quantize exactly to the 4-step-per-quarter grid.
            # --------------------------------------------------------
            start_step = quantize_beat(onset)
            end_step = quantize_beat(offset)

            # Location of the resulting grid points in beats.
            quantized_onset = start_step * STEP_BEATS
            quantized_offset = end_step * STEP_BEATS

            # --------------------------------------------------------
            # Quantization error.
            # --------------------------------------------------------
            onset_error = abs(onset - quantized_onset)
            offset_error = abs(offset - quantized_offset)

            onset_errors.append(onset_error)
            offset_errors.append(offset_error)

            if onset_error < 1e-6:
                exact_onsets += 1

            if offset_error < 1e-6:
                exact_offsets += 1

            # --------------------------------------------------------
            # Duration after quantization.
            # --------------------------------------------------------
            quantized_duration = end_step - start_step

            duration_steps[quantized_duration] += 1

            # --------------------------------------------------------
            # Inter-onset interval after quantization.
            # --------------------------------------------------------
            if previous_start is not None:
                ioi_steps[start_step - previous_start] += 1

            previous_start = start_step

        if len(examples) < args.examples:
            examples.append(
                (record_id, notes)
            )

    # ==================================================================
    # SUMMARY
    # ==================================================================

    print(f"MIDI-backed records used: {records_used}")
    print(f"Total notes: {total_notes}")
    print()

    if total_notes == 0:
        print("ERROR: No MIDI-backed notes found.")
        print(
            "Check that the input file is the MIDI-backed cleaned dataset."
        )
        return

    # ==================================================================
    # GRID ALIGNMENT
    # ==================================================================

    print("=" * 70)
    print("GRID ALIGNMENT")
    print("=" * 70)

    print(
        f"Onsets already exactly on 0.25-beat grid: "
        f"{exact_onsets / total_notes * 100:.2f}%"
    )

    print(
        f"Offsets already exactly on 0.25-beat grid: "
        f"{exact_offsets / total_notes * 100:.2f}%"
    )

    print()

    # ==================================================================
    # QUANTIZATION ERROR
    # ==================================================================

    print("=" * 70)
    print("QUANTIZATION ERROR")
    print("=" * 70)

    print("Onset error (beats):")
    print(
        f"  mean   = {np.mean(onset_errors):.6f}"
    )
    print(
        f"  median = {np.median(onset_errors):.6f}"
    )
    print(
        f"  p95    = {np.percentile(onset_errors, 95):.6f}"
    )
    print(
        f"  max    = {np.max(onset_errors):.6f}"
    )

    print()

    print("Offset error (beats):")
    print(
        f"  mean   = {np.mean(offset_errors):.6f}"
    )
    print(
        f"  median = {np.median(offset_errors):.6f}"
    )
    print(
        f"  p95    = {np.percentile(offset_errors, 95):.6f}"
    )
    print(
        f"  max    = {np.max(offset_errors):.6f}"
    )

    # ==================================================================
    # QUANTIZED DURATIONS
    # ==================================================================

    print()
    print("=" * 70)
    print("QUANTIZED NOTE DURATIONS")
    print("=" * 70)

    for steps, count in sorted(duration_steps.items()):

        percentage = 100.0 * count / total_notes

        print(
            f"{steps:3d} step(s) "
            f"= {steps * STEP_BEATS:6.2f} beats : "
            f"{count:8d} ({percentage:6.2f}%)"
        )

    # ==================================================================
    # QUANTIZED INTER-ONSET INTERVALS
    # ==================================================================

    print()
    print("=" * 70)
    print("QUANTIZED INTER-ONSET INTERVALS")
    print("=" * 70)

    total_ioi = sum(ioi_steps.values())

    for steps, count in sorted(ioi_steps.items())[:30]:

        percentage = 100.0 * count / total_ioi

        print(
            f"{steps:3d} step(s) "
            f"= {steps * STEP_BEATS:6.2f} beats : "
            f"{count:8d} ({percentage:6.2f}%)"
        )

    # ==================================================================
    # EXAMPLES
    # ==================================================================

    print()
    print("=" * 70)
    print("EXAMPLES")
    print("=" * 70)

    for example_number, (record_id, notes) in enumerate(
        examples,
        start=1,
    ):

        print()
        print(f"Example {example_number}")
        print(f"ID: {record_id}")

        print()
        print(
            "pitch | raw_onset | raw_offset | "
            "start_step | end_step | duration"
        )
        print("-" * 70)

        for note in notes[:30]:

            onset = float(note["onset"])
            offset = float(note["offset"])

            start_step = quantize_beat(onset)
            end_step = quantize_beat(offset)

            duration = end_step - start_step

            print(
                f"{int(note['midi_pitch']):5d} | "
                f"{onset:9.4f} | "
                f"{offset:10.4f} | "
                f"{start_step:10d} | "
                f"{end_step:8d} | "
                f"{duration:8d}"
            )


if __name__ == "__main__":
    main()