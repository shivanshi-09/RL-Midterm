#!/usr/bin/env python3

"""
Clean HookTheory melodies using the released MIDI files as the authoritative
source of melody pitch/timing for TRAIN and TEST.

Input:
    Hooktheory.json.gz
    Hooktheory_Train_MIDI.tar.gz
    Hooktheory_Test_MIDI.tar.gz

Output:
    cleaned_melodies_midi_backed.json.gz
    cleaning_report_midi_backed.json

For TRAIN and TEST:
    - filter the JSON records using AUDIO_AVAILABLE + MELODY
    - require a matching <record_id>.mid in the supplied MIDI archives
    - read the track named MELODY from the MIDI
    - extract actual MIDI note numbers (0--127)
    - convert MIDI ticks to beats using ticks_per_beat
    - remove malformed/non-positive notes
    - sort notes by onset
    - remove exact duplicate note events
    - trim leading silence
    - require at least MIN_NOTES melody notes

For VALID:
    No VALID MIDI archive was supplied, so VALID records are retained from
    Hooktheory.json.gz and explicitly marked with pitch_source="json".

No quantization, octave folding, or 38-class encoding is performed here.
"""

import argparse
import gzip
import io
import json
import math
import tarfile
from collections import Counter
from pathlib import Path

import mido


VALID_SPLITS = {"TRAIN", "VALID", "TEST"}


def load_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def save_json_gz(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))


def load_midi_members(tar_paths):
    """
    Return:
        {
            record_id: (archive_path, tar_member_name)
        }

    for every .mid file found in the supplied archives.
    """
    members = {}

    for tar_path in tar_paths:
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue

                name = Path(member.name).name

                if not name.lower().endswith(".mid"):
                    continue

                record_id = name[:-4]
                members[record_id] = (str(tar_path), member.name)

    return members


def read_midi_melody_from_bytes(midi_bytes):
    """
    Read the MELODY track from MIDI bytes.

    Returns notes with:
        onset       -- beats
        offset      -- beats
        midi_pitch  -- actual MIDI note number
    """

    midi = mido.MidiFile(file=io.BytesIO(midi_bytes))

    melody_tracks = [
        track for track in midi.tracks
        if track.name == "MELODY"
    ]

    if len(melody_tracks) != 1:
        raise ValueError(
            f"Expected exactly one MELODY track, "
            f"found {len(melody_tracks)}"
        )

    track = melody_tracks[0]

    active = {}
    notes = []

    absolute_tick = 0

    for message in track:
        absolute_tick += message.time

        if message.type == "note_on" and message.velocity > 0:

            active.setdefault(message.note, []).append(
                absolute_tick
            )

        elif (
            message.type == "note_off"
            or (
                message.type == "note_on"
                and message.velocity == 0
            )
        ):

            starts = active.get(message.note)

            if not starts:
                continue

            start_tick = starts.pop(0)

            if not starts:
                del active[message.note]

            if absolute_tick <= start_tick:
                continue

            notes.append(
                {
                    "onset": start_tick / midi.ticks_per_beat,
                    "offset": absolute_tick / midi.ticks_per_beat,
                    "midi_pitch": int(message.note),
                }
            )

    return notes


def clean_midi_melody(raw_notes):
    stats = Counter()

    if not isinstance(raw_notes, list):
        stats["invalid_melody"] += 1
        return None, stats

    stats["raw_notes"] = len(raw_notes)

    valid = []

    for note in raw_notes:

        if not isinstance(note, dict):
            stats["invalid_notes"] += 1
            continue

        onset = note.get("onset")
        offset = note.get("offset")
        midi_pitch = note.get("midi_pitch")

        if (
            not is_finite_number(onset)
            or not is_finite_number(offset)
            or not isinstance(midi_pitch, int)
        ):
            stats["invalid_notes"] += 1
            continue

        if offset <= onset:
            stats["nonpositive_duration"] += 1
            continue

        if not 0 <= midi_pitch <= 127:
            stats["invalid_midi_pitch"] += 1
            continue

        valid.append(
            {
                "onset": float(onset),
                "offset": float(offset),
                "midi_pitch": midi_pitch,
            }
        )

    if not valid:
        stats["empty_after_validation"] += 1
        return None, stats

    valid.sort(
        key=lambda n: (
            n["onset"],
            n["offset"],
            n["midi_pitch"],
        )
    )

    unique = []
    seen = set()

    for note in valid:

        key = (
            note["onset"],
            note["offset"],
            note["midi_pitch"],
        )

        if key in seen:
            stats["duplicate_notes_removed"] += 1
            continue

        seen.add(key)
        unique.append(note)

    if not unique:
        stats["empty_after_deduplication"] += 1
        return None, stats

    first_onset = unique[0]["onset"]

    if first_onset != 0.0:
        stats["leading_silence_trimmed"] += 1
        stats["leading_silence_beats"] = first_onset

    cleaned = [
        {
            "onset": note["onset"] - first_onset,
            "offset": note["offset"] - first_onset,
            "midi_pitch": note["midi_pitch"],
        }
        for note in unique
    ]

    stats["cleaned_notes"] = len(cleaned)
    stats["melody_span_beats"] = cleaned[-1]["offset"]

    return cleaned, stats


def clean_json_valid_melody(raw_notes):
    """
    Fallback used only for VALID because no VALID MIDI archive was supplied.

    This deliberately retains the original JSON pitch representation.
    """

    stats = Counter()

    if not isinstance(raw_notes, list):
        stats["invalid_melody"] += 1
        return None, stats

    valid = []

    for note in raw_notes:

        if not isinstance(note, dict):
            stats["invalid_notes"] += 1
            continue

        required = (
            "onset",
            "offset",
            "octave",
            "pitch_class",
        )

        if any(key not in note for key in required):
            stats["invalid_notes"] += 1
            continue

        onset = note["onset"]
        offset = note["offset"]
        octave = note["octave"]
        pitch_class = note["pitch_class"]

        if (
            not is_finite_number(onset)
            or not is_finite_number(offset)
            or not is_finite_number(octave)
            or not is_finite_number(pitch_class)
        ):
            stats["invalid_notes"] += 1
            continue

        if offset <= onset:
            stats["nonpositive_duration"] += 1
            continue

        if (
            int(pitch_class) != pitch_class
            or not 0 <= int(pitch_class) <= 11
        ):
            stats["invalid_pitch_class"] += 1
            continue

        valid.append(
            {
                "onset": float(onset),
                "offset": float(offset),
                "octave": int(octave),
                "pitch_class": int(pitch_class),
            }
        )

    if not valid:
        stats["empty_after_validation"] += 1
        return None, stats

    valid.sort(
        key=lambda n: (
            n["onset"],
            n["offset"],
            n["octave"],
            n["pitch_class"],
        )
    )

    unique = []
    seen = set()

    for note in valid:

        key = (
            note["onset"],
            note["offset"],
            note["octave"],
            note["pitch_class"],
        )

        if key in seen:
            stats["duplicate_notes_removed"] += 1
            continue

        seen.add(key)
        unique.append(note)

    first_onset = unique[0]["onset"]

    cleaned = [
        {
            "onset": note["onset"] - first_onset,
            "offset": note["offset"] - first_onset,
            "octave": note["octave"],
            "pitch_class": note["pitch_class"],
        }
        for note in unique
    ]

    stats["cleaned_notes"] = len(cleaned)
    stats["melody_span_beats"] = cleaned[-1]["offset"]

    return cleaned, stats


def is_finite_number(x):
    return (
        isinstance(x, (int, float))
        and math.isfinite(float(x))
    )


def clean_dataset(
    input_path,
    output_path,
    report_path,
    min_notes,
    midi_tar_paths,
):

    data = load_json_gz(input_path)

    # Build a lookup from record ID -> MIDI archive/member.
    midi_members = load_midi_members(midi_tar_paths)

    cleaned_data = {}

    report = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "min_notes": min_notes,
        "midi_archives": [
            str(path) for path in midi_tar_paths
        ],
        "records": {
            split: {
                "raw": 0,
                "tag_filtered": 0,
                "cleaned": 0,
                "removed_too_few_notes": 0,
                "removed_invalid": 0,
                "removed_no_midi": 0,
            }
            for split in VALID_SPLITS
        },
        "global": Counter(),
        "pitch_sources": Counter(),
        "examples_removed": [],
    }

    # ------------------------------------------------------------------
    # First pass:
    # process VALID directly from the JSON because there is no VALID MIDI.
    # TRAIN/TEST are processed below from the MIDI archives.
    # ------------------------------------------------------------------

    train_test_ids = set()

    for record_id, record in data.items():

        split = record.get("split")

        if split not in VALID_SPLITS:
            continue

        report["records"][split]["raw"] += 1

        tags = set(record.get("tags", []))

        if (
            "AUDIO_AVAILABLE" not in tags
            or "MELODY" not in tags
            or "TEMPO_CHANGES" in tags
        ):
            report["records"][split]["tag_filtered"] += 1
            report["global"]["tag_filtered"] += 1
            continue

        if split in {"TRAIN", "TEST"}:
            train_test_ids.add(record_id)
            continue

        # --------------------------------------------------------------
        # VALID
        # --------------------------------------------------------------

        raw_melody = (
            record
            .get("annotations", {})
            .get("melody")
        )

        cleaned_melody, stats = clean_json_valid_melody(
            raw_melody
        )

        pitch_source = "json"

        report["global"].update(stats)
        report["pitch_sources"][pitch_source] += 1

        if cleaned_melody is None:
            report["records"][split]["removed_invalid"] += 1
            continue

        if len(cleaned_melody) < min_notes:
            report["records"][split]["removed_too_few_notes"] += 1
            report["global"]["removed_too_few_notes"] += 1
            continue

        new_record = dict(record)

        new_record["annotations"] = dict(
            record.get("annotations", {})
        )

        new_record["annotations"]["melody"] = cleaned_melody
        new_record["pitch_source"] = pitch_source

        cleaned_data[record_id] = new_record

        report["records"][split]["cleaned"] += 1
        report["global"]["kept"] += 1

    # ------------------------------------------------------------------
    # Process TRAIN and TEST MIDI archives.
    #
    # IMPORTANT:
    # Each tar.gz archive is opened ONCE, rather than once per MIDI file.
    # This is the major performance improvement.
    # ------------------------------------------------------------------

    for archive_path in midi_tar_paths:

        archive_name = Path(archive_path).name

        # Determine which record IDs belong to this archive.
        archive_ids = {
            record_id
            for record_id in train_test_ids
            if record_id in midi_members
            and midi_members[record_id][0] == str(archive_path)
        }

        if not archive_ids:
            continue

        processed = 0

        print(
            f"\n[{archive_name}] "
            f"Processing {len(archive_ids)} MIDI files...",
            flush=True,
        )

        with tarfile.open(archive_path, "r:gz") as tar:

            # Map archive member name -> record ID for quick lookup.
            member_to_id = {
                midi_members[record_id][1]: record_id
                for record_id in archive_ids
            }

            for member in tar:

                if not member.isfile():
                    continue

                if not member.name.lower().endswith(".mid"):
                    continue

                record_id = member_to_id.get(member.name)

                if record_id is None:
                    continue

                record = data[record_id]
                split = record.get("split")

                processed += 1

                try:
                    extracted = tar.extractfile(member)

                    if extracted is None:
                        raise ValueError(
                            f"Could not extract MIDI member: "
                            f"{member.name}"
                        )

                    midi_bytes = extracted.read()

                    raw_melody = read_midi_melody_from_bytes(
                        midi_bytes
                    )

                    cleaned_melody, stats = clean_midi_melody(
                        raw_melody
                    )

                except Exception as exc:

                    report["records"][split]["removed_invalid"] += 1
                    report["global"]["midi_read_errors"] += 1

                    if len(report["examples_removed"]) < 100:
                        report["examples_removed"].append(
                            {
                                "id": record_id,
                                "split": split,
                                "reason": "midi_read_error",
                                "error": str(exc),
                            }
                        )

                    if processed % 500 == 0:
                        print(
                            f"[{archive_name}] "
                            f"Processed {processed} MIDI files",
                            flush=True,
                        )

                    continue

                report["global"].update(stats)
                report["pitch_sources"]["midi"] += 1

                if cleaned_melody is None:

                    report["records"][split]["removed_invalid"] += 1

                    if processed % 500 == 0:
                        print(
                            f"[{archive_name}] "
                            f"Processed {processed} MIDI files",
                            flush=True,
                        )

                    continue

                if len(cleaned_melody) < min_notes:

                    report["records"][split][
                        "removed_too_few_notes"
                    ] += 1

                    report["global"][
                        "removed_too_few_notes"
                    ] += 1

                    if processed % 500 == 0:
                        print(
                            f"[{archive_name}] "
                            f"Processed {processed} MIDI files",
                            flush=True,
                        )

                    continue

                new_record = dict(record)

                new_record["annotations"] = dict(
                    record.get("annotations", {})
                )

                new_record["annotations"]["melody"] = (
                    cleaned_melody
                )

                new_record["pitch_source"] = "midi"

                new_record["midi_source"] = {
                    "archive": str(archive_path),
                    "member": member.name,
                    "track": "MELODY",
                }

                cleaned_data[record_id] = new_record

                report["records"][split]["cleaned"] += 1
                report["global"]["kept"] += 1

                # ------------------------------------------------------
                # Progress counter
                # ------------------------------------------------------

                if processed % 500 == 0:
                    print(
                        f"[{archive_name}] "
                        f"Processed {processed} MIDI files",
                        flush=True,
                    )

        print(
            f"[{archive_name}] "
            f"Finished: {processed} MIDI files processed",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Records that passed filtering but had no matching MIDI file.
    # ------------------------------------------------------------------

    for record_id in train_test_ids:

        if record_id in cleaned_data:
            continue

        record = data[record_id]
        split = record.get("split")

        if record_id not in midi_members:

            report["records"][split]["removed_no_midi"] += 1
            report["global"]["removed_no_midi"] += 1

    # ------------------------------------------------------------------
    # Save output and report.
    # ------------------------------------------------------------------

    report["global"] = dict(report["global"])
    report["pitch_sources"] = dict(report["pitch_sources"])

    save_json_gz(
        cleaned_data,
        output_path,
    )

    report_path = Path(report_path)
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

    return report


def print_summary(report):

    print("\n=== MIDI-BACKED CLEANING SUMMARY ===")

    print(
        "Input records: "
        f"{sum(s['raw'] for s in report['records'].values())}"
    )

    print(
        "Kept:          "
        f"{report['global'].get('kept', 0)}"
    )

    print(
        "Tag filtered:  "
        f"{report['global'].get('tag_filtered', 0)}"
    )

    print(
        "No MIDI:       "
        f"{report['global'].get('removed_no_midi', 0)}"
    )

    print(
        "Too few notes: "
        f"{report['global'].get('removed_too_few_notes', 0)}"
    )

    print(
        "MIDI errors:   "
        f"{report['global'].get('midi_read_errors', 0)}"
    )

    print(
        "Pitch sources: "
        f"{report['pitch_sources']}"
    )

    print("\nPer split:")

    for split in ("TRAIN", "VALID", "TEST"):

        s = report["records"][split]

        print(
            f"  {split:5s}: "
            f"raw={s['raw']:5d}  "
            f"tag-filtered={s['tag_filtered']:5d}  "
            f"kept={s['cleaned']:5d}  "
            f"short={s['removed_too_few_notes']:5d}  "
            f"invalid={s['removed_invalid']:5d}  "
            f"no-midi={s['removed_no_midi']:5d}"
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="cleaned_melodies_midi_backed.json.gz",
    )

    parser.add_argument(
        "--report",
        default="cleaning_report_midi_backed.json",
    )

    parser.add_argument(
        "--midi-tar",
        action="append",
        required=True,
        help="MIDI tar.gz archive. Can be supplied multiple times.",
    )

    parser.add_argument(
        "--min-notes",
        type=int,
        default=6,
        help="Minimum number of melody notes.",
    )

    args = parser.parse_args()

    if args.min_notes < 1:
        parser.error(
            "--min-notes must be >= 1"
        )

    report = clean_dataset(
        input_path=args.input,
        output_path=args.output,
        report_path=args.report,
        min_notes=args.min_notes,
        midi_tar_paths=args.midi_tar,
    )

    print_summary(report)


if __name__ == "__main__":
    main()