"""Build fixed MIDI-backed splits and Magenta RL Tuner token sequences."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

import torch

from clean_melodies import clean_midi_melody, read_midi_melody_from_bytes
from melody_encoding import ENCODING, encode_notes

ROOT = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=ROOT / "cleaned_melodies_midi_backed.json.gz")
    p.add_argument("--train-midi", type=Path, default=ROOT / "Hooktheory_Train_MIDI.tar.gz")
    p.add_argument("--valid-midi", type=Path, default=ROOT / "Hooktheory_Valid_MIDI.tar.gz")
    p.add_argument("--test-midi", type=Path, default=ROOT / "Hooktheory_Test_MIDI.tar.gz")
    p.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    args = p.parse_args()
    with gzip.open(args.manifest, "rt") as f:
        manifest = json.load(f)
    report = {"encoding": ENCODING, "sources": {"manifest_sha256": sha256(args.manifest)},
              "splits": {}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assigned_ids = set()
    song_splits = {}
    for split, archive in [("TRAIN", args.train_midi), ("VALID", args.valid_midi),
                           ("TEST", args.test_midi)]:
        expected = {rid for rid, r in manifest.items() if r["split"] == split}
        items, found, seen = [], set(), set()
        totals, discarded, token_counts = Counter(), Counter(), Counter()
        failures = []
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.lower().endswith(".mid"):
                    continue
                rid = Path(member.name).stem
                if rid in seen:
                    raise ValueError(f"Duplicate MIDI ID {rid} in {archive}")
                seen.add(rid)
                if rid not in expected:
                    discarded["not_in_retained_manifest"] += 1
                    continue
                found.add(rid)
                try:
                    with tar.extractfile(member) as f:
                        raw = read_midi_melody_from_bytes(f.read())
                    notes, cleaning = clean_midi_melody(raw)
                    if not notes or len(notes) < 6:
                        discarded["fewer_than_six_clean_notes"] += 1
                        continue
                    tokens, transposition, stats = encode_notes(notes)
                    # x=events[:-1], y=events[1:]; eight predictions warm up.
                    if len(tokens) <= 9:
                        discarded["no_targets_after_warmup"] += 1
                        continue
                except Exception as e:
                    failures.append({"id": rid, "error": str(e)})
                    continue
                record = manifest[rid]
                h = record.get("hooktheory", {})
                song = (h.get("artist", ""), h.get("song", ""))
                if all(song):
                    if song in song_splits and song_splits[song] != split:
                        raise ValueError(f"Song occurs in multiple splits: {song}")
                    song_splits[song] = split
                if rid in assigned_ids:
                    raise ValueError(f"Record occurs in multiple splits: {rid}")
                assigned_ids.add(rid)
                items.append({"id": rid, "tokens": torch.tensor(tokens, dtype=torch.uint8),
                              "length": len(tokens), "transpose_amount": transposition,
                              "artist": song[0], "song": song[1],
                              "archive_member": member.name})
                for k, v in stats.items():
                    if k.endswith("max_beats"):
                        totals[k] = max(totals[k], v)
                    else:
                        totals[k] += v
                totals["duplicate_notes_removed"] += cleaning["duplicate_notes_removed"]
                token_counts.update(tokens)
        if failures:
            raise ValueError(f"Failed reading {split} MIDI: {failures[:5]}")
        if not items:
            raise ValueError(f"No usable MIDI records for {split}")
        items.sort(key=lambda item: item["id"])
        lengths = sorted(item["length"] for item in items)
        summary = {"manifest_records": len(expected), "archive_records": len(seen),
                   "matched_midi": len(found), "retained_records": len(items),
                   "missing_midi_ids": sorted(expected - found),
                   "discarded": dict(discarded), "conversion": dict(totals),
                   "tokens": sum(lengths),
                   "length_min": lengths[0], "length_median": lengths[len(lengths)//2],
                   "length_max": lengths[-1], "token_counts": dict(token_counts)}
        report["splits"][split] = summary
        report["sources"][split] = {"path": str(archive), "sha256": sha256(archive)}
        torch.save({"split": split, "encoding": ENCODING, "records": items},
                   args.output_dir / f"{split.lower()}.pt")
        print(f"{split}: {len(items)} melodies, {sum(lengths):,} events, "
              f"{len(expected-found)} unmatched records", flush=True)
    (args.output_dir / "preprocessing_report.json").write_text(json.dumps(report, indent=2))
    print(f"Saved dataset and report to {args.output_dir}")


if __name__ == "__main__":
    main()
