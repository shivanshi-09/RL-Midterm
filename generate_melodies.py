"""Sample the MLE Note RNN autoregressively and export its events as MIDI."""

import argparse
import json
import math
from pathlib import Path

import note_seq
import mido
import torch

from clean_melodies import clean_midi_melody, read_midi_melody_from_bytes
from melody_encoding import ENCODING, decode_tokens, encode_notes
from note_rnn import NoteRNN

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "runs/note_rnn/best.pt")
    p.add_argument("--output-dir", type=Path, default=ROOT / "generated/mle")
    p.add_argument("--num-outputs", type=int, default=10)
    p.add_argument("--steps", type=int, default=128, help="Total events including primer; 128 = 8 bars of 4/4")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--qpm", type=float, default=80)
    p.add_argument("--seed", type=int, default=42)
    primer = p.add_mutually_exclusive_group()
    primer.add_argument("--primer-tokens", default=None, help='JSON event IDs, e.g. "[14,0,16,0]"')
    primer.add_argument("--primer-midi", type=Path, help="MIDI with a track named MELODY; normalize as in training")
    args = p.parse_args()
    if (not math.isfinite(args.temperature) or args.temperature <= 0 or
            args.num_outputs < 1 or args.steps < 1):
        p.error("Temperature, num-outputs, and steps must be positive")
    torch.set_num_threads(4)
    if not math.isfinite(args.qpm) or args.qpm <= 0:
        p.error("QPM must be finite and positive")
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["encoding"] != ENCODING:
        p.error("Checkpoint encoding differs from installed preprocessing")
    model = NoteRNN(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    if args.primer_midi:
        notes, _ = clean_midi_melody(read_midi_melody_from_bytes(args.primer_midi.read_bytes()))
        tokens, _, _ = encode_notes(notes)
    else:
        tokens = json.loads(args.primer_tokens) if args.primer_tokens is not None else [14]  # MIDI 60
    if (not isinstance(tokens, list) or not tokens or
        any(type(t) is not int or not 0 <= t < 38 for t in tokens) or len(tokens) > args.steps):
        p.error("Primer must be a nonempty list of event IDs 0..37 no longer than --steps")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    with torch.no_grad():
        for i in range(args.num_outputs):
            events = list(tokens)
            logits, state = model(torch.tensor([tokens]))
            while len(events) < args.steps:
                probabilities = (logits[:, -1] / args.temperature).softmax(-1)
                event = int(torch.multinomial(probabilities, 1).item())
                events.append(event)
                logits, state = model(torch.tensor([[event]]), state)
            sequence = decode_tokens(events, qpm=args.qpm)
            path = args.output_dir / f"melody_{i:03d}.mid"
            note_seq.sequence_proto_to_midi_file(sequence, str(path))
            # Make our output usable as a MELODY-track primer on the next run.
            midi = mido.MidiFile(path)
            note_tracks = [track for track in midi.tracks
                           if any(m.type == "note_on" for m in track)]
            if note_tracks:
                note_tracks[0].name = "MELODY"
            else:
                track = mido.MidiTrack()
                track.name = "MELODY"
                midi.tracks.append(track)
            midi.save(path)
            outputs.append({"file": path.name, "tokens": events,
                            "notes": len(sequence.notes), "duration_seconds": sequence.total_time})
    (args.output_dir / "samples.json").write_text(json.dumps(
        {"checkpoint": str(args.checkpoint), "temperature": args.temperature,
         "seed": args.seed, "qpm": args.qpm, "encoding": ENCODING, "outputs": outputs}, indent=2))
    print(f"Saved {len(outputs)} MIDI melodies to {args.output_dir}")


if __name__ == "__main__":
    main()
