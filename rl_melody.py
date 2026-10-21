"""Train, evaluate, or sample a recurrent Sequence Tutor Psi melody model."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

import torch

from melody_encoding import ENCODING, decode_tokens
from music_rewards import RewardConfig, reward_metadata
from psi_learning import (Collector, Replay, evaluate, frozen_copy, load_model,
                          save_checkpoint, save_json, sha256, update)


ROOT = Path(__file__).resolve().parent
DEFAULT_PRIOR = ROOT / "checkpoints/mle_256.pt"
DEFAULT_PSI = (ROOT / "runs/psi_seed42/best.pt" if (ROOT / "runs/psi_seed42/best.pt").exists()
               else ROOT / "checkpoints/psi_seed42.pt")


def device_name(value):
    return ("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
            if torch.cuda.is_available() else "cpu") if value == "auto" else value


def setup(args):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # Deterministic recurrent kernels where supported; CPU/CUDA results can differ.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch.device(device_name(args.device))


def train(args):
    device = setup(args)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists() and args.resume is None:
        raise ValueError("Output already has a run; use --resume or a new output directory")
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved.get("algorithm") != "psi" or saved["encoding"] != ENCODING:
            raise ValueError("Resume requires a compatible Psi checkpoint")
        config = saved["config"]
        metadata = saved["metadata"]
        # Only total budget and output location may change on resume.
        online, _ = load_model(args.resume, device)
        target, prior = frozen_copy(online), frozen_copy(online)
        target.load_state_dict(saved["target_state"])
        prior.load_state_dict(saved["prior_state"])
        events, updates = saved["events"], saved["updates"]
        history, best = saved["history"], saved["best_objective"]
    else:
        online, source = load_model(args.checkpoint, device)
        target, prior = frozen_copy(online), frozen_copy(online)
        reward_config = RewardConfig(**json.loads(args.reward_config.read_text())) if args.reward_config else RewardConfig()
        config = {"horizon": args.steps, "c": args.c, "gamma": args.gamma,
                  "learning_rate": args.learning_rate, "target_rate": args.target_rate,
                  "batch_size": args.batch_size, "capacity": args.replay_capacity,
                  "warmup": args.warmup, "train_every": 4, "seed": args.seed,
                  "eval_every": args.eval_every, "eval_count": args.eval_count,
                  "eval_seed": 100_000, "reward": asdict(reward_config)}
        metadata = {"prior_path": str(args.checkpoint.resolve()),
                    "prior_sha256": sha256(args.checkpoint), "source_epoch": source.get("epoch"),
                    "device": str(device), "torch_version": str(torch.__version__),
                    "reward": reward_metadata(reward_config), "primer": [14]}
        events, updates, history, best = 0, 0, [], -math.inf
    config["total_events"] = args.total_events
    rewards = RewardConfig(**config["reward"])
    optimizer = torch.optim.Adam(online.parameters(), lr=config["learning_rate"])
    replay = Replay(config["capacity"])
    collector = Collector(prior, config["horizon"], rewards)
    if args.resume:
        optimizer.load_state_dict(saved["optimizer_state"])
        replay.items = saved["replay"]
        replay.position = saved["replay_position"]
        collector.load_state_dict(saved["collector"])
        random.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    save_json(output / "config.json", {"config": config, "metadata": metadata})
    start = time.monotonic()
    diagnostic = {}

    def checkpoint(name):
        save_checkpoint(output / name, online, target, prior, optimizer, replay, collector,
                        config, metadata, events, updates, history, best)

    def assess():
        nonlocal best
        result = evaluate(online, prior, config["eval_count"], config["eval_seed"],
                          config["horizon"], rewards, config["c"], config["gamma"])
        summary = result["summary"]
        entry = {"events": events, "updates": updates, "elapsed_seconds": time.monotonic() - start,
                 **summary, "training": dict(diagnostic)}
        history.append(entry)
        improved = summary["objective"] > best
        if improved:
            best = summary["objective"]
            checkpoint("best.pt")
        checkpoint("last.pt")
        save_json(output / "history.json", history)
        print(json.dumps({"events": events, "updates": updates,
                          "music_reward": round(summary["music_reward"], 3),
                          "objective": round(summary["objective"], 3),
                          "kl": round(summary["kl"], 3), "best": improved,
                          "elapsed_s": round(time.monotonic() - start, 1)}), flush=True)

    print(f"Psi training on {device}; prior={metadata['prior_path']}", flush=True)
    try:
        if not history:
            assess()
        next_eval = (events // config["eval_every"] + 1) * config["eval_every"]
        while events < args.total_events:
            online.eval()
            collected = collector.collect(online, replay, min(4, args.total_events - events))
            events += collected
            if len(replay) >= max(config["warmup"], config["batch_size"]) and events % 4 == 0:
                online.train()
                diagnostic = update(online, target, optimizer,
                                    replay.batch(config["batch_size"], random, device),
                                    config["c"], config["gamma"], config["target_rate"])
                updates += 1
            if events >= next_eval:
                assess()
                next_eval += config["eval_every"]
        if history[-1]["events"] != events:
            assess()
        save_json(output / "status.json", {"status": "complete", "events": events,
                                          "updates": updates, "best_objective": best,
                                          "elapsed_seconds": time.monotonic() - start})
    except (Exception, KeyboardInterrupt) as error:
        checkpoint("interrupted.pt")
        save_json(output / "failure.json", {"events": events, "updates": updates,
                                           "error": repr(error), "last_training": diagnostic})
        raise


def export_midi(tokens, path, qpm=80):
    import mido
    import note_seq

    sequence = decode_tokens(tokens, qpm=qpm)
    note_seq.sequence_proto_to_midi_file(sequence, str(path))
    midi = mido.MidiFile(path)
    note_tracks = [t for t in midi.tracks if any(m.type == "note_on" for m in t)]
    if note_tracks:
        note_tracks[0].name = "MELODY"
    else:
        track = mido.MidiTrack()
        track.name = "MELODY"
        midi.tracks.append(track)
    midi.save(path)
    return validate_midi(path, tokens, qpm)


def validate_midi(path, tokens, qpm=80):
    """Read back both ticks and notes; validate grid, durations and track naming."""
    import mido
    import note_seq

    midi = mido.MidiFile(path)
    melody_tracks = [t for t in midi.tracks if t.name == "MELODY"]
    if len(melody_tracks) != 1:
        raise AssertionError("MIDI must have one named MELODY track")
    for track in midi.tracks:
        tick = 0
        for event in track:
            tick += event.time
            if event.type in ("note_on", "note_off"):
                if abs(tick * 4 / midi.ticks_per_beat - round(tick * 4 / midi.ticks_per_beat)) > 1e-6:
                    raise AssertionError("MIDI note not aligned to sixteenth grid")
    sequence = note_seq.midi_file_to_note_sequence(str(path))
    notes = sorted(sequence.notes, key=lambda n: n.start_time)
    expected = sorted(decode_tokens(tokens, qpm=qpm).notes, key=lambda n: n.start_time)
    if len(notes) != len(expected):
        raise AssertionError("MIDI note count differs from token decoding")
    for i, (note, source) in enumerate(zip(notes, expected, strict=True)):
        if not 48 <= note.pitch <= 83 or note.end_time <= note.start_time:
            raise AssertionError("Invalid MIDI pitch or duration")
        if i and note.start_time < notes[i - 1].end_time - 0.002:
            raise AssertionError("Polyphonic output")
        if note.pitch != source.pitch or abs(note.start_time - source.start_time) > 0.002 or abs(note.end_time - source.end_time) > 0.002:
            raise AssertionError("MIDI timing differs from token decoding")
    return {"file": str(path), "notes": len(notes), "validated": True,
            "duration_seconds": max((n.end_time for n in notes), default=0.0)}


def curves(history, output, selected_events=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, key, label in zip(axes.flat,
                             ("music_reward", "objective", "kl", "entropy"),
                             ("Music reward / melody", "Discounted KL objective", "KL / event", "Entropy / event")):
        ax.plot([x["events"] for x in history], [x[key] for x in history], marker=".")
        if selected_events is not None:
            ax.axvline(selected_events, color="tab:orange", linestyle="--", alpha=.7)
        ax.set(xlabel="Generated training events", ylabel=label)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))
        ax.grid(alpha=0.2)
    fig.savefig(output / "learning_curves.png", dpi=160)
    fig.savefig(output / "learning_curves.svg")
    plt.close(fig)


def compare(args):
    device = setup(args)
    online, saved = load_model(args.checkpoint, device)
    if saved.get("algorithm") != "psi":
        raise ValueError("Comparison requires a trained Psi checkpoint")
    prior = frozen_copy(online)
    prior.load_state_dict(saved["prior_state"])
    config = saved["config"]
    rewards = RewardConfig(**config["reward"])
    output = args.output_dir
    if (output / "comparison.json").exists():
        raise ValueError("Comparison already exists; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    common = {"count": args.count, "seed_start": args.seed_start, "horizon": config["horizon"],
              "config": rewards, "c": config["c"], "gamma": config["gamma"],
              "temperature": args.temperature}
    baseline = evaluate(prior, prior, **common)
    tuned = evaluate(online, prior, **common)
    report = {"prior": saved["metadata"], "checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_sha256": sha256(args.checkpoint), "selected_at_events": saved["events"],
              "training_budget": config["total_events"], "training_seed": config["seed"], "count": args.count,
              "horizon": config["horizon"], "c": config["c"], "gamma": config["gamma"],
              "qpm": args.qpm,
              "seed_start": args.seed_start, "temperature": args.temperature,
              "baseline": baseline["summary"], "psi": tuned["summary"],
              "delta_psi_minus_baseline": {k: tuned["summary"][k] - baseline["summary"][k]
                                           for k in baseline["summary"]}}
    report["paired_standard_errors"] = {}
    for key in ("music_reward", "objective", "entropy", "prior_log_probability"):
        diffs = torch.tensor([b[key] - a[key] for a, b in zip(baseline["records"], tuned["records"], strict=True)])
        report["paired_standard_errors"][key] = float(diffs.std(unbiased=True) / math.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
    validations = []
    for name, result in (("baseline", baseline), ("psi", tuned)):
        destination = output / name
        destination.mkdir(exist_ok=True)
        save_json(destination / "sequences.json", result)
        for i, record in enumerate(result["records"][:args.num_samples]):
            validations.append(export_midi(record["tokens"], destination / f"melody_{i:03d}.mid", args.qpm))
    save_json(output / "midi_validation.json", validations)
    save_json(output / "comparison.json", report)
    history_file = args.checkpoint.parent / "history.json"
    full_history = json.loads(history_file.read_text()) if history_file.exists() else saved["history"]
    curves(full_history, output, saved["events"])
    save_json(output / "evaluation_config.json", {**common, "config": asdict(rewards), "qpm": args.qpm})
    write_report(report, output)
    print(json.dumps({k: report[k] for k in ("selected_at_events", "baseline", "psi")}, indent=2), flush=True)


def write_report(report, output):
    labels = [("music_reward", "Music reward per melody"), ("objective", "Discounted objective"),
              ("prior_log_probability", "Prior log probability per event"), ("kl", "KL per event"),
              ("entropy", "Entropy per event"), ("repetition_rate", "Excessive repetition rate"),
              ("silence_fraction", "Silence fraction"), ("out_of_key_fraction", "Out of key onsets"),
              ("motif_fraction", "Motif frequency"), ("repeated_motif_fraction", "Repeated motif frequency"),
              ("unique_pitches", "Distinct pitches per melody"), ("pitch_range", "Pitch range"),
              ("duplicate_melody_rate", "Duplicate melody rate")]
    rows = ["# Initial Psi learning experiment", "",
            f"Compared {report['count']} paired seeds at temperature {report['temperature']}. "
            f"Selected checkpoint: {report['selected_at_events']:,} generated training events.", "",
            "| Metric | Baseline | Psi | Difference |", "|---|---:|---:|---:|"]
    for key, label in labels:
        rows.append(f"| {label} | {report['baseline'][key]:.4f} | {report['psi'][key]:.4f} | {report['delta_psi_minus_baseline'][key]:+.4f} |")
    reward_delta = report["delta_psi_minus_baseline"]["music_reward"]
    objective_delta = report["delta_psi_minus_baseline"]["objective"]
    rows.extend(["", f"Music reward changed by {reward_delta:+.2f} per melody; "
                 f"the selection objective changed by {objective_delta:+.3f}.", "",
                 f"Mean distinct pitches changed from {report['baseline']['unique_pitches']:.2f} "
                 f"to {report['psi']['unique_pitches']:.2f}; mean event entropy changed from "
                 f"{report['baseline']['entropy']:.3f} to {report['psi']['entropy']:.3f}. "
                 "These describe within-melody pitch variety and policy uncertainty; "
                 "neither establishes whether listeners prefer the results."])
    rows.extend(["", f"The selection objective discounts each event by gamma={report['gamma']}. "
                 "It therefore emphasizes early events; full-melody reward and diversity are reported separately.", "",
                 "The fixed C4 primer receives no reward. Starting on the tonic is not a learned improvement. "
                 "Reward rules are documented adaptations, not an exact reproduction of the paper's experiment.", "",
                 "This is a single training seed. Improvements in proxy rewards do not establish listening preference. "
                 "Paired standard errors cover generation variability, not training variability.", "",
                 "Listen to paired MIDI files in baseline/ and psi/. "
                 f"The event horizon is {report['horizon'] / 16:g} bars "
                 f"({report['horizon'] / 4 * 60 / report['qpm']:g} seconds at {report['qpm']:g} QPM); "
                 "audible endings may be earlier.", ""])
    (output / "report.md").write_text("\n".join(rows))


def generate(args):
    device = setup(args)
    model, saved = load_model(args.checkpoint, device)
    reference = frozen_copy(model)
    if saved.get("algorithm") == "psi":
        reference.load_state_dict(saved["prior_state"])
        rewards = RewardConfig(**saved["config"]["reward"])
    else:
        rewards = RewardConfig()
    result = evaluate(model, reference, args.count, args.seed_start, args.steps,
                      rewards, temperature=args.temperature)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i, record in enumerate(result["records"]):
        export_midi(record["tokens"], args.output_dir / f"melody_{i:03d}.mid", args.qpm)
    save_json(args.output_dir / "samples.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "compare", "generate"):
        p = sub.add_parser(name)
        p.add_argument("--checkpoint", type=Path, default=DEFAULT_PRIOR if name == "train" else DEFAULT_PSI)
        p.add_argument("--output-dir", type=Path, default=ROOT / ("runs/psi_seed42" if name == "train" else f"generated/{name}"))
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--device", default="auto")
        if name == "train":
            p.add_argument("--resume", type=Path)
            p.add_argument("--total-events", type=int, default=200_000)
            p.add_argument("--steps", type=int, default=128)
            p.add_argument("--c", type=float, default=0.5)
            p.add_argument("--gamma", type=float, default=0.5)
            p.add_argument("--learning-rate", type=float, default=1e-4)
            p.add_argument("--target-rate", type=float, default=0.01)
            p.add_argument("--batch-size", type=int, default=32)
            p.add_argument("--replay-capacity", type=int, default=100_000)
            p.add_argument("--warmup", type=int, default=1024)
            p.add_argument("--eval-every", type=int, default=10_000)
            p.add_argument("--eval-count", type=int, default=100)
            p.add_argument("--reward-config", type=Path, help="JSON object overriding RewardConfig coefficients")
            p.set_defaults(func=train)
        else:
            p.add_argument("--count", type=int, default=1000 if name == "compare" else 10)
            p.add_argument("--seed-start", type=int, default=200_000)
            p.add_argument("--temperature", type=float, default=1.0)
            p.add_argument("--qpm", type=float, default=80)
            if name == "compare":
                p.add_argument("--num-samples", type=int, default=20)
                p.set_defaults(func=compare)
            else:
                p.add_argument("--steps", type=int, default=128)
                p.set_defaults(func=generate)
    args = parser.parse_args()
    if args.command == "train":
        if (args.total_events <= 0 or args.total_events % 4 or args.batch_size <= 0 or
            args.replay_capacity < max(args.batch_size, args.warmup) or args.warmup < 0 or
            args.eval_every <= 0 or args.eval_count <= 0 or args.steps < 16 or args.steps % 16 or
            not math.isfinite(args.c) or args.c <= 0 or not 0 <= args.gamma <= 1 or
            not 0 < args.target_rate <= 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
            parser.error("Invalid training settings; events must be a positive multiple of four and steps whole bars")
    else:
        if args.count <= 0 or not math.isfinite(args.temperature) or args.temperature <= 0 or not math.isfinite(args.qpm) or args.qpm <= 0:
            parser.error("Count, temperature and QPM must be positive and finite")
    args.func(args)


if __name__ == "__main__":
    main()
