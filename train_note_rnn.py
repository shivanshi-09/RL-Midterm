"""Train the MLE Note RNN and retain the best validation checkpoint."""

import argparse
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from note_rnn import NUM_CLASSES, NoteRNN, collate, masked_targets, next_event_loss

ROOT = Path(__file__).resolve().parent


def batches(records, batch_size, rng=None):
    indices = list(range(len(records)))
    if rng is not None:
        rng.shuffle(indices)
    # Buckets reduce padding while retaining shuffled training batches.
    result = []
    for start in range(0, len(indices), batch_size * 16):
        pool = sorted(indices[start:start + batch_size * 16],
                      key=lambda i: len(records[i]["tokens"]))
        result.extend(pool[i:i + batch_size] for i in range(0, len(pool), batch_size))
    if rng is not None:
        rng.shuffle(result)
    for ids in result:
        yield collate([records[i] for i in ids])


@torch.no_grad()
def evaluate(model, records, batch_size, device, skip_first):
    model.eval()
    nll = correct = count = pitch_nll = pitch_correct = pitch_count = 0
    for x, y in batches(records, batch_size):
        x, y = x.to(device), y.to(device)
        targets = masked_targets(y, skip_first)
        logits, _ = model(x)
        losses = F.cross_entropy(logits.transpose(1, 2), targets,
                                 ignore_index=-100, reduction="none")
        valid = targets != -100
        pitches = targets >= 2
        predicted = logits.argmax(-1)
        count += int(valid.sum())
        nll += float(losses.sum())
        correct += int(((predicted == targets) & valid).sum())
        pitch_count += int(pitches.sum())
        pitch_nll += float(losses[pitches].sum())
        pitch_correct += int(((predicted == targets) & pitches).sum())
    if not count:
        raise ValueError("No evaluation targets after warmup")
    ce = nll / count
    return {"cross_entropy": ce, "perplexity": math.exp(ce),
            "accuracy": correct / count, "scored_events": count,
            "note_on_accuracy": pitch_correct / pitch_count if pitch_count else None,
            "note_on_cross_entropy": pitch_nll / pitch_count if pitch_count else None,
            "scored_note_on_events": pitch_count}


def unigram_baseline(train, valid, skip_first):
    counts = torch.ones(NUM_CLASSES, dtype=torch.float64)
    for r in train:
        counts += torch.bincount(r["tokens"][1 + skip_first:].long(), minlength=NUM_CLASSES)
    logp = (counts / counts.sum()).log()
    total = nll = 0
    for r in valid:
        targets = r["tokens"][1 + skip_first:].long()
        nll -= float(logp[targets].sum())
        total += len(targets)
    return {"cross_entropy": nll / total, "perplexity": math.exp(nll / total)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    p.add_argument("--output-dir", type=Path, default=ROOT / "runs/note_rnn")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--hidden-size", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=0.001,
                   help="Stable PyTorch default; use 0.5 to reproduce the paper's reported setting.")
    p.add_argument("--weight-decay", type=float, default=2.5e-5)
    p.add_argument("--decay-steps", type=int, default=1000)
    p.add_argument("--decay-rate", type=float, default=0.85)
    p.add_argument("--clip-norm", type=float, default=5)
    p.add_argument("--skip-first", type=int, default=8,
                   help="Mask eight initial predictions, following the paper (upstream code uses 32).")
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=30000)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if min(args.hidden_size, args.batch_size, args.max_epochs, args.max_steps,
           args.patience, args.decay_steps) <= 0 or args.skip_first < 0:
        p.error("Sizes/budgets/patience must be positive; skip-first must be nonnegative")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        p.error("Learning rate must be finite and positive")
    if (not math.isfinite(args.decay_rate) or not 0 < args.decay_rate <= 1 or
            not math.isfinite(args.clip_norm) or args.clip_norm <= 0 or
            not math.isfinite(args.weight_decay) or args.weight_decay < 0):
        p.error("Decay rate must be in (0,1], clip norm positive, and weight decay nonnegative")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable in the installed PyTorch build")
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = {s: torch.load(args.data_dir / f"{s}.pt", weights_only=True)
            for s in ("train", "valid", "test")}
    encoding = data["train"]["encoding"]
    excluded_by_warmup = {}
    for split, d in data.items():
        if d["encoding"] != encoding:
            raise ValueError(f"Encoding mismatch in {split}")
        kept = [r for r in d["records"] if len(r["tokens"]) > args.skip_first + 1]
        excluded_by_warmup[split] = len(d["records"]) - len(kept)
        d["records"] = kept
        if not d["records"]:
            raise ValueError(f"Empty {split} split")
    model = NoteRNN(args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    initial = evaluate(model, data["valid"]["records"], args.batch_size, device, args.skip_first)
    unigram = unigram_baseline(data["train"]["records"], data["valid"]["records"], args.skip_first)
    history = {"config": config, "encoding": encoding, "device": device,
               "excluded_by_warmup": excluded_by_warmup,
               "torch_version": str(torch.__version__), "initial_validation": initial,
               "unigram_validation": unigram, "epochs": []}
    print(f"Device={device}; parameters={sum(p.numel() for p in model.parameters()):,}; "
          f"initial val CE={initial['cross_entropy']:.4f}; unigram CE={unigram['cross_entropy']:.4f}",
          flush=True)
    best_ce, stale, step, best_epoch = float("inf"), 0, 0, 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        started = time.monotonic()
        train_nll = train_count = 0
        for x, y in batches(data["train"]["records"], args.batch_size, rng):
            lr = args.learning_rate * args.decay_rate ** (step // args.decay_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = next_event_loss(logits, y, args.skip_first)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm,
                                           error_if_nonfinite=True)
            optimizer.step()
            count = int((masked_targets(y, args.skip_first) != -100).sum())
            train_nll += float(loss.detach()) * count
            train_count += count
            step += 1
            if step >= args.max_steps:
                break
        validation = evaluate(model, data["valid"]["records"], args.batch_size, device, args.skip_first)
        entry = {"epoch": epoch, "step": step, "learning_rate": lr,
                 "train_cross_entropy": train_nll / train_count,
                 "validation": validation, "seconds": time.monotonic() - started}
        history["epochs"].append(entry)
        payload = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                   "model_config": {"hidden_size": args.hidden_size}, "train_config": config,
                   "encoding": encoding, "epoch": epoch, "step": step,
                   "validation": validation}
        torch.save(payload, args.output_dir / "last.pt")
        significant_improvement = validation["cross_entropy"] < best_ce - 1e-4
        if validation["cross_entropy"] < best_ce:
            best_ce, best_epoch = validation["cross_entropy"], epoch
            torch.save(payload, args.output_dir / "best.pt")
        stale = 0 if significant_improvement else stale + 1
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"Epoch {epoch:3d} step {step:5d}: train CE={entry['train_cross_entropy']:.4f} "
              f"val CE={best_ce if best_epoch == epoch else validation['cross_entropy']:.4f} "
              f"acc={validation['accuracy']:.3%} note-on acc={validation['note_on_accuracy']:.3%} "
              f"({entry['seconds']:.1f}s)", flush=True)
        if stale >= args.patience or step >= args.max_steps:
            break
    best = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["model_state"])
    metrics = {"best_epoch": best_epoch, "best_step": best["step"],
               "excluded_by_warmup": excluded_by_warmup,
               "initial_validation": initial, "unigram_validation": unigram,
               "validation": evaluate(model, data["valid"]["records"], args.batch_size, device, args.skip_first),
               "test": evaluate(model, data["test"]["records"], args.batch_size, device, args.skip_first)}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
