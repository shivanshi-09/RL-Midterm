"""Replay-based recurrent Psi learning and reproducible melody evaluation."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from melody_encoding import ENCODING
from music_rewards import MusicReward, RewardConfig
from note_rnn import NoteRNN


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["encoding"] != ENCODING:
        raise ValueError("Checkpoint event encoding does not match this implementation")
    model = NoteRNN(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state"])
    return model, payload


def frozen_copy(model):
    result = deepcopy(model).eval()
    result.requires_grad_(False)
    result.lstm.flatten_parameters()
    return result


def psi_targets(log_prior, rewards, done, next_scores, c=0.5, gamma=0.5):
    # torch.where prevents terminal bootstraps from propagating even NaN values.
    bootstrap = torch.where(done, torch.zeros_like(rewards),
                            torch.logsumexp(next_scores, dim=-1))
    return log_prior + rewards / c + gamma * bootstrap


@torch.no_grad()
def interpolate_target(target, online, rate=0.01):
    for dest, source in zip(target.parameters(), online.parameters(), strict=True):
        dest.lerp_(source, rate)


class Replay:
    """Compact full histories; never cache an online/target recurrent state.

    Each tuple is (prefix plus action bytes, reward, frozen log prior, terminal).
    Frozen prior values may be cached because its weights never change.
    """

    def __init__(self, capacity=100_000):
        self.items = []
        self.capacity = capacity
        self.position = 0

    def __len__(self):
        return len(self.items)

    def add(self, tokens, reward, log_prior, done):
        if len(tokens) < 2 or any(not 0 <= x < 38 for x in tokens):
            raise ValueError("Replay needs a valid prefix followed by an action")
        record = (bytes(tokens), float(reward), float(log_prior), bool(done))
        if len(self.items) < self.capacity:
            self.items.append(record)
        else:
            self.items[self.position] = record
        self.position = (self.position + 1) % self.capacity

    def batch(self, size, rng, device):
        records = [self.items[i] for i in rng.sample(range(len(self.items)), size)]
        lengths = torch.tensor([len(x[0]) for x in records], device=device)
        tokens = torch.zeros((size, int(lengths.max())), dtype=torch.long)
        for i, record in enumerate(records):
            tokens[i, :len(record[0])] = torch.tensor(list(record[0]))
        return (tokens.to(device), lengths,
                torch.tensor([x[1] for x in records], device=device),
                torch.tensor([x[2] for x in records], device=device),
                torch.tensor([x[3] for x in records], device=device, dtype=torch.bool))

def update(online, target, optimizer, batch, c=0.5, gamma=0.5, target_rate=0.01):
    tokens, lengths, rewards, log_prior, done = batch
    row = torch.arange(len(tokens), device=tokens.device)
    # For [primer, a1, a2], index 1 predicts a2; index 2 predicts the next action.
    current, _ = online(tokens)
    action = tokens[row, lengths - 1]
    prediction = current[row, lengths - 2, action]
    with torch.no_grad():
        future, _ = target(tokens)
        expected = psi_targets(log_prior, rewards, done,
                               future[row, lengths - 1], c, gamma)
    loss = F.mse_loss(prediction, expected)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite Psi loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(online.parameters(), 5, error_if_nonfinite=True)
    optimizer.step()
    interpolate_target(target, online, target_rate)
    return {"loss": float(loss.detach()), "grad_norm": float(grad),
            "prediction": float(prediction.detach().mean()), "target": float(expected.mean())}


class Collector:
    """Four synchronized environments allow one update per four new events."""

    def __init__(self, prior, horizon=128, config=None, batch_size=4):
        self.prior = prior
        self.horizon, self.config, self.batch_size = horizon, config or RewardConfig(), batch_size
        self.reset()

    def reset(self):
        self.histories = [[14] for _ in range(self.batch_size)]
        self.environments = [MusicReward(self.horizon, self.config) for _ in self.histories]
        self.prior_state = None

    @torch.no_grad()
    def collect(self, online, replay, count=None):
        device = next(online.parameters()).device
        prefixes = torch.tensor(self.histories, device=device)
        # Recompute after every optimizer update; prior alone can retain state.
        scores, _ = online(prefixes)
        probs = scores[:, -1].softmax(-1)
        actions = torch.multinomial(probs, 1).squeeze(-1)
        previous = prefixes[:, -1:]
        reference, self.prior_state = self.prior(previous, self.prior_state)
        lp = reference[:, -1].log_softmax(-1).gather(1, actions[:, None]).squeeze(1)
        values = list(zip(actions.tolist(), lp.tolist()))
        terminal = len(self.histories[0]) + 1 == self.horizon
        for i, (action, log_prior) in enumerate(values):
            reward, _ = self.environments[i].step(action)
            self.histories[i].append(action)
            if count is None or i < count:
                replay.add(self.histories[i], reward, log_prior, terminal)
        if terminal:
            self.reset()
        return len(values) if count is None else count

    def state_dict(self):
        # Reconstruct the frozen state and reward state from the full prefix.
        return {"histories": self.histories}

    def load_state_dict(self, state):
        self.histories = [list(x) for x in state["histories"]]
        self.environments = []
        for history in self.histories:
            env = MusicReward(self.horizon, self.config)
            for action in history[1:]:
                env.step(action)
            self.environments.append(env)
        self.prior_state = None
        if len(self.histories[0]) > 1:
            device = next(self.prior.parameters()).device
            with torch.no_grad():
                _, self.prior_state = self.prior(torch.tensor(
                    [x[:-1] for x in self.histories], device=device))


@torch.no_grad()
def evaluate(policy, prior, count=100, seed_start=100_000, horizon=128,
             config=None, c=0.5, gamma=0.5, temperature=1.0, batch_size=100):
    """Independent per-melody random streams; same uniforms for paired models."""
    policy.eval()
    prior.eval()
    device = next(policy.parameters()).device
    all_records = []
    for offset in range(0, count, batch_size):
        n = min(batch_size, count - offset)
        seeds = list(range(seed_start + offset, seed_start + offset + n))
        streams = [np.random.default_rng(s) for s in seeds]
        envs = [MusicReward(horizon, config) for _ in range(n)]
        previous = torch.full((n, 1), 14, device=device, dtype=torch.long)
        state, prior_state = None, None
        statistics = torch.zeros((n, 5), device=device)
        reward_sums = np.zeros(n)
        discounted_music = np.zeros(n)
        for t in range(1, horizon):
            scores, state = policy(previous, state)
            reference, prior_state = prior(previous, prior_state)
            log_policy = (scores[:, -1] / temperature).log_softmax(-1)
            log_prior = reference[:, -1].log_softmax(-1)
            probs = log_policy.exp()
            uniforms = torch.tensor([s.random() for s in streams], device=device)
            actions = (probs.cumsum(-1) < uniforms[:, None]).sum(-1).clamp_max(37)
            entropy = -(probs * log_policy).sum(-1)
            kl = (probs * (log_policy - log_prior)).sum(-1)
            sampled_prior = log_prior.gather(1, actions[:, None]).squeeze(1)
            statistics[:, 0] += sampled_prior
            statistics[:, 1] += entropy
            statistics[:, 2] += kl
            statistics[:, 3] += gamma ** (t - 1) * kl
            statistics[:, 4] += gamma ** (t - 1) * sampled_prior
            for i, action in enumerate(actions.tolist()):
                reward, _ = envs[i].step(action)
                reward_sums[i] += reward
                discounted_music[i] += gamma ** (t - 1) * reward
            previous = actions[:, None]
        stats = statistics.cpu().numpy()
        for i, env in enumerate(envs):
            record = env.metrics()
            record.update({"seed": seeds[i], "tokens": env.tokens,
                           "prior_log_probability": float(stats[i, 0] / (horizon - 1)),
                           "entropy": float(stats[i, 1] / (horizon - 1)),
                           "kl": float(stats[i, 2] / (horizon - 1)),
                           "objective": float(discounted_music[i] / c - stats[i, 3]),
                           "undiscounted_objective": float(reward_sums[i] / c - stats[i, 2])})
            all_records.append(record)
    keys = [k for k in all_records[0] if k not in ("seed", "tokens")]
    summary = {k: float(np.mean([r[k] for r in all_records])) for k in keys}
    summary["duplicate_melody_rate"] = 1 - len({bytes(r["tokens"]) for r in all_records}) / count
    for key in ("music_reward", "objective", "kl", "entropy", "prior_log_probability"):
        summary[f"{key}_std"] = float(np.std([r[key] for r in all_records]))
    if not all(np.isfinite(v) for v in summary.values()):
        raise FloatingPointError("Nonfinite evaluation statistic")
    return {"count": count, "seed_start": seed_start, "summary": summary,
            "records": all_records}


def save_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def save_checkpoint(path, online, target, prior, optimizer, replay, collector,
                    config, metadata, events, updates, history, best_objective):
    payload = {"format_version": 1, "algorithm": "psi", "encoding": ENCODING,
               "model_config": {"hidden_size": online.hidden_size},
               "model_state": online.state_dict(), "target_state": target.state_dict(),
               "prior_state": prior.state_dict(), "optimizer_state": optimizer.state_dict(),
               "replay": replay.items, "replay_position": replay.position,
               "collector": collector.state_dict(),
               "config": config, "metadata": metadata, "events": events, "updates": updates,
               "history": history, "best_objective": best_objective,
               "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)
