"""Sequence Tutor music rules adapted to our sixteenth-note event grid.

Adapted from Magenta's Apache-2.0 RL Tuner, copyright The Magenta Authors.
See references/README.md for revision, license, and deliberate adaptations.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np


COMPONENTS = ("key", "tonic", "repetition", "autocorrelation", "motif",
              "repeated_motif", "interval", "leap", "extrema")


@dataclass(frozen=True)
class RewardConfig:
    steps_per_bar: int = 16
    steps_per_quarter: int = 4
    key_penalty: float = -1.0
    tonic_bonus: float = 3.0
    repetition_penalty: float = -100.0
    autocorrelation_weight: float = 3.0
    autocorrelation_threshold: float = 0.15
    motif_bonus: float = 3.0
    motif_complexity: float = 0.3
    repeated_motif_bonus: float = 4.0
    interval_scale: float = 5.0
    leap_bonus: float = 5.0
    double_leap_penalty: float = -5.0
    leap_return_wait_beats: int = 6
    extrema_bonus: float = 3.0

    def __post_init__(self):
        if self.steps_per_bar != 16 or self.steps_per_quarter != 4:
            raise ValueError("Reward timing must match the checkpoint's sixteenth-note encoding")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Reward coefficients must be finite numbers")
        if self.leap_return_wait_beats < 0:
            raise ValueError("Leap return wait cannot be negative")


def pitch(token):
    return int(token) + 46 if token >= 2 else None


def autocorrelations(sounding, steps_per_quarter=4):
    """Upstream covariance/variance estimate, masking pairs involving silence.

    Constant sounding pitch has correlation one; silence has correlation zero.
    Neither condition can produce NaN or masquerade as a varying melody.
    """
    result = []
    x = np.asarray(sounding, dtype=np.float64)
    voiced = x[x >= 0]
    mean = voiced.mean() if len(voiced) else 0.0
    variance = voiced.var() if len(voiced) else 0.0
    for lag in (steps_per_quarter, 2 * steps_per_quarter, 3 * steps_per_quarter):
        if len(x) <= lag:
            result.append(0.0)
            continue
        valid = (x[lag:] >= 0) & (x[:-lag] >= 0)
        if not valid.any():
            result.append(0.0)
        elif variance < 1e-12:
            result.append(float(valid.sum()) / len(x))
        else:
            cov = ((x[lag:][valid] - mean) * (x[:-lag][valid] - mean)).sum()
            result.append(float(np.clip(cov / len(x) / variance, -1, 1)))
    return result


class MusicReward:
    """State belongs to one episode; step returns only the new action's reward."""

    def __init__(self, horizon=128, config=None, primer=(14,)):
        self.config = config or RewardConfig()
        if horizon < self.config.steps_per_bar or horizon % self.config.steps_per_bar:
            raise ValueError("Horizon must be a positive whole number of bars")
        if not primer or len(primer) >= horizon:
            raise ValueError("Primer must leave at least one generated event")
        self.horizon = horizon
        self.tokens = []
        self.notes = []
        self.sounding = []
        self.active = None
        self.direction = 0
        self.leapt_from = None
        self.since_leap = 0
        self.leaps = 0
        self.resolved = 0
        self.repeated_count = 0
        self.motif_count = 0
        self.repeated_motif_count = 0
        self.totals = dict.fromkeys(COMPONENTS, 0.0)
        for token in primer:
            self._append(int(token))

    def _append(self, token):
        if not 0 <= token < 38:
            raise ValueError("Event ID must be 0..37")
        self.tokens.append(token)
        if token >= 2:
            self.active = pitch(token)
            self.notes.append(self.active)
        elif token == 1:
            self.active = None
        self.sounding.append(-1 if self.active is None else self.active)

    def _repeating(self, action):
        # Count repeated onsets, permitting longer runs when holds/rests occur.
        # NO_EVENT never means another onset, even during a long sustained note.
        if action == 0:
            return False
        count, separated = 0, False
        for token in reversed(self.tokens):
            if token == action:
                count += 1
            elif token in (0, 1):
                separated = True
            else:
                break
        return count > (1 if action == 1 else 6 if separated else 4)

    def _interval(self, token, previous, active_before):
        if token == 0:
            if active_before is None:
                return 0.0
            return 0.3 if active_before % 12 in (0, 7) else 0.075
        if token == 1:
            if active_before is None:  # redundant note-off is not a fresh rest
                return 0.0
            return 0.15 if active_before % 12 in (0, 7) else 0.05
        if previous is None:
            return 0.0
        current = pitch(token)
        interval = abs(current - previous)
        if interval > 12:
            return -1.0
        if interval == 11:
            return -0.3
        if interval == 7 and previous % 12 in (0, 7):
            return 0.1
        if interval == 4 and previous % 12 in (0, 4):
            return 0.15
        return {2: 0.08, 4: 0.09, 5: 0.07, 7: 0.02, 9: 0.05}.get(interval, 0.0)

    def _leap(self, token, previous):
        c = self.config
        if token < 2 or previous is None:
            self.since_leap += 1
            return 0.0
        current = pitch(token)
        delta = current - previous
        reward = 0.0
        if abs(delta) >= 7:
            self.leaps += 1
            direction = 1 if delta > 0 else -1
            if self.direction:
                if direction != self.direction:
                    if self.since_leap > c.leap_return_wait_beats * c.steps_per_quarter:
                        reward = c.leap_bonus
                        self.resolved += 1
                    self.direction, self.leapt_from = 0, None
                else:
                    reward = c.double_leap_penalty
            else:
                self.direction, self.leapt_from = direction, previous
            self.since_leap = 0
        else:
            self.since_leap += 1
            if self.direction and self.direction * (current - self.leapt_from) <= 0:
                reward = c.leap_bonus
                self.resolved += 1
                self.direction, self.leapt_from = 0, None
        return reward

    def step(self, token):
        if len(self.tokens) >= self.horizon:
            raise ValueError("Episode already terminated")
        if not isinstance(token, (int, np.integer)) or not 0 <= token < 38:
            raise ValueError("Event ID must be 0..37")
        c, t = self.config, len(self.tokens)
        r = dict.fromkeys(COMPONENTS, 0.0)
        previous = self.notes[-1] if self.notes else None
        active_before = self.active
        onset = pitch(token)
        if onset is not None and onset % 12 not in (0, 2, 4, 5, 7, 9, 11):
            r["key"] = c.key_penalty
        final_bar = self.horizon - c.steps_per_bar
        if t in (0, final_bar):
            if onset is not None and onset % 12 == 0:
                r["tonic"] = c.tonic_bonus
        elif t > final_bar and active_before is not None and active_before % 12 == 0:
            # Continuation bonus requires an actually sounding tonic.
            if token == 0 or (token == 1 and t > final_bar + 1):
                r["tonic"] = c.tonic_bonus
        if self._repeating(token):
            r["repetition"] = c.repetition_penalty
            self.repeated_count += 1
        r["interval"] = c.interval_scale * self._interval(token, previous, active_before)
        r["leap"] = self._leap(token, previous)
        self._append(token)
        correlations = autocorrelations(self.sounding, c.steps_per_quarter)
        r["autocorrelation"] = -c.autocorrelation_weight * sum(
            abs(x) for x in correlations if abs(x) > c.autocorrelation_threshold)
        if len(self.tokens) >= c.steps_per_bar:
            motif = self.tokens[-c.steps_per_bar:]
            distinct = len({x for x in motif if x >= 2})
            if distinct >= 3:
                self.motif_count += 1
                r["motif"] = c.motif_bonus + c.motif_complexity * (distinct - 3)
                before = self.tokens[:-c.steps_per_bar]
                if any(before[i:i + c.steps_per_bar] == motif
                       for i in range(len(before) - c.steps_per_bar + 1)):
                    self.repeated_motif_count += 1
                    r["repeated_motif"] = c.repeated_motif_bonus + distinct - 3
        if len(self.tokens) == self.horizon and self.notes:
            r["extrema"] = c.extrema_bonus * (
                int(self.notes.count(max(self.notes)) == 1) +
                int(self.notes.count(min(self.notes)) == 1))
        if not all(math.isfinite(x) for x in r.values()):
            raise FloatingPointError("Nonfinite musical reward")
        for key in COMPONENTS:
            self.totals[key] += r[key]
        return float(sum(r.values())), r

    def metrics(self):
        notes = self.notes
        count = max(len(notes), 1)
        generated = max(len(self.tokens) - 1, 1)
        result = {
            "music_reward": sum(self.totals.values()),
            "note_density": len(notes) / len(self.tokens),
            "silence_fraction": sum(x < 0 for x in self.sounding) / len(self.sounding),
            "repetition_rate": self.repeated_count / generated,
            "out_of_key_fraction": sum(x % 12 not in (0, 2, 4, 5, 7, 9, 11) for x in notes) / count,
            "pitch_range": float(max(notes) - min(notes)) if notes else 0.0,
            "unique_pitches": float(len(set(notes))),
            "motif_fraction": self.motif_count / generated,
            "repeated_motif_fraction": self.repeated_motif_count / generated,
            "leaps": float(self.leaps), "resolved_leaps": float(self.resolved),
            "unique_max": float(bool(notes) and notes.count(max(notes)) == 1),
            "unique_min": float(bool(notes) and notes.count(min(notes)) == 1),
        }
        result.update({f"reward_{k}": v for k, v in self.totals.items()})
        result.update({f"autocorr_lag_{i + 1}_beats": v for i, v in enumerate(
            autocorrelations(self.sounding, self.config.steps_per_quarter))})
        return result


def reward_metadata(config):
    return {"version": 1, "config": asdict(config),
            "upstream_revision": "c15687ebd1c1cc12658cd468c737d3c24c02b918",
            "adaptations": ["16 steps per bar; autocorrelation lags 4/8/12",
                            "Pitch rules use onsets; hold and silence are distinct",
                            "Autocorrelation uses sounding pitch and masks silent pairs",
                            "Constant pitch has finite maximal correlation; silence zero",
                            "Tonic bonus recognizes C in any octave and actual continuation",
                            "Leap return wait is six quarter-note beats",
                            "Fixed primer contributes context but receives no reward"]}
