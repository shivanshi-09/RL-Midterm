"""Numerical and behavioral checks for the recurrent Psi implementation."""

from copy import deepcopy
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from music_rewards import MusicReward, RewardConfig, autocorrelations
from note_rnn import NoteRNN
from psi_learning import (Collector, Replay, evaluate, frozen_copy,
                          interpolate_target, psi_targets, save_checkpoint, update)
from rl_melody import export_midi


class PsiTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        random.seed(42)

    def test_numerical_target_and_terminal_mask(self):
        future = torch.tensor([[0., np.log(3.)], [float("nan"), float("nan")]])
        actual = psi_targets(torch.tensor([-2., -3.]), torch.tensor([2., 1.]),
                             torch.tensor([False, True]), future)
        expected = torch.tensor([2. + 0.5 * np.log(4.), -1.], dtype=actual.dtype)
        torch.testing.assert_close(actual, expected)

    def test_replay_prediction_alignment_and_padding(self):
        net = NoteRNN(8)
        target = frozen_copy(net)
        replay = Replay(3)
        replay.add([14, 0], 1., -1., False)
        replay.add([14, 2, 4, 6], 2., -2., True)
        batch = replay.batch(2, random, "cpu")
        tokens, lengths, rewards, lp, done = batch
        individual_predictions, expected_values = [], []
        with torch.no_grad():
            for i, length in enumerate(lengths.tolist()):
                scores, _ = net(tokens[i:i+1, :length])
                individual_predictions.append(scores[0, length - 2, tokens[i, length - 1]])
                expected_values.append(psi_targets(lp[i:i+1], rewards[i:i+1], done[i:i+1], scores[:, -1])[0])
        expected_loss = torch.mean((torch.stack(individual_predictions) - torch.stack(expected_values)) ** 2)
        result = update(net, target, torch.optim.Adam(net.parameters()), batch)
        self.assertAlmostEqual(result["loss"], float(expected_loss), places=5)

    def test_target_interpolation(self):
        online = NoteRNN(4)
        target = frozen_copy(online)
        before = deepcopy(target.state_dict())
        with torch.no_grad():
            for parameter in online.parameters():
                parameter.add_(2)
        interpolate_target(target, online, .01)
        for key, value in target.state_dict().items():
            torch.testing.assert_close(value, before[key] + .02)

    def test_prior_frozen_and_collector_uses_current_weights(self):
        online = NoteRNN(8)
        prior = frozen_copy(online)
        before = deepcopy(prior.state_dict())
        collector = Collector(prior, horizon=16)
        replay = Replay(100)
        collector.collect(online, replay)
        update(online, frozen_copy(online), torch.optim.Adam(online.parameters()),
               replay.batch(4, random, "cpu"))
        collector.collect(online, replay)
        for sequence, _, cached_lp, _ in replay.items:
            with torch.no_grad():
                scores, _ = prior(torch.tensor([list(sequence[:-1])]))
                correct = scores[0, -1].log_softmax(-1)[sequence[-1]].item()
            self.assertAlmostEqual(cached_lp, correct, places=5)
        for key, value in prior.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in prior.parameters()))

    def test_resumption_reproduces_next_collection_and_update(self):
        net = NoteRNN(8)
        target, prior = frozen_copy(net), frozen_copy(net)
        optimizer = torch.optim.Adam(net.parameters())
        replay, collector = Replay(20), Collector(prior, horizon=16)
        for _ in range(7):
            collector.collect(net, replay)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_checkpoint(path, net, target, prior, optimizer, replay, collector,
                            {}, {}, 28, 0, [], -100.)
            saved = torch.load(path, weights_only=True)
            collector.collect(net, replay)
            update(net, target, optimizer, replay.batch(4, random, "cpu"))
            expected = deepcopy(net.state_dict())
            new_net = NoteRNN(8)
            new_net.load_state_dict(saved["model_state"])
            new_target = frozen_copy(new_net)
            new_target.load_state_dict(saved["target_state"])
            new_prior = frozen_copy(new_net)
            new_prior.load_state_dict(saved["prior_state"])
            new_optimizer = torch.optim.Adam(new_net.parameters())
            new_optimizer.load_state_dict(saved["optimizer_state"])
            new_replay, new_collector = Replay(20), Collector(new_prior, horizon=16)
            new_replay.items, new_replay.position = saved["replay"], saved["replay_position"]
            new_collector.load_state_dict(saved["collector"])
            torch.set_rng_state(saved["torch_rng"])
            random.setstate(saved["python_rng"])
            new_collector.collect(new_net, new_replay)
            self.assertEqual(new_replay.items, replay.items)
            update(new_net, new_target, new_optimizer, new_replay.batch(4, random, "cpu"))
            for key, value in new_net.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)

    def test_seeded_evaluation_is_batch_independent(self):
        net = NoteRNN(8)
        prior = frozen_copy(net)
        a = evaluate(net, prior, count=3, horizon=16, batch_size=3)
        b = evaluate(net, prior, count=3, horizon=16, batch_size=1)
        self.assertEqual([r["tokens"] for r in a["records"]], [r["tokens"] for r in b["records"]])
        self.assertAlmostEqual(a["summary"]["kl"], 0., places=6)
        self.assertAlmostEqual(a["summary"]["objective"], b["summary"]["objective"], places=5)

    def test_midi_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            for i, sequence in enumerate(([14, 0, 0, 0, 1, 0, 18, 0] * 2, [1] + [0] * 15)):
                result = export_midi(sequence, Path(directory) / f"sample_{i}.mid")
                self.assertTrue(result["validated"])


class RewardTests(unittest.TestCase):
    def test_reward_overrides_validate_encoding_and_finiteness(self):
        self.assertEqual(RewardConfig(motif_bonus=2.).motif_bonus, 2.)
        for override in ({"steps_per_bar": 8}, {"key_penalty": float("nan")},
                         {"leap_return_wait_beats": -1}):
            with self.assertRaises(ValueError):
                RewardConfig(**override)

    def test_leap_resolves_by_returning_in_steps(self):
        env = MusicReward(32)
        env.step(21)  # C4 -> G4, one ascending fifth
        self.assertEqual(env.leaps, 1)
        env.step(19)  # F4
        env.step(18)  # E4
        env.step(16)  # D4
        _, rewards = env.step(14)  # Return to C4 without another leap
        self.assertEqual(rewards["leap"], 5.)
        self.assertEqual(env.resolved, 1)

    def test_final_bar_timing(self):
        env = MusicReward(128)
        for _ in range(111):
            env.step(0)
        self.assertEqual(len(env.tokens), 112)
        _, r = env.step(14)
        self.assertEqual(r["tonic"], 3.)
        _, r = env.step(0)
        self.assertEqual(r["tonic"], 3.)

    def test_hold_rest_and_onsets_are_distinct(self):
        env = MusicReward(16)
        _, hold = env.step(0)
        _, off = env.step(1)
        _, silent = env.step(0)
        self.assertEqual(hold["interval"], 1.5)
        self.assertEqual(off["interval"], .75)
        self.assertEqual(silent["interval"], 0.)
        self.assertEqual(len(env.notes), 1)
        self.assertEqual(env.sounding, [60, 60, -1, -1])
        self.assertEqual(env.totals["repetition"], 0.)

    def test_repetition_and_key(self):
        env = MusicReward(16)
        for _ in range(4):
            _, r = env.step(14)
            self.assertEqual(r["repetition"], 0.)
        _, r = env.step(14)
        self.assertEqual(r["repetition"], -100.)
        _, r = env.step(15)  # C sharp
        self.assertEqual(r["key"], -1.)
        _, r = env.step(1)
        self.assertEqual(r["key"], 0.)

    def test_finite_constant_silent_autocorrelation_and_actual_lag(self):
        self.assertTrue(all(np.isfinite(autocorrelations([60] * 32))))
        self.assertGreater(autocorrelations([60] * 32)[0], .8)
        self.assertEqual(autocorrelations([-1] * 32), [0., 0., 0.])
        signal = [60, 62, 64, 65] * 8
        self.assertGreater(autocorrelations(signal)[0], .8)

    def test_motif_windows_extrema_and_termination(self):
        pattern = [14, 0, 18, 0, 21, 0, 18, 0, 14, 0, 18, 0, 21, 0, 18, 0]
        env = MusicReward(32)
        for event in (pattern + pattern)[1:]:
            _, r = env.step(event)
        self.assertGreater(r["repeated_motif"], 0.)
        self.assertEqual(r["extrema"], 0.)
        with self.assertRaises(ValueError):
            env.step(14)


if __name__ == "__main__":
    unittest.main()
