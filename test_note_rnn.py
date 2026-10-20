"""Checks for timing, normalization, causal state, and the MLE loss contract."""

import unittest

import torch
from torch.nn import functional as F

from melody_encoding import decode_tokens, encode_notes
from note_rnn import NoteRNN, collate, next_event_loss


def note(pitch, start, end):
    return {"midi_pitch": pitch, "onset": start, "offset": end}


class EncodingTests(unittest.TestCase):
    def test_note_duration_rest_and_rearticulation(self):
        source = [note(60, 0, 1), note(60, 1, 2), note(64, 2.5, 3)]
        tokens, shift, stats = encode_notes(source)
        self.assertEqual(stats["note_on_events"], 3)
        self.assertEqual(tokens[0], tokens[4])  # Repeated pitch has a new onset.
        self.assertEqual(tokens[1:4], [0, 0, 0])
        self.assertEqual(tokens[8:10], [1, 0])  # Note-off followed by continuing rest.
        decoded = decode_tokens(tokens, qpm=60, transpose_amount=shift)
        self.assertEqual([n.pitch for n in decoded.notes], [60, 60, 64])
        self.assertEqual([(n.start_time, n.end_time) for n in decoded.notes],
                         [(0, 1), (1, 2), (2.5, 3)])

    def test_short_notes_and_same_step_onsets_follow_magenta(self):
        tokens, _, stats = encode_notes([note(60, 0, .02), note(64, .04, .09)])
        self.assertEqual(stats["notes_extended_to_one_step"], 2)
        self.assertEqual(stats["notes_discarded_same_step"], 1)
        self.assertEqual(stats["note_on_events"], 1)
        self.assertEqual(len(tokens), 1)
        self.assertEqual(len(decode_tokens(tokens).notes), 1)

    def test_long_rest_is_retained_and_wide_pitches_fold_by_octave(self):
        tokens, shift, stats = encode_notes([note(21, 0, 1), note(108, 12, 13)])
        self.assertEqual(len(tokens), 52)
        self.assertEqual(stats["note_on_events"], 2)
        self.assertGreater(stats["octave_folded_note_on_events"], 0)
        pitches = [n.pitch for n in decode_tokens(tokens).notes]
        self.assertTrue(all(48 <= p < 84 for p in pitches))
        self.assertTrue(all((p - original - shift) % 12 == 0
                            for p, original in zip(pitches, [21, 108])))


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_shift_padding_and_warmup_loss_and_gradients(self):
        records = [{"tokens": torch.tensor([14, 0, 16, 1, 18])},
                   {"tokens": torch.tensor([21, 0, 1])}]
        x, y = collate(records)
        self.assertEqual(x[0].tolist(), [14, 0, 16, 1])
        self.assertEqual(y[0].tolist(), [0, 16, 1, 18])
        self.assertEqual(y[1].tolist(), [0, 1, -100, -100])
        logits = torch.randn(2, 4, 38, requires_grad=True)
        loss = next_event_loss(logits, y, skip_first=1)
        expected = F.cross_entropy(torch.cat([logits[0, 1:], logits[1, 1:2]]),
                                   torch.tensor([16, 1, 18, 1]))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(float(logits.grad[:, 0].abs().sum()), 0)
        self.assertEqual(float(logits.grad[1, 2:].abs().sum()), 0)

    def test_incremental_state_matches_full_causal_sequence(self):
        model = NoteRNN().eval()
        tokens = torch.tensor([[14, 0, 18, 1, 21]])
        whole, _ = model(tokens)
        state, results = None, []
        for token in tokens.unbind(1):
            logits, state = model(token[:, None], state)
            results.append(logits)
        torch.testing.assert_close(whole, torch.cat(results, 1), atol=1e-6, rtol=1e-5)

    def test_tiny_motif_is_learnable(self):
        model = NoteRNN(hidden_size=16)
        x = torch.tensor([[14, 0, 18, 0, 21, 0] * 3] * 4)
        y = x.roll(-1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=.03)
        initial = float(next_event_loss(model(x)[0], y, skip_first=0).detach())
        for _ in range(30):
            optimizer.zero_grad()
            loss = next_event_loss(model(x)[0], y, skip_first=0)
            loss.backward()
            optimizer.step()
        final = float(next_event_loss(model(x)[0], y, skip_first=0).detach())
        self.assertLess(final, initial * .5)


if __name__ == "__main__":
    unittest.main()
