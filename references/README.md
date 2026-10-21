# Reward source and adaptations

The reference Python files are unmodified Magenta RL Tuner source, copyright
The Magenta Authors, distributed under the Apache License 2.0 (see LICENSE).
They are retained for attribution and audit and are not imported by our runner.

Upstream revision: `c15687ebd1c1cc12658cd468c737d3c24c02b918`.

- https://github.com/magenta/magenta/blob/c15687ebd1c1cc12658cd468c737d3c24c02b918/magenta/models/rl_tuner/rl_tuner.py
- https://github.com/magenta/magenta/blob/c15687ebd1c1cc12658cd468c737d3c24c02b918/magenta/models/rl_tuner/rl_tuner_ops.py
- Paper: https://proceedings.mlr.press/v70/jaques17a/jaques17a.pdf

The upstream component reward coefficients are retained. Our implementation
uses the paper's gamma=0.5, rather than the archive's different default.
The following reward adaptations are deliberate:

- NO_EVENT=0 continues the current sound or silence; NOTE_OFF=1 begins silence.
  The archived tuner locally interchanges these constants.
- A bar is 16 sixteenth-note steps. Motif windows use 16 steps, final-bar tonic
  rewards begin at step 112 for 128-event episodes, and autocorrelation lags
  are 4, 8, and 12 steps (one, two, and three quarter-note beats).
- Key and extrema rules operate on note onsets; special events cannot be
  mistaken for low pitches. Any octave of C is a tonic.
- Interval hold/rest bonuses require a sounding note before the action.
  Continuing silence and redundant note-offs earn no interval bonus.
- A final-bar tonic continuation bonus requires an actually sounding C.
  Continuing silence cannot keep earning tonic rewards after a note-off.
- Autocorrelation uses sounding MIDI pitches, masking pairs containing silence.
  Its covariance is normalized by full sequence length, as upstream. Constant
  sounding pitch receives its finite constant-signal limit; silence scores zero.
- Repetition penalties count repeated onsets, allowing more when separated by
  holds/rests. Holding a note is not itself another onset. Redundant note-offs
  retain the original repetition threshold.
- Leap state is updated once per event, with the original return wait interpreted
  as six quarter-note beats. Interval lookup and leap detection share the last
  actual onset; perfect fifth and major third special cases retain upstream logic.
- Exact repeated motifs must occur in nonoverlapping windows. Unique extrema
  are rewarded only at episode termination.

The fixed C4 primer is context only: its initial tonic is not rewarded, nor is
it presented as a learned benefit. Primer onsets remain part of phrase context
and full-melody descriptive statistics. Evaluation excludes the primer from
per-action log probabilities, entropy, KL, and reward averages.
