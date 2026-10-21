# RL-Midterm

MLE melody baseline for [Sequence Tutor](https://proceedings.mlr.press/v70/jaques17a/jaques17a.pdf):
a PyTorch LSTM predicts the next event in a monophonic melody. This is the
supervised prior model for a later RL stage.

## Setup and run

The project virtual environment is `.venv` (Python 3.13). The pinned CUDA
12.8 PyTorch build works with the installed NVIDIA driver and H100 GPUs.

```bash
cd .
source .venv/bin/activate
pip install -r requirements.txt
python prepare_melody_data.py
python -m unittest -v test_note_rnn
python train_note_rnn.py --device cuda
python generate_melodies.py
```

Preparation writes `data/processed/{train,valid,test}.pt` and a JSON conversion
report. Training writes `runs/note_rnn/best.pt`, `last.pt`, `history.json`, and
`metrics.json`. Sampling writes ten MIDI files and their event sequences to
`generated/mle/`. These generated artifacts and the environment are Git ignored.

## Data and encoding

The retained records in `cleaned_melodies_midi_backed.json.gz` supply IDs, splits,
and song metadata. Pitch and timing are re-read from each archive's `MELODY`
track, including `Hooktheory_Valid_MIDI.tar.gz`; the old cleaner's validation
records used JSON pitches. Unmatched validation records are excluded. The
original song splits are preserved and checked for cross-split song overlap.

MIDI ticks are converted to quarter-note beats. Cleaning removes exact duplicate
events and leading silence, and requires at least six notes. A source melody
remains one sequence, including its internal rests; no additional minimum bar
length or pitch-diversity filtering is applied.

Use the actual `note_seq` implementation of Magenta's relative quantizer and
[`Melody.squash(48, 84, transpose_to_key=0)`](https://github.com/magenta/note-seq/blob/main/note_seq/melodies_lib.py).
Squash estimates a major-key center, transposes it to C, centers the register,
and moves remaining out-of-range notes by octaves into MIDI 48–83. It preserves
pitch class after the global transposition, but individual octave folds can
change intervals; those folds cannot be inverted by the saved global shift.

One time step is a sixteenth note (four steps per quarter). Event IDs follow
[Magenta RL Tuner](https://github.com/magenta/magenta/blob/main/magenta/models/rl_tuner/rl_tuner_ops.py):

| ID | Meaning |
| --- | --- |
| 0 | No event: continue the current note or rest |
| 1 | Note-off: end the current note |
| 2–37 | Start MIDI pitch `ID + 46` (48–83) |

A quarter-note MIDI 60 followed by a quarter rest is `[14,0,0,0,1,0,0,0]`.
Repeated note-on tokens rearticulate a pitch. Magenta extends durations that
round to zero to one step. When onsets round to the same step it retains the
highest pitch; new onsets also end the preceding note. The preprocessing report
counts duration extensions, discarded onset collisions, and octave folds.

## Model, loss, and evaluation

The network uses 38-wide one-hot inputs, one LSTM layer with 100 units and a
38-logit output head. `NoteRNN.forward(tokens, state=None)` returns logits and
`(h,c)`, so the same model can supply a recurrent prior in a future RL port.
Each batch contains independent full melodies, with length buckets to limit
padding. Targets are shifted by one event. Cross-entropy averages over actual
targets after eight initial prediction positions; padding and warmup positions
contribute no loss. For `[e0,e1,...]`, the first scored target is `e9`.

Defaults use Adam, batch 128, learning rate 0.001, staircase decay
0.85 every 1,000 updates, gradient norm clip 5, and L2 weight decay 2.5e-5. The
paper reports learning rate 0.5; a full run at that rate was unstable here and
its best model had zero note-on argmax accuracy. That run is preserved in
`runs/paper_lr_0p5/`. The recommended PyTorch run uses 0.001; all other settings
above follow the paper. Reproduce the reported setting explicitly with
`--learning-rate 0.5 --output-dir runs/paper_lr_0p5_reproduction`.
The paper says to skip eight initial notes; the archived RL Tuner code sets its
warmup to 32 events. This implementation uses eight event predictions per the
accepted baseline plan; `--skip-first 32` selects the code's setting, filtering
sequences that are too short to have targets after that warmup.

Training evaluates validation every epoch and stops after 15 epochs without a
cross-entropy improvement of at least 1e-4, or at 200 epochs / 30,000 updates.
Seed 42 is used by default. The best checkpoint is chosen by validation loss;
test is evaluated only after selection. Reports include cross-entropy,
perplexity, overall event accuracy, and note-on-only accuracy/cross-entropy so
frequent no-event tokens do not conceal poor pitch prediction. A training-only
unigram prior supplies a validation reference.

Checkpoints contain the state dict, optimizer state, model/training configuration,
event encoding, and validation metrics. PyTorch checkpoints require a PyTorch
RL Tuner implementation or explicit conversion to use Magenta's TensorFlow
runner.

## Completed baseline run

The prepared splits contain 13,491 / 1,326 / 1,466 train / validation / test
melodies. Fourteen training records were too short after quantization to score
with the eight-position warmup. Across the retained splits, quantization merges
106,807 colliding onsets and extends 116,892 short durations to one step;
squash octave-folds 4,139 retained note-on events. Duration extensions and
onset collisions overlap and should not be added as a count of lost notes.

The seed-42 run at learning rate 0.001 completed 200 epochs; the best checkpoint
was epoch 197 (20,882 updates). Cross-entropy below uses natural logarithms.

| Model | Validation CE | Test CE |
| --- | ---: | ---: |
| Random initial LSTM | 3.6822 | — |
| Training unigram prior | 2.2521 | — |
| LSTM, reported LR 0.5 | 2.1238 | 2.1726 |
| LSTM, recommended LR 0.001 | 1.6739 | 1.7046 |

The recommended run's test perplexity is 5.4993, event accuracy 46.54%, and
note-on event accuracy 15.15%. This is a working MLE baseline with substantial
room to improve pitch prediction. The test set was scored after checkpoint
selection; the learning-rate correction was based on training instability and
validation note-on accuracy, without using test scores to choose the rate.

Ten samples in `generated/mle/` were produced with a nine-event C-major primer
`[14,0,18,0,21,0,18,0,14]`, temperature 1, and 128 total steps. Their MIDI files
were read back to verify valid pitch bounds, monophony, timing, and named MELODY
tracks, and one output was successfully reused as a primer. Six unit checks
cover rhythm decoding, short-note collisions, octave folding, loss masks,
incremental recurrent state, and learning a tiny motif.

## Sampling

The default is 128 total events (eight bars of 4/4), QPM 80, temperature 1,
and a MIDI 60 primer (`[14]`). Generated melodies use the normalized register
and key. Samples are unconstrained draws from the learned MLE distribution.

```bash
python generate_melodies.py --primer-tokens '[14,0,18,0,21,0,18,0]' --steps 128
python generate_melodies.py --primer-midi melody.mid --steps 256 --temperature 0.8
```

MIDI primers must have a track named `MELODY` and are quantized/squashed exactly
as the training data. The step limit includes the entire primer. A sampler
always closes any sustained note at the end of the output.
