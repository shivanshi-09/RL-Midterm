# Initial Psi learning experiment

Compared 1000 paired seeds at temperature 1.0. Selected checkpoint: 200,000 generated training events.

| Metric | Baseline | Psi | Difference |
|---|---:|---:|---:|
| Music reward per melody | 201.2751 | 253.5366 | +52.2615 |
| Discounted objective | -3.3873 | 0.4015 | +3.7888 |
| Prior log probability per event | -2.1005 | -1.0421 | +1.0583 |
| KL per event | 0.0000 | 0.3752 | +0.3752 |
| Entropy per event | 2.1064 | 0.6696 | -1.4368 |
| Excessive repetition rate | 0.0078 | 0.0000 | -0.0078 |
| Silence fraction | 0.0839 | 0.0156 | -0.0683 |
| Out of key onsets | 0.0998 | 0.0100 | -0.0898 |
| Motif frequency | 0.8442 | 0.6190 | -0.2252 |
| Repeated motif frequency | 0.0000 | 0.0044 | +0.0044 |
| Distinct pitches per melody | 17.1790 | 8.7810 | -8.3980 |
| Pitch range | 25.7340 | 16.3580 | -9.3760 |
| Duplicate melody rate | 0.0000 | 0.0000 | +0.0000 |

Music reward changed by +52.26 per melody; the selection objective changed by +3.789.

Mean distinct pitches changed from 17.18 to 8.78; mean event entropy changed from 2.106 to 0.670. These describe within-melody pitch variety and policy uncertainty; neither establishes whether listeners prefer the results.

The selection objective discounts each event by gamma=0.5. It therefore emphasizes early events; full-melody reward and diversity are reported separately.

The fixed C4 primer receives no reward. Starting on the tonic is not a learned improvement. Reward rules are documented adaptations, not an exact reproduction of the paper's experiment.

This is a single training seed. Improvements in proxy rewards do not establish listening preference. Paired standard errors cover generation variability, not training variability.

Listen to paired MIDI files in baseline/ and psi/. The event horizon is 8 bars (24 seconds at 80 QPM); audible endings may be earlier.
