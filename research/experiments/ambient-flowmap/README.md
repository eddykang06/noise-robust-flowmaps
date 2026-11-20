---
slug: ambient-flowmap
status: running
hypothesis: On the two-armed spiral with a quarter of the training points noisy, ambient iMF gives 1-step samples closer in W2 to the clean distribution than iMF trained on the same data as if clean.
kill_criterion: Ambient MeanFlow 1-step energy distance not below the better of flowmap-corrupt and clean-only MeanFlow by more than 2 SE (3 seeds) in at least 2 of 3 noise levels.
best_metric:
started: 2026-11-09
---

# ambient-flowmap

## Hypothesis

On the two-armed spiral with 25% of training points corrupted by noise of known level σ,
ambient iMF (noisy points supervise the velocity only above their noise level, through an
unbiased target) gives 1-step samples with lower W2 to the clean distribution than iMF that
treats the noisy points as clean.

## Kill criterion

Fixed before the first run (checkerboard, MeanFlow, energy distance): the direction is dead if,
in at least 2 of 3 noise levels σ ∈ {0.1, 0.3, 1.0}, ambient MeanFlow's mean 1-step energy
distance is not lower than the better of naive and clean-only MeanFlow by more than 2 SE over
seeds 42–44, at matched forward passes. The human ruled (2026-11-10) that the first
experiment, on the spiral with iMF and W2, runs without a kill criterion; this one does not
apply to it.

## Current state

**Method.** Exact iMF (Geng et al., 2025): network from the official code (`src/models/`),
objective and sampler ported from the official JAX code and checked against it
(`tests/test_imf_reference.py`). One stated deviation: plain squared error (`norm_p: 0`, iMF
uses 1.0), so the ambient target stays unbiased. Ambient iMF (`src/objectives/ambient_imf.py`)
replaces the velocity target e − x of a noisy point by an unbiased estimate, used only for
t > σ/(1+σ) + 0.05.

**Setup.** Two-armed spiral, 10,000 points, scaled to unit variance per coordinate (arm gap
≈ 0.67), observed in D = 64 dimensions through a fixed rotation (`src/embed.py`), because exact
iMF never sees t and cannot read it off a 2-D point. Toy MLP: hidden 256, 4 shared blocks, 2 per
head. 100k steps, batch 1024, Adam lr 1e-4 (β₂ 0.95, 100 warmup steps), EMA 0.9999, fp32,
seed 42. Metric: exact W2 in the spiral's plane on 4096 samples (floor 0.122, two independent
clean samples), and off-plane RMS (0 for clean data).

**Running: fixed-σ sweep** (human, 2026-11-19), 75% clean, one noise level per job,
σ ∈ {0.05, 0.1, 0.2, 0.33}, the five arms of the brief:

| arm | data | objective |
|---|---|---|
| 1 | all clean | iMF (finished run, 60038059) |
| 2 | Gaussian noise on 25% | iMF, noisy points as clean |
| 3 | Gaussian noise on 25% | ambient iMF |
| 4 | uniform noise (variance σ²) on 25% | ambient iMF |
| 5 | uniform noise (variance σ²) on 25% | iMF, noisy points as clean |

`sweep.sbatch` with `DIM=64,SIGMA=<σ>`; jobs 60187230 (0.05), 60187234 (0.1), 60187287 (0.2),
60187288 (0.33), all started 20:21–20:24, each ~1 h 55 min. Tests of the new fixed-σ uniform
corruption pass (60187390: 58/58 non-GPU).

```
#SBATCH --partition=gpu,gpu_quad
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=6G
#SBATCH --time=02:30:00   (submitted at 04:00:00, lowered with scontrol while pending)
```

**Standing decisions by the human**
- 2026-11-19: one noise level per run, 75% clean (σ² ~ U[0, 1] per point left most noisy points
  without structure: median σ 0.71 against an arm gap of 0.67). Sweep 60185808 cancelled.
- 2026-11-19: D = 64; all figures are matplotlib (`figures.py`, run in the agent session).
- 2026-11-18: 100k steps; seed 42 only, so differences are single-seed signals, not results.
- 2026-11-18: observe the spiral in D dimensions (Pixel MeanFlow toy, arXiv 2601.22158 Sec. 5).
- 2026-11-11: exact iMF, so the network does not see t.
- 2026-11-10: W2 as the only metric; plain squared error; spiral keeps its 0.1 radial jitter;
  `research/baseline.md` set aside; job emails at start and end; `--mem` ≈ 1.5× measured peak.

## Results

**Clean iMF by dimension D**, 100k steps, seed 42 (60027106, 60038059; re-scored 60033355):

| D | W2@1 | W2@2 | W2@4 | W2@8 | off-plane RMS @8 |
|---|---|---|---|---|---|
| 2 | 0.724 | 0.341 | 0.218 | 0.244 | – |
| 8 | 0.519 | 0.264 | 0.190 | 0.191 | 0.094 |
| 16 | 0.389 | 0.219 | 0.166 | 0.147 | 0.077 |
| 32 | 0.293 | 0.172 | 0.131 | 0.107 | 0.059 |
| 64 | 0.236 | 0.155 | 0.121 | 0.100 | 0.045 |

Larger D helps up to 64; all runs have plateaued (W2@1 changes < 3% from 90k to 100k). At
D = 64, 4 steps reach the W2 floor, but 1 step (0.236) does not.

**Five arms, 33% clean, σ² ~ U[0, 1]**, D = 64, 100k steps, seed 42 (60098738; re-scored 60108782):

| arm | W2@1 | W2@2 | W2@8 | off-plane RMS @8 |
|---|---|---|---|---|
| 1 iMF, clean | 0.243 | 0.160 | 0.104 | 0.047 |
| 2 iMF, Gaussian | 0.334 | 0.297 | 0.299 | 0.379 |
| 3 ambient iMF, Gaussian | 0.358 | 0.256 | 0.251 | 0.302 |
| 4 ambient iMF, uniform | 0.353 | 0.249 | 0.256 | 0.318 |
| 5 iMF, uniform | 0.356 | 0.307 | 0.292 | 0.390 |

No corrupted arm learns the spiral: samples are a blob with no arms at every step count. The
ambient arms' lower W2 from 2 steps on is a tighter blob nearer the plane, not a recovered
spiral. The noise was too strong (see above), which the running sweep corrects. Evaluation
differs across GPU nodes by up to 0.012 in W2 (not bitwise reproducible across nodes).

## Next step

When the four σ jobs finish, re-score every arm at 1–8 steps and give the sweep its own page.

## Blocked on

**One-step quality of exact iMF on the toy (2026-11-18).** Even clean iMF does not reach the
W2 floor in one step at any D tried. Options, each a departure from exact iMF for the human to
rule on: x-prediction as in Pixel MeanFlow; a larger network or more steps; giving the network t.

## Dead ends

- MeanFlow with adaptive weighting (p = 1) on the checkerboard degrades with training (59157265).
- Exact iMF on the 2-D spiral: W2@1 ≈ 0.72 at every lr and EMA tried (59281303); the network cannot read t.
- Giving the network t fixes the 2-D toy (W2@1 0.113, 59285877) but is not exact iMF (ruled out 2026-11-11).
- D = 512: nothing learned in the plane at 50k steps (59987798), as Pixel MeanFlow reports for u-prediction.
- σ² ~ U[0, 1] per noisy point: too strong to leave any structure (60098738).
