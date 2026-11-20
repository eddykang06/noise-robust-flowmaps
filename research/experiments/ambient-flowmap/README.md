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
seed 42. Metric (human, 2026-11-20): exact W2 in all 64 dimensions on 4096 samples
(`wasserstein2_full`). Since clean data lie on the plane, W2(64-D)² = W2(plane)² + mean squared
distance from the plane, and both parts are reported. "Off-plane RMS" in older tables is per
coordinate: distance from the plane is √62 times it.

**Fixed-σ sweep** (human, 2026-11-19), 75% clean, one noise level per job,
σ ∈ {0.05, 0.1, 0.2, 0.33}, the five arms of the brief:

| arm | data | objective |
|---|---|---|
| 1 | all clean | iMF (finished run, 60038059) |
| 2 | Gaussian noise on 25% | iMF, noisy points as clean |
| 3 | Gaussian noise on 25% | ambient iMF |
| 4 | uniform noise (variance σ²) on 25% | ambient iMF |
| 5 | uniform noise (variance σ²) on 25% | iMF, noisy points as clean |

`sweep.sbatch` with `DIM=64,SIGMA=<σ>`: jobs 60187230 (0.05), 60187234 (0.1), 60187287 (0.2),
60187288 (0.33), all completed in 1 h 35 min – 1 h 53 min, peak 4.1–4.4 GB. Re-scoring at
1–8 steps: 60194400 (`more-steps.sbatch`, `--mem=1200M` from 60108782's 0.79 GB peak).

```
#SBATCH --partition=gpu,gpu_quad
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=6G
#SBATCH --time=02:30:00   (submitted at 04:00:00, lowered with scontrol while pending)
```

**Running: strong-noise sweep** (human, 2026-11-20: coarser grid reaching higher noise), same
five arms and setup, σ ∈ {0.25, 0.5, 0.75, 1.0}. σ = 1.0 is the top: there t_n = 0.5, ambient
keeps 47% of noisy draws and their target's E|·|² is 4.4× a clean point's; at σ = 1.5 it would
keep 29% at 7×. Jobs 60207440 (0.25), 60207441 (0.5), 60207442 (0.75), 60207443 (1.0), header as
above with `--mem=7G` (1.5 × 4.4 GB) and `--time=02:30:00`. Tests of the 64-D metric: 60207586
(`gpu-tests.sbatch`, now on gpu,gpu_quad; 60204481 cancelled after Slurm put its start 2 days out).

**Standing decisions by the human**
- 2026-11-20: no further seeds for the fixed-σ sweep.
- 2026-11-19: one noise level per run, 75% clean (σ² ~ U[0, 1] per point left most noisy points
  without structure: median σ 0.71 against an arm gap of 0.67). Sweep 60185808 cancelled.
- 2026-11-19: D = 64; all figures are matplotlib (`figures.py`, run in the agent session).
- 2026-11-18: 100k steps; seed 42 only, so differences are single-seed signals, not results.
- 2026-11-18: observe the spiral in D dimensions (Pixel MeanFlow toy, arXiv 2601.22158 Sec. 5).
- 2026-11-11: exact iMF, so the network does not see t.
- 2026-11-20: W2 in all 64 dimensions is the metric for everything from now on; in-plane W2
  misses the off-plane error that full-dimensional noise causes.
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

**Fixed-σ sweep, 75% clean**, D = 64, 100k steps, seed 42 (60187230/34/87/88; all arms
re-scored in 64-D by 60204483). W2 in 64-D at 1 / 2 / 8 steps; clean iMF: 1.148 / 0.623 / 0.372.

| σ | 2 naive G | 3 ambient G | 4 ambient U | 5 naive U |
|---|---|---|---|---|
| 0.05 | 1.104 / 0.633 / 0.394 | 1.126 / 0.638 / 0.384 | 1.139 / 0.641 / 0.380 | 1.161 / 0.660 / 0.388 |
| 0.1 | 1.039 / 0.592 / 0.416 | 1.041 / 0.585 / 0.386 | 1.104 / 0.603 / 0.388 | 1.048 / 0.587 / 0.412 |
| 0.2 | 1.046 / 0.617 / 0.464 | 1.005 / 0.597 / 0.392 | 0.997 / 0.564 / 0.388 | 1.075 / 0.650 / 0.457 |
| 0.33 | 0.996 / 0.693 / 0.743 | 0.837 / 0.570 / 0.514 | 0.743 / 0.545 / 0.494 | 0.984 / 0.667 / 0.719 |

From 4 steps on, ambient iMF is below naive iMF in every cell, for both noise types, and the gap
grows with σ; ambient stays within 0.02 of clean iMF up to σ = 0.2. The gain is off the plane:
at σ = 0.33 naive iMF puts 5% of its 8-step samples more than 2 from the plane (the noisy
training points sit 2.6 off it), ambient 1%. In the plane ambient is equal or slightly worse
(8 steps, σ = 0.33: 0.164 against 0.140), plausibly the cost of its noisier target (noisy
points' E|target|² 102 against 66 for clean at σ = 0.33). One step is far from clean for every
arm, clean included (W2 ≈ 1.0–1.15, almost all off the plane). Seed 42 only: signals, not results.

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

When the strong-noise jobs finish, re-score them at 1–8 steps in 64-D and add them to the sweep page.

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
