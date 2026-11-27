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

On the two-armed spiral with 25% of training points corrupted by noise of known level $\sigma$,
ambient iMF (noisy points supervise the velocity only above their noise level, through an
unbiased target) gives 1-step samples with lower $W_2$ to the clean distribution than iMF that
treats the noisy points as clean.

## Kill criterion

Fixed before the first run (checkerboard, MeanFlow, energy distance): the direction is dead if,
in at least 2 of 3 noise levels $\sigma \in \{0.1, 0.3, 1.0\}$, ambient MeanFlow's mean 1-step
energy distance is not lower than the better of naive and clean-only MeanFlow by more than 2 SE
over seeds 42–44, at matched forward passes. The human ruled (2026-11-10) that the first
experiment, on the spiral with iMF and $W_2$, runs without a kill criterion; this one does not
apply to it.

## Current state

**In short.** On the toy, ambient iMF trained on 75% clean / 25% noisy data matches clean iMF at
2 and 8 sampling steps for every $\sigma$ up to 1.0, Gaussian or uniform noise, while standard iMF
degrades with $\sigma$ (single seed). At 1 step no arm, clean included, is good: exact iMF's
one-step map is weak on this toy (Blocked on). The shared code is on main: spiral, MNIST and
ImageNet 256 (as iMF sets it up), class-conditional generation with guidance, FID-type metrics.
ImageNet is ready for real runs on GPUs with at least 40 GB. Waiting on the human's choice of the
next experiment.

### Setup

**Data.** A two-armed spiral of $n$ points, each coordinate of unit variance, arms about 0.67
apart:

$$
\theta = 3\pi\sqrt{U},\ U \sim \mathcal{U}[0,1], \qquad
\rho = \theta + 0.1\,\xi,\ \xi \sim \mathcal{N}(0,1), \qquad
x_{2\text{D}} = \pm\,\frac{\rho\,(\cos\theta,\ \sin\theta)}{4.71}
$$

It is observed in $D = 64$: padded with zeros and rotated by a fixed random orthogonal $Q$
(`src/embed.py`), $x = Q\,(x_{2\text{D}}, 0, \dots, 0)$. Exact iMF never sees $t$; in 64-D the
network can read $t$ from the noise off the plane.

**Corruption.** Each point stays clean with probability 0.75; otherwise it is observed as
$y = x + \sigma\eta$ with $\eta \sim \mathcal{N}(0, I_{64})$ (Gaussian) or
$\eta \sim \mathcal{U}[-\sqrt3, \sqrt3]^{64}$ (uniform, same variance). One $\sigma$ per run.

**iMF.** An MLP $f_\theta(z, h) = (u, v)$, $h = t - r$, from the official iMF parts (width 256,
4 shared blocks, a $u$-head and a $v$-head of 2 blocks each; `src/models/mlp.py`); it never
receives $t$. Per step, on a batch of 1024:

1. $a, b \sim$ logit-normal$(-0.4, 1)$, $t = \max(a,b)$, $r = \min(a,b)$; $r = t$ for half the batch.
2. $e \sim \mathcal{N}(0, I)$, $z = (1-t)\,x + t\,e$, $v_t = e - x$.
3. $v_c$ = $v$-head of $f_\theta(z, 0)$, no gradient.
4. $u, v$ from $f_\theta(z, t-r)$ and $\frac{du}{dt} = \partial_z u\, v_c + \partial_t u$ by one JVP.
5. $V = u + (t-r)\,\operatorname{sg}[\frac{du}{dt}]$.
6. $\mathcal{L} = \mathbb{E}\big[\lVert V - v_t\rVert^2 + \lVert v - v_t\rVert^2\big]$, plain squared error (`norm_p = 0`).
7. Adam, lr $10^{-4}$ (100-step warm-up), $\beta = (0.9, 0.95)$, EMA 0.9999, 100k steps, fp32.

Standard iMF ("naive") uses the noisy $y$ as if it were $x$.

**Ambient iMF** changes steps 2 and 6 for a point with noise $\sigma$ (both reduce to iMF at
$\sigma = 0$). It adds only the missing noise and uses an unbiased target:

$$
c = \frac{\sigma^2 (1-t)^2}{t^2}, \qquad
z = (1-t)\,y + t\sqrt{1-c}\;e, \qquad
\hat v = \sqrt{1-c}\left(1 + \frac{c}{(1-t)(1-c)}\right) e - y,
\qquad \mathbb{E}[\hat v \mid z] = \mathbb{E}[e - x \mid z]
$$

A noisy point contributes only when $t > \sigma/(1+\sigma) + 0.05$ (the margin because
$\operatorname{Var}[\hat v]$ grows like $1/(1-c)$). Plain squared error is required for the
unbiased target to give the right minimiser. The target assumes Gaussian noise, so uniform noise
tests it off-model.

**Sampling and metric.** $k$ steps on $t_i = 1 - i/k$ from $z \sim \mathcal{N}(0, I_{64})$:
$z \leftarrow z - (t_i - t_{i+1})\,u_\theta(z, t_i - t_{i+1})$, EMA weights, $k \in \{1,2,3,4,8\}$.
$W_2$ in 64-D between 4096 samples and 4096 fresh clean points, exact by linear assignment
(`wasserstein2_full`). It splits exactly into an in-plane part and the mean squared distance from
the plane: $(W_2^{64\text{D}})^2 = (W_2^{\text{plane}})^2 + \mathbb{E}[\operatorname{dist}^2]$.
Evaluation differs across GPU types by up to 0.012.

### Code on main

- **Real data** (c7ddd3e, 6147a1b): MNIST and ImageNet 256 latents, FID, precision, recall, IS.
  Per-step GPU memory (60706992): MNIST patch 4 at batch 64 = 17.2 GiB; ImageNet patch 2 at
  batch 32 = 29.6 GiB. MNIST runs at about 1.2 s per step (50k steps ≈ 17 h).
- **Class-conditional generation** (a34ba56, thresholds 1fdefbe): guided ambient iMF with target
  $\hat v + (1 - 1/\omega)(v_c - v_u)$; spiral arms as classes 0 and 1 with an `arm_accuracy`
  metric; `sampler.omega` a list, scored as `<metric>@<steps>_w<omega>`; class-balanced reference
  labels.
- **ImageNet as iMF does it** (c6539c5, README 6817950): SD-VAE ft-MSE posteriors from
  `ChocolateDave/imagenet-256-sd-vae-latents` (1,281,167 train, 50,000 val), iMF's per-channel
  normalization, a fresh latent $x + x_\text{std}\,\epsilon$ each step, no flips; corruption added
  once to the normalized mean.

**Checks.** Every change to main was verified bitwise against the previous main on all nine
spiral cases (unit fingerprints, training sets, references, 50-step training), with only intended
fields differing:
- consolidation 60632190 (CPU), 60636197 (GPU);
- real-data pipeline 60639921, 60641517 (CPU), 60707401 (GPU);
- conditioning 60841856 (CPU), 60842135 (GPU);
- ImageNet setup 60846014 (CPU), 60846098 (cache verification), 60846467 (real-data GPU).

The cache verification matched our PyTorch ft-MSE encoding (correlation 1.000000, std to 2e-5),
confirmed labels against the raw validation shard, and found normalized train latents with mean
−0.001 and std 1.000. The real-data check trained 50 steps on 2,048 images (1,022 noisy at
$\sigma = 0.5$) and decoded its reference images as real ImageNet pictures of classes 0–7 in
order.

GPU check 60842135 trained conditional iMF and ambient iMF for 4,000 steps on the spiral. Arm
accuracy was 0.67–0.68 at 1 step and 0.84–0.87 at 2 steps ($\omega$ = 1, 2); ignoring labels
would give 0.50. My original threshold of 0.9 was unreachable: spiral points blurred to the
samples' in-plane $W_2$ cap accuracy at 0.919 for $W_2$ 0.171 and 0.858 for 0.197, and the 2-step
samples sit at $W_2$ 0.20–0.25. The thresholds are now 0.6 at 1 step and 0.75 at 2 steps
(main 1fdefbe).

### Decisions by the human

- 2026-11-27: the ambient-protein / AlphaFold-database question is dropped. The README is to be
  concise and readable for others (asked 2026-11-26).
- 2026-11-26: class-conditional generation as iMF does it; ImageNet in the standard iMF setup.
- 2026-11-25: only the data and training pipelines need to work for now, not MNIST quality.
- 2026-11-24: shared code consolidated on main (spiral only, behaviour verified unchanged).
  Real-data pipeline built on main.
- 2026-11-23: sweep at $n = 100{,}000$, all eight $\sigma$, seed 42 only; speed of iteration
  first.
- 2026-11-20: $W_2$ in 64-D is the metric for everything; strong-noise sweep up to $\sigma = 1.0$.
- 2026-11-19: 75% clean, one $\sigma$ per run ($\sigma^2 \sim \mathcal{U}[0,1]$ per point was too
  strong); $D = 64$; figures in matplotlib (`figures.py`).
- 2026-11-18: 100k steps; observe the spiral in $D$ dimensions (Pixel MeanFlow toy,
  arXiv 2601.22158 Sec. 5).
- 2026-11-11: exact iMF (the network does not see $t$).
- 2026-11-10: plain squared error; spiral keeps its radial jitter; the spiral experiment runs
  without a kill criterion; job emails at start and end; `--mem` about 1.5× the measured peak.

### Jobs

GPU header unless noted: `--partition=gpu,gpu_quad --qos=gpuquad_qos --gres=gpu:1
--cpus-per-task=2 --mail-type=BEGIN,END`, plus `--exclude=compute-g-17-168` from 60246613 on
(60246388 hit an uncorrectable ECC error there). CPU jobs: short partition, 4 CPUs.

| job | what | script | `--mem` / `--time` | ran |
|---|---|---|---|---|
| 60187230/34/87/88 | fixed-σ sweep, $n$ = 10k | `sweep.sbatch` (8 CPUs) | 6G / 2:30 | 1 h 35 – 1 h 53 |
| 60207440–43 | strong-noise sweep, $n$ = 10k | `sweep.sbatch` (8 CPUs) | 7G / 2:30 | 1 h 43 – 1 h 56 |
| 60204483, 60224909 | re-score both sweeps | `more-steps.sbatch` | 1200M / 0:30 | 18 min |
| 60226916 | five sampling seeds, $\sigma = 1.0$ | `resample.sbatch` | 1200M / 0:30 | 23 min |
| 60233034 | localize the break (ambient, eval every 2k) | `localize.sbatch` | 2G / 2:30 | 2 h 00 |
| 60246613 | half lr | `localize.sbatch` | 1800M / 2:30 | 1 h 58 |
| 60256817 | naive control | `localize.sbatch` | 1700M / 2:30 | 1 h 45 |
| 60264427 | batch 4096 | `localize.sbatch` | 1700M / 4:00 | 2 h 23 |
| 60274565, 60291354, 60291357 | $n$ = 100k: ambient, clean, naive Gaussian | `localize.sbatch` | 1600–1800M / 3:00 | 1 h 52 – 2 h 02 |
| 60313028, 60313029 | $n$ = 100k: ambient and naive uniform | `localize.sbatch` | 1600M / 3:00 | 1 h 51 – 2 h 16 |
| 60311981, 60321685 | re-score $n$ = 100k, $\sigma = 1.0$ | `more-steps.sbatch` | 1200M / 0:30 | 3–5 min |
| 60555244–74 | sweep at $n$ = 100k, 28 jobs (`af-n100k-s<σ>-a<arm>`) | `localize.sbatch` | 1900M / 3:00 | 1 h 14 – 1 h 53 |
| 60627029 | re-score all 33 runs at $n$ = 100k | `more-steps.sbatch` | 1200M / 0:50 | 28 min |
| 60632190, 60639921, 60841856, 60846014 | CPU checks | `consolidate-check-cpu.sbatch` | 1500M / 0:20 | – |
| 60636197 | GPU check, consolidation | `consolidate-check.sbatch` | – / 0:30 | 12 min |
| 60641517 | CPU data smoke | `data-smoke-cpu.sbatch` | 6G / 1:00 | 11 min |
| 60639945, 60704833 | GPU data smoke, out of GPU memory | `data-smoke.sbatch` (4 CPUs) | 3G / 1:30 | – |
| 60706992 | GPU memory probe | `mem-probe.sbatch` | 3G / 0:20 | 1 min |
| 60707401 | GPU data smoke | `data-smoke.sbatch` (4 CPUs) | 3G / 1:30 | 1 h 14 |
| 60842135 | GPU check, conditioning | `consolidate-check.sbatch` | – / 0:15 | 6 min |
| 60844562 | raw ImageNet download | `download-imagenet.sbatch` | – | 7 min |
| 60845969 | latent cache download (CPU) | `download-latents.sbatch` | 2G / 1:00 | 3 min |
| 60846098 | cache verification and assembly (CPU) | `verify-latent-cache.sbatch` | 4G / 2:00 | – |
| 60846467 | real-data GPU check | `imagenet-run-check.sbatch` | 3G / 0:10 | 1 min |

Cancelled or failed with no result: 60626861, 60628326, 60636214 (duplicates or oversized);
60712569 (not needed); 60841574 (three test bugs), 60841575, 60842116; 60845032 (obsolete);
60846066 (out of memory, rerun as 60846098).

## Results

All numbers are seed 42 only: signals, not results. $W_2^{64\text{D}}$ at $k = 1\,/\,2\,/\,8$ steps.

**Sweep at $n = 100{,}000$**, 75% clean (60555244–74 and the $\sigma = 1.0$ runs; all 33 re-scored
on one node by 60627029). Clean iMF: 1.166 / 0.639 / 0.374.

| $\sigma$ | iMF, Gaussian | ambient iMF, Gaussian | ambient iMF, uniform | iMF, uniform |
|---|---|---|---|---|
| 0.05 | 1.130 / 0.639 / 0.391 | 1.206 / 0.690 / 0.382 | 1.161 / 0.657 / 0.378 | 1.123 / 0.631 / 0.390 |
| 0.1 | 0.999 / 0.572 / 0.417 | 1.087 / 0.605 / 0.387 | 1.074 / 0.595 / 0.391 | 1.085 / 0.658 / 0.414 |
| 0.2 | 1.282 / 0.782 / 0.459 | 1.079 / 0.603 / 0.385 | 1.021 / 0.579 / 0.396 | 1.037 / 0.593 / 0.461 |
| 0.25 | 1.095 / 0.632 / 0.489 | 1.041 / 0.613 / 0.387 | 1.012 / 0.621 / 0.397 | 1.040 / 0.592 / 0.488 |
| 0.33 | 0.997 / 0.585 / 0.555 | 0.986 / 0.572 / 0.388 | 0.977 / 0.565 / 0.390 | 1.086 / 0.597 / 0.559 |
| 0.5 | 0.919 / 0.643 / 0.880 | 1.009 / 0.551 / 0.378 | 1.017 / 0.541 / 0.383 | 0.912 / 0.676 / 0.927 |
| 0.75 | 1.124 / 1.314 / 1.990 | 1.030 / 0.558 / 0.395 | 1.053 / 0.540 / 0.392 | 1.275 / 1.416 / 1.862 |
| 1.0 | 1.841 / 2.508 / 3.657 | 1.146 / 0.661 / 0.406 | 1.102 / 0.601 / 0.405 | 2.339 / 2.879 / 3.650 |

- **8 steps.** Ambient iMF is within 0.035 of clean iMF at every $\sigma$ for both noise types;
  standard iMF degrades from $\sigma = 0.1$ (Gaussian 0.417 at 0.1, 0.880 at 0.5, 3.657 at 1.0).
  In the plane, ambient stays at 0.10–0.15; standard iMF rises from 0.11 to 0.46–0.50.
- **2 steps.** The two are within 0.07 up to $\sigma = 0.33$ (except Gaussian 0.2) and part from
  $\sigma = 0.5$.
- **1 step.** Neither is consistently better below $\sigma = 0.75$ (differences up to 0.2 either
  way); all values sit near clean iMF's weak 1.17.
- Uniform noise, off-model for the ambient target, behaves like Gaussian.
- At $\sigma = 1.0$ ambient iMF improves with more steps while standard iMF gets worse as it
  learns the noisy data. Across five sampling seeds (60226916) $\sigma = 1.0$ runs vary by up to
  0.15 at 2 and 8 steps.

**In-plane $W_2$ at $\sigma = 1.0$, $n = 100{,}000$** (60321685), $k = 1 / 2 / 4 / 8$: clean
0.234 / 0.160 / 0.125 / 0.107; ambient Gaussian 0.246 / 0.185 / 0.169 / 0.131; ambient uniform
0.250 / 0.179 / 0.154 / 0.138; standard Gaussian 0.390 / 0.296 / 0.443 / 0.495; standard uniform
0.387 / 0.311 / 0.417 / 0.461.

**Sweeps at $n = 10{,}000$** (60187230–88, 60207440–43; re-scored 60204483, 60224909). Ambient iMF
is below standard iMF at 2 and 8 steps from $\sigma = 0.2$ on, as at $n = 100{,}000$. But from
$\sigma = 0.5$ no arm recovers the spiral: ambient Gaussian is 0.865 / 0.929 at 2 / 8 steps at
$\sigma = 0.5$ and gets worse with more steps at $\sigma = 1.0$ (1.332 / 2.030). Ten times the
points closes that gap to clean iMF.

**One-step break at $n = 10{,}000$.** Ambient Gaussian at $\sigma = 1.0$ scored 5.507 at 1 step at
100k after 0.94–1.29 from 30k to 90k. The retrain with
evaluation every 2k steps (60233034) reproduces it exactly. It is a progressive drift of the
one-step map from 92k to 100k (1.65 → 5.84), with 96% of samples more than 2 from the plane. Each
of three changes removes it (1-step $W_2^{64\text{D}}$):

| run | lowest, step | 80k | 100k | 2-step, 100k |
|---|---|---|---|---|
| $n$ = 10k, batch 1024, lr $10^{-4}$ (60233034) | 0.91, 32k | 1.20 | 5.51 | 1.33 |
| half lr (60246613) | 0.75, 50k | 1.06 | 1.30 | 1.49 |
| batch 4096 (60264427) | 1.30, 24k | 1.54 | 1.60 | 2.02 |
| $n$ = 100k (60274565) | 0.79, 40k | 1.13 | 1.15 | 0.66 |

A slower rise of the 1-step off-plane error is not ambient-specific: clean iMF has it too
(per-coordinate RMS 0.093 at 40k → 0.144 at 100k) while its in-plane $W_2$ improves.

**Clean iMF by $D$** (60027106, 60038059): in-plane $W_2$ at 1 step is 0.724 / 0.519 / 0.389 /
0.293 / 0.236 at $D$ = 2 / 8 / 16 / 32 / 64, and 0.100 at 8 steps for $D = 64$. Hence $D = 64$.

## Next step

Wait for the human's choice of the next experiment.

## Blocked on

**Late one-step break of ambient iMF at $\sigma = 1.0$, $n = 10{,}000$.** Half the lr, 4× the
batch or 10× the training points each remove it. Keeping the original data and step size would
need gradient clipping or a decaying lr in `build_optimizer` (`src/training.py`), a departure from
the official iMF optimizer; a larger training set needs no code change.

**One-step quality of exact iMF on the toy.** Even clean iMF is far from clean data at 1 step at
every $D$ tried. Options, each a departure from exact iMF: $x$-prediction as in Pixel MeanFlow; a
larger network or more steps; giving the network $t$.

## Dead ends

- MeanFlow with adaptive weighting ($p = 1$) on the checkerboard degrades with training (59157265).
- Exact iMF on the 2-D spiral: in-plane $W_2 \approx 0.72$ at 1 step for every lr and EMA tried (59281303); the network cannot read $t$.
- Giving the network $t$ fixes the 2-D toy ($W_2$ 0.113 at 1 step, 59285877) but is not exact iMF (ruled out 2026-11-11).
- $D = 512$: nothing learned in the plane at 50k steps (59987798), as Pixel MeanFlow reports for $u$-prediction.
- 33% clean with $\sigma^2 \sim \mathcal{U}[0,1]$ per noisy point: no arm learns the spiral, samples are a blob (60098738).
- Batch 256 of DiT-B on one GPU for MNIST or ImageNet: out of memory (60639945, 60704833); configs use batch 64 and 32.
