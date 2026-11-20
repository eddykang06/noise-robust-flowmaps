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

### Data

**Clean data.** A two-armed spiral of 10,000 points in $\mathbb{R}^2$:

$$
\theta = 3\pi\sqrt{U},\quad U \sim \mathcal{U}[0,1], \qquad
\rho = \theta + 0.1\,\xi,\quad \xi \sim \mathcal{N}(0,1), \qquad
x_{2\text{D}} = \pm\,\frac{\rho\,(\cos\theta,\ \sin\theta)}{4.71}
$$

The sign picks the arm; dividing by 4.71 gives each coordinate unit variance. The two arms are
about 0.67 apart.

**Observed in $D = 64$.** Each point is padded with 62 zeros and rotated by a fixed random
orthogonal matrix $Q$ (`src/embed.py`), so clean data lie on a 2-D plane in $\mathbb{R}^{64}$:

$$
x = Q\,(x_{2\text{D}},\ 0,\dots,0) \in \mathbb{R}^{64}
$$

This is needed because exact iMF never sees $t$; in 64-D the network can read $t$ from how much
noise sits off the plane.

**Corruption.** Each point stays clean with probability 0.75. Otherwise it is observed as

$$
y = x + \sigma\,\eta, \qquad
\eta \sim \mathcal{N}(0, I_{64}) \ \text{(Gaussian)} \quad\text{or}\quad
\eta \sim \mathcal{U}[-\sqrt3, \sqrt3]^{64} \ \text{(uniform, same variance)}
$$

The noise is in all 64 coordinates. Every training point carries its own $\sigma$ ($0$ if clean);
each run uses one $\sigma$.

### Training: iMF (arms 1, 2, 5)

**Network.** $f_\theta(z, h) = (u, v)$ with $h = t - r$: a toy MLP (`src/models/mlp.py`) built from
the official iMF parts (hidden width 256; 4 shared blocks, then a $u$-head and a $v$-head of 2
blocks each). It never receives $t$ itself. $u$ is the average velocity over $[r, t]$, $v$ the
instantaneous velocity at $t$.

**One training step**, on a batch of 1024 points $x$ (the naive arms use the noisy $y$ as if it
were $x$):

1. **Times.** Draw $a, b \sim \text{logit-normal}(-0.4,\ 1)$ and set
   $t = \max(a,b)$, $r = \min(a,b)$. For half of the batch set $r = t$.
2. **Noise the data.**
   $$
   e \sim \mathcal{N}(0, I_{64}), \qquad z = (1-t)\,x + t\,e, \qquad v_t = e - x
   $$
3. **Tangent.** $v_c = v\text{-head of } f_\theta(z, 0)$, with no gradient.
4. **Forward and JVP.** $u$, $v$ from $f_\theta(z, t - r)$, and
   $$
   \frac{du}{dt} = \frac{\partial u}{\partial z}\,v_c + \frac{\partial u}{\partial t}
   \qquad \text{(one Jacobian–vector product)}
   $$
5. **Compound prediction.**
   $$
   V = u + (t - r)\,\operatorname{sg}\!\left[\frac{du}{dt}\right]
   $$
   where $\operatorname{sg}$ is stop-gradient.
6. **Loss** (plain squared error, summed over the 64 coordinates, averaged over the batch):
   $$
   \mathcal{L} = \mathbb{E}\Big[\ \lVert V - v_t \rVert^2 + \lVert v - v_t \rVert^2\ \Big]
   $$
   Official iMF divides by an adaptive weight (`norm_p = 1`); here `norm_p = 0`.
7. **Update.** Adam, learning rate $10^{-4}$ (linear warm-up over 100 steps, then constant),
   $\beta = (0.9,\ 0.95)$; EMA of the weights with decay 0.9999. 100k steps, fp32.

### Training: ambient iMF (arms 3, 4)

Same network, times, JVP and optimizer. Only steps 2 and 6 change, per point with noise level
$\sigma$; for $\sigma = 0$ both reduce exactly to iMF.

**2′. Noise the data, and the target.** Add only the noise the point is still missing:

$$
c = \frac{\sigma^2 (1-t)^2}{t^2}, \qquad
z = (1-t)\,y + t\sqrt{1-c}\;e
$$

so $z$ has the same distribution as a clean point's $z_t$. The target is

$$
\hat v = \sqrt{1-c}\left(1 + \frac{c}{(1-t)(1-c)}\right) e \;-\; y,
\qquad \mathbb{E}\big[\hat v \mid z\big] = \mathbb{E}\big[e - x \mid z\big]
$$

an unbiased estimate of the clean velocity. It replaces $v_t$ in steps 5 and 6.

**6′. Which points count.** A noisy point contributes only when

$$
t > \frac{\sigma}{1+\sigma} + 0.05
$$

i.e. above its own noise level on the path, plus a margin because $\operatorname{Var}[\hat v]$
grows like $1/(1-c)$. Other points contribute $0$ to the batch mean.

Plain squared error is required: with adaptive weighting an unbiased target no longer gives the
right minimiser. Ambient iMF assumes Gaussian noise, so arm 4 (uniform) tests it off-model.

### Sampling and metric

**Sampler.** With $k$ steps on the grid $t_i = 1 - i/k$, start from $z \sim \mathcal{N}(0, I_{64})$
and repeat for $i = 0, \dots, k-1$:

$$
z \leftarrow z - (t_i - t_{i+1})\; u_\theta\big(z,\ h = t_i - t_{i+1}\big)
$$

using the EMA weights. Scored at $k = 1, 2, 3, 4, 8$.

**Metric** (human, 2026-11-20). $W_2$ in all 64 dimensions between 4096 samples and 4096 fresh
clean points, exact by linear assignment (`wasserstein2_full`). Because clean data lie on the
plane, it splits exactly:

$$
\big(W_2^{64\text{D}}\big)^2 = \big(W_2^{\text{plane}}\big)^2 + \mathbb{E}\big[\operatorname{dist}(\text{sample}, \text{plane})^2\big]
$$

and both parts are reported. "Off-plane RMS" in older tables is per coordinate; the distance
from the plane is $\sqrt{62}$ times it.

### Experiments

Five arms (brief), seed 42, 75% clean, one $\sigma$ per job (`sweep.sbatch`, `DIM=64,SIGMA=<σ>`):

| arm | data | training |
|---|---|---|
| 1 | all clean | iMF (run 60038059, shared by every $\sigma$) |
| 2 | Gaussian noise on 25% | iMF, noisy points as clean |
| 3 | Gaussian noise on 25% | ambient iMF |
| 4 | uniform noise on 25% | ambient iMF |
| 5 | uniform noise on 25% | iMF, noisy points as clean |

- **Fixed-σ sweep**, $\sigma \in \{0.05, 0.1, 0.2, 0.33\}$: jobs 60187230, 60187234, 60187287,
  60187288 (1 h 35 min – 1 h 53 min, peak 4.1–4.4 GB); re-scored in 64-D by 60204483
  (`more-steps.sbatch`).
- **Strong-noise sweep** (human, 2026-11-20), $\sigma \in \{0.25, 0.5, 0.75, 1.0\}$: jobs
  60207440, 60207441, 60207442, 60207443, all completed (1 h 43 min – 1 h 56 min, peak 4.2 GB).
  $\sigma = 1.0$ is the top: there ambient keeps 47% of noisy draws and their target's
  $\mathbb{E}\lVert\hat v\rVert^2$ is $4.4\times$ a clean point's (at $\sigma = 1.5$: 29%, $7\times$).
- **Re-score of the strong-noise sweep** at $k = 1, 2, 3, 4, 8$ in 64-D, job 60224909 (17 min 54 s,
  peak 0.76 GB; `more-steps.sbatch`, keys `<g|u><σ·100><n|a>`), submitted with
  `--mem=1200M --mail-type=BEGIN,END` over the header
  `--partition=gpu,gpu_quad --qos=gpuquad_qos --gres=gpu:1 --cpus-per-task=2 --time=00:30:00`.
- **Fresh-seed resample** of the $\sigma = 1.0$ runs (all four corrupted arms and clean iMF), job
  60226916 (22 min 43 s, peak 0.84 GB; `resample.sbatch`, sampling seeds 42, 1, 2, 3, 4 at
  $k = 1, 2, 8$), submitted with `--mail-type=BEGIN,END` over the header
  `--partition=gpu,gpu_quad --qos=gpuquad_qos --gres=gpu:1 --cpus-per-task=2 --mem=1200M --time=00:30:00`.
- **Localize the one-step break**: ambient iMF, Gaussian, $\sigma = 1.0$ retrained with
  evaluation every 2k steps (otherwise the sweep's config; evaluation does not touch the training
  RNG), job 60233034 (2 h 00 min, peak 1.16 GB; `localize.sbatch`, `DIM=64,SIGMA=1.0,ARM=3,EVAL_EVERY=2000`), submitted with
  `--mail-type=BEGIN,END` over the header
  `--partition=gpu,gpu_quad --qos=gpuquad_qos --gres=gpu:1 --cpus-per-task=2 --mem=2G --time=02:30:00`.
- **Running: half learning rate**: the same arm retrained with lr $5 \times 10^{-5}$ and evaluation
  every 2k steps, job 60246388 (`localize.sbatch`, `OVERRIDES="train.eval_every=2000 train.lr=0.00005"`),
  submitted with `--mail-type=BEGIN,END` over the header
  `--partition=gpu,gpu_quad --qos=gpuquad_qos --gres=gpu:1 --cpus-per-task=2 --mem=1800M --time=02:30:00`.
- **Results page** for both sweeps (human, 2026-11-20): artifact "Ambient iMF Noise Sweep",
  figures from `figures.py` (all eight $\sigma$, both noise types, 64-D and in-plane $W_2$).
- **Tests** of the 64-D metric: GPU suite 60207586 (65/67; the new split test had a bug in its
  generator use, fixed and passing in CPU job 60211648, 59/59; `test_learns_the_checkerboard_in_one_step`
  has failed since 2026-11-10, 59185847, unrelated to this direction's changes).

Current `sweep.sbatch` header (the fixed-σ jobs ran with `--mem=6G`):

```
#SBATCH --partition=gpu,gpu_quad
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=7G
#SBATCH --time=02:30:00
```

### Standing decisions by the human

- 2026-11-20: $W_2$ in all 64 dimensions is the metric for everything; no further seeds for the
  fixed-σ sweep; strong-noise sweep up to $\sigma = 1.0$; math in the note renders as equations.
- 2026-11-19: one noise level per run, 75% clean ($\sigma^2 \sim \mathcal{U}[0,1]$ per point was too
  strong: median $\sigma$ 0.71 against an arm gap of 0.67; sweep 60185808 cancelled). $D = 64$.
  All figures are matplotlib (`figures.py`, run in the agent session).
- 2026-11-18: 100k steps; seed 42 only, so differences are single-seed signals, not results.
  Observe the spiral in $D$ dimensions (Pixel MeanFlow toy, arXiv 2601.22158 Sec. 5).
- 2026-11-11: exact iMF, so the network does not see $t$.
- 2026-11-10: plain squared error; spiral keeps its 0.1 radial jitter; `research/baseline.md`
  set aside; job emails at start and end; `--mem` ≈ 1.5× measured peak.

## Results

**Clean iMF by dimension $D$**, 100k steps, seed 42 (60027106, 60038059; re-scored 60033355).
In-plane $W_2$ at $k$ steps and per-coordinate off-plane RMS (older metric):

| $D$ | $k=1$ | $k=2$ | $k=4$ | $k=8$ | off-plane RMS, $k=8$ |
|---|---|---|---|---|---|
| 2 | 0.724 | 0.341 | 0.218 | 0.244 | – |
| 8 | 0.519 | 0.264 | 0.190 | 0.191 | 0.094 |
| 16 | 0.389 | 0.219 | 0.166 | 0.147 | 0.077 |
| 32 | 0.293 | 0.172 | 0.131 | 0.107 | 0.059 |
| 64 | 0.236 | 0.155 | 0.121 | 0.100 | 0.045 |

Larger $D$ helps up to 64; all runs have plateaued ($W_2$ at $k=1$ changes < 3% from 90k to 100k).
At $D = 64$, 4 steps reach the in-plane floor, but 1 step (0.236) does not.

**Fixed-σ sweep, 75% clean**, $D = 64$, 100k steps, seed 42 (60187230/34/87/88; all arms
re-scored by 60204483). $W_2^{64\text{D}}$ at $k = 1\,/\,2\,/\,8$ steps; clean iMF:
1.148 / 0.623 / 0.372.

| $\sigma$ | 2 naive, Gaussian | 3 ambient, Gaussian | 4 ambient, uniform | 5 naive, uniform |
|---|---|---|---|---|
| 0.05 | 1.104 / 0.633 / 0.394 | 1.126 / 0.638 / 0.384 | 1.139 / 0.641 / 0.380 | 1.161 / 0.660 / 0.388 |
| 0.1 | 1.039 / 0.592 / 0.416 | 1.041 / 0.585 / 0.386 | 1.104 / 0.603 / 0.388 | 1.048 / 0.587 / 0.412 |
| 0.2 | 1.046 / 0.617 / 0.464 | 1.005 / 0.597 / 0.392 | 0.997 / 0.564 / 0.388 | 1.075 / 0.650 / 0.457 |
| 0.33 | 0.996 / 0.693 / 0.743 | 0.837 / 0.570 / 0.514 | 0.743 / 0.545 / 0.494 | 0.984 / 0.667 / 0.719 |

From 4 steps on, ambient iMF is below naive iMF in every cell, for both noise types, and the gap
grows with $\sigma$; ambient stays within 0.02 of clean iMF up to $\sigma = 0.2$. The gain is off
the plane: at $\sigma = 0.33$ naive iMF puts 5% of its 8-step samples more than 2 from the plane
(the noisy training points sit 2.6 off it), ambient 1%. In the plane ambient is equal or slightly
worse ($k = 8$, $\sigma = 0.33$: 0.164 against 0.140), plausibly the cost of its noisier target
($\mathbb{E}\lVert\hat v\rVert^2$ 102 for noisy points against 66 for clean at $\sigma = 0.33$).
One step is far from clean for every arm, clean included ($W_2^{64\text{D}} \approx 1.0$–$1.15$,
almost all off the plane). Seed 42 only: signals, not results.

**Strong-noise sweep, 75% clean**, $D = 64$, 100k steps, seed 42 (60207440–43; re-scored by
60224909). $W_2^{64\text{D}}$ at $k = 1\,/\,2\,/\,8$ steps; clean iMF: 1.148 / 0.623 / 0.372.

| $\sigma$ | 2 naive, Gaussian | 3 ambient, Gaussian | 4 ambient, uniform | 5 naive, uniform |
|---|---|---|---|---|
| 0.25 | 1.036 / 0.642 / 0.521 | 1.028 / 0.613 / 0.408 | 0.970 / 0.555 / 0.388 | 1.064 / 0.670 / 0.523 |
| 0.5 | 1.054 / 1.151 / 1.300 | 0.991 / 0.865 / 0.929 | 0.924 / 0.848 / 0.920 | 1.137 / 1.141 / 1.282 |
| 0.75 | 1.638 / 2.198 / 2.156 | 1.122 / 1.123 / 1.469 | 1.065 / 1.130 / 1.374 | 1.714 / 2.108 / 2.114 |
| 1.0 | 2.551 / 3.179 / 3.072 | 5.507 / 1.332 / 2.030 | 1.349 / 1.493 / 1.919 | 2.613 / 3.161 / 3.000 |

Across both sweeps ($\sigma$ from 0.05 to 1.0), ambient iMF is below naive iMF at 2 and 8 steps in
every cell from $\sigma = 0.2$ on, for both noise types, and the gap grows with $\sigma$; below 0.2 the
two are within 0.03. All of the gain is off the plane: in-plane $W_2$ at 8 steps is within about 0.02
between ambient and naive at every $\sigma \le 0.75$, and at $\sigma = 1.0$ ambient Gaussian is worse in
the plane (0.415 against 0.326). From $\sigma = 0.5$ on no arm recovers the spiral (plane samples are
a blob at every step count), so the 64-D differences there measure distance from the plane, not
shape. At high $\sigma$ ambient iMF gets worse with more steps ($\sigma = 1.0$, Gaussian: 1.332 at
2 steps, 2.030 at 8). Ambient Gaussian at $\sigma = 1.0$, 1 step, is anomalous: 5.507 at 100k after
0.94–1.29 at every evaluation from 30k to 90k, with a normal 2-step value. It is a property of the
final checkpoint, not of one sampling draw: with five sampling seeds (60226916) its 1-step $W_2^{64\text{D}}$
is 5.47–5.54 and 96% of its 1-step samples lie more than 2 from the plane (naive iMF: 22–26%;
clean iMF: 2%). Retrained with evaluation every 2k steps (60233034, same GPU type, Tesla V100S),
the run reproduces the original evaluations exactly (3.499 at 10k, 0.936 at 30k) and the break: 1-step
$W_2^{64\text{D}}$ drifts slowly up from 0.91 at 32k to 1.29 at 90k, then rises at every evaluation:
1.65 (92k), 2.73 (94k), 4.50 (96k), 5.84 (98k), 5.507 (100k, identical to the sweep run), with off-plane RMS 0.155 → 0.735. The 2-step value
jumps too (1.30 → 2.32 at 94k) and then recovers to 1.38. It is a progressive drift of the one-step map
over about 8k steps, not a single bad update. The logged training loss (one batch every 1k steps) has
its largest value of the run at 72k (130, against 50–70 typical) and more values above 85 after 70k,
but these are single-batch samples, so they do not locate the cause. Across the same five seeds the other $\sigma = 1.0$ runs vary by up to 0.15 at
2 and 8 steps, which bounds how far single-draw differences at this $\sigma$ can be read. Seed 42
training only: signals, not results.

**Five arms, 33% clean, $\sigma^2 \sim \mathcal{U}[0,1]$**, $D = 64$, 100k steps, seed 42
(60098738; re-scored 60108782). In-plane $W_2$ and per-coordinate off-plane RMS (older metric):

| arm | $k=1$ | $k=2$ | $k=8$ | off-plane RMS, $k=8$ |
|---|---|---|---|---|
| 1 iMF, clean | 0.243 | 0.160 | 0.104 | 0.047 |
| 2 iMF, Gaussian | 0.334 | 0.297 | 0.299 | 0.379 |
| 3 ambient iMF, Gaussian | 0.358 | 0.256 | 0.251 | 0.302 |
| 4 ambient iMF, uniform | 0.353 | 0.249 | 0.256 | 0.318 |
| 5 iMF, uniform | 0.356 | 0.307 | 0.292 | 0.390 |

No corrupted arm learns the spiral: samples are a blob with no arms at every step count. The
ambient arms' lower $W_2$ from 2 steps on is a tighter blob nearer the plane, not a recovered
spiral. The noise was too strong, which the fixed-σ sweeps correct. Evaluation differs across
GPU nodes by up to 0.012 in $W_2$ (not bitwise reproducible across nodes).

## Next step

When 60246388 finishes, compare its 1-step $W_2^{64\text{D}}$ curve with 60233034's: if the drift is gone or delayed, the late break is step-size driven; gradient clipping or a decaying schedule would need a change to `src/training.py` and goes to the human first.

## Blocked on

**One-step quality of exact iMF on the toy (2026-11-18).** Even clean iMF is far from clean data
in one step at every $D$ tried. Options, each a departure from exact iMF for the human to rule
on: $x$-prediction as in Pixel MeanFlow; a larger network or more steps; giving the network $t$.

## Dead ends

- MeanFlow with adaptive weighting ($p = 1$) on the checkerboard degrades with training (59157265).
- Exact iMF on the 2-D spiral: in-plane $W_2 \approx 0.72$ at 1 step for every lr and EMA tried (59281303); the network cannot read $t$.
- Giving the network $t$ fixes the 2-D toy ($W_2$ 0.113 at 1 step, 59285877) but is not exact iMF (ruled out 2026-11-11).
- $D = 512$: nothing learned in the plane at 50k steps (59987798), as Pixel MeanFlow reports for $u$-prediction.
- $\sigma^2 \sim \mathcal{U}[0,1]$ per noisy point: too strong to leave any structure (60098738).
