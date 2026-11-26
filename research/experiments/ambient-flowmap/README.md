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
- **Half learning rate**: the same arm retrained with lr $5 \times 10^{-5}$ and evaluation
  every 2k steps, job 60246613 (1 h 58 min, peak 1.11 GB; `localize.sbatch`,
  `OVERRIDES="train.eval_every=2000 train.lr=0.00005"`), submitted with
  `--mail-type=BEGIN,END --exclude=compute-g-17-168` over the header
  `--partition=gpu,gpu_quad --qos=gpuquad_qos --gres=gpu:1 --cpus-per-task=2 --mem=1800M --time=02:30:00`.
  The first submission, 60246388, failed after 22 s with an uncorrectable ECC error on the GPU of
  compute-g-17-168 (hardware), hence the exclusion.
- **Naive control**: naive iMF, Gaussian, $\sigma = 1.0$ (arm 2) retrained with evaluation every 2k
  steps at the sweep's lr, job 60256817 (1 h 45 min, peak 1.04 GB, Quadro RTX 8000; `localize.sbatch`,
  `ARM=2`, `OVERRIDES="train.eval_every=2000"`), submitted with `--mail-type=BEGIN,END --exclude=compute-g-17-168`
  over the header
  `--partition=gpu,gpu_quad --qos=gpuquad_qos --gres=gpu:1 --cpus-per-task=2 --mem=1700M --time=02:30:00`.
- **Larger batch**: ambient iMF, Gaussian, $\sigma = 1.0$ at batch 4096 (full lr, same 100k
  updates), job 60264427 (2 h 23 min, peak 1.19 GB; `localize.sbatch`, `ARM=3`,
  `OVERRIDES="train.eval_every=2000 train.batch_size=4096"`), submitted with
  `--mail-type=BEGIN,END --exclude=compute-g-17-168` over the header
  `--partition=gpu,gpu_quad --qos=gpuquad_qos --gres=gpu:1 --cpus-per-task=2 --mem=1700M --time=04:00:00`.
- **More training points**: ambient iMF, Gaussian, $\sigma = 1.0$ with $n = 100{,}000$
  training points (10×, same 75% clean; batch 1024, full lr), job 60274565 (1 h 57 min, peak 1.06 GB,
  Quadro RTX 8000; `localize.sbatch`, `ARM=3`, `OVERRIDES="train.eval_every=2000 data.n=100000"`),
  submitted with `--mail-type=BEGIN,END --exclude=compute-g-17-168` over the header
  `--partition=gpu,gpu_quad --qos=gpuquad_qos --gres=gpu:1 --cpus-per-task=2 --mem=1800M --time=03:00:00`.
- **Matched controls at $n = 100{,}000$**: clean iMF (arm 1) and naive iMF, Gaussian, $\sigma = 1.0$
  (arm 2), same overrides, jobs 60291354 (2 h 02 min, peak 1.10 GB, Quadro RTX 8000) and 60291357
  (1 h 52 min, peak 1.09 GB, Tesla V100S) (`localize.sbatch`), submitted with
  `--mail-type=BEGIN,END --exclude=compute-g-17-168` over the header
  `--partition=gpu,gpu_quad --qos=gpuquad_qos --gres=gpu:1 --cpus-per-task=2 --mem=1600M --time=03:00:00`.
- **Re-score at $n = 100{,}000$** of those three runs at $k = 1, 2, 3, 4, 8$, job 60311981
  (2 min 44 s, peak 0.76 GB; `more-steps.sbatch`, keys `c100k`, `g100n100k`, `g100a100k`), submitted with
  `--mem=1200M --mail-type=BEGIN,END --exclude=compute-g-17-168` over the header
  `--partition=gpu,gpu_quad --qos=gpuquad_qos --gres=gpu:1 --cpus-per-task=2 --time=00:30:00`.
- **Uniform arms at $n = 100{,}000$**: ambient iMF (arm 4) and naive iMF (arm 5), uniform noise,
  $\sigma = 1.0$, same overrides, jobs 60313028 (1 h 51 min, peak 1.25 GB, Quadro RTX 8000) and
  60313029 (2 h 16 min, peak 1.12 GB, Tesla V100S) (`localize.sbatch`), submitted with
  `--mail-type=BEGIN,END --exclude=compute-g-17-168` over the header
  `--partition=gpu,gpu_quad --qos=gpuquad_qos --gres=gpu:1 --cpus-per-task=2 --mem=1600M --time=03:00:00`.
- **Re-score of all five $n = 100{,}000$ arms** on one node at $k = 1, 2, 3, 4, 8$, job 60321685 (5 min 04 s, peak 0.76 GB)
  (`more-steps.sbatch`, keys `c100k`, `g100n100k`, `g100a100k`, `u100a100k`, `u100n100k`), submitted with
  `--mem=1200M --mail-type=BEGIN,END --exclude=compute-g-17-168` over the header
  `--partition=gpu,gpu_quad --qos=gpuquad_qos --gres=gpu:1 --cpus-per-task=2 --time=00:30:00`.
- **Sweep at $n = 100{,}000$** (human, 2026-11-23), $\sigma \in \{0.05, 0.1, 0.2, 0.25, 0.33, 0.5, 0.75\}$,
  arms 2–5, one arm per job (arm 1 and $\sigma = 1.0$ are the runs above): jobs 60555244–60555274
  (28 jobs, named `af-n100k-s<σ>-a<arm>`), `localize.sbatch` with `OVERRIDES="data.n=100000"`
  (evaluation every 10k; evaluation does not touch training), submitted with
  `--mem=1900M --mail-type=BEGIN,END --exclude=compute-g-17-168` over the header
  `--partition=gpu,gpu_quad --qos=gpuquad_qos --gres=gpu:1 --cpus-per-task=2 --time=03:00:00`.
  All 28 completed (1 h 14 min – 1 h 53 min).
- **Re-score of all 33 runs at $n = 100{,}000$** on one node at $k = 1, 2, 3, 4, 8$, job 60627029
  (`more-steps.sbatch`, keys `c100k` and `<g|u><σ·100><n|a>100k`), submitted with
  `--mem=1200M --mail-type=BEGIN,END --exclude=compute-g-17-168` over the header
  `--partition=gpu,gpu_quad --qos=gpuquad_qos --gres=gpu:1 --cpus-per-task=2 --time=00:50:00`.
  Completed in 28 min (duplicate 60636214 on gpu_requeue cancelled when it started).
- **Consolidated main** (human, 2026-11-24): this branch's shared code moved onto main as 3530c9d
  (pushed; worktree `/n/scratch/users/e/edk005/nrfm/main-consolidate`). CPU check 60632190
  (`consolidate-check-cpu.sbatch`; short partition, 4 CPUs, `--mem=1500M --time=00:20:00`,
  `--mail-type=BEGIN,END`): unit fingerprint and all nine cases (run ids, training sets,
  references) bitwise identical to b9ea442, run ids equal the existing $n = 100{,}000$ runs', CPU tests
  58/58. GPU check 60628326 (`consolidate-check.sbatch`, 50-step training per case, re-score, GPU arm
  tests; `--time=00:30:00`, otherwise the header above) ran as its gpu_requeue duplicate 60636197
  (60628326 cancelled; Quadro RTX 6000, 12 min) against main c7ddd3e: unit fingerprints, all nine 50-step
  training runs (weights, EMA, metrics) and the re-score bitwise identical to b9ea442; GPU tests 66 + 6 pass.
  60626861 cancelled as larger than needed.
- **Real-data pipeline on main** (human, 2026-11-24): MNIST, ImageNet 256 latents, FID-type metrics
  (c7ddd3e, pushed). CPU check 60639921 (`consolidate-check-cpu.sbatch`, OLD = main 0c1b13f,
  `OLD_MODE=new`): spiral bitwise identical, CPU tests 65/65. CPU smoke 60641517
  (`data-smoke-cpu.sbatch`, `data_smoke_cpu.py`; short partition, 4 CPUs, `--mem=6G
  --time=01:00:00`, `--mail-type=BEGIN,END`; 10 min 49 s, peak 1.80 GB): MNIST classifier 98.87%;
  MNIST reference vs itself FID 0 / P = R = 1, vs train digits FID 18.6, vs noise FID 2,157;
  fake ImageNet VAE round trip PSNR 39.7 dB, Inception FID 0 vs itself, 563 vs noise; CPU tests 66/66.
  GPU smoke 60639945 (`data-smoke.sbatch`, Quadro RTX 6000, 24 GB) failed: MNIST iMF (imfDiT_B_2,
  patch 2, 256 tokens, batch 256, fp32) ran out of GPU memory in the first step. Fix on main's
  worktree, not yet pushed: `configs/mnist.yaml` patch size 4 (64 tokens; 8c8de02). Resubmitted as
  60704833 with `--mem=3G` (1.5× the CPU smoke peak), otherwise the header of 60639945
  (`--partition=gpu,gpu_quad --qos=gpuquad_qos --gres=gpu:1 --cpus-per-task=4 --time=01:30:00`):
  out of memory again, on a 44 GB GPU at patch 4. Batch 256 of B/2 (12 blocks plus 8 auxiliary-head
  blocks, activations held through the JVP) does not fit one GPU; the iMF paper splits that batch
  across devices. Memory probe 60706992 (`mem-probe.sbatch`, `mem_probe.py`; one fp32 step at batch
  32–256, MNIST patch 2 and 4, ImageNet patch 2; `--cpus-per-task=2 --mem=3G --time=00:20:00`,
  otherwise the header above; Quadro RTX 8000, 44 GiB, 1 min): peak GPU memory per step, MNIST patch 4 at batch
  32 / 64 / 128 = 9.1 / 17.2 / 33.9 GiB (256 out of memory; patch 2 out of memory from 128), ImageNet
  patch 2 at batch 16 / 32 = 15.0 / 29.6 GiB (64 out of memory). Main's configs set to MNIST patch 4,
  batch 64 (fits 24 GB) and ImageNet batch 32 (needs 40 GB or more); 6147a1b, pushed to main after GPU
  smoke 60707401 (header of 60704833; Quadro RTX 8000, 1 h 14 min, peak 1.54 GB) passed: MNIST 2,000
  steps and fake ImageNet 50 steps train and score, tests 77/77. MNIST runs at about 1.2 s per step at
  batch 64 (B/2), so the config's 50,000 steps would take about 17 h. Its 2,000-step EMA samples are
  noise (FID 2,701, accuracy 10%): EMA 0.9999 still holds 82% of the initial weights at 2,000 steps, so
  the smoke cannot show learning. Not needed (human, 2026-11-25: only the data and training pipelines
  have to work for now, not MNIST quality), so raw-weight check 60712569 was cancelled.
- **Class-conditional generation on main** (human, 2026-11-26), as iMF does it: a34ba56, pushed to
  main after the CPU check and before the GPU check (human: push now, confirm later). The human chose:
  - guided ambient iMF with target v_hat + (1 − 1/ω)(v_c − v_u);
  - spiral arms as classes 0 and 1, with an `arm_accuracy` metric;
  - `sampler.omega` as a list, each scale scored under `<metric>@<steps>_w<omega>`;
  - reference labels at evaluation, class-balanced for ImageNet.

  Conditional sampling follows `objective.guidance`, and `scripts.sample` takes `--omega` and draws
  balanced labels. `configs/ambient_imf.yaml` gains `cfg_beta`, `class_dropout_prob` and
  `guidance: false`, which changes ambient run ids only. The checks compare against main 6147a1b
  (worktree `main-before-cond`):
  - CPU check 60841856 (`consolidate-check-cpu.sbatch` with `OLD=main-before-cond OLD_MODE=new`,
    header as before; `diff_cases.py` lists differing fields): unit fingerprints identical, all nine
    training sets and references identical, and only the six ambient run-id fields differ; CPU tests
    75/75. The first run, 60841574, had three failing tests, all test bugs: two old ambient tests
    lacked the new arguments, and one spiral test assumed the wrong arm sign.
  - GPU check 60842135 (`consolidate-check.sbatch`, same OLD, `TESTS=conditional_run`;
    `--time=00:15:00`, otherwise the earlier header). The human asked for short checks that queue
    quickly, so it covers only what the CPU cannot: the nine 50-step training cases, bitwise
    against OLD, and conditional iMF and ambient iMF trained for 4,000 steps on the spiral. Those
    two must reach arm accuracy > 0.9 at 1 and 2 steps, at ω = 1 and 2, and land within half the
    prior's W2. Earlier attempts 60841575 (whole suite) and 60842116 were cancelled while queued.
- **ImageNet, set up as iMF does** (human, 2026-11-26: "use the standard iMF setup"; noise on the
  normalized latents, the precomputed cache). The official iMF code (`configs/default.py`,
  `utils/vae_util.py`) uses SD-VAE ft-MSE, stores the posterior mean and std, samples a latent each
  step, normalizes per channel ((z - m)/s, s = 3.9-5.3) and uses no flips; main had ft-EMA,
  mean-only latents x 0.18215 and flips. Main's worktree `main-parquet`, commit c6539c5 (local, not
  pushed) now:
  - takes posteriors from ChocolateDave/imagenet-256-sd-vae-latents (Flax ft-MSE, MeanFlow's ADM
    crop, unscaled mean and std; complete for 1,281,167 train and 50,000 validation rows);
  - stores normalized mean `x` and std `x_std`, and `Batch.resample` draws x + x_std·eps each step;
  - adds corruption once to the normalized mean (equal in law to noising each fresh latent).

  The ambient objective's docstring now cites Consistent Diffusion Meets Tweedie (Daras et al.,
  ICML 2024) and says its consistency loss is not used. Our own Parquet encoding path (0326d2d) is
  removed. The first ImageNet download (raw images, 60844562: 294 train and 14 validation shards,
  143 GiB, 6 min 44 s) is kept for verification.
  - Cache download: 60845969 (`download-latents.sbatch`; short partition, 4 CPUs, `--mem=2G
    --time=01:00:00`).
  - CPU check 60846014 (`consolidate-check-cpu.sbatch`, OLD = a34ba56, NEW = `main-parquet`):
    unit fingerprints and all nine spiral cases identical, 0 differing fields; CPU tests 79/79.
  - Cache download 60845969 done in 3 min 28 s (294 + 14 shards, 39 GiB). Verification and assembly 60846066 (`verify-latent-cache.sbatch`, `verify_latent_cache.py`;
    short partition, 4 CPUs, `--mem=4G --time=02:00:00`). It compares 16 cached posteriors with
    our PyTorch ft-MSE encoding and checks iMF's channel statistics. Then the real-data GPU check
    (`imagenet-run-check.sbatch`; `--mem=3G --time=00:10:00`, 1 sampling step).
  - The obsolete ft-EMA shard check 60845032 was cancelled while queued.
- **Results page** for both sweeps (human, 2026-11-20): artifact "Ambient iMF Noise Sweep",
  figures from `figures.py` (all eight $\sigma$, both noise types, 64-D and in-plane $W_2$; since
  2026-11-20 also $\sigma = 1.0$ at $n = 10{,}000$ against $n = 100{,}000$, `fig_data_size`).
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

- 2026-11-24: build the real-data loading pipeline on main (MNIST, ImageNet 256 as SD-VAE
  latents, AFDB later); keep all labels; FID-type metrics plus baseline metrics; protein
  settings to follow Ambient Protein Diffusion, decided by the human.
- 2026-11-24: consolidate the code on main: spiral only (checkerboard removed), current
  pipeline included, behaviour verified unchanged; push straight to main.
- 2026-11-23: the sweep moves to $n = 100{,}000$ training points (all eight $\sigma$); a single
  seed (42) for now, no triplicates; speed of iteration is the priority. Results page for the
  $n = 100{,}000$, $\sigma = 1.0$ runs: artifact "Ambient iMF at 100k Points".
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
but these are single-batch samples, so they do not locate the cause. Three retrains of this arm with evaluation every 2k
steps separate the break from a slower drift (1-step $W_2^{64\text{D}}$ / per-coordinate off-plane RMS):

| run | lowest, step | 80k | 100k | 2-step, 100k |
|---|---|---|---|---|
| $n = 10{,}000$, batch 1024, lr $10^{-4}$ (60233034, V100S) | 0.91, 32k | 1.20 / 0.146 | 5.51 / 0.693 | 1.33 |
| half lr (60246613, V100S) | 0.75, 50k | 1.06 / 0.131 | 1.30 / 0.161 | 1.49 |
| batch 4096 (60264427, V100S) | 1.30, 24k | 1.54 / 0.190 | 1.60 / 0.198 | 2.02 |
| $n = 100{,}000$ (60274565, RTX 8000) | 0.79, 40k | 1.13 / 0.140 | 1.15 / 0.142 | 0.66 |

The break needs the small training set and the full step: half the lr, 4× the batch or 10× the
points each remove it. The slow drift is not ambient-specific: clean iMF (60038059, no noise) has the
same rise of the 1-step off-plane RMS, 0.093 at 40k to 0.144 at 100k, while its in-plane 1-step
$W_2$ keeps improving (0.299 → 0.236); it is part of exact iMF's weak one-step map (Blocked on). With
10× the points, ambient iMF follows clean iMF's 1-step curve (off-plane 0.092 → 0.142, in-plane
0.303 → 0.246) and its drift stops at 80k; at 100k it scores 1.146 at 1 step and 0.661 at 2 steps,
against clean iMF's 1.148 and 0.623 (at $n = 10{,}000$, another GPU type) and naive iMF's 2.53 and 3.15
(60256817, $n = 10{,}000$). Naive iMF's 1-step value falls through the whole run (4.43 at 30k, 2.53 at
100k). The larger batch is worse than batch 1024 throughout after 24k.

**Fixed-σ sweep at $n = 100{,}000$**, 75% clean, $D = 64$, 100k steps, seed 42 (60555244–74 and the
$\sigma = 1.0$ runs below; all 33 re-scored on one node by 60627029). $W_2^{64\text{D}}$ at $k = 1\,/\,2\,/\,8$
steps; clean iMF: 1.166 / 0.639 / 0.374.

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

At 8 steps ambient iMF is within 0.035 of clean iMF at every $\sigma$ for both noise types (0.378–0.406
against 0.374), while standard iMF degrades steadily from $\sigma = 0.1$ (Gaussian 0.417 at 0.1, 0.880 at 0.5,
3.657 at 1.0; uniform similar). At 2 steps the two are within 0.07 up to $\sigma = 0.33$ except Gaussian 0.2 (standard 0.782, ambient 0.603), and part from
$\sigma = 0.5$. At 1 step neither is consistently better below $\sigma = 0.75$ (differences up to 0.2 either
way, e.g. ambient worse at $\sigma = 0.05$–$0.1$); all 1-step values sit near clean iMF's weak 1.17. In the plane
at 8 steps ambient stays at 0.10–0.15 at every $\sigma$; standard iMF rises from 0.11 to 0.46–0.50.
Seed 42 only: signals, not results.

**$\sigma = 1.0$, $n = 100{,}000$** (10× the sweep's training set, 75% clean), $D = 64$, 100k steps,
seed 42 (60291354, 60291357, 60274565, 60313028, 60313029; all re-scored on one node by 60321685).
$W_2^{64\text{D}}$ (in-plane $W_2$) at $k$ steps:

| arm | $k=1$ | $k=2$ | $k=4$ | $k=8$ |
|---|---|---|---|---|
| 1 iMF, clean | 1.166 (0.234) | 0.639 (0.160) | 0.398 (0.125) | 0.374 (0.107) |
| 2 iMF, Gaussian | 1.841 (0.390) | 2.508 (0.296) | 3.315 (0.443) | 3.657 (0.495) |
| 3 ambient iMF, Gaussian | 1.146 (0.246) | 0.661 (0.185) | 0.444 (0.169) | 0.406 (0.131) |
| 4 ambient iMF, uniform | 1.102 (0.250) | 0.601 (0.179) | 0.430 (0.154) | 0.405 (0.138) |
| 5 iMF, uniform | 2.339 (0.387) | 2.879 (0.311) | 3.416 (0.417) | 3.650 (0.461) |

With 10× the points, ambient iMF is within 0.05 of clean iMF in 64 dimensions at every step count,
for Gaussian and for uniform noise of the same variance (off-model for the ambient target), and
improves with more steps (at $n = 10{,}000$ it got worse: Gaussian 1.33 at 2 steps, 2.03 at 8). Naive
iMF gets worse with more steps (1.84 → 3.66 Gaussian, 2.34 → 3.65 uniform), as it learns the noisy
data. In the plane ambient trails clean by 0.01–0.04. Naive iMF improves through the whole run
(Gaussian: 2.50 at 20k), so it has not converged; ambient and clean have the same shape over
training (e.g. 0.787 and 0.774 at 40k). Seed 42 only: a signal, not a result. Across the same five seeds the other $\sigma = 1.0$ runs vary by up to 0.15 at
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

When GPU check 60842135 finishes, record it (fix or revert a34ba56 on main if it fails). Verify the posterior cache and assemble it; run the real-data GPU check; then push the iMF ImageNet setup to main.

## Blocked on

**Late one-step break of ambient iMF at $\sigma = 1.0$, $n = 10{,}000$ (2026-11-20).** Half the lr,
4× the batch or 10× the training points each remove it. Keeping the sweep's data and step size would
need gradient-norm clipping or a decaying lr in `build_optimizer` (`src/training.py`, which has
neither), a departure from the official iMF optimizer; a larger training set needs no code change.

**One-step quality of exact iMF on the toy (2026-11-18).** Even clean iMF is far from clean data
in one step at every $D$ tried. Options, each a departure from exact iMF for the human to rule
on: $x$-prediction as in Pixel MeanFlow; a larger network or more steps; giving the network $t$.

## Dead ends

- MeanFlow with adaptive weighting ($p = 1$) on the checkerboard degrades with training (59157265).
- Exact iMF on the 2-D spiral: in-plane $W_2 \approx 0.72$ at 1 step for every lr and EMA tried (59281303); the network cannot read $t$.
- Giving the network $t$ fixes the 2-D toy ($W_2$ 0.113 at 1 step, 59285877) but is not exact iMF (ruled out 2026-11-11).
- $D = 512$: nothing learned in the plane at 50k steps (59987798), as Pixel MeanFlow reports for $u$-prediction.
- $\sigma^2 \sim \mathcal{U}[0,1]$ per noisy point: too strong to leave any structure (60098738).
