---
slug: ambient-flowmap
status: running
hypothesis: On a 2-D toy with a fraction of samples Gaussian-corrupted, MeanFlow trained with an ambient target gives 1-step samples closer to the clean distribution than MeanFlow on all data or on the clean subset alone, at matched compute.
kill_criterion: Ambient MeanFlow 1-step energy distance not below the better of flowmap-corrupt and clean-only MeanFlow by more than 2 SE (3 seeds) in at least 2 of 3 noise levels.
best_metric:
started: 2026-11-09
---

# ambient-flowmap

## Hypothesis

On a 2-D toy distribution in which a fraction 1−p of the training samples
carry additive Gaussian noise of known level σ_n, MeanFlow trained with an
ambient target produces 1-step samples with lower energy distance to the
clean distribution than MeanFlow trained on all samples as if clean
(`flowmap-corrupt`) and than MeanFlow trained on the clean subset alone, at
matched training forward passes.

## Kill criterion

Fixed before the first run. Setting: checkerboard toy, unit-variance data,
p = 0.1 clean, σ_n ∈ {0.1, 0.3, 1.0}, seeds 42–44, 1-step sampling, energy
distance on 10k samples against fresh clean samples. The direction is dead if,
in at least 2 of the 3 noise levels, mean ambient-MeanFlow energy distance is
not lower than the better of `flowmap-corrupt` and clean-only MeanFlow by more
than 2 standard errors (SE of the difference over the 3 seeds), at matched
forward passes.

## Current state

**Blocked before the first run: there is no code to run.** `scripts/train.py`,
`sample.py`, `eval.py`, `make_data.py`, `configs/*.yaml`, `src/toy.py`,
`src/corrupt.py`, `src/training.py` and `src/sampling.py` are empty stubs on
`origin/eddy`, and `research/baseline.md` has no protocol or overrides. No jobs
have been submitted.

Method design, fixed before any run. MeanFlow (Geng et al., 2025) uses the
linear path z_t = (1−t)x + tε and learns the average velocity u(z_t, r, t).
Its regression target is v_t − (t−r)(v_t·∂_z u_θ + ∂_t u_θ), with stop-grad,
where v_t = ε − x. Sampling is one step: x = ε − u_θ(ε, 0, 1). The ambient
version follows Daras et al.'s ambient diffusion for noisy data. A noisy
sample y = x + σ_n η sits at path time t_n = σ_n/(1+σ_n). For t > t_n we
noise it to z_t = (1−t)y + √(t² − (1−t)²σ_n²) ε′. With c = σ_n²(1−t)²/t²,
the substitute x̃ = (y − c z_t/(1−t))/(1−c) satisfies E[x̃ | z_t] = E[x | z_t].
So ṽ = (z_t − x̃)/t is an unbiased estimate of the marginal velocity. The
MeanFlow target is linear in v_t given z_t, because the JVP is linear in its
tangent, so the loss keeps its minimiser. Clean samples train at all (r, t).
Noisy samples train only at t > t_n + δ, where δ bounds the 1/(1−c) variance
blow-up. Because (t−r)u vanishes at t = r, intervals with r < t_n still need
velocity below t_n, which only the clean samples supply. That is why the clean
fraction p matters, and it is the first thing the results should show.

Comparisons: `flowmap-clean` (all samples clean; the upper bound),
`flowmap-corrupt` (all samples treated as clean) and clean-only MeanFlow (the
noisy samples discarded). Clean-only MeanFlow is not in `baseline.md`, but
without it a win over `flowmap-corrupt` cannot be credited to the ambient
target.

Shared pipeline (2026-11-09, at the human's request). The human directed
that the repo's structure be settled before any direction is implemented. The
shared code now lives on this branch: `src/` (config, Batch, toys, corruption,
MLP flow map, flow-matching and MeanFlow objectives, samplers, energy distance,
training loop), `scripts/` (train, eval, sample, check), `configs/toy.yaml`
and `jobs/train.sbatch`. The human approved creating conda env `nrfm` as a
clone of `big_things`. Check job, not yet verified:

```
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
CMD="python -m scripts.check"    job 59122018: queued 16:06-20:47 (group GPU limit), ran 16.5 min,
                                 12 checks passed; failed on a float32 comparison in the check itself
CMD="python -m scripts.check"    job 59153948: all 25 unit checks pass; end-to-end failed, V100
                                 unsupported by torch cu130 -> nrfm reinstalled with torch 2.14.1+cu126
CMD="python -m scripts.check"    job 59155309: found MeanFlow not learning; the MLP's DiT-style 1000t
                                 time embedding made the JVP target ~1000x data scale -> raw t inputs
CMD="python -m scripts.check"    job 59155890: 31/32 pass (bitwise reproducibility, bitwise resume,
                                 re-scoring, all arms, flow matching learns: ED@50 0.0036 vs 0.0146
                                 for N(0,I)). Fails "MeanFlow learns in 1 step": ED@1 0.0073 after
                                 2000 steps (threshold 0.3x prior = 0.0044), and ED@2 0.024 is worse
                                 than ED@1 and than N(0,I).
Diagnostic pilots, configs/toy.yaml (MeanFlow, clean checkerboard, 20k steps), seed 42:
CMD="python -m scripts.train --config configs/toy.yaml slug=ambient-flowmap seed=42"                     job 59156303
CMD="python -m scripts.train --config configs/toy.yaml slug=ambient-flowmap seed=42 objective.ratio=0.75" job 59156304
Both degrade with training: ED@1 0.0069 -> 0.0170 (ratio 0.25) and 0.0091 -> 0.0174 (ratio 0.75)
from 5k to 20k steps; N(0,I) scores 0.0146.
research/experiments/ambient-flowmap/diag.sbatch                                          job 59157265
ED at 20k steps (seed 42): flow matching 0.0005 (50 Euler, stable from 5k); MeanFlow lr 1e-4 p=1
ED@1 0.0069, ED@2 0.0182 (erratic); MeanFlow lr 1e-3 p=0 ED@1 0.0021, ED@2 0.0010 (stable);
MeanFlow lr 1e-4 p=0 ED@1 0.0047, ED@2 0.0042. The adaptive weighting (p = 1) causes the
degradation, so configs/toy.yaml now uses p = 0. One seed: a diagnosis, not a result.
CMD="python -m scripts.check"    job 59159091: **all 34 checks pass** (78 s). MeanFlow ED@1 0.0035 and
                                 flow matching ED@50 0.0036 after 2000 steps vs 0.0146 for N(0,I);
                                 bitwise reproducibility and resume hold. The 300-step arms only
                                 check that runs finish (EMA 0.999 still dominated by init).

**The shared pipeline is verified.**

2026-11-10, at the human's request: the pipeline now wraps the official improved
MeanFlow (iMF) code. The network is vendored unchanged, and the objective and sampler
are ported from the official JAX repo. The MeanFlow results below are superseded and
must be rerun with iMF. Ambient iMF (`src/objectives/ambient_imf.py`) replaces e - x
with an unbiased target for noisy samples above their noise level. CPU tests job
59177930: 21/21 ambient + iMF tests pass on this branch; 33/33 non-GPU tests pass in the
research-free main candidate (9140d1b). An earlier run (59177592) found float32
cancellation in the target at small t, fixed by an algebraically equal form.
At the human's instruction, `main` was fast-forwarded to 9140d1b (code only, no research
files) before the GPU tests ran. GPU job 59171059 queued ~5 h, then failed in 3 s on main
20623bc without running any test: its absence check used `ls`, which exits nonzero when
the files are absent, so `set -e` stopped the job because main was clean. Fixed in 2bbae24.
2026-11-10: at the human's request, history on main, eddy and this branch was rewritten
to drop Claude co-author trailers; trees, authors, dates and subjects are unchanged.
Commit IDs cited here and in `runs/ambient-flowmap.jsonl` before that date are the old
IDs; `sha-map.txt` maps old -> new (main 9140d1b -> 6f79037).
main now at 04df8a6 (models under src/models; 33/33 CPU tests, job 59180764).
iMF pilot, configs/toy.yaml (official lr 1e-4, EMA 0.9999, p = 1), clean, seed 42:
research/experiments/ambient-flowmap/pilot-imf.sbatch    job 59181941 (gpu, 2G, 3h)
2026-11-10, at the human's request: pilot cancelled after 7 s, before any record was
written. No experiments run until the setup and experimental framework are redesigned.
GPU test suite on main 20623bc (research-free checkout, asserts no research files), not an
experiment (gpu, V100, 4 CPU, 2G, 30 min):
research/experiments/ambient-flowmap/test-main.sbatch    job 59185847: resubmission of
                                 59171059; 38/39 pass in 13 min. Fails "learns the checkerboard in
                                 one step": after 4000 steps (lr 1e-3, EMA 0.99, official iMF
                                 objective with adaptive weight p = 1) ED@1 0.054 vs 0.0146 for
                                 N(0,I); ED@2 0.009. Same pattern as the p = 1 degradation found
                                 earlier with MeanFlow (job 59157265). main's default config does
                                 not yet give a working 1-step model on the toy; not fixed, since
                                 no runs happen before the framework is settled.
Full test suite on this branch with norm_p 0 (gpu, 4 CPU, 2G, 30 min, mail BEGIN,END):
research/experiments/ambient-flowmap/gpu-tests.sbatch    job 59190716: 45/46 pass in 13 min
                                 (peak RAM 1.0 GB). The one-step test fails worse than with
                                 p = 1: ED@1 0.157 and ED@2 0.028 after 4000 steps (lr 1e-3,
                                 EMA 0.99), vs 0.054 / 0.009 with p = 1 (59185847) and 0.0146
                                 for N(0,I). So p = 1 is not what breaks one-step iMF here, and
                                 p = 0 does not transfer from the old MeanFlow code. Cause unknown;
                                 the test's shortened lr/EMA/steps are the next suspect.
Same script, after the W2 change (mem 2G: 1.06 GB measured plus the W2 cost matrix and
plotting): job 59273327, running (tests 84f9e9d). Job 59274501, same script, tests the
per-run figure (e42283b).

First experiment (brief): MeanFlow (p = 0) on clean, corrupted (sigma 0.3, 10% clean, noisy
used as clean) and clean-only checkerboard data, seeds 42-44, 20k steps each:

```
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
research/experiments/ambient-flowmap/exp1.sbatch    job 59163437
```
```

Redirections and events:
- 2026-11-10 03:35: the brief was revised and is still being written. Mechanism:
  MeanFlow and ambient diffusion for now, as the lowest-hanging fruit, with more novel
  methods hoped for later. Synthetic setup: two target 2-D distributions, the two-armed
  spiral and the checkerboard, under several corruption schemes; the first axis is
  Gaussian vs non-Gaussian (e.g. masking) corruption. The second axis and the first
  experiment are blank in the brief, so nothing is acted on beyond recording it.
- 2026-11-10 ~21:10: the brief's first experiment: port ambient training directly to
  MeanFlow on the two-armed spiral. Gaussian case: 33% clean, the rest with noise of
  variance sigma^2 ~ U[0, 10] drawn per point (implemented as `gaussian_uniform_var`,
  07e0e52). Non-Gaussian case: 33% clean, the rest with additive uniform noise (sentence
  unfinished). Arms: (1) iMF on clean data; (2) iMF on Gaussian-corrupted data, no
  adjustment; (3) ambient iMF, Gaussian; (4) ambient iMF, non-Gaussian.
- 2026-11-10: the human asked for a true vs generated comparison in every toy run's
  output. e42283b: each toy run writes `true-vs-generated.png` to its run directory after
  the last step (training data, true samples, generated samples per step count, W2 over
  training with its floor); `scripts/plot.py` puts several runs in one figure.
- 2026-11-10: the human chose Wasserstein-2 as the only metric, with plots of every
  result. 84f9e9d: `src/metrics.py` computes exact W2 by linear assignment on 4096
  samples (O(n^3), so not 10k); energy distance is removed; `scripts/plot.py` draws each
  run's training data, clean reference, samples per step count and W2 over training
  against the floor of two independent clean samples. Earlier energy-distance numbers
  are not comparable with W2.
- 2026-11-10: the human set every synthetic experiment to 50k training steps for now
  (was 100k), to be revised as results come in. Applied to `configs/toy.yaml` and this
  direction's `corrupt.yaml` and `ambient.yaml`, which also now carry `norm_p: 0`
  (they had been left at 1.0).
- 2026-11-10: at the human's instruction, for compute-heavy runs (pretraining from
  scratch), this branch's `research/bin/submit` limits match learned-cascade's: 5 days
  per job (the gpu/gpu_quad partition limit, was 12 h) and 64 jobs per direction (was
  2); still 1 GPU and 8G RAM per job. The account's 12,000 GPU run-minute cap is
  unchanged.
- 2026-11-10: at the human's instruction, `research/CLAUDE.md` on eddy (7f3215e) now
  requires `--mem` at about 1.5x the last comparable job's measured peak RAM; this
  direction's test scripts were resized to match (35d0851).
- 2026-11-10: the human set the loss to plain squared error, `norm_p: 0` in
  `configs/toy.yaml` (fdce3d6), replacing the official iMF default 1.0 that the port had
  restored. Reasons: it works for MeanFlow on this toy (job 59157265) and keeps the
  ambient target unbiased; with p = 1 the minimiser is a loss-weighted mean. This is the
  one stated deviation from iMF's released code (the iMF paper writes Eq. (10) as plain
  L2 and inherits the weighting from MeanFlow, arXiv:2512.02012 App. A). Against it:
  MeanFlow's ImageNet ablation has p = 0 worst (arXiv:2505.13447, Table 1(e)). main
  still has 1.0.
- 2026-11-10: the human asked to remove `eddy` and base this branch directly on main,
  then withdrew the request after seeing what depends on `eddy`. The workflow is
  unchanged; nothing was modified.
- 2026-11-10: the human chose the toys for the framework: checkerboard and the two-armed
  spiral (`src/toy.py`). Rings, the 2-D Swiss roll and the circle embedded in d dimensions
  were considered and not chosen. Still open: whether the spiral keeps its 0.1 radial jitter.
- 2026-11-09: the brief's slug changed from `example` to `ambient-flowmap`.
  The question and mechanism are unchanged.
- 2026-11-09: the host's first launch failed because `origin` used HTTPS and
  the batch job had no credentials. The human approved switching `origin` to
  SSH, and the branch now tracks `origin/claude/ambient-flowmap`. The session
  started at 15:38.

## Results

First experiment (job 59163437): MeanFlow (p = 0), checkerboard, 20k steps, energy distance on
10k samples against fresh clean samples, mean ± SE over seeds 42-44. These are not baseline
results yet: `research/baseline.md` has no protocol.

| arm | ED@1 | ED@2 |
|---|---|---|
| clean | 0.00206 ± 0.00001 | 0.00118 ± 0.00015 |
| corrupt (sigma 0.3, 10% clean, noisy used as clean) | 0.00273 ± 0.00045 | 0.00230 ± 0.00014 |
| clean-only (~1k clean samples) | 0.00343 ± 0.00057 | 0.00333 ± 0.00097 |

At sigma = 0.3 the corruption clearly hurts 2-step samples (0.00230 vs 0.00118, about 6 SE),
but the 1-step gap is inside seed variance. Clean MeanFlow's own 1-step error (0.0021, about
5x the ED floor for two 10k samples) is about as large as the corruption effect, so 1-step ED
is a weak probe at this noise level. Clean-only is the worst arm on average but varies most.

## Next step

With the human, finish the experimental framework (toys chosen: checkerboard and
two-armed spiral; still open: spiral jitter, metrics, arms, budgets, evaluation) before
running any experiment.

## Blocked on

0. **GPU allowance.** Check jobs can wait hours with reason
   `AssocGrpGRESRunMinutes` (59122018 waited 16:06-20:47): the group's GPU run-minute limit is used
   up, apparently by the nine running `learned-cascade` jobs, which are not
   this direction's.

1. **Shared code.** The run needs:
   - `src/toy.py`: checkerboard and spiral samplers;
   - `src/corrupt.py`: adds Gaussian noise of level σ_n to a fraction 1−p of
     samples and returns a per-sample noise level (0 for clean);
   - `src/training.py`: the MeanFlow loss, with an option for the ambient
     target above;
   - `src/sampling.py`: 1-step and k-step flow-map sampling;
   - `scripts/train.py`, `sample.py` and `eval.py`, where `eval.py` computes
     energy distance against fresh clean samples;
   - a toy `configs/*.yaml`;
   - training code that appends to `research/runs/<slug>.jsonl`.

   Either you write these, or you authorize me to write them on this branch.
   The second option is faster, and it is what I would propose: about 300
   lines, with the MeanFlow loss and the ambient target in `src/training.py`.
2. **Baseline protocol.** `research/baseline.md` needs the overrides for
   `flowmap-clean` and `flowmap-corrupt`, the update or forward-pass budget,
   and the metric. I propose energy distance, 10k samples, 1-step and 2-step
   sampling, seeds 42–44, and adding clean-only MeanFlow as a third baseline
   row.

## Dead ends

None yet.
