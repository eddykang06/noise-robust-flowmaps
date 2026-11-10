---
slug: ambient-flowmap
status: blocked
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
```

Redirections and events:
- 2026-11-09: the brief's slug changed from `example` to `ambient-flowmap`.
  The question and mechanism are unchanged.
- 2026-11-09: the host's first launch failed because `origin` used HTTPS and
  the batch job had no credentials. The human approved switching `origin` to
  SSH, and the branch now tracks `origin/claude/ambient-flowmap`. The session
  started at 15:38.

## Results

None: no runs yet.

## Next step

Once unblocked, implement MeanFlow and the ambient target on the checkerboard
toy. Then submit a single 1-GPU pilot covering σ_n = 0.3, p = 0.1, seed 42, for
all four arms. Its purpose is to check that clean MeanFlow reaches a low energy
distance in 1 step before running the full grid.

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
