#!/usr/bin/env bash
# Golden outputs of the official JAX iMF objective (in a separate JAX venv), then the
# torch tests that compare our port with them and with the official torch sampler.
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=512M
#SBATCH --time=00:45:00
#SBATCH --output=/n/scratch/users/e/edk005/nrfm/slurm/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
NRFM=$HOME/.conda/envs/nrfm/bin
VENV=/n/scratch/users/e/edk005/nrfm/jax-ref
[[ -x $VENV/bin/python ]] || { $NRFM/python -m venv $VENV && $VENV/bin/pip install -q -r tests/reference/requirements-jax.txt; }
$VENV/bin/pip freeze | grep -iE "^(jax|jaxlib|flax|numpy)=="
$VENV/bin/python tests/reference/make_imf_reference.py
$NRFM/python -m pytest -v tests/test_imf_reference.py tests/test_imf.py tests/test_models.py
PYTHONPATH=. $NRFM/python research/experiments/ambient-flowmap/imf_mutants.py 2>&1 | grep -E "^(caught|MISSED)"
