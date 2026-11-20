# The first experiment's five arms (brief, eddy 90d5ebd) as scripts.train arguments, with
# the spiral observed in $DIM dimensions (src/embed.py).
# 75% clean, one noise level $SIGMA per run (human, 2026-11-19; the brief and job 60098738
# used 33% clean with sigma^2 ~ U[0, 1] per point). Source it, then `arm_args <1-5>` sets ARGS.
D=research/experiments/ambient-flowmap
: "${SIGMA:?set SIGMA}"
GAUSSIAN="corruption={name: gaussian, sigma: $SIGMA, clean_frac: 0.75}"
UNIFORM="corruption={name: uniform, sigma: $SIGMA, clean_frac: 0.75}"

arm_args() {
  case $1 in
    1) ARGS=(--config $D/spiral-imf.yaml) ;;                  # clean data
    2) ARGS=(--config $D/spiral-imf.yaml "$GAUSSIAN") ;;      # naive, Gaussian
    3) ARGS=(--config $D/spiral-ambient.yaml "$GAUSSIAN") ;;  # ambient, Gaussian
    4) ARGS=(--config $D/spiral-ambient.yaml "$UNIFORM") ;;   # ambient, uniform
    5) ARGS=(--config $D/spiral-imf.yaml "$UNIFORM") ;;       # naive, uniform
  esac
  ARGS+=(data.dim=$DIM model.dim=$DIM)
}

# Runs `<command> <arm>` for every arm in parallel on this node (one node, so appends to
# the shared runs file stay whole on NFS), logging to $LOGS/arm<arm>.log; fails if any arm fails.
all_arms() {
  local arm pids=() status=0
  for arm in 1 2 3 4 5; do
    "$1" $arm > "$LOGS/arm$arm.log" 2>&1 & pids+=($!)
  done
  for pid in "${pids[@]}"; do wait $pid || status=1; done
  return $status
}
