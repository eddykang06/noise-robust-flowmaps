# The first experiment's five arms (brief, eddy 90d5ebd) as scripts.train arguments.
# Source it, then `arm_args <1-5>` sets ARGS.
D=research/experiments/ambient-flowmap
GAUSSIAN="corruption={name: gaussian_uniform_var, var_max: 1.0, clean_frac: 0.33}"
UNIFORM="corruption={name: box_uniform_var, var_max: 1.0, clean_frac: 0.33}"

arm_args() {
  case $1 in
    1) ARGS=(--config $D/spiral-imf.yaml) ;;                  # clean data
    2) ARGS=(--config $D/spiral-imf.yaml "$GAUSSIAN") ;;      # naive, Gaussian
    3) ARGS=(--config $D/spiral-ambient.yaml "$GAUSSIAN") ;;  # ambient, Gaussian
    4) ARGS=(--config $D/spiral-ambient.yaml "$UNIFORM") ;;   # ambient, uniform
    5) ARGS=(--config $D/spiral-imf.yaml "$UNIFORM") ;;       # naive, uniform
  esac
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
