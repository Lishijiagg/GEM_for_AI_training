#!/bin/bash
#SBATCH --job-name=dfba
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# If your cluster requires a partition, add a line such as:  #SBATCH -p <partition>
#
# Stage 2 (dFBA). Run after stage 1 (run_random_media.sbatch) has finished.
# Submit with submit_dfba.sh, which sets the array size and the merge job:
#   bash submit_dfba.sh
# Array layout: task = (model index) x N_CHUNKS + (part index).
# With N_CHUNKS parts per model and M models there are M x N_CHUNKS tasks;
# task t runs model ((t-1) / N_CHUNKS) + 1, part ((t-1) % N_CHUNKS) + 1.

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

# ---- settings: edit these lines ------------------------------------------
N_CONC=10          # random concentration sets per growing substrate combination
CONC_MIN=0.01      # mM
CONC_MAX=10        # mM
X0=0.01            # initial biomass, gDW/L
T_END=24           # h
DT=0.01            # h, integration step
RECORD_DT=0.1      # h, interval of written time points
METHODS=fba,pfba
CARBON_CAP=60      # mmol C/gDW/h; 0 = off
SOLVER=cplex       # LP solver: cplex, glpk, gurobi or auto
ENV_NAME=gemai
# --------------------------------------------------------------------------
N_CHUNKS="${N_CHUNKS:-1}"   # parts per model; set by submit_dfba.sh

MODELS=(models/*.xml)
TASK=$(( ${SLURM_ARRAY_TASK_ID:-1} - 1 ))
IDX=$(( TASK / N_CHUNKS ))
CHUNK=$(( TASK % N_CHUNKS + 1 ))
if [ "$IDX" -ge "${#MODELS[@]}" ]; then
  echo "ERROR: task $((TASK + 1)) exceeds ${#MODELS[@]} models x $N_CHUNKS parts" >&2
  exit 1
fi
MODEL=${MODELS[$IDX]}
NAME=$(basename "$MODEL" .xml)

MM="${MAMBA_EXE:-$(command -v micromamba || true)}"
if [ -z "$MM" ]; then
  for p in "$HOME/.local/bin/micromamba" "$HOME/bin/micromamba" "$HOME/micromamba/bin/micromamba"; do
    if [ -x "$p" ]; then MM="$p"; break; fi
  done
fi
if [ -n "$MM" ]; then
  eval "$("$MM" shell hook --shell bash)"
  micromamba activate "$ENV_NAME"
else
  echo "micromamba not found; using python3 from the submitting shell" >&2
fi
python3 -c "import cobra; print('cobra', cobra.__version__)"

if [ ! -f "results/$NAME/substrates_presence.tsv" ]; then
  echo "ERROR: stage-1 results missing for $NAME (results/$NAME); run run_random_media.sbatch first" >&2
  exit 1
fi

echo "== dFBA model $((IDX + 1)) of ${#MODELS[@]}: $MODEL, part $CHUNK of $N_CHUNKS =="
mkdir -p results_dfba
python3 dfba_random_conc.py --model "$MODEL" --stage1 "results/$NAME" --out "results_dfba/$NAME" \
  --n-conc "$N_CONC" --conc-min "$CONC_MIN" --conc-max "$CONC_MAX" --x0 "$X0" \
  --t-end "$T_END" --dt "$DT" --record-dt "$RECORD_DT" --methods "$METHODS" --carbon-cap "$CARBON_CAP" \
  --n-chunks "$N_CHUNKS" --chunk "$CHUNK" --solver "$SOLVER" \
  --processes "${SLURM_CPUS_PER_TASK:-1}"
echo "== done =="
