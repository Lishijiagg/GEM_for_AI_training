#!/bin/bash
#SBATCH --job-name=random_media
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# One array task per model in models/*.xml:
#   sbatch --array=1-$(ls models/*.xml | wc -l) run_random_media.sbatch
# Without --array, the first model is run.

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

# ---- settings: edit these lines ------------------------------------------
N=1000             # conditions per model
FRAC_DROPOUT=0.5   # share of dropout conditions (rich media with 1..M_MAX nutrients removed)
M_MAX=20           # dropout design: max number of nutrients removed
ENV_NAME=gemai
# --------------------------------------------------------------------------

MODELS=(models/*.xml)
IDX=$(( ${SLURM_ARRAY_TASK_ID:-1} - 1 ))
MODEL=${MODELS[$IDX]}
NAME=$(basename "$MODEL" .xml)
OUT=results/$NAME

# activate the micromamba environment
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

echo "python: $(command -v python3)"
python3 -c "import cobra; print('cobra', cobra.__version__)" || {
  echo "ERROR: cobra not importable; activate the $ENV_NAME environment before sbatch" >&2
  exit 1
}

echo "== model $((IDX + 1)) of ${#MODELS[@]}: $MODEL -> $OUT =="
mkdir -p results
python3 random_media_pfba.py --model "$MODEL" --out "$OUT" --n "$N" \
  --frac-dropout "$FRAC_DROPOUT" --m-max "$M_MAX"
echo "== done =="
