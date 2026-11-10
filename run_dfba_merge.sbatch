#!/bin/bash
#SBATCH --job-name=dfba_merge
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# If your cluster requires a partition, add a line such as:  #SBATCH -p <partition>
#
# Merges the parts of every model in results_dfba/ (merge_dfba.py).
# Submitted by submit_dfba.sh; can also be run by hand:  sbatch run_dfba_merge.sbatch

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
ENV_NAME=gemai

MM="${MAMBA_EXE:-$(command -v micromamba || true)}"
if [ -z "$MM" ]; then
  for p in "$HOME/.local/bin/micromamba" "$HOME/bin/micromamba" "$HOME/micromamba/bin/micromamba"; do
    if [ -x "$p" ]; then MM="$p"; break; fi
  done
fi
if [ -n "$MM" ]; then
  eval "$("$MM" shell hook --shell bash)"
  micromamba activate "$ENV_NAME"
fi

python3 merge_dfba.py results_dfba/*/
echo "== done =="
