#!/bin/bash
# Submit stage 2 (dFBA) for every model in models/, split into parts, plus a
# merge job that starts after all parts have ended.
#   bash submit_dfba.sh
# Edit N_CHUNKS and MAX_RUNNING below; resources per part are set in
# run_dfba.sbatch (#SBATCH lines), simulation settings there as well.

set -eo pipefail
cd "$(dirname "$0")"

# ---- settings: edit these lines ------------------------------------------
N_CHUNKS=2        # parts per model (array tasks per model)
MAX_RUNNING=0     # max tasks running at the same time (0 = no limit)
# --------------------------------------------------------------------------

N_MODELS=$(ls models/*.xml | wc -l)
N_TASKS=$(( N_MODELS * N_CHUNKS ))
ARRAY="1-${N_TASKS}"
if [ "$MAX_RUNNING" -gt 0 ]; then ARRAY="${ARRAY}%${MAX_RUNNING}"; fi
mkdir -p logs

JID=$(sbatch --parsable --array="$ARRAY" --export=ALL,N_CHUNKS="$N_CHUNKS" run_dfba.sbatch)
echo "submitted dFBA job $JID: $N_MODELS models x $N_CHUNKS parts = $N_TASKS tasks (array $ARRAY)"

MJID=$(sbatch --parsable --dependency=afterany:"$JID" run_dfba_merge.sbatch)
echo "submitted merge job $MJID (starts after all tasks of $JID have ended)"
