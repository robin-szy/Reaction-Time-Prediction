#!/bin/bash -l
#SBATCH --job-name=DL_reaction_times
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
# SBATCH --partition=gpu
# SBATCH --gpus-per-task=1
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=0-0:30:00 #DD-HH:MM:SS
#SBATCH --array=1-15
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

module --force purge
module load env/development/2025a
module load lang/Python/3.13.1-GCCcore-14.2.0


source $HOME/deep_learning/DL_env_latest/bin/activate

PROJECT_DIR="$HOME/deep_learning/reaction_time_HW"
CONFIG_FILE="$PROJECT_DIR/configs.csv"
DATA_DIR="$PROJECT_DIR/scanpaths"
RUNS_DIR="$PROJECT_DIR/runs"

cd "$PROJECT_DIR"
mkdir -p "$RUNS_DIR" logs

CONFIG_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG_FILE")

IFS=',' read -r RUN_NAME HIDDEN_SIZE LR WEIGHT_DECAY BATCH_SIZE DROPOUT SEED LOSS HUBER_DELTA BIDIRECTIONAL EPOCHS PATIENCE BOTTLENECK_DIM GLOBAL_DROPOUT GLOBAL_ALPHA GLOBAL_WARMUP <<< "$CONFIG_LINE"

MODEL_FILE="$RUNS_DIR/${RUN_NAME}.pth"

echo "======================================"
echo "SLURM job id:        ${SLURM_JOB_ID}"
echo "SLURM array task:    ${SLURM_ARRAY_TASK_ID}"
echo "Host:                $(hostname)"
echo "Date:                $(date)"
echo "Project dir:         ${PROJECT_DIR}"
echo "Data dir:            ${DATA_DIR}"
echo "Run name:            ${RUN_NAME}"
echo "Model file:          ${MODEL_FILE}"
echo "Hidden size:         ${HIDDEN_SIZE}"
echo "LR:                  ${LR}"
echo "Weight decay:        ${WEIGHT_DECAY}"
echo "Batch size:          ${BATCH_SIZE}"
echo "Dropout:             ${DROPOUT}"
echo "Seed:                ${SEED}"
echo "CUDA available:"
python - <<'PY'
import torch
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
echo "======================================"

BIDIRECTIONAL_ARG=""
if [[ "$BIDIRECTIONAL" == "true" || "$BIDIRECTIONAL" == "1" ]]; then
    BIDIRECTIONAL_ARG="--bidirectional"
fi

srun python -u train.py \
    --data-dir "$DATA_DIR" \
    --metadata-file "train_val_labels.csv" \
    --scanpath-dir "train_val" \
    --model-file "$MODEL_FILE" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --seq-max-len 512 \
    --hidden-size "$HIDDEN_SIZE" \
    --min-delta 1e-4 \
    --dropout "$DROPOUT" \
    --seed "$SEED" \
    --val-frac 0.2 \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --loss "$LOSS" \
    --huber-delta "$HUBER_DELTA" \
    --bottleneck-dim "$BOTTLENECK_DIM" \
    --global-dropout "$GLOBAL_DROPOUT" \
    --global-alpha "$GLOBAL_ALPHA" \
    --global-warmup "$GLOBAL_WARMUP" \
    $BIDIRECTIONAL_ARG

echo "Finished run: ${RUN_NAME}"
echo "Saved model: ${MODEL_FILE}"




