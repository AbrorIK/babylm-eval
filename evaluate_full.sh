#!/bin/bash
#SBATCH -J babylm-eval-full                             # Job name (overridden by submit_all.sh)
#SBATCH -p a40                                          # Use a40 partition
#SBATCH --gres=gpu:a40:1                                # Request 1 GPU
#SBATCH --cpus-per-task=4                               # CPUs for data loading
#SBATCH --time=03:00:00                                 # zero-shot + PIQA + MECO + finetune
#SBATCH -o /home/hpc/b279bb/b279bb26/thesis/babylm-eval/logs/eval_%x_%j.out  # %x = job name
#SBATCH -e /home/hpc/b279bb/b279bb26/thesis/babylm-eval/logs/eval_%x_%j.err

# Usage: sbatch eval_job.sh <model_name> [checkpoint-dir-name]
MODEL_NAME="${1:?Usage: sbatch eval_job.sh <model_name> [checkpoint-XXXX]}"
CHECKPOINT="${2:-}"

mkdir -p "$HOME/thesis/babylm-eval/logs"

echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Partition:    $SLURM_JOB_PARTITION"
echo "GPUs:         $SLURM_GPUS_ON_NODE"
echo "CPUs:         $SLURM_CPUS_PER_TASK"
echo "Start time:   $(date)"
echo "=========================================="

# 1. FAU Internet Proxy (CRITICAL: Required to download eval datasets!)
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

# Disable wandb logging (no network calls, no local run folders)
export WANDB_MODE=disabled

# Tokenizers fork warnings during finetuning dataloaders
export TOKENIZERS_PARALLELISM=false

source "$HOME/thesis/.venv/bin/activate"

# Pick the checkpoint: the one given, or else the highest-numbered one
RUN_DIR="$WORK/output/$MODEL_NAME"
if [[ -n "$CHECKPOINT" ]]; then
    MY_MODEL_PATH="$RUN_DIR/$CHECKPOINT"
else
    MY_MODEL_PATH=$(ls -d "$RUN_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
fi

if [[ -z "$MY_MODEL_PATH" || ! -d "$MY_MODEL_PATH" ]]; then
    echo "ERROR: no checkpoint found under $RUN_DIR"
    exit 1
fi

LANGS="eng nld zho"

EVAL_DIR="$HOME/thesis/babylm-eval/multilingual"
cd "$EVAL_DIR"

LINK_DIR="$WORK/outputs"
EVAL_MODEL_PATH="$LINK_DIR/$MODEL_NAME"
mkdir -p "$LINK_DIR"
ln -sfn "$MY_MODEL_PATH" "$EVAL_MODEL_PATH"

if [[ ! -f "$EVAL_MODEL_PATH/config.json" ]]; then
    echo "ERROR: no config.json under $EVAL_MODEL_PATH (checked $MY_MODEL_PATH)"
    exit 1
fi

echo ""
echo "Model path:   $MY_MODEL_PATH"
echo "Eval as:      $EVAL_MODEL_PATH"
echo "Model name:   $MODEL_NAME"
echo "Languages:    $LANGS"
echo ""

echo "--- Running FULL evaluation (zero-shot + Global PIQA + MECO + finetune) ---"
bash scripts/eval_model_full.sh \
    --model_name "$EVAL_MODEL_PATH" \
    --langs "$LANGS" \
    --bos_fix 1
EVAL_STATUS=$?

echo ""
echo "eval_model_full.sh finished with status $EVAL_STATUS at $(date)"

# Steps inside eval_model_full.sh run independently, so collate whatever
# succeeded even if one of them failed.
echo ""
echo "--- Collating Results ---"
python scripts/collate_results.py \
    --model_name "$MODEL_NAME" \
    --output "$EVAL_DIR/results/${MODEL_NAME}_submission.json" \
    --output_predictions "$EVAL_DIR/results/${MODEL_NAME}_predictions.json"
COLLATE_STATUS=$?

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "End time:      $(date)"
echo "Eval status:   $EVAL_STATUS"
echo "Collate status:$COLLATE_STATUS"
echo ""
echo "Results written to:"
echo "  Zero-shot + PIQA : $EVAL_DIR/results/main/"
echo "  MECO             : $EVAL_DIR/meco/results/main/"
echo "  Finetune         : $EVAL_DIR/finetune/results/$MODEL_NAME/"
echo "  Submission JSON  : $EVAL_DIR/results/${MODEL_NAME}_submission.json"
echo ""
echo "=========================================="

exit $EVAL_STATUS