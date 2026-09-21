#!/bin/bash
#SBATCH -J babylm-eval-full                             # Job name (overridden by the caller)
#SBATCH -p a40                                          # Use a40 partition
#SBATCH --gres=gpu:a40:1                                # Request 1 GPU
#SBATCH --cpus-per-task=4                               # CPUs for data loading
#SBATCH --time=03:00:00                                 # zero-shot + PIQA + MECO + finetune
#SBATCH -o /home/hpc/b279bb/b279bb26/thesis/babylm-eval/logs/eval_%x_%j.out  # %x = job name
#SBATCH -e /home/hpc/b279bb/b279bb26/thesis/babylm-eval/logs/eval_%x_%j.err

# Usage: sbatch evaluate_full.sh [model_path] [model_name]
#   model_path  directory holding config.json  (default: $WORK/output1/baseline/seed0)
#   model_name  label the results are filed under (default: basename-of-parent + basename,
#               i.e. "baseline-seed0" for the default path)

MODEL_PATH="${1:-$WORK/output1/baseline/seed0}"
MODEL_PATH="${MODEL_PATH%/}"
MODEL_NAME="${2:-$(basename "$(dirname "$MODEL_PATH")")-$(basename "$MODEL_PATH")}"

LANGS="eng nld zho"
EVAL_DIR="$HOME/thesis/babylm-eval/multilingual"
LINK_DIR="$WORK/outputs"

mkdir -p "$HOME/thesis/babylm-eval/logs"

echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Partition:    $SLURM_JOB_PARTITION"
echo "GPUs:         $SLURM_GPUS_ON_NODE"
echo "CPUs:         $SLURM_CPUS_PER_TASK"
echo "Start time:   $(date)"
echo "=========================================="

# FAU Internet Proxy (CRITICAL: Required to download eval datasets!)
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

source "$HOME/thesis/.venv/bin/activate"

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "ERROR: no config.json under $MODEL_PATH"
    exit 1
fi

cd "$EVAL_DIR" || exit 1

mkdir -p "$LINK_DIR"
EVAL_MODEL_PATH="$LINK_DIR/$MODEL_NAME"
ln -sfn "$MODEL_PATH" "$EVAL_MODEL_PATH"

echo ""
echo "Model path:   $MODEL_PATH"
echo "Eval as:      $EVAL_MODEL_PATH"
echo "Model name:   $MODEL_NAME"
echo "Languages:    $LANGS"
echo ""

echo "--- Full evaluation (zero-shot + Global PIQA + MECO + finetune) ---"
bash scripts/eval_model_full.sh \
    --model_name "$EVAL_MODEL_PATH" \
    --langs "$LANGS" \
    --bos_fix 1
EVAL_STATUS=$?


echo ""
echo "--- Collating results ---"
python scripts/collate_results.py \
    --model_name "$MODEL_NAME" \
    --output "$EVAL_DIR/results/${MODEL_NAME}_submission.json" \
    --output_predictions "$EVAL_DIR/results/${MODEL_NAME}_predictions.json"
COLLATE_STATUS=$?

echo ""
echo "=========================================="
echo "Full evaluation finished (eval $EVAL_STATUS, collate $COLLATE_STATUS)"
echo "End time: $(date)"
echo ""
echo "Results written to:"
echo "  Zero-shot + PIQA : $EVAL_DIR/results/main/*${MODEL_NAME}/"
echo "  MECO             : $EVAL_DIR/meco/results/main/"
echo "  Finetune         : $EVAL_DIR/finetune/results/$MODEL_NAME/"
echo "  Submission JSON  : $EVAL_DIR/results/${MODEL_NAME}_submission.json"
echo "=========================================="

exit $EVAL_STATUS
