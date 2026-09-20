#!/bin/bash
#SBATCH -J babylm-zeroshot                              # Job name (overridden by the caller)
#SBATCH -p a40                                          # Use a40 partition
#SBATCH --gres=gpu:a40:1                                # Request 1 GPU
#SBATCH --cpus-per-task=4                               # CPUs for data loading
#SBATCH --time=02:00:00                                 # Zero-shot only
#SBATCH -o /home/hpc/b279bb/b279bb26/thesis/babylm-eval/logs/eval_%x_%j.out  # %x = job name
#SBATCH -e /home/hpc/b279bb/b279bb26/thesis/babylm-eval/logs/eval_%x_%j.err

# Zero-shot evaluation of a locally trained checkpoint.
#
# Usage: sbatch evaluate_zero_shot.sh [model_path] [model_name]
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

# lm-eval names its output folder after the *basename* of the model path, and
# the default path's basename is just "seed0". Symlink the checkpoint to a path
# whose basename is $MODEL_NAME so the results stay identifiable.
mkdir -p "$LINK_DIR"
EVAL_MODEL_PATH="$LINK_DIR/$MODEL_NAME"
ln -sfn "$MODEL_PATH" "$EVAL_MODEL_PATH"

echo ""
echo "Model path:   $MODEL_PATH"
echo "Eval as:      $EVAL_MODEL_PATH"
echo "Model name:   $MODEL_NAME"
echo "Languages:    $LANGS"
echo ""

echo "--- Zero-shot evaluation ---"
bash scripts/zeroshot_model.sh \
    --model_name "$EVAL_MODEL_PATH" \
    --langs "$LANGS" \
    --bos_fix 1
STATUS=$?

echo ""
echo "=========================================="
echo "Zero-shot evaluation finished (status $STATUS)"
echo "End time: $(date)"
echo ""
echo "Results: $EVAL_DIR/results/main/*${MODEL_NAME}/"
echo "Read them with:"
echo "  cd $EVAL_DIR && python3 scripts/print_results_table.py"
echo "=========================================="

exit $STATUS
