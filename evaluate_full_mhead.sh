#!/bin/bash
#SBATCH -J babylm-eval-multihead                        # Job name (overridden by the caller)
#SBATCH -p a40                                          # Use a40 partition
#SBATCH --gres=gpu:a40:1                                # Request 1 GPU
#SBATCH --cpus-per-task=4                               # CPUs for data loading
#SBATCH --time=06:00:00                                 # Same total work as eval_job.sh, but 3 model loads + 3x POS
#SBATCH -o /home/hpc/b279bb/b279bb26/thesis/babylm-eval/logs/eval_%x_%j.out  # %x = job name
#SBATCH -e /home/hpc/b279bb/b279bb26/thesis/babylm-eval/logs/eval_%x_%j.err

# Usage: sbatch eval_job_multihead.sh <model_name> [checkpoint-XXXX]
MODEL_NAME="${1:?Usage: sbatch eval_job_multihead.sh <model_name> [checkpoint-XXXX]}"
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

# FAU Internet Proxy (CRITICAL: Required to download eval datasets!)
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

source "$HOME/thesis/.venv/bin/activate"

RUN_DIR="$WORK/output/$MODEL_NAME"


if [[ -n "$CHECKPOINT" ]]; then
    EXPORT_DIR="$RUN_DIR/${CHECKPOINT%-export}-export"
else
    EXPORT_DIR=$(ls -d "$RUN_DIR"/checkpoint-*-export 2>/dev/null | sort -V | tail -n 1)
fi

if [[ -z "$EXPORT_DIR" || ! -d "$EXPORT_DIR" ]]; then
    echo "ERROR: no '<checkpoint>-export' directory under $RUN_DIR"
    echo "Create one first, with the training environment:"
    echo "  python export_mhead.py --checkpoint $RUN_DIR/<checkpoint-N> --tokenizer <tokenizer>"
    exit 1
fi

LANGS="eng nld zho"
EVAL_DIR="$HOME/thesis/babylm-eval/multilingual"
LINK_DIR="$WORK/outputs"
cd "$EVAL_DIR" || exit 1
mkdir -p "$LINK_DIR"

echo ""
echo "Export dir:   $EXPORT_DIR"
echo "Model name:   $MODEL_NAME"
echo "Languages:    $LANGS"
echo ""

STATUS=0

for LANG in $LANGS; do
    LANG_DIR="$EXPORT_DIR/$LANG"
    if [[ ! -f "$LANG_DIR/config.json" ]]; then
        echo "ERROR: no config.json under $LANG_DIR — skipping $LANG"
        STATUS=1
        continue
    fi

    EVAL_MODEL_PATH="$LINK_DIR/${MODEL_NAME}-${LANG}"
    ln -sfn "$LANG_DIR" "$EVAL_MODEL_PATH"

    echo ""
    echo "=========================================="
    echo "### $LANG -> $EVAL_MODEL_PATH"
    echo "=========================================="

    echo "--- [1/4] Zero-shot ($LANG) ---"
    bash scripts/zeroshot_model.sh --model_name "$EVAL_MODEL_PATH" --langs "$LANG" --bos_fix 1 || STATUS=1

    echo "--- [2/4] Global PIQA ($LANG) ---"
    bash scripts/global_piqa_model.sh --model_name "$EVAL_MODEL_PATH" --langs "$LANG" --bos_fix 1 || STATUS=1

    echo "--- [3/4] MECO ($LANG) ---"
    bash scripts/meco_model.sh --model_name "$EVAL_MODEL_PATH" --langs "$LANG" || STATUS=1

    echo "--- [4/4] Finetune ($LANG) ---"
    bash scripts/finetune_model.sh --model_name "$EVAL_MODEL_PATH" --langs "$LANG" || STATUS=1
done

echo ""
echo "=========================================="
echo "Evaluation complete (status $STATUS)"
echo "End time: $(date)"
echo ""
echo "Results (one set per language):"
echo "  Zero-shot + PIQA : $EVAL_DIR/results/main/*${MODEL_NAME}-{eng,nld,zho}/"
echo "  MECO             : $EVAL_DIR/meco/results/main/"
echo "  Finetune         : $EVAL_DIR/finetune/results/${MODEL_NAME}-{eng,nld,zho}/"
echo "=========================================="

exit $STATUS
