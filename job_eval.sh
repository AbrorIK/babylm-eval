#!/bin/bash
#SBATCH -J babylm-eval                                  # Job name
#SBATCH -p a40                                          # Use a40 partition
#SBATCH --gres=gpu:a40:1                                # Request 1 GPU
#SBATCH --cpus-per-task=4                               # CPUs for data loading
#SBATCH --time=06:00:00                                 # 8 hours (zero-shot + finetune for 3 langs)
#SBATCH -o /home/hpc/b279bb/b279bb26/thesis/babylm-eval/logs/eval_log_%j.out  # Standard output
#SBATCH -e /home/hpc/b279bb/b279bb26/thesis/babylm-eval/logs/eval_log_%j.err  # Standard error


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

source $HOME/thesis/babylm26/.venv/bin/activate

MY_MODEL_PATH="$HOME/thesis/babylm26/output/multihead-gpt2-complex/-export/nld"
MODEL_NAME="multihead-gpt2-complex"

LANGS="nld"

EVAL_DIR="$HOME/thesis/babylm-eval/multilingual"
cd "$EVAL_DIR"

echo ""
echo "--- Running Zero-shot Evaluation ---"
bash scripts/zeroshot_model.sh \
    --model_name "$MY_MODEL_PATH" \
    --langs "$LANGS" \
    --revision "$MODEL_NAME"

echo "Zero-shot evaluation complete at $(date)"

echo ""
echo "--- Running Fine-tuning Evaluation ---"
bash scripts/finetune_model.sh \
    --model_name "$MY_MODEL_PATH" \
    --langs "$LANGS" \
    --lr 5e-5 \
    --bsz 64 \
    --max_epochs 10 \
    --patience 3 \
    --seed 12

echo "Fine-tuning evaluation complete at $(date)"

echo ""
echo "--- Collating Results ---"
python scripts/collate_results.py \
    --model_name "$MODEL_NAME" \
    --fast

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "End time: $(date)"
echo ""
echo "Results written to:"
echo "  Zero-shot : $EVAL_DIR/../results/$MODEL_NAME/"
echo "  Finetune  : $EVAL_DIR/finetune/results/$MODEL_NAME/"
echo ""
echo "To inspect results locally, run from babylm-eval/multilingual/:"
echo "  python scripts/print_results_table.py"
echo "  python scripts/print_finetune_results.py"
echo "=========================================="
