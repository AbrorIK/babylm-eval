#!/bin/bash
#SBATCH -J babylm-eval-multihead                        # Job name (overridden by the caller)
#SBATCH -p a40                                          # Use a40 partition
#SBATCH --gres=gpu:a40:1                                # Request 1 GPU
#SBATCH --cpus-per-task=4                               # CPUs for data loading
#SBATCH --time=02:00:00                                 # Same work as evaluate_full.sh, but 3 model loads + 3x finetune
#SBATCH -o /home/hpc/b279bb/b279bb26/thesis/babylm-eval/logs/eval_%x_%j.out  # %x = job name
#SBATCH -e /home/hpc/b279bb/b279bb26/thesis/babylm-eval/logs/eval_%x_%j.err

# Usage: sbatch evaluate_full_mhead.sh [model_path] [model_name]

MODEL_PATH="${1:-$WORK/output/multihead/seed0}"
MODEL_PATH="${MODEL_PATH%/}"
# Name after the run directory, not after the checkpoint-*-export inside it.
NAME_PATH="$MODEL_PATH"
[[ "$(basename "$NAME_PATH")" == checkpoint-*-export ]] && NAME_PATH="$(dirname "$NAME_PATH")"
MODEL_NAME="${2:-$(basename "$(dirname "$NAME_PATH")")-$(basename "$NAME_PATH")}"

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

# Accept either the export directory itself or the run directory above it.
if [[ -f "$MODEL_PATH/eng/config.json" ]]; then
    EXPORT_PATH="$MODEL_PATH"
else
    EXPORT_PATH=$(ls -d "$MODEL_PATH"/checkpoint-*-export 2>/dev/null | sort -V | tail -n 1)
fi

if [[ -z "$EXPORT_PATH" || ! -d "$EXPORT_PATH" ]]; then
    echo "ERROR: no per-language export under $MODEL_PATH"
    echo "Expected $MODEL_PATH/{eng,nld,zho}/config.json, or a checkpoint-*-export directory."
    echo "Create one first, with the training environment:"
    echo "  python export_mhead.py --checkpoint $MODEL_PATH/<checkpoint-N> --tokenizer <tokenizer>"
    exit 1
fi

cd "$EVAL_DIR" || exit 1

mkdir -p "$LINK_DIR"

echo ""
echo "Model path:   $MODEL_PATH"
echo "Export dir:   $EXPORT_PATH"
echo "Model name:   $MODEL_NAME"
echo "Languages:    $LANGS"
echo ""

EVAL_STATUS=0
COLLATE_STATUS=0

for LANG in $LANGS; do
    LANG_DIR="$EXPORT_PATH/$LANG"
    if [[ ! -f "$LANG_DIR/config.json" ]]; then
        echo "ERROR: no config.json under $LANG_DIR — skipping $LANG"
        EVAL_STATUS=1
        continue
    fi

    EVAL_MODEL_PATH="$LINK_DIR/${MODEL_NAME}-${LANG}"
    ln -sfn "$LANG_DIR" "$EVAL_MODEL_PATH"

    echo ""
    echo "=========================================="
    echo "### $LANG head"
    echo "Eval as:      $EVAL_MODEL_PATH"
    echo "=========================================="

    echo "--- Full evaluation (zero-shot + Global PIQA + MECO + finetune) ---"
    bash scripts/eval_model_full.sh \
        --model_name "$EVAL_MODEL_PATH" \
        --langs "$LANG" \
        --bos_fix 1 || EVAL_STATUS=1

    # Each head covers one language, so the collator's "missing task" warnings
    # for the other two languages are expected here; the merge below fills them in.
    echo ""
    echo "--- Collating results ($LANG) ---"
    python scripts/collate_results.py \
        --model_name "${MODEL_NAME}-${LANG}" \
        --output "$EVAL_DIR/results/${MODEL_NAME}-${LANG}_submission.json" \
        --output_predictions "$EVAL_DIR/results/${MODEL_NAME}-${LANG}_predictions.json" || COLLATE_STATUS=1
done

echo ""
echo "--- Merging the per-language submissions ---"
python - "$EVAL_DIR/results" "$MODEL_NAME" $LANGS <<'PY'
import json
import sys
from pathlib import Path

results_dir, model_name, *langs = sys.argv[1:]
results_dir = Path(results_dir)

# Submissions are {task: {subtask: score}} and predictions are
# {zeroshot, finetune, hidden}; the heads contribute disjoint keys, except for
# meco_l1/meco_l2, which are keyed by language and merge one level deeper.
submission = {}
predictions = {"zeroshot": {}, "finetune": {}, "hidden": {}}

for lang in langs:
    sub_path = results_dir / f"{model_name}-{lang}_submission.json"
    if not sub_path.exists():
        print(f"Warning: {sub_path.name} is missing — {lang} left out of the merge.")
        continue
    for task, entries in json.loads(sub_path.read_text()).items():
        submission.setdefault(task, {}).update(entries)

    pred_path = results_dir / f"{model_name}-{lang}_predictions.json"
    if not pred_path.exists():
        print(f"Warning: {pred_path.name} is missing — {lang} predictions left out.")
        continue
    head = json.loads(pred_path.read_text())
    predictions["zeroshot"].update(head.get("zeroshot", {}))
    predictions["finetune"].update(head.get("finetune", {}))
    for task, value in head.get("hidden", {}).items():
        if task.startswith("meco"):
            predictions["hidden"].setdefault(task, {}).update(value)
        else:
            predictions["hidden"][task] = value

(results_dir / f"{model_name}_submission.json").write_text(json.dumps(submission, indent=2))
(results_dir / f"{model_name}_predictions.json").write_text(json.dumps(predictions, indent=2))

print(f"Merged {len(submission)} task(s): "
      f"{len(predictions['zeroshot'])} zeroshot, "
      f"{len(predictions['finetune'])} finetune task-lang pairs, "
      f"{len(predictions['hidden'])} hidden tasks")
for task, entries in sorted(submission.items()):
    if len(entries) > 1:
        print(f"  {task}: {', '.join(sorted(entries))}")
PY
MERGE_STATUS=$?

echo ""
echo "=========================================="
echo "Full evaluation finished (eval $EVAL_STATUS, collate $COLLATE_STATUS, merge $MERGE_STATUS)"
echo "End time: $(date)"
echo ""
echo "Results written to:"
echo "  Zero-shot + PIQA : $EVAL_DIR/results/main/*${MODEL_NAME}-{eng,nld,zho}/"
echo "  MECO             : $EVAL_DIR/meco/results/main/"
echo "  Finetune         : $EVAL_DIR/finetune/results/${MODEL_NAME}-{eng,nld,zho}/"
echo "  Per-head JSON    : $EVAL_DIR/results/${MODEL_NAME}-{eng,nld,zho}_submission.json"
echo "  Submission JSON  : $EVAL_DIR/results/${MODEL_NAME}_submission.json"
echo "=========================================="

exit $EVAL_STATUS
