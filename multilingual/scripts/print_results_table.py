#!/usr/bin/env python3
"""Print an aggregate markdown table of lm-eval results from results/."""

import json
from glob import glob
from pathlib import Path


def _score(task_name: str, entry: dict) -> float | None:
    """Score for one task, in percent. Global PIQA reports acc_norm only."""
    if task_name.startswith("global_piqa"):
        val = entry.get("acc_norm,none", entry.get("acc,none"))
    else:
        val = entry.get("acc,none", entry.get("acc_norm,none"))
    return None if val is None else round(val * 100, 2)


def parse_groups(data: dict) -> dict[str, dict[str, float]]:
    """Parse a results JSON into {top-level group -> {task -> score}}.

    The hierarchy comes from `group_subtasks` ({group: [children]}), not from
    the leading spaces in `alias`: newer lm-eval versions stopped indenting
    aliases, which made every row look like a top-level group.

    A top-level group (e.g. zeroshot_eng) is one that is nobody's child. Its
    direct children become the task rows; a child that is itself a group
    (e.g. blimp_babylm_filtered) stays one aggregated row, and its own
    subtasks are not expanded.
    """
    results = {**data.get("results", {}), **data.get("groups", {})}
    group_subtasks: dict[str, list[str]] = data.get("group_subtasks", {})

    if not group_subtasks:
        # Task-only run with no group wrapper: one row per task.
        scores = {t: _score(t, e) for t, e in results.items()}
        return {"ungrouped": {t: s for t, s in scores.items() if s is not None}}

    children = {c for subtasks in group_subtasks.values() for c in subtasks}
    top_level = [g for g in group_subtasks if g not in children]

    groups: dict[str, dict[str, float]] = {}
    for group in top_level:
        rows: dict[str, float] = {}
        for task in group_subtasks[group]:
            score = _score(task, results.get(task, {}))
            if score is not None:
                rows[task] = score
        groups[group] = rows
    return groups


def load_model(json_path: str) -> dict[str, dict[str, float]]:
    with open(json_path) as f:
        return parse_groups(json.load(f))


def fmt_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


_LANG_COMBO_ORDER = [
    frozenset({"eng"}),
    frozenset({"nld"}),
    frozenset({"zho"}),
    frozenset({"eng", "nld"}),
    frozenset({"eng", "zho"}),
    frozenset({"nld", "zho"}),
    frozenset({"eng", "nld", "zho"}),
]


def training_languages(model_name: str) -> frozenset[str]:
    """Infer the training language(s) from a model name."""
    name = model_name.lower()

    # "Strict" / "Strict-Small" → English only
    if "strict" in name:
        return frozenset({"eng"})

    # *babylm-nld / *babylm-zho → single non-English language
    for lang in ("nld", "zho"):
        if f"babylm-{lang}" in name:
            return frozenset({lang})

    # en_nld_equal / nld_zho_equal / en_nld_zho_equal style
    langs: set[str] = set()
    if "en_" in name or "_en_" in name or name.startswith("en"):
        langs.add("eng")
    if "nld" in name:
        langs.add("nld")
    if "zho" in name:
        langs.add("zho")
    if langs:
        return frozenset(langs)

    return frozenset()


def model_sort_key(model_name: str) -> tuple[int, str]:
    langs = training_languages(model_name)
    try:
        order = _LANG_COMBO_ORDER.index(langs)
    except ValueError:
        order = len(_LANG_COMBO_ORDER)
    return (order, model_name)


def main():
    results_dir = Path(__file__).parent.parent / "results"
    json_files = sorted(glob(str(results_dir / "**/results*.json"), recursive=True))

    # model name = name of folder directly containing the results JSON
    # multiple JSON files can share the same folder (one per language run) — merge them
    all_models: dict[str, dict[str, dict[str, float]]] = {}
    for json_path in json_files:
        model_name = Path(json_path).parent.name.split("__")[-1]
        if model_name not in all_models:
            all_models[model_name] = {}
        for group, tasks in load_model(json_path).items():
            all_models[model_name][group] = tasks

    if not all_models:
        print("No results found.")
        return

    # Collect all groups and tasks, preserving first-seen order
    group_tasks: dict[str, list[str]] = {}
    for model_groups in all_models.values():
        for group, tasks in model_groups.items():
            if group not in group_tasks:
                group_tasks[group] = []
            for task in tasks:
                if task not in group_tasks[group]:
                    group_tasks[group].append(task)

    models = sorted(all_models.keys(), key=model_sort_key)

    # Header: task | model1 | model2 | ...
    print(fmt_row(["task"] + models))
    print(fmt_row(["---"] + ["---"] * len(models)))

    for group, tasks in group_tasks.items():
        # Language group header row
        print(fmt_row([f"**{group}**"] + [""] * len(models)))

        # One row per task
        for task in tasks:
            vals = [all_models[model].get(group, {}).get(task) for model in models]
            best = max((v for v in vals if v is not None), default=None)
            row = [task]
            for val in vals:
                if val is None:
                    row.append("")
                elif val == best:
                    row.append(f"**{val:.2f}**")
                else:
                    row.append(f"{val:.2f}")
            print(fmt_row(row))

        # Average row for this language group
        avgs = []
        for model in models:
            scores = [all_models[model].get(group, {}).get(t) for t in tasks]
            valid = [s for s in scores if s is not None]
            avgs.append(round(sum(valid) / len(valid), 2) if valid else None)
        best_avg = max((a for a in avgs if a is not None), default=None)
        avg_row = ["*avg*"]
        for avg in avgs:
            if avg is None:
                avg_row.append("")
            elif avg == best_avg:
                avg_row.append(f"**{avg:.2f}**")
            else:
                avg_row.append(f"{avg:.2f}")
        print(fmt_row(avg_row))


if __name__ == "__main__":
    main()
