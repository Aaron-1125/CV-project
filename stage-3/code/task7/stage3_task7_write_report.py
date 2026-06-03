#!/usr/bin/env python3
"""Write the final Stage3 Task7 StarGAN report."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from stage3_task7_common import load_config, read_json, report_dir, stage3_root, summary_dir, write_json


def maybe_json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def pct(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "N/A"


def num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "N/A"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task7_stargan/a800_full.py")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def md_path(value: Any, output: Path) -> str:
    if not value:
        return ""
    path = Path(str(value))
    if not path.is_absolute():
        path = stage3_root() / path
    try:
        return Path(os.path.relpath(path, output.parent)).as_posix()
    except Exception:
        return str(value)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    sdir = summary_dir(cfg)
    prepare = maybe_json(sdir / "celeba_prepare_summary.json")
    fixed = maybe_json(sdir / "fixed_samples_manifest.json")
    pretrained = maybe_json(sdir / "pretrained_sanity_summary.json")
    train = maybe_json(sdir / "stargan_train_summary.json")
    test = maybe_json(sdir / "self_trained_test_summary.json")
    monitor = maybe_json(sdir / "monitor_samples_summary.json")
    classifier = maybe_json(sdir / "attribute_classifier_summary.json")
    attr = maybe_json(sdir / "attribute_success_summary.json")
    identity = maybe_json(sdir / "identity_retention_summary.json")
    fid_is = maybe_json(sdir / "fid_is_summary.json")
    evaluation = maybe_json(sdir / "task7_evaluation_summary.json")

    attr_rows = []
    for direction, row in attr.get("per_direction", {}).items():
        attr_rows.append(
            f"| {direction} | {row.get('total', 0)} | {pct(row.get('primary_success_rate'))} | {pct(row.get('strict_success_rate'))} |"
        )
    if not attr_rows:
        attr_rows.append("| N/A | 0 | N/A | N/A |")

    monitor_rows = []
    for row in monitor.get("checkpoints", []):
        monitor_rows.append(f"- iter `{row.get('iters')}`: ![monitor iter {row.get('iters')}]({Path(row.get('grid', '')).as_posix()})")
    if not monitor_rows:
        monitor_rows.append("- No monitor grids have been generated yet.")

    low_identity = []
    threshold = identity.get("warning_threshold", 0.35)
    for row in identity.get("records", []):
        sim = row.get("identity_cosine")
        if sim is not None and sim < threshold:
            low_identity.append(row)
    failure_lines = []
    attr_failures = [
        row for row in attr.get("records", [])
        if not row.get("primary_success", False)
    ][:8]
    for row in attr_failures:
        failure_lines.append(
            f"- Attribute miss: `{row.get('filename')}` direction `{row.get('direction')}`, "
            f"target `{row.get('target_label')}`, predicted `{row.get('predicted_label')}`."
        )
    for row in low_identity[:8]:
        failure_lines.append(
            f"- Identity drift: `{row.get('filename')}` direction `{row.get('direction')}`, "
            f"cosine `{num(row.get('identity_cosine'), 3)}`."
        )
    if not failure_lines:
        failure_lines.append("- No failure records are available yet; rerun evaluation after full training to populate this section.")

    output = args.output or report_dir(cfg) / "stage3_task7_stargan_attribute_editing_report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    fixed_source_grid = md_path(fixed.get("source_grid", "reports/task7/assets/fixed_samples/fixed_source_grid.jpg"), output)
    pretrained_grid = md_path(pretrained.get("fixed_grid", ""), output)
    self_grid = md_path(test.get("fixed_grid", ""), output)
    monitor_rows = [
        row.replace(str(Path(row.split("](")[-1][:-1]).as_posix()), md_path(row.split("](")[-1][:-1], output))
        if "](" in row and row.endswith(")") else row
        for row in monitor_rows
    ]
    markdown = f"""# Stage3 Task7 StarGAN Attribute Editing Report

## 1. Scope

This report covers only Stage3 Task 7.x: face attribute editing with StarGAN on CelebA. The official StarGAN source is kept outside this Git deliverable at `/Users/aaron/Documents/字节实习/task/StarGAN`; `stage-3` contains wrappers, configs, reports, summaries, and generated evidence.

## 2. Setup And Fixed Samples

- CelebA prepared: `{prepare.get('ready', False)}`
- fixed sample count: `{fixed.get('fixed_sample_count', len(fixed.get('samples', [])))}`
- selected attrs: `{fixed.get('selected_attrs', [])}`
- train command return code: `{train.get('returncode', 'N/A')}`
- final checkpoint: `{test.get('checkpoint', evaluation.get('checkpoint', 'N/A'))}`

Fixed input faces:

![fixed source grid]({fixed_source_grid})

Target labels are saved before generation in `{fixed.get('target_labels', {}).get('json', 'N/A')}` and `{fixed.get('target_labels', {}).get('csv', 'N/A')}`. Hair-color target labels are validated as mutually exclusive for every generated direction.

## 3. Pretrained Sanity Check

The official CelebA 128 pretrained checkpoint is used as a reference before accepting the self-trained run.

- pretrained checkpoint: `{pretrained.get('checkpoint', 'N/A')}`
- pretrained fixed grid: `{pretrained.get('fixed_grid', 'N/A')}`

![official pretrained fixed grid]({pretrained_grid})

## 4. Self-trained Results

Final self-trained fixed grid:

![self trained fixed grid]({self_grid})

Intermediate fixed-sample monitoring:

{chr(10).join(monitor_rows)}

## 5. Quantitative Evaluation

Attribute classifier:

- checkpoint: `{classifier.get('checkpoint', 'N/A')}`
- final exact-match accuracy: `{pct(classifier.get('final', {}).get('exact_match_accuracy'))}`
- per-attribute accuracy: `{classifier.get('final', {}).get('per_attr_accuracy', {})}`

Attribute edit success:

| Direction | Samples | Primary success | Strict 5-attr success |
| --- | ---: | ---: | ---: |
{chr(10).join(attr_rows)}

Identity retention with InsightFace `buffalo_l`:

- valid pairs: `{identity.get('valid_pairs', 'N/A')}` / `{identity.get('pairs', 'N/A')}`
- mean cosine: `{num(identity.get('mean'), 4)}`
- median cosine: `{num(identity.get('median'), 4)}`
- p10 cosine: `{num(identity.get('p10'), 4)}`
- no generated face: `{identity.get('no_generated_face', 'N/A')}`

FID/IS auxiliary metrics:

- available: `{fid_is.get('available', False)}`
- FID: `{num(fid_is.get('fid'), 4)}`
- Inception Score: `{num(fid_is.get('inception_score_mean'), 4)} +/- {num(fid_is.get('inception_score_std'), 4)}`
- note: FID/IS are auxiliary; acceptance focuses on attribute success and identity retention.

## 6. Failure Case Analysis

{chr(10).join(failure_lines)}

Common failure modes to inspect manually are: target attribute not activated, identity drift after large gender/age edits, hair-color ambiguity, and local artifacts around hairline, glasses, or background.
"""
    output.write_text(markdown, encoding="utf-8")
    print(f"Wrote {output}")
    write_json(sdir / "report_summary.json", {"report": str(output), "ready": True})


if __name__ == "__main__":
    main()
