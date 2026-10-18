"""Conservative local/CI quality gate for a saved serving evaluation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report",
        type=Path,
        nargs="?",
        default=Path("reports/serving/api_evaluation.json"),
    )
    parser.add_argument("--min-macro-f1", type=float, default=0.85)
    parser.add_argument("--min-neutral-f1", type=float, default=0.80)
    args = parser.parse_args()
    if not args.report.is_file():
        raise SystemExit(
            f"Quality report not found: {args.report}. Run scripts/evaluate_served_model.py first."
        )
    metrics = json.loads(args.report.read_text(encoding="utf-8"))
    if not metrics.get("complete_test_set", False):
        raise SystemExit(
            "Quality gate requires a complete held-out test-set evaluation, not a smoke-test subset."
        )
    macro_f1 = float(metrics["f1_macro"])
    neutral_f1 = float(metrics["per_class_f1"]["NEUTRAL"])
    print(f"macro-F1={macro_f1:.4f} (minimum {args.min_macro_f1:.4f})")
    print(f"NEUTRAL F1={neutral_f1:.4f} (minimum {args.min_neutral_f1:.4f})")
    failures = []
    if macro_f1 < args.min_macro_f1:
        failures.append("macro-F1 below threshold")
    if neutral_f1 < args.min_neutral_f1:
        failures.append("NEUTRAL F1 below threshold")
    if failures:
        print("QUALITY GATE FAILED: " + "; ".join(failures))
        return 1
    print("QUALITY GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
