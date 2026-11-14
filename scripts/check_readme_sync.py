"""
CI verification script to ensure README.md benchmark numbers strictly match evals/results/rag_report.json.
Fails if report is not 'live' or if any number in README was edited manually.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.render_readme_tables import START_MARKER, END_MARKER, generate_results_block
from evals.rag.metrics import compute_file_sha256


def check_sync(
    readme_path: Path,
    report_path: Path,
    allow_mock: bool = False,
    dataset_path: Path | None = None,
    manifest_path: Path | None = None,
) -> tuple[bool, str]:
    if not report_path.exists():
        return False, f"Report file missing: {report_path}. Run 'python -m evals.rag.runner' first."

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    mode = report.get("mode", "").lower()
    if not allow_mock and mode != "live":
        return False, f"Report mode is '{mode}'. README must only be synced with 'live' report."

    # Task A: Freshness and integrity validation against live dataset & manifest
    meta = report.get("metadata", {}) or {}
    report_dataset_sha = report.get("dataset_sha256") or meta.get("dataset_sha256")
    report_total_cases = report.get("total_cases")
    report_manifest_sha = report.get("corpus_manifest_sha256") or meta.get("corpus_manifest_sha256")

    if dataset_path and dataset_path.exists():
        actual_dataset_sha = compute_file_sha256(dataset_path)
        if report_dataset_sha and report_dataset_sha != actual_dataset_sha:
            return (
                False,
                f"STALE REPORT: report dataset_sha256 ({report_dataset_sha[:10]}...) does not match "
                f"current {dataset_path.name} sha256 ({actual_dataset_sha[:10]}...).",
            )
        with open(dataset_path, "r", encoding="utf-8") as f:
            actual_case_count = sum(1 for line in f if line.strip())
        if report_total_cases is not None and report_total_cases != actual_case_count:
            return (
                False,
                f"CASE COUNT MISMATCH: report has {report_total_cases} cases, but {dataset_path.name} has {actual_case_count} cases.",
            )

    if manifest_path and manifest_path.exists():
        actual_manifest_sha = compute_file_sha256(manifest_path)
        if report_manifest_sha and report_manifest_sha != actual_manifest_sha:
            return (
                False,
                f"STALE MANIFEST: report corpus_manifest_sha256 ({report_manifest_sha[:10]}...) does not match "
                f"current {manifest_path.name} sha256 ({actual_manifest_sha[:10]}...).",
            )

    if not readme_path.exists():
        return False, f"README file missing: {readme_path}"

    readme_content = readme_path.read_text(encoding="utf-8")
    if START_MARKER not in readme_content or END_MARKER not in readme_content:
        return False, f"README.md is missing required boundary markers: {START_MARKER} and {END_MARKER}"

    pattern = re.compile(
        rf"{re.escape(START_MARKER)}.*?{re.escape(END_MARKER)}",
        re.DOTALL,
    )
    match = pattern.search(readme_content)
    if not match:
        return False, "Could not extract benchmark block between markers in README.md"

    actual_block = match.group(0).strip()
    expected_block = generate_results_block(report).strip()

    if actual_block != expected_block:
        return False, "README.md benchmark block is OUT OF SYNC with rag_report.json! Run 'python scripts/render_readme_tables.py' to regenerate."

    return True, "README.md is in perfect sync with live rag_report.json."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify README.md sync with rag_report.json.")
    parser.add_argument("--readme", default="README.md", help="Path to README.md")
    parser.add_argument("--report", default="evals/results/rag_report.json", help="Path to report JSON")
    parser.add_argument("--dataset", default="evals/rag/rag_dataset.jsonl", help="Path to dataset")
    parser.add_argument("--manifest", default="data/manifest.json", help="Path to corpus manifest")
    parser.add_argument("--allow-mock", action="store_true", help="Allow mock mode report for local testing")
    args = parser.parse_args()

    ok, msg = check_sync(
        Path(args.readme),
        Path(args.report),
        allow_mock=args.allow_mock,
        dataset_path=Path(args.dataset) if args.dataset else None,
        manifest_path=Path(args.manifest) if args.manifest else None,
    )
    if ok:
        print(f"[OK] {msg}")
        sys.exit(0)
    else:
        print(f"[FAIL] {msg}", file=sys.stderr)
        sys.exit(1)
