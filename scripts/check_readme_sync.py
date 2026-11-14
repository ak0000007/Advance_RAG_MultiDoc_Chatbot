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


def check_sync(readme_path: Path, report_path: Path, allow_mock: bool = False) -> tuple[bool, str]:
    if not report_path.exists():
        return False, f"Report file missing: {report_path}. Run 'python -m evals.rag.runner' first."

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    mode = report.get("mode", "").lower()
    if not allow_mock and mode != "live":
        return False, f"Report mode is '{mode}'. README must only be synced with 'live' report."

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
    parser.add_argument("--allow-mock", action="store_true", help="Allow mock mode report for local testing")
    args = parser.parse_args()

    ok, msg = check_sync(Path(args.readme), Path(args.report), allow_mock=args.allow_mock)
    if ok:
        print(f"[OK] {msg}")
        sys.exit(0)
    else:
        print(f"[FAIL] {msg}", file=sys.stderr)
        sys.exit(1)
