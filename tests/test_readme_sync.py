"""
Unit test to verify README sync script and boundary markers.
"""

import json
from pathlib import Path
import pytest

from scripts.check_readme_sync import check_sync
from scripts.render_readme_tables import START_MARKER, END_MARKER, generate_results_block


def test_readme_contains_markers():
    readme_path = Path("README.md")
    assert readme_path.exists()
    content = readme_path.read_text(encoding="utf-8")
    assert START_MARKER in content, f"Missing {START_MARKER} in README.md"
    assert END_MARKER in content, f"Missing {END_MARKER} in README.md"


def test_check_sync_detects_tampering(tmp_path):
    report_file = tmp_path / "rag_report.json"
    readme_file = tmp_path / "README.md"

    fake_report = {
        "mode": "live",
        "timestamp": "2026-11-14T00:00:00Z",
        "total_cases": 10,
        "answerable_cases": 8,
        "unanswerable_cases": 2,
        "metadata": {
            "device": "cpu",
            "device_name": "CPU",
            "repeats": 1,
            "seed": 0,
            "component_latencies": {"query_embedding": {"median_sec": 0.01, "p95_sec": 0.02}},
        },
        "markdown_table": "| Pipeline | Hit@1 |\n| :--- | :---: |\n| Dense | 1.000 |",
        "category_markdown_table": "| Category | Pipeline | N | Hit@1 |\n| single | Dense | 8 | 1.000 |",
    }
    report_file.write_text(json.dumps(fake_report), encoding="utf-8")

    # Correct sync
    valid_block = generate_results_block(fake_report)
    readme_file.write_text(f"Prefix\n\n{valid_block}\n\nSuffix", encoding="utf-8")

    ok, msg = check_sync(readme_file, report_file, allow_mock=False)
    assert ok is True

    # Tampered sync (someone hand-edited 1.000 to 0.999)
    tampered_block = valid_block.replace("1.000", "0.999")
    readme_file.write_text(f"Prefix\n\n{tampered_block}\n\nSuffix", encoding="utf-8")

    ok, msg = check_sync(readme_file, report_file, allow_mock=False)
    assert ok is False
    assert "OUT OF SYNC" in msg


def test_check_sync_fails_on_mock_mode(tmp_path):
    report_file = tmp_path / "rag_report.json"
    readme_file = tmp_path / "README.md"
    fake_report = {
        "mode": "mock",
        "timestamp": "2026-11-14T00:00:00Z",
        "total_cases": 10,
        "metadata": {},
        "markdown_table": "",
        "category_markdown_table": "",
    }
    report_file.write_text(json.dumps(fake_report), encoding="utf-8")
    readme_file.write_text(f"{START_MARKER}\n{END_MARKER}", encoding="utf-8")

    ok, msg = check_sync(readme_file, report_file, allow_mock=False)
    assert ok is False
    assert "must only be synced with 'live' report" in msg

    # Allowed when allow_mock=True
    ok, _ = check_sync(readme_file, report_file, allow_mock=True)
    # Block is out of sync or missing, but mode check passed


def test_check_sync_fails_on_stale_hash_or_count(tmp_path):
    report_file = tmp_path / "rag_report.json"
    readme_file = tmp_path / "README.md"
    fake_dataset = tmp_path / "dataset.jsonl"
    fake_dataset.write_text('{"id": "1"}\n{"id": "2"}\n', encoding="utf-8")

    fake_report = {
        "mode": "live",
        "timestamp": "2026-11-14T00:00:00Z",
        "dataset_sha256": "wrong_sha256",
        "total_cases": 2,
        "metadata": {},
        "markdown_table": "",
        "category_markdown_table": "",
    }
    report_file.write_text(json.dumps(fake_report), encoding="utf-8")
    readme_file.write_text(f"{START_MARKER}\n{END_MARKER}", encoding="utf-8")

    ok, msg = check_sync(readme_file, report_file, dataset_path=fake_dataset)
    assert ok is False
    assert "STALE REPORT" in msg

    # Case count mismatch
    fake_report["dataset_sha256"] = ""  # skip sha check
    fake_report["total_cases"] = 999
    report_file.write_text(json.dumps(fake_report), encoding="utf-8")
    ok, msg = check_sync(readme_file, report_file, dataset_path=fake_dataset)
    assert ok is False
    assert "CASE COUNT MISMATCH" in msg

