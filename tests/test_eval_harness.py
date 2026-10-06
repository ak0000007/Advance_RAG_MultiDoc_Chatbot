"""
Pytest integration test for the Evaluation Benchmark Harness.
"""

import pytest
from evals.runner import EvalHarnessRunner


@pytest.mark.asyncio
async def test_full_eval_benchmark_accuracy():
    """Run full evaluation suite and assert 100% accuracy on golden dataset."""
    runner = EvalHarnessRunner()
    report = await runner.run_all()
    summary = report["summary"]

    assert summary["total_cases"] == 16
    assert summary["failed"] == 0, f"Eval cases failed: {[c for c in report['cases'] if not c['passed']]}"
    assert summary["overall_accuracy_pct"] == 100.0

    # Ensure every single category achieved 100%
    for category, score in summary["category_scores"].items():
        assert score == 100.0, f"Category '{category}' scored {score}%"
