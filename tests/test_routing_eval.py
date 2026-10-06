"""
End-to-End Routing Evaluation Benchmark Test.

Runs the routing evaluation suite against the real LLM and mock CRM client,
and asserts that overall routing accuracy meets or exceeds the production threshold (70.0%).
Excluded from default CI/local pytest runs; invoked explicitly with: pytest -m routing
"""

import pytest
from evals.routing.runner import RoutingEvalRunner


@pytest.mark.routing
@pytest.mark.asyncio
async def test_agent_routing_benchmark_accuracy():
    """Assert that LLM agent routing decision accuracy exceeds 70% threshold."""
    runner = RoutingEvalRunner()
    report = await runner.run_all()
    accuracy = report["summary"]["overall_accuracy_pct"]

    threshold = 70.0
    failed_cases = [c for c in report["cases"] if not c["passed"]]
    failure_details = "\n".join(
        f"  - [{c['case_id']}] ({c['expected_behavior']}): {c['failure_reason']}"
        for c in failed_cases
    )

    assert accuracy >= threshold, (
        f"Routing evaluation accuracy {accuracy}% fell below threshold {threshold}%.\n"
        f"Failed cases ({len(failed_cases)}/{report['summary']['total_cases']}):\n"
        f"{failure_details}"
    )
