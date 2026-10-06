"""
Anthropic/OpenAI-Grade Evaluation Harness Runner.

Executes deterministic benchmark tests against the mock Salesforce client,
measures accuracy, safety guard adherence, and HITL execution metrics,
and outputs a structured scorecard and JSON benchmark report.
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from evals.mock_salesforce import MockSalesforceAsyncClient
from evals.golden_dataset import GOLDEN_TEST_CASES, EvalTestCase
from src.tools.salesforce_tools import (
    build_salesforce_opportunities_tool,
    build_salesforce_update_tool,
)
from src.tools.travel_tools import (
    build_get_booking_tool,
    build_update_booking_tool,
    build_get_travel_packages_tool,
    build_update_travel_package_tool,
    build_get_payments_tool,
    build_update_payment_tool,
)
from src.graph.approval import build_human_approval_node


class EvalHarnessRunner:
    def __init__(self, username: str = "eval_agent@travelcorp.com"):
        self.username = username
        self.mock_sf = MockSalesforceAsyncClient()
        self._init_tools()

    def _init_tools(self) -> None:
        self.tools = {
            "search_salesforce_opportunities": build_salesforce_opportunities_tool(self.mock_sf),
            "update_salesforce_opportunity_status": build_salesforce_update_tool(self.mock_sf, max_writes_per_session=5),
            "get_booking": build_get_booking_tool(self.mock_sf),
            "update_booking": build_update_booking_tool(self.mock_sf, max_writes_per_session=5),
            "get_travel_packages": build_get_travel_packages_tool(self.mock_sf),
            "update_travel_package": build_update_travel_package_tool(self.mock_sf, max_writes_per_session=5),
            "get_payments": build_get_payments_tool(self.mock_sf),
            "update_payment": build_update_payment_tool(self.mock_sf, max_writes_per_session=5),
        }
        self.approval_node = build_human_approval_node(self.mock_sf)

    async def execute_case(self, case: EvalTestCase) -> dict[str, Any]:
        """Execute a single eval test case and return detailed results."""
        self.mock_sf.reset()
        tool = self.tools[case.tool_name]
        config: RunnableConfig = {
            "configurable": {
                "sf_username": self.username,
                "write_count": 0,
                "thread_id": case.case_id,
            }
        }

        start_time = time.perf_counter()
        passed = False
        failure_reason = ""
        output_str = ""

        try:
            # 1. Invoke tool
            output_str = await tool.ainvoke(case.input_kwargs, config=config)

            # 2. Check approval requirement
            if case.requires_approval:
                try:
                    payload = json.loads(output_str)
                    if not (isinstance(payload, dict) and payload.get("__requires_approval__")):
                        failure_reason = f"Expected __requires_approval__: True, got: {output_str}"
                    else:
                        # 3. Simulate Human Approval Node execution
                        tool_msg = ToolMessage(
                            content=output_str,
                            tool_call_id=f"call_{case.case_id}",
                            id=f"msg_{case.case_id}",
                        )
                        state_input = {"messages": [tool_msg], "write_count": 0}

                        # Mock interrupt to return the golden case decision
                        from unittest.mock import patch
                        with patch("src.graph.approval.interrupt", return_value=case.decision):
                            node_result = await self.approval_node(state_input, config=config)

                        # Check decision outcome
                        res_msg = node_result.get("messages", [tool_msg])[-1].content
                        if case.decision == "Approve" and "Success:" not in res_msg:
                            failure_reason = f"Approved update failed: {res_msg}"
                        elif case.decision == "Reject" and "Action aborted:" not in res_msg:
                            failure_reason = f"Rejected update did not abort: {res_msg}"
                        elif case.verification_check and not case.verification_check(self.mock_sf):
                            failure_reason = "Verification check failed on Mock Salesforce state."
                        else:
                            passed = True
                except Exception as exc:
                    failure_reason = f"Approval lifecycle exception: {str(exc)}"

            else:
                # 4. Standard (non-approval) verification
                if case.expected_to_succeed:
                    if case.expected_output_substr and case.expected_output_substr not in output_str:
                        failure_reason = f"Missing expected substr '{case.expected_output_substr}' in '{output_str[:120]}'"
                    else:
                        passed = True
                else:
                    if case.expected_error_substr and case.expected_error_substr not in output_str:
                        failure_reason = f"Expected error '{case.expected_error_substr}' not found in '{output_str[:120]}'"
                    else:
                        passed = True

        except Exception as e:
            failure_reason = f"Execution exception: {str(e)}"

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "case_id": case.case_id,
            "category": case.category,
            "description": case.description,
            "tool": case.tool_name,
            "passed": passed,
            "latency_ms": round(elapsed_ms, 2),
            "failure_reason": failure_reason,
            "output_preview": output_str[:100] if output_str else "",
        }

    async def run_all(self) -> dict[str, Any]:
        """Run full evaluation suite and compile performance scorecard."""
        results = []
        for case in GOLDEN_TEST_CASES:
            res = await self.execute_case(case)
            results.append(res)

        total = len(results)
        passed_count = sum(1 for r in results if r["passed"])
        accuracy_pct = (passed_count / total * 100.0) if total else 0.0
        avg_latency = sum(r["latency_ms"] for r in results) / total if total else 0.0

        # Category breakdown
        categories: dict[str, dict[str, Any]] = {}
        for r in results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = {"total": 0, "passed": 0}
            categories[cat]["total"] += 1
            if r["passed"]:
                categories[cat]["passed"] += 1

        category_scores = {
            cat: round((data["passed"] / data["total"]) * 100.0, 1)
            for cat, data in categories.items()
        }

        report = {
            "summary": {
                "total_cases": total,
                "passed": passed_count,
                "failed": total - passed_count,
                "overall_accuracy_pct": round(accuracy_pct, 2),
                "avg_latency_ms": round(avg_latency, 2),
                "category_scores": category_scores,
            },
            "cases": results,
        }

        return report


def print_scorecard(report: dict[str, Any]) -> None:
    """Print an Anthropic/OpenAI grade evaluation scorecard in the console."""
    summary = report["summary"]
    print("\n" + "=" * 80)
    print(" 🚀 ADVANCED RAG & CRM COPILOT — EVALUATION BENCHMARK SCORECARD")
    print("=" * 80)
    print(f" TOTAL EVAL CASES   : {summary['total_cases']}")
    print(f" PASSED             : {summary['passed']}")
    print(f" FAILED             : {summary['failed']}")
    print(f" OVERALL ACCURACY   : \033[1;32m{summary['overall_accuracy_pct']}%\033[0m" if summary['overall_accuracy_pct'] == 100 else f" OVERALL ACCURACY   : {summary['overall_accuracy_pct']}%")
    print(f" AVERAGE LATENCY    : {summary['avg_latency_ms']} ms")
    print("-" * 80)
    print(" CATEGORY ACCURACY BREAKDOWN:")
    for cat, score in summary["category_scores"].items():
        bar = "█" * int(score // 10) + "░" * (10 - int(score // 10))
        print(f"  • {cat:<18}: [{bar}] {score:>5.1f}%")
    print("-" * 80)
    print(" INDIVIDUAL TEST RUNS:")
    for r in report["cases"]:
        status_sym = "\033[32m✔ PASS\033[0m" if r["passed"] else "\033[31m✖ FAIL\033[0m"
        print(f"  [{r['case_id']}] {status_sym} | {r['tool']:<36} | {r['latency_ms']:>6.2f}ms | {r['description']}")
        if not r["passed"]:
            print(f"        ↳ Reason: {r['failure_reason']}")
    print("=" * 80 + "\n")


async def main():
    runner = EvalHarnessRunner()
    report = await runner.run_all()
    print_scorecard(report)

    # Save JSON report
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "eval_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"📄 Detailed evaluation report written to: {report_file}")


if __name__ == "__main__":
    asyncio.run(main())
