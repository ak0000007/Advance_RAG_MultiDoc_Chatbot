"""
Golden Evaluation Benchmark Dataset for Travel RAG & CRM Copilot.

Covers:
1. Opportunity tools (search, update stage)
2. Travel Booking tools (fetch all, lookup by ID/name, update with approval)
3. Travel Package tools (list all, lookup by ID, price update)
4. Payment tools (list all, lookup by booking, status update)
5. Safety Guards (single-record enforcement, bulk intent protection, session write caps)
6. HITL lifecycle (approval interrupt triggering, Approve execution, Reject abortion)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class EvalTestCase:
    case_id: str
    category: str
    description: str
    tool_name: str
    input_kwargs: dict[str, Any]
    requires_approval: bool = False
    decision: Optional[str] = None  # "Approve" or "Reject" if requires_approval
    expected_to_succeed: bool = True
    expected_error_substr: Optional[str] = None
    expected_output_substr: Optional[str] = None
    verification_check: Optional[Callable[[Any], bool]] = None


GOLDEN_TEST_CASES: list[EvalTestCase] = [
    # ── Category 1: Opportunity Operations ────────────────────────────────────
    EvalTestCase(
        case_id="OPP-01",
        category="Opportunity",
        description="Search all recent opportunities with empty query",
        tool_name="search_salesforce_opportunities",
        input_kwargs={"search": ""},
        requires_approval=False,
        expected_output_substr="Found 2 opportunity(ies)",
    ),
    EvalTestCase(
        case_id="OPP-02",
        category="Opportunity",
        description="Filter opportunities by company keyword 'Acme'",
        tool_name="search_salesforce_opportunities",
        input_kwargs={"search": "Acme"},
        requires_approval=False,
        expected_output_substr="Acme Global Retreat",
    ),
    EvalTestCase(
        case_id="OPP-03",
        category="Opportunity",
        description="Request valid Opportunity stage update to 'Closed Won' (HITL)",
        tool_name="update_salesforce_opportunity_status",
        input_kwargs={
            "opportunity_id": "0065g000001AAAAAA1",
            "new_status": "Closed Won",
            "opportunity_name": "Acme Global Retreat",
        },
        requires_approval=True,
        decision="Approve",
        expected_to_succeed=True,
        verification_check=lambda mock_sf: any(
            opp["id"] == "0065g000001AAAAAA1" and opp["stageName"] == "Closed Won"
            for opp in mock_sf.opportunities
        ),
    ),
    EvalTestCase(
        case_id="OPP-04",
        category="Safety Guards",
        description="Block bulk opportunity update attempt ('Close all deals')",
        tool_name="update_salesforce_opportunity_status",
        input_kwargs={
            "opportunity_id": "0065g000001AAAAAA1",
            "new_status": "Closed Won for all deals",
        },
        requires_approval=False,
        expected_to_succeed=False,
        expected_error_substr="Hard cap: bulk operations are not allowed",
    ),

    # ── Category 2: Travel Booking Operations ─────────────────────────────────
    EvalTestCase(
        case_id="BKG-01",
        category="Booking",
        description="Retrieve all user bookings without parameters",
        tool_name="get_booking",
        input_kwargs={"booking_id_or_name": ""},
        requires_approval=False,
        expected_output_substr="Found 2 booking(s)",
    ),
    EvalTestCase(
        case_id="BKG-02",
        category="Booking",
        description="Retrieve booking by human-readable booking number BK-000001",
        tool_name="get_booking",
        input_kwargs={"booking_id_or_name": "BK-000001"},
        requires_approval=False,
        expected_output_substr="Booking: BK-000001",
    ),
    EvalTestCase(
        case_id="BKG-03",
        category="Booking",
        description="Update booking status to Confirmed with Human Approval (Approved)",
        tool_name="update_booking",
        input_kwargs={
            "booking_id": "a005g000001CCCCCC1",
            "booking_name": "BK-000001",
            "status": "Confirmed",
        },
        requires_approval=True,
        decision="Approve",
        expected_to_succeed=True,
        verification_check=lambda mock_sf: any(
            b["Id"] == "a005g000001CCCCCC1" and b["Status__c"] == "Confirmed"
            for b in mock_sf.bookings
        ),
    ),
    EvalTestCase(
        case_id="BKG-04",
        category="Booking",
        description="Reject booking cancellation request via HITL (Status preserved)",
        tool_name="update_booking",
        input_kwargs={
            "booking_id": "a005g000002DDDDDD1",
            "booking_name": "BK-000002",
            "status": "Cancelled",
        },
        requires_approval=True,
        decision="Reject",
        expected_to_succeed=True,
        verification_check=lambda mock_sf: any(
            b["Id"] == "a005g000002DDDDDD1" and b["Status__c"] == "Confirmed"
            for b in mock_sf.bookings
        ),
    ),

    # ── Category 3: Travel Package Operations ─────────────────────────────────
    EvalTestCase(
        case_id="PKG-01",
        category="Package",
        description="List all active travel packages",
        tool_name="get_travel_packages",
        input_kwargs={"package_id": ""},
        requires_approval=False,
        expected_output_substr="Bali Serenity Package",
    ),
    EvalTestCase(
        case_id="PKG-02",
        category="Package",
        description="Get specific package details by 18-char ID",
        tool_name="get_travel_packages",
        input_kwargs={"package_id": "a015g000002FFFFFF1"},
        requires_approval=False,
        expected_output_substr="Swiss Alpine Expedition",
    ),
    EvalTestCase(
        case_id="PKG-03",
        category="Package",
        description="Update travel package base price with Approval (Approved)",
        tool_name="update_travel_package",
        input_kwargs={
            "package_id": "a015g000001EEEEEE1",
            "package_name": "Bali Serenity Package",
            "base_price": 1699.0,
        },
        requires_approval=True,
        decision="Approve",
        expected_to_succeed=True,
        verification_check=lambda mock_sf: any(
            p["Id"] == "a015g000001EEEEEE1" and p["Base_Price__c"] == 1699.0
            for p in mock_sf.packages
        ),
    ),

    # ── Category 4: Payment Operations ────────────────────────────────────────
    EvalTestCase(
        case_id="PAY-01",
        category="Payment",
        description="Retrieve all transaction records",
        tool_name="get_payments",
        input_kwargs={},
        requires_approval=False,
        expected_output_substr="Found 2 payment(s)",
    ),
    EvalTestCase(
        case_id="PAY-02",
        category="Payment",
        description="Filter payments by booking reference number BK-000001",
        tool_name="get_payments",
        input_kwargs={"booking_id_or_name": "BK-000001"},
        requires_approval=False,
        expected_output_substr="PAY-000001",
    ),
    EvalTestCase(
        case_id="PAY-03",
        category="Payment",
        description="Update payment status to Completed with Human Approval (Approved)",
        tool_name="update_payment",
        input_kwargs={
            "payment_id": "a025g000001GGGGGG1",
            "payment_name": "PAY-000001",
            "status": "Completed",
        },
        requires_approval=True,
        decision="Approve",
        expected_to_succeed=True,
        verification_check=lambda mock_sf: any(
            pay["Id"] == "a025g000001GGGGGG1" and pay["Status__c"] == "Completed"
            for pay in mock_sf.payments
        ),
    ),

    # ── Category 5: Safety Guardrails & Validation ────────────────────────────
    EvalTestCase(
        case_id="SFT-01",
        category="Safety Guards",
        description="Enforce single-record lock: Reject comma-separated IDs",
        tool_name="update_booking",
        input_kwargs={
            "booking_id": "a005g000001CCCCCC1, a005g000002DDDDDD1",
            "status": "Confirmed",
        },
        requires_approval=False,
        expected_to_succeed=False,
        expected_error_substr="Hard cap: this tool updates exactly ONE record per call",
    ),
    EvalTestCase(
        case_id="SFT-02",
        category="Safety Guards",
        description="Prevent benign substring false positive: 'Small group' note passes",
        tool_name="update_booking",
        input_kwargs={
            "booking_id": "a005g000001CCCCCC1",
            "special_requests": "Small group tour",
        },
        requires_approval=True,
        decision="Approve",
        expected_to_succeed=True,
    ),
]
