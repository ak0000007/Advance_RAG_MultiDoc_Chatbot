"""
LangChain tools for Booking, TravelPackage, and Payment Apex REST APIs.

GET tools  — read-only, no approval needed.
UPDATE tools — require human approval before DML (same pattern as opportunity updates).
"""

import json
import re
from typing import Optional
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from src.write_guards import single_record_guard, bulk_intent_guard, session_write_cap

_BOOKING_DISPLAY = "booking"
_PACKAGE_DISPLAY = "travel package"
_PAYMENT_DISPLAY = "payment"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_booking(b: dict) -> str:
    parts = []
    if b.get("Id"):           parts.append(f"ID: {b['Id']}")
    if b.get("Name"):         parts.append(f"Booking: {b['Name']}")
    if b.get("Status__c"):    parts.append(f"Status: {b['Status__c']}")
    if b.get("Booking_Date__c"): parts.append(f"Date: {b['Booking_Date__c']}")
    if b.get("Destination__c"):  parts.append(f"Dest: {b['Destination__c']}")
    if b.get("Total_Amount__c"): parts.append(f"Total: ₹{b['Total_Amount__c']}")
    if b.get("Paid_Amount__c"):  parts.append(f"Paid: ₹{b['Paid_Amount__c']}")
    if b.get("Balance_Due__c"):  parts.append(f"Balance: ₹{b['Balance_Due__c']}")
    if b.get("Number_of_Travelers__c"): parts.append(f"Travelers: {b['Number_of_Travelers__c']}")
    if b.get("Travel_Start_Date__c"):   parts.append(f"Start: {b['Travel_Start_Date__c']}")
    if b.get("Travel_End_Date__c"):     parts.append(f"End: {b['Travel_End_Date__c']}")
    pkg = b.get("Travel_Package__r") or {}
    if pkg.get("Name"):        parts.append(f"Package: {pkg['Name']}")
    contact = b.get("Contact__r") or {}
    if contact.get("Name"):    parts.append(f"Contact: {contact['Name']}")
    return " | ".join(parts)


def _fmt_package(p: dict) -> str:
    parts = []
    if p.get("Id"):              parts.append(f"ID: {p['Id']}")
    if p.get("Name"):            parts.append(f"Package: {p['Name']}")
    if p.get("Package_Type__c"): parts.append(f"Type: {p['Package_Type__c']}")
    if p.get("Destination__r"): parts.append(f"Dest: {p['Destination__r'].get('Name', p.get('Destination__c', ''))}")
    if p.get("Base_Price__c"):  parts.append(f"Price: ₹{p['Base_Price__c']}")
    if p.get("Duration_Days__c"): parts.append(f"Days: {p['Duration_Days__c']}")
    if p.get("Max_Capacity__c"):  parts.append(f"Capacity: {p['Max_Capacity__c']}")
    if p.get("Difficulty_Level__c"): parts.append(f"Difficulty: {p['Difficulty_Level__c']}")
    if p.get("Rating__c"):        parts.append(f"Rating: {p['Rating__c']}")
    active = p.get("Is_Active__c")
    if active is not None:        parts.append(f"Active: {active}")
    return " | ".join(parts)


def _fmt_payment(p: dict) -> str:
    parts = []
    if p.get("Id"):           parts.append(f"ID: {p['Id']}")
    if p.get("Name"):         parts.append(f"Payment: {p['Name']}")
    if p.get("Status__c"):    parts.append(f"Status: {p['Status__c']}")
    if p.get("Amount__c"):    parts.append(f"Amount: ₹{p['Amount__c']}")
    if p.get("Payment_Date__c"): parts.append(f"Date: {p['Payment_Date__c']}")
    if p.get("Method__c"):    parts.append(f"Method: {p['Method__c']}")
    if p.get("Gateway__c"):   parts.append(f"Gateway: {p['Gateway__c']}")
    if p.get("Transaction_ID__c"): parts.append(f"TxnID: {p['Transaction_ID__c']}")
    if p.get("Is_Refund__c"): parts.append("REFUND")
    bk = p.get("Booking__r") or {}
    if bk.get("Name"):        parts.append(f"Booking: {bk['Name']}")
    return " | ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# GET TOOLS  (read-only, no approval)
# ═══════════════════════════════════════════════════════════════════════════════

def build_get_booking_tool(salesforce_client):
    """Read Booking__c record(s) by ID, name (BK-XXXXXX), or list all user bookings."""

    @tool
    async def get_booking(config: RunnableConfig, booking_id_or_name: str = "") -> str:
        """
        Retrieve Salesforce Booking records for the current user.

        - Call with NO arguments to fetch ALL user bookings automatically.
        - Call with booking_id_or_name (e.g. 'BK-000001') to get a specific booking.

        IMPORTANT: Never ask the user for a booking ID or number first.
        Always call this tool immediately; let the user choose from the results.
        Call this before any update_booking call to get the correct Salesforce ID.

        Args:
            booking_id_or_name: (optional) Booking number (e.g. 'BK-000001') or 18-char Salesforce ID.
                                Leave blank to fetch all bookings.
        """
        sf_username = config.get("configurable", {}).get("sf_username")
        if not sf_username:
            return "System Error: Missing sf_username in configuration."

        try:
            resp = await salesforce_client.get_booking(
                username=sf_username,
                booking_id_or_name=booking_id_or_name if booking_id_or_name else None,
            )
            if not resp.get("isSuccess"):
                return f"Salesforce Error: {resp.get('message')}"

            data = resp.get("data")
            if not data:
                return "No bookings found." if not booking_id_or_name else f"No booking found for '{booking_id_or_name}'."

            if isinstance(data, list):
                lines = [f"Found {len(data)} booking(s):"]
                lines += [_fmt_booking(b) for b in data]
                return "\n".join(lines)

            return f"Booking Details:\n{_fmt_booking(data)}"
        except Exception as e:
            return f"Failed to retrieve booking(s): {str(e)}"

    return get_booking


def build_get_travel_packages_tool(salesforce_client):
    """List all active travel packages, or get one by ID."""

    @tool
    async def get_travel_packages(config: RunnableConfig, package_id: str = "") -> str:
        """
        Retrieve Salesforce Travel Package records.
        Without a package_id, returns all active packages (up to 100).
        With a package_id, returns details for that specific package.

        Use when the user asks about available travel packages, destinations, prices, or itineraries.

        Args:
            package_id: (optional) 18-char Salesforce Travel_Package__c ID.
                        Leave blank to list all active packages.
        """
        sf_username = config.get("configurable", {}).get("sf_username")
        if not sf_username:
            return "System Error: Missing sf_username in configuration."

        try:
            resp = await salesforce_client.get_travel_packages(
                username=sf_username,
                package_id=package_id if package_id else None,
            )
            if not resp.get("isSuccess"):
                return f"Salesforce Error: {resp.get('message')}"

            data = resp.get("data")
            if not data:
                return "No travel packages found."

            if isinstance(data, list):
                lines = [f"Found {len(data)} active travel package(s):"]
                lines += [_fmt_package(p) for p in data]
                return "\n".join(lines)
            return f"Travel Package:\n{_fmt_package(data)}"
        except Exception as e:
            return f"Failed to retrieve travel packages: {str(e)}"

    return get_travel_packages


def build_get_payments_tool(salesforce_client):
    """Get payments for all user bookings, a specific booking, or a single payment by ID."""

    @tool
    async def get_payments(
        config: RunnableConfig,
        booking_id_or_name: str = "",
        payment_id: str = "",
    ) -> str:
        """
        Retrieve Salesforce Payment records for the current user.

        - Call with NO arguments to fetch ALL payments automatically.
        - Provide booking_id_or_name (e.g. 'BK-000001') to filter by booking.
        - Provide payment_id to fetch a specific payment.

        IMPORTANT: Never ask the user for a payment ID or booking ID first.
        Always call this tool immediately; let the user choose from the results.
        Call this before any update_payment call to get the correct Salesforce ID.

        Args:
            booking_id_or_name: (optional) Booking number or ID to filter payments.
            payment_id: (optional) 18-char Salesforce Payment__c ID.
        """
        sf_username = config.get("configurable", {}).get("sf_username")
        if not sf_username:
            return "System Error: Missing sf_username in configuration."

        try:
            resp = await salesforce_client.get_payments(
                username=sf_username,
                booking_id=booking_id_or_name if booking_id_or_name else None,
                payment_id=payment_id if payment_id else None,
            )
            if not resp.get("isSuccess"):
                return f"Salesforce Error: {resp.get('message')}"

            data = resp.get("data")
            if not data:
                return "No payments found."

            if isinstance(data, list):
                lines = [f"Found {len(data)} payment(s):"]
                lines += [_fmt_payment(p) for p in data]
                return "\n".join(lines)
            return f"Payment Details:\n{_fmt_payment(data)}"
        except Exception as e:
            return f"Failed to retrieve payments: {str(e)}"

    return get_payments


# ═══════════════════════════════════════════════════════════════════════════════
# UPDATE TOOLS  (require human approval before DML)
# ═══════════════════════════════════════════════════════════════════════════════

def build_update_booking_tool(salesforce_client, max_writes_per_session: int = 5):
    """
    Update a Booking__c record. Requires human approval before DML.
    Only non-empty fields are applied (partial update).
    """

    @tool
    async def update_booking(
        booking_id: str,
        config: RunnableConfig,
        booking_name: str = "",
        status: str = "",
        special_requests: str = "",
        purpose_of_visit: str = "",
        internal_notes: str = "",
        travel_start_date: str = "",
        travel_end_date: str = "",
        cancellation_reason: str = "",
        cancellation_types: str = "",
        number_of_travelers: Optional[int] = None,
    ) -> str:
        """
        Request a Salesforce Booking record update. The system will pause for human approval.

        PRE-CONDITION: Always call get_booking first to get the 18-char Salesforce ID.
        Never guess or fabricate a booking_id. If multiple bookings exist, confirm
        which specific one the user wants to update before calling this tool.
        Only accepts a single Booking ID. All fields except booking_id are optional.

        Args:
            booking_id: 18-char Salesforce Booking__c ID (starts with a00).
            booking_name: Human-readable booking number (e.g. 'BK-000001'). Optional but preferred.
            status: New status (e.g. 'Confirmed', 'Cancelled', 'Completed').
            special_requests: Special requirements for the trip.
            purpose_of_visit: e.g. 'Leisure', 'Business', 'Honeymoon'.
            internal_notes: Internal agent notes on the booking.
            travel_start_date: ISO date YYYY-MM-DD.
            travel_end_date: ISO date YYYY-MM-DD.
            cancellation_reason: Reason if cancelling.
            cancellation_types: Cancellation type/category.
            number_of_travelers: Number of travelers (integer).
        """
        sf_username = config.get("configurable", {}).get("sf_username")
        write_count: int = config.get("configurable", {}).get("write_count", 0)

        if not sf_username:
            return "System Error: Missing sf_username in configuration."

        if err := single_record_guard(booking_id, expected_prefix=None):
            return err
        if status and (err := bulk_intent_guard(status)):
            return err
        if err := session_write_cap(write_count, max_writes_per_session):
            return err

        # Build only the fields explicitly provided
        update_fields: dict = {}
        if status:               update_fields["status"] = status
        if special_requests:     update_fields["specialRequests"] = special_requests
        if purpose_of_visit:     update_fields["purposeOfVisit"] = purpose_of_visit
        if internal_notes:       update_fields["internalNotes"] = internal_notes
        if travel_start_date:    update_fields["travelStartDate"] = travel_start_date
        if travel_end_date:      update_fields["travelEndDate"] = travel_end_date
        if cancellation_reason:  update_fields["cancellationReason"] = cancellation_reason
        if cancellation_types:   update_fields["cancellationTypes"] = cancellation_types
        if number_of_travelers is not None:
            update_fields["numberOfTravelers"] = number_of_travelers

        if not update_fields:
            return "No fields to update. Please provide at least one field to change."

        display_name = booking_name.strip() if booking_name.strip() else booking_id
        changes = ", ".join(f"{k}='{v}'" for k, v in update_fields.items())

        return json.dumps({
            "__requires_approval__": True,
            "action_type": "booking_update",
            "sf_username": sf_username,
            "record_id": booking_id,
            "record_name": display_name,
            "update_fields": update_fields,
            "message": f"Update booking '{display_name}': {changes}?",
        })

    return update_booking


def build_update_travel_package_tool(salesforce_client, max_writes_per_session: int = 5):
    """
    Update a Travel_Package__c record. Requires human approval before DML.
    """

    @tool
    async def update_travel_package(
        package_id: str,
        config: RunnableConfig,
        package_name: str = "",
        name: str = "",
        base_price: Optional[float] = None,
        max_capacity: Optional[int] = None,
        duration_days: Optional[int] = None,
        difficulty_level: str = "",
        minimum_age: Optional[int] = None,
        package_type: str = "",
        inclusions: str = "",
        is_active: Optional[bool] = None,
        rating: Optional[float] = None,
    ) -> str:
        """
        Request a Salesforce Travel Package update. The system will pause for human approval.

        PRE-CONDITION: Always call get_travel_packages first to get the 18-char Salesforce ID.
        Never guess or fabricate a package_id. If multiple packages exist, confirm
        which specific one the user wants to update before calling this tool.
        Only accepts a single Travel_Package__c ID.

        Args:
            package_id: 18-char Salesforce Travel_Package__c ID.
            package_name: Human-readable package name for display. Optional.
            name: New name for the package.
            base_price: New base price (decimal).
            max_capacity: Maximum number of travelers.
            duration_days: Duration of the package in days.
            difficulty_level: e.g. 'Easy', 'Moderate', 'Hard'.
            minimum_age: Minimum age requirement (integer).
            package_type: e.g. 'Adventure', 'Cultural', 'Beach'.
            inclusions: What is included (meals, transport, etc.).
            is_active: True to activate, False to deactivate.
            rating: Package rating (decimal, e.g. 4.5).
        """
        sf_username = config.get("configurable", {}).get("sf_username")
        write_count: int = config.get("configurable", {}).get("write_count", 0)

        if not sf_username:
            return "System Error: Missing sf_username in configuration."

        if err := single_record_guard(package_id, expected_prefix=None):
            return err
        if err := session_write_cap(write_count, max_writes_per_session):
            return err

        update_fields: dict = {}
        if name:               update_fields["name"] = name
        if base_price is not None:   update_fields["basePrice"] = base_price
        if max_capacity is not None: update_fields["maxCapacity"] = max_capacity
        if duration_days is not None: update_fields["durationDays"] = duration_days
        if difficulty_level:   update_fields["difficultyLevel"] = difficulty_level
        if minimum_age is not None:  update_fields["minimumAge"] = minimum_age
        if package_type:       update_fields["packageType"] = package_type
        if inclusions:         update_fields["inclusions"] = inclusions
        if is_active is not None:    update_fields["isActive"] = is_active
        if rating is not None:       update_fields["rating"] = rating

        if not update_fields:
            return "No fields to update. Please provide at least one field to change."

        display_name = package_name.strip() if package_name.strip() else package_id
        changes = ", ".join(f"{k}='{v}'" for k, v in update_fields.items())

        return json.dumps({
            "__requires_approval__": True,
            "action_type": "travel_package_update",
            "sf_username": sf_username,
            "record_id": package_id,
            "record_name": display_name,
            "update_fields": update_fields,
            "message": f"Update travel package '{display_name}': {changes}?",
        })

    return update_travel_package


def build_update_payment_tool(salesforce_client, max_writes_per_session: int = 5):
    """
    Update a Payment__c record. Requires human approval before DML.
    """

    @tool
    async def update_payment(
        payment_id: str,
        config: RunnableConfig,
        payment_name: str = "",
        status: str = "",
        amount: Optional[float] = None,
        payment_date: str = "",
        method: str = "",
        gateway: str = "",
        transaction_id: str = "",
        receipt_number: str = "",
        failure_reason: str = "",
        is_refund: Optional[bool] = None,
        refund_reference: str = "",
    ) -> str:
        """
        Request a Salesforce Payment record update. The system will pause for human approval.

        PRE-CONDITION: Always call get_payments first to get the 18-char Salesforce ID.
        Never guess or fabricate a payment_id. If multiple payments exist, confirm
        which specific one the user wants to update before calling this tool.
        Only accepts a single Payment__c ID.

        Args:
            payment_id: 18-char Salesforce Payment__c ID.
            payment_name: Human-readable payment name (e.g. 'PAY-000001'). Optional.
            status: New status (e.g. 'Completed', 'Failed', 'Pending').
            amount: Payment amount (decimal).
            payment_date: ISO date YYYY-MM-DD.
            method: Payment method (e.g. 'Credit Card', 'UPI', 'Bank Transfer').
            gateway: Payment gateway (e.g. 'Stripe', 'Razorpay').
            transaction_id: Gateway transaction ID.
            receipt_number: Receipt number.
            failure_reason: Reason for failure (if applicable).
            is_refund: True if this is a refund transaction.
            refund_reference: Refund reference number.
        """
        sf_username = config.get("configurable", {}).get("sf_username")
        write_count: int = config.get("configurable", {}).get("write_count", 0)

        if not sf_username:
            return "System Error: Missing sf_username in configuration."

        if err := single_record_guard(payment_id, expected_prefix=None):
            return err
        if status and (err := bulk_intent_guard(status)):
            return err
        if err := session_write_cap(write_count, max_writes_per_session):
            return err

        update_fields: dict = {}
        if status:            update_fields["status"] = status
        if amount is not None:       update_fields["amount"] = amount
        if payment_date:      update_fields["paymentDate"] = payment_date
        if method:            update_fields["method"] = method
        if gateway:           update_fields["gateway"] = gateway
        if transaction_id:    update_fields["transactionId"] = transaction_id
        if receipt_number:    update_fields["receiptNumber"] = receipt_number
        if failure_reason:    update_fields["failureReason"] = failure_reason
        if is_refund is not None:    update_fields["isRefund"] = is_refund
        if refund_reference:  update_fields["refundReference"] = refund_reference

        if not update_fields:
            return "No fields to update. Please provide at least one field to change."

        display_name = payment_name.strip() if payment_name.strip() else payment_id
        changes = ", ".join(f"{k}='{v}'" for k, v in update_fields.items())

        return json.dumps({
            "__requires_approval__": True,
            "action_type": "payment_update",
            "sf_username": sf_username,
            "record_id": payment_id,
            "record_name": display_name,
            "update_fields": update_fields,
            "message": f"Update payment '{display_name}': {changes}?",
        })

    return update_payment
