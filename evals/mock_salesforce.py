"""
Mock Salesforce Async Client for Deterministic Evaluation & Testing.

Provides an in-memory, stateful replica of Salesforce Apex REST endpoints
matching actual Apex REST schema definitions and custom field names (__c).
"""

import copy
import re
from typing import Any, Optional


_DEFAULT_OPPORTUNITIES = [
    {
        "id": "0065g000001AAAAAA1",
        "name": "Acme Global Retreat",
        "stageName": "Prospecting",
        "amount": 75000.0,
        "closeDate": "2026-12-01",
    },
    {
        "id": "0065g000002BBBBBB1",
        "name": "Stark Tech Summit",
        "stageName": "Negotiation/Review",
        "amount": 120000.0,
        "closeDate": "2026-11-15",
    },
]

_DEFAULT_BOOKINGS = [
    {
        "Id": "a005g000001CCCCCC1",
        "id": "a005g000001CCCCCC1",
        "Name": "BK-000001",
        "name": "BK-000001",
        "Status__c": "Pending",
        "status": "Pending",
        "Number_of_Travelers__c": 2,
        "numberOfTravelers": 2,
        "Destination__c": "Bali, Indonesia",
        "purposeOfVisit": "Leisure",
        "specialRequests": "Ocean view room",
        "internalNotes": "VIP customer",
        "Travel_Start_Date__c": "2026-11-01",
        "Travel_End_Date__c": "2026-11-08",
    },
    {
        "Id": "a005g000002DDDDDD1",
        "id": "a005g000002DDDDDD1",
        "Name": "BK-000002",
        "name": "BK-000002",
        "Status__c": "Confirmed",
        "status": "Confirmed",
        "Number_of_Travelers__c": 4,
        "numberOfTravelers": 4,
        "Destination__c": "Swiss Alps",
        "purposeOfVisit": "Corporate",
        "specialRequests": "Ski passes included",
        "internalNotes": "Group booking",
        "Travel_Start_Date__c": "2026-12-10",
        "Travel_End_Date__c": "2026-12-18",
    },
]

_DEFAULT_PACKAGES = [
    {
        "Id": "a015g000001EEEEEE1",
        "id": "a015g000001EEEEEE1",
        "Name": "Bali Serenity Package",
        "name": "Bali Serenity Package",
        "Base_Price__c": 1499.0,
        "basePrice": 1499.0,
        "Duration_Days__c": 7,
        "durationDays": 7,
        "Max_Capacity__c": 20,
        "maxCapacity": 20,
        "Is_Active__c": True,
    },
    {
        "Id": "a015g000002FFFFFF1",
        "id": "a015g000002FFFFFF1",
        "Name": "Swiss Alpine Expedition",
        "name": "Swiss Alpine Expedition",
        "Base_Price__c": 2899.0,
        "basePrice": 2899.0,
        "Duration_Days__c": 10,
        "durationDays": 10,
        "Max_Capacity__c": 15,
        "maxCapacity": 15,
        "Is_Active__c": True,
    },
]

_DEFAULT_PAYMENTS = [
    {
        "Id": "a025g000001GGGGGG1",
        "id": "a025g000001GGGGGG1",
        "Name": "PAY-000001",
        "name": "PAY-000001",
        "Booking__c": "a005g000001CCCCCC1",
        "bookingId": "a005g000001CCCCCC1",
        "Amount__c": 2998.0,
        "amount": 2998.0,
        "Status__c": "Pending",
        "status": "Pending",
        "Method__c": "Credit Card",
        "Booking__r": {"Name": "BK-000001"},
    },
    {
        "Id": "a025g000002HHHHHH1",
        "id": "a025g000002HHHHHH1",
        "Name": "PAY-000002",
        "name": "PAY-000002",
        "Booking__c": "a005g000002DDDDDD1",
        "bookingId": "a005g000002DDDDDD1",
        "Amount__c": 11596.0,
        "amount": 11596.0,
        "Status__c": "Completed",
        "status": "Completed",
        "Method__c": "Wire Transfer",
        "Booking__r": {"Name": "BK-000002"},
    },
]


class MockSalesforceAsyncClient:
    """
    In-memory stateful Salesforce client replica.
    Adheres strictly to SalesforceAsyncClient public interface.
    """

    def __init__(self, domain: str = "https://mock.salesforce.com"):
        self.domain = domain
        self.call_history: list[dict[str, Any]] = []
        self.mutation_history: list[dict[str, Any]] = []
        self.reset()

    def reset(self) -> None:
        """Reset in-memory database to initial state and clear telemetry."""
        self.opportunities: list[dict[str, Any]] = copy.deepcopy(_DEFAULT_OPPORTUNITIES)
        self.bookings: list[dict[str, Any]] = copy.deepcopy(_DEFAULT_BOOKINGS)
        self.packages: list[dict[str, Any]] = copy.deepcopy(_DEFAULT_PACKAGES)
        self.payments: list[dict[str, Any]] = copy.deepcopy(_DEFAULT_PAYMENTS)
        self.call_history.clear()
        self.mutation_history.clear()

    async def get_access_token(self, username: str) -> str:
        self.call_history.append({"method": "get_access_token", "username": username})
        return f"mock_token_{username}"

    # ── Opportunity Operations ────────────────────────────────────────────────

    async def get_opportunities(self, username: str, search: Optional[str] = None) -> dict[str, Any]:
        self.call_history.append({"method": "get_opportunities", "username": username, "search": search})
        if not search or not search.strip():
            return {"success": True, "data": list(self.opportunities)}

        term = search.strip().lower()
        filtered = [
            opp for opp in self.opportunities
            if term in opp["name"].lower() or term in opp["id"].lower()
        ]
        return {"success": True, "data": filtered}

    async def update_opportunity_status(
        self, username: str, opportunity_id: str, new_status: str, idempotency_key: Optional[str] = None
    ) -> dict[str, Any]:
        call_info = {
            "method": "update_opportunity_status",
            "username": username,
            "opportunity_id": opportunity_id,
            "new_status": new_status,
            "idempotency_key": idempotency_key,
        }
        self.call_history.append(call_info)
        self.mutation_history.append(call_info)

        for opp in self.opportunities:
            if opp["id"] == opportunity_id:
                opp["stageName"] = new_status
                return {
                    "isSuccess": True,
                    "statusCode": 200,
                    "message": f"Updated Opportunity '{opp['name']}' to '{new_status}'.",
                    "data": dict(opp),
                }

        return {
            "isSuccess": False,
            "statusCode": 404,
            "message": f"Opportunity with ID '{opportunity_id}' not found.",
        }

    # ── Internal SOQL ID Resolver ───────────────────────────────────────────

    async def _resolve_id_by_name(self, token: str, sobject: str, name: str) -> Optional[str]:
        self.call_history.append({
            "method": "_resolve_id_by_name",
            "sobject": sobject,
            "name": name,
        })
        clean_name = re.sub(r'[\x00-\x1f\x7f]', '', name).strip().lower()

        if sobject in ("Booking__c", "Travel_Booking__c"):
            for b in self.bookings:
                if b["Name"].lower() == clean_name:
                    return b["Id"]
        elif sobject == "Travel_Package__c":
            for p in self.packages:
                if p["Name"].lower() == clean_name:
                    return p["Id"]
        elif sobject == "Payment__c":
            for pay in self.payments:
                if pay["Name"].lower() == clean_name:
                    return pay["Id"]
        elif sobject == "Opportunity":
            for opp in self.opportunities:
                if opp["name"].lower() == clean_name:
                    return opp["id"]
        return None

    # ── Booking Operations ────────────────────────────────────────────────────

    async def get_booking(self, username: str, booking_id_or_name: Optional[str] = None) -> dict[str, Any]:
        self.call_history.append({"method": "get_booking", "username": username, "target": booking_id_or_name})
        if not booking_id_or_name or not booking_id_or_name.strip():
            return {"isSuccess": True, "message": "Retrieved all bookings.", "data": list(self.bookings)}

        target = booking_id_or_name.strip()
        for b in self.bookings:
            if b["Id"] == target or b["Name"].lower() == target.lower():
                return {"isSuccess": True, "message": "Booking found.", "data": dict(b)}

        return {"isSuccess": False, "message": f"No booking found matching '{target}'.", "data": None}

    async def update_booking(
        self,
        username: str,
        booking_id: str,
        update_fields: dict[str, Any],
        idempotency_key: Optional[str] = None,
    ) -> dict[str, Any]:
        call_info = {
            "method": "update_booking",
            "username": username,
            "booking_id": booking_id,
            "update_fields": update_fields,
            "idempotency_key": idempotency_key,
        }
        self.call_history.append(call_info)
        self.mutation_history.append(call_info)

        for b in self.bookings:
            if b["Id"] == booking_id:
                if "status" in update_fields:
                    b["Status__c"] = update_fields["status"]
                    b["status"] = update_fields["status"]
                b.update(update_fields)
                return {
                    "isSuccess": True,
                    "statusCode": 200,
                    "message": f"Booking '{b['Name']}' updated successfully.",
                    "data": dict(b),
                }

        return {
            "isSuccess": False,
            "statusCode": 404,
            "message": f"Booking ID '{booking_id}' not found.",
        }

    # ── Travel Package Operations ─────────────────────────────────────────────

    async def get_travel_packages(self, username: str, package_id: Optional[str] = None) -> dict[str, Any]:
        self.call_history.append({"method": "get_travel_packages", "username": username, "package_id": package_id})
        if not package_id or not package_id.strip():
            return {"isSuccess": True, "message": "Retrieved packages.", "data": list(self.packages)}

        pid = package_id.strip()
        for p in self.packages:
            if p["Id"] == pid:
                return {"isSuccess": True, "message": "Package found.", "data": dict(p)}

        return {"isSuccess": False, "message": f"Package ID '{pid}' not found.", "data": None}

    async def update_travel_package(
        self,
        username: str,
        package_id: str,
        update_fields: dict[str, Any],
        idempotency_key: Optional[str] = None,
    ) -> dict[str, Any]:
        call_info = {
            "method": "update_travel_package",
            "username": username,
            "package_id": package_id,
            "update_fields": update_fields,
            "idempotency_key": idempotency_key,
        }
        self.call_history.append(call_info)
        self.mutation_history.append(call_info)

        for p in self.packages:
            if p["Id"] == package_id:
                if "basePrice" in update_fields:
                    p["Base_Price__c"] = update_fields["basePrice"]
                p.update(update_fields)
                return {
                    "isSuccess": True,
                    "statusCode": 200,
                    "message": f"Package '{p['Name']}' updated successfully.",
                    "data": dict(p),
                }

        return {
            "isSuccess": False,
            "statusCode": 404,
            "message": f"Package ID '{package_id}' not found.",
        }

    # ── Payment Operations ────────────────────────────────────────────────────

    async def get_payments(
        self,
        username: str,
        booking_id: Optional[str] = None,
        payment_id: Optional[str] = None,
    ) -> dict[str, Any]:
        self.call_history.append({
            "method": "get_payments",
            "username": username,
            "booking_id": booking_id,
            "payment_id": payment_id,
        })

        if payment_id and payment_id.strip():
            pid = payment_id.strip()
            for pay in self.payments:
                if pay["Id"] == pid or pay["Name"].lower() == pid.lower():
                    return {"isSuccess": True, "message": "Payment found.", "data": dict(pay)}
            return {"isSuccess": False, "message": f"Payment '{pid}' not found.", "data": None}

        if booking_id and booking_id.strip():
            bid = booking_id.strip()
            # If passed human-readable booking number, resolve to ID first
            if not bid.startswith("a00"):
                resolved = await self._resolve_id_by_name("token", "Booking__c", bid)
                if resolved:
                    bid = resolved
            matched = [pay for pay in self.payments if pay.get("Booking__c") == bid or pay.get("bookingId") == bid]
            return {"isSuccess": True, "message": "Payments retrieved.", "data": matched}

        return {"isSuccess": True, "message": "All payments retrieved.", "data": list(self.payments)}

    async def update_payment(
        self,
        username: str,
        payment_id: str,
        update_fields: dict[str, Any],
        idempotency_key: Optional[str] = None,
    ) -> dict[str, Any]:
        call_info = {
            "method": "update_payment",
            "username": username,
            "payment_id": payment_id,
            "update_fields": update_fields,
            "idempotency_key": idempotency_key,
        }
        self.call_history.append(call_info)
        self.mutation_history.append(call_info)

        for pay in self.payments:
            if pay["Id"] == payment_id:
                if "status" in update_fields:
                    pay["Status__c"] = update_fields["status"]
                    pay["status"] = update_fields["status"]
                pay.update(update_fields)
                return {
                    "isSuccess": True,
                    "statusCode": 200,
                    "message": f"Payment '{pay['Name']}' updated successfully.",
                    "data": dict(pay),
                }

        return {
            "isSuccess": False,
            "statusCode": 404,
            "message": f"Payment ID '{payment_id}' not found.",
        }
