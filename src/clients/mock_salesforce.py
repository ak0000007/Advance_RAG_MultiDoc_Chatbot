"""
Mock Salesforce Client wrapper for local testing and zero-credential demos.
Re-exports MockSalesforceAsyncClient from evals.mock_salesforce.
"""

from evals.mock_salesforce import MockSalesforceAsyncClient

__all__ = ["MockSalesforceAsyncClient"]
