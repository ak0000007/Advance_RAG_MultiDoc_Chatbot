/**
 * @description REST API endpoint to update Opportunity StageName.
 *              Used for external system / Agent integration.
 *
 * Endpoint : PATCH /services/apexrest/Opportunity/StatusUpdate/
 * Body     : { "opportunityId": "006xxx", "newStatus": "Closed Won" }
 *
 * Security layers (in order):
 *   1. with sharing          — record-level sharing rules enforced.
 *   2. WITH USER_MODE (SOQL) — CRUD + FLS on read.
 *   3. stripInaccessible     — FLS on update (removes fields user can't write).
 *   4. AccessLevel.USER_MODE — CRUD on DML.
 *
 * Idempotency:
 *   Reads the "Idempotency-Key" request header (a stable tool_call_id from the agent).
 *   Checks a custom field Opportunity.AgentIdempotencyKey__c before writing.
 *   If the key already matches, the DML is skipped and a 200 "already applied" is returned.
 *   This prevents duplicate writes on double-clicks, network retries, and LangGraph re-runs.
 *
 *   Prerequisites:
 *     - Create custom field: Opportunity.AgentIdempotencyKey__c (Text 255, External ID)
 */
@RestResource(urlMapping='/Opportunity/StatusUpdate/*')
global with sharing class OpportunityStatusAPI {

    /**
     * @description Standard JSON response envelope.
     */
    global class ApiResponse {
        global Boolean isSuccess;
        global Integer statusCode;
        global String  message;

        global ApiResponse(Boolean isSuccess, Integer statusCode, String message) {
            this.isSuccess  = isSuccess;
            this.statusCode = statusCode;
            this.message    = message;
        }
    }

    /**
     * @description Deserializes JSON request body.
     *              Both fields are required.
     */
    global class RequestBody {
        global String opportunityId;
        global String newStatus;
    }

    /**
     * @description PATCH handler. Updates StageName on a single Opportunity.
     *
     * NOTE: Apex REST ignores method-level parameters for JSON body fields on PATCH/POST.
     *       Body is always read from RestContext.request.requestBody via JSON.deserialize().
     *       Do NOT add body-field parameters to this method signature.
     */
    @HttpPatch
    global static ApiResponse updateStatus() {
        RestRequest  req = RestContext.request;
        RestResponse res = RestContext.response;

        // ── 1. Parse & Validate Request Body ──────────────────────────────
        RequestBody reqBody;
        try {
            reqBody = (RequestBody) JSON.deserialize(
                req.requestBody.toString(),
                RequestBody.class
            );
        } catch (Exception e) {
            res.statusCode = 400;
            return new ApiResponse(false, 400, 'Invalid JSON body: ' + e.getMessage());
        }

        if (String.isBlank(reqBody?.opportunityId) || String.isBlank(reqBody?.newStatus)) {
            res.statusCode = 400;
            return new ApiResponse(false, 400, 'opportunityId and newStatus are required.');
        }

        // ── 2. Validate opportunityId is a valid Salesforce Opportunity ID ─
        try {
            Id oppId = (Id) reqBody.opportunityId;
            if (oppId.getSObjectType() != Opportunity.SObjectType) {
                res.statusCode = 400;
                return new ApiResponse(false, 400, 'opportunityId does not belong to Opportunity object.');
            }
        } catch (Exception e) {
            res.statusCode = 400;
            return new ApiResponse(false, 400, 'opportunityId is not a valid Salesforce ID.');
        }

        // ── 3. Read idempotency key from request header ────────────────────
        String idempotencyKey = req.headers.get('Idempotency-Key');

        try {
            // ── 4. Query with USER_MODE (Enforces CRUD + FLS on read) ─────
            //       Also selects AgentIdempotencyKey__c for duplicate check.
            List<Opportunity> opps = [
                SELECT Id, StageName, AgentIdempotencyKey__c
                FROM   Opportunity
                WHERE  Id = :reqBody.opportunityId
                WITH   USER_MODE
                LIMIT  1
            ];

            if (opps.isEmpty()) {
                res.statusCode = 404;
                return new ApiResponse(false, 404, 'Opportunity not found or you do not have read access.');
            }

            Opportunity oppToUpdate = opps[0];

            // ── 5. Idempotency check — skip DML if already applied ─────────
            // If the key on record matches the incoming key, this exact operation
            // already succeeded. Return 200 with "already applied" — safe for
            // double-clicks, retries, and LangGraph re-runs.
            if (String.isNotBlank(idempotencyKey)
                    && idempotencyKey == oppToUpdate.AgentIdempotencyKey__c) {
                res.statusCode = 200;
                return new ApiResponse(
                    true,
                    200,
                    'Opportunity status was already updated to "' + reqBody.newStatus + '" (idempotent replay).'
                );
            }

            oppToUpdate.StageName              = reqBody.newStatus;
            oppToUpdate.AgentIdempotencyKey__c = idempotencyKey; // stamp key with this write

            // ── 6. Strip fields the running user cannot update (Enforces FLS) ─
            SObjectAccessDecision decision = Security.stripInaccessible(
                AccessType.UPDATABLE,
                new List<Opportunity>{ oppToUpdate }
            );

            // ── 7. DML with USER_MODE (Enforces edit CRUD) ────────────────
            Database.SaveResult saveResult = Database.update(
                decision.getRecords()[0],
                false,
                AccessLevel.USER_MODE
            );

            if (saveResult.isSuccess()) {
                res.statusCode = 200;
                return new ApiResponse(true, 200, 'Opportunity status updated to "' + reqBody.newStatus + '" successfully.');
            }

            // Collect all DML errors into one message
            List<String> errors = new List<String>();
            for (Database.Error err : saveResult.getErrors()) {
                errors.add(err.getStatusCode() + ': ' + err.getMessage());
            }
            res.statusCode = 400;
            return new ApiResponse(false, 400, 'DML failed — ' + String.join(errors, ' | '));

        } catch (System.QueryException qe) {
            res.statusCode = 403;
            return new ApiResponse(false, 403, 'Access denied during query: ' + qe.getMessage());
        } catch (System.NoAccessException nae) {
            res.statusCode = 403;
            return new ApiResponse(false, 403, 'Insufficient access: ' + nae.getMessage());
        } catch (Exception e) {
            res.statusCode = 500;
            return new ApiResponse(false, 500, 'Unexpected server error: ' + e.getMessage());
        }
    }
}
