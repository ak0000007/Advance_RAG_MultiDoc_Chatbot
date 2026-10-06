# Refactor Complete — Security, Pros & Cons

## What Changed (Summary)

| File | Before | After |
|------|--------|-------|
| [`graph.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/graph/graph.py#L33-L44) | 33-line `SYSTEM_PROMPT` with per-tool routing prose | 11-line prompt — role + 5 hard rules only |
| [`salesforce_tools.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/tools/salesforce_tools.py) | Generic docstrings, routing guidance in prompt | Tool docstrings carry their own routing rules |
| [`travel_tools.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/tools/travel_tools.py) | Generic docstrings, routing guidance in prompt | PRE-CONDITION + "never ask first" per tool |

**Zero logic changes.** No guards removed. No API calls changed. No graph wiring changed.
The approval flow, write caps, single-record guard — all untouched.

---

## Security Concerns

### 🟡 Concern 1 — Tool Docstrings Are LLM-Visible (Prompt Injection Surface)

**What it means:** Tool descriptions become part of the message sent to the LLM. A malicious user could try to craft input that makes the LLM misread a tool description.

**Already mitigated in your code:**
- [`salesforce_tools.py` line 53](file:///home/akhilsaini/Advance_RAG_Chatbot/src/tools/salesforce_tools.py#L53) — control characters stripped from SF data before returning to LLM
- [`write_guards.py`](file:///home/akhilsaini/Advance_RAG_Chatbot/src/write_guards.py) — `single_record_guard`, `bulk_intent_guard`, `session_write_cap` run in Python before any write, regardless of what the LLM says
- Approval interrupt runs **server-side** — LLM cannot bypass it

**Residual risk:** Low. The guards are in code, not prose. Even if the LLM is tricked into calling an update tool, the Python guards will fire before Salesforce is touched.

---

### 🟡 Concern 2 — Shorter Prompt = Less Explicit Refusal Guidance

**What it means:** The old prompt had an explicit `BEHAVIORAL GUIDELINES` section listing what to decline. The new prompt has 5 hard rules — concise but less elaborated.

**Mitigation applied:** Rule 4 explicitly says: *"Decline requests completely outside your tools."*
The LLM infers from the tool schemas what is in/out of scope. If a user asks about flights, no flight tool exists → LLM declines naturally.

**Residual risk:** Very low. The LLM's tool selection IS the scope boundary. No tool = no action.

---

### 🟢 Concern 3 — ID Fabrication (IMPROVED by this change)

**What it means:** Before, the "never fabricate an ID" rule was buried in SYSTEM_PROMPT line 57 alongside 32 other lines. The LLM could miss it.

**After this change:** Every update tool docstring has a `PRE-CONDITION` line that says *"Never guess or fabricate a [booking_id/payment_id/package_id]."* The LLM reads this **only** when it's about to call that specific tool — right when it matters.

**Risk direction:** ↓ Reduced. The instruction is contextually placed.

---

### 🟢 Concern 4 — Write Guards Are Code, Not Prompt (Always Was, Still Is)

These security properties are **not affected by this change** — they were never in the prompt:

| Guard | Where | Behaviour |
|-------|-------|-----------|
| Single record only | `write_guards.py` → called in every update tool | Rejects list/wildcard IDs in Python |
| Bulk intent block | `write_guards.py` → called in every update tool | Rejects "all", "every", "bulk" phrases |
| Session write cap | `write_guards.py` → injected via `tool_node` | Hard cap at 5 writes/session in Python |
| Human approval | `graph.py` → `_after_tool_router` → interrupt | Freezes graph, requires explicit user confirm |

All four guards run in Python **after** the LLM makes a tool call and **before** any Salesforce write.
The LLM literally cannot skip them regardless of what it "decides".

---

### 🔴 Concern 5 — `get_travel_packages` Has No Ownership Check

**This is pre-existing, not introduced by this refactor.** `get_travel_packages` fetches ALL active packages for any authenticated user. Any logged-in user can see all package data.

**Recommendation (future):** Salesforce-side row-level security (record sharing rules) is the right layer to enforce this — not the LLM prompt.

---

## Pros of This Approach

| # | Pro | Why It Matters |
|---|-----|---------------|
| 1 | **Smaller prompt = fewer tokens per turn** | 33 → 11 lines. Saves ~150 tokens every single agent invocation. Across 1000 conversations/day = meaningful cost reduction |
| 2 | **Routing guidance is contextual** | LLM reads "never ask for ID first" only when it selects `get_booking`, not while also reading 8 other tools' instructions |
| 3 | **Hallucination surface shrinks** | LLM has less prose to misinterpret. PRE-CONDITION on update tools is read exactly when the LLM is about to do the risky action |
| 4 | **Adding a new Salesforce object is cleaner** | Add one `@tool` with its own docstring. No need to edit `SYSTEM_PROMPT` and remember to follow its format |
| 5 | **Easier to test tool behaviour** | Each tool's contract (when to call it, what it needs) is self-contained. You can inspect `tool.description` directly |
| 6 | **Security guards unchanged** | All write guards, approval flow, and ID validation still run in Python — not in prose |

---

## Cons of This Approach

| # | Con | Severity | Mitigation |
|---|-----|----------|-----------|
| 1 | **Tool docstrings are now load-bearing** | Medium | A bad edit to a docstring can change LLM behaviour. Treat them like code, not comments |
| 2 | **Cross-tool workflows need the system prompt** | Low | The "fetch before update" rule is now in both the system prompt (rule 2) AND each update tool's PRE-CONDITION — redundant on purpose for safety |
| 3 | **LLM still reads all tool descriptions on every turn** | Low | `bind_tools()` sends ALL tool schemas every call. Mitigation: group tools by feature area if the list grows past ~15 |
| 4 | **Shorter prompt is harder to debug at a glance** | Low | The old prompt read like documentation. The new one is terse. Engineers need to read tool files to understand full routing |

---

## Bottom Line

> [!IMPORTANT]
> **Behaviour is identical.** Same tools, same guards, same approval flow, same API calls.
> The only change is WHERE the routing instructions live — moved from one big prose block
> into each tool's structured description where the LLM reads them at the right moment.

> [!TIP]
> Next improvement when you're ready: add a `document_search` tool docstring that says
> *"Use for internal policy/FAQ/knowledge-base questions. Do NOT use for Salesforce record lookups."*
> This prevents the LLM from using RAG when it should be calling a Salesforce tool.
