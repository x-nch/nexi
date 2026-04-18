## Nexi — Decision Engine Architecture

### 1. Architecture: Modules + Flow

Nexi is a **stateless reasoning orchestrator with ephemeral working context**. It holds no persistent state between sessions. Everything it needs to reason is loaded at session initialization from xnch. Everything it produces is either submitted to xnch for execution or written back to xnch memory through the governed write path.

---

**Module Map**

**1.1 Intent Interpreter**

Entry point. Receives raw input — from a user, an agent, or a system event — and produces a structured intent object. This is not NLP for its own sake. It is normalization into a schema Nexi can reason over.

Output:
```
{
  intent_id: uuid,
  intent_class: enum(QUERY | DECISION | EXECUTION | ESCALATION),
  target_entity: string,
  constraints_declared: [string],
  urgency: LOW | NORMAL | HIGH | CRITICAL,
  ambiguity_score: float,          // if above threshold, trigger clarification path
  raw_input_hash: sha256
}
```

If `ambiguity_score` exceeds threshold, Nexi does not proceed to option generation. It returns a structured clarification request to xnch, which routes it back to the originating actor. Reasoning on ambiguous intent produces garbage options — this gate is cheap insurance.

**1.2 Context Loader**

Calls `POST /memory/read` on xnch. Does not pull everything — pulls a **context manifest**: the minimum structured memory required to reason about this specific intent class and target entity.

Context manifest contents:
- Entity history: prior decisions involving the target
- Outcome records: what happened after previous similar decisions
- Active constraints: policies relevant to this intent class (pulled from xnch, not self-evaluated)
- System state version: pinned at load time, must match at submission time
- Role/capability scope of the requesting actor

The context manifest is immutable for the duration of the reasoning session. If xnch signals a system state version change mid-session, the session is invalidated and restarted — not patched.

**1.3 Option Generator**

Calls the model layer. This is the only module that touches models. It submits a **constrained generation request** — not an open-ended prompt — and receives a structured set of candidate actions.

The generation request includes:
- Normalized intent
- Relevant context (subset of manifest, scoped to what the model needs)
- Output schema: model must return N structured options, not prose
- Explicit instruction: do not evaluate, do not select, generate only

Model output is treated as **raw candidate material**, not as a recommendation. The model has no authority here. It is a generation substrate.

Minimum viable option set: 3. Maximum practical: 7. Below 3, ranking is meaningless. Above 7, the evaluation cost outweighs the marginal option quality gain.

Each option schema:
```
{
  option_id: uuid,
  action_type: enum,
  action_spec: structured_object,
  stated_rationale: string,         // model's stated reasoning, stored for audit
  estimated_side_effects: [string], // model's prediction, not trusted, used as signal
  payload_hash: sha256
}
```

**1.4 Policy Alignment Filter**

Before ranking, every option is run through `GET /policy/check` on xnch — the dry-run verdict interface. Options that return `BLOCK` are dropped immediately. Options that return `MODIFY` are updated with xnch's modified action spec and flagged. Options that return `DEFER` are held separately — they remain candidates but require secondary authorization.

This filter runs in parallel across all options. The output is a reduced, policy-clean candidate set. Nexi never ranks or evaluates options that xnch would block — it would be wasted computation and produces a misleading ranking artifact in the audit trail.

**1.5 Option Evaluator**

Scores the policy-clean candidate set across four dimensions:

- **Policy alignment score**: derived from xnch dry-run response quality (ALLOW clean vs ALLOW with warnings vs MODIFY)
- **Outcome prediction score**: similarity match against past decisions with recorded outcomes in xnch memory — how often did structurally similar actions produce good outcomes?
- **Risk score**: composite of estimated side effects, entity sensitivity, action reversibility, and actor capability scope
- **Context fit score**: how well does this option address the stated intent given the loaded context?

Scoring is deterministic given the same inputs. Weights are configurable per intent class — a `CRITICAL` urgency decision weights reversibility higher; a `QUERY` intent weights context fit highest. Weight configurations are versioned and stored in xnch.

**1.6 Outcome Simulator (Optional, Conditional)**

Activated when: risk score exceeds threshold OR action type is `EXECUTION` with irreversible flag OR actor is an agent (not a human).

Runs a lightweight forward projection: given the selected action and current system state, what is the expected next system state? This is not a deep simulation — it is a structured state diff prediction using the context manifest and outcome history as reference.

If the projected state violates any known constraint (pulled from context manifest), the option is re-scored with a risk penalty. If all options project to constraint-violating states, Nexi escalates rather than selecting — it does not force a selection when all paths are problematic.

**1.7 Decision Selector**

Selects the highest-scoring, non-blocked option. Produces a **decision record** — not just the selected action, but the full reasoning artifact:

```
{
  decision_id: uuid,
  session_id: uuid,
  intent_ref: uuid,
  context_manifest_ref: uuid,
  system_state_version: string,
  options_generated: int,
  options_blocked: int,
  options_evaluated: [{ option_id, scores }],
  selected_option_id: uuid,
  selection_rationale: structured_object,   // not prose — structured scoring summary
  confidence: float,
  escalation_triggered: bool
}
```

This record is submitted to xnch via `POST /verdict` as the action payload. Nexi does not submit the action directly — it submits the decision record, which contains the action. xnch evaluates the full record, not just the action leaf.

---

**Session Flow**

```
Input
  → Intent Interpreter          [normalize, gate on ambiguity]
  → Context Loader              [pull manifest from xnch, pin state version]
  → Option Generator            [call model layer, receive N candidates]
  → Policy Alignment Filter     [parallel dry-run against xnch, drop blocked]
  → Option Evaluator            [score remaining candidates]
  → Outcome Simulator           [conditional, on high-risk or agent actor]
  → Decision Selector           [select, build decision record]
  → Submit to xnch              [POST /verdict with full decision record]
```

---

### 2. Interaction with Models vs xnch

These two interactions are architecturally opposite in character and must never be conflated.

**Interaction with Models: Bounded, Distrusted, Output-Constrained**

Nexi calls models exactly once per session, in the Option Generator module. The call is constrained:

- Output schema is enforced — the model returns structured JSON conforming to the option schema, not free text
- The model is not told which option will be selected — this prevents anchoring bias in option generation
- Model output is never surfaced directly to the user or the execution layer
- Model output is never written to xnch memory directly — only Nexi's evaluated, selected decision record is

The model relationship is: **untrusted generator with constrained output contract**. Nexi treats model output the way a compiler treats user input — with schema validation before any further processing.

If the model returns malformed output, Nexi retries once with a stricter prompt. If it fails again, Nexi falls back to a reduced option set from a secondary model or a rule-based option generator. Model unavailability is not a Nexi failure — Nexi degrades gracefully.

**Interaction with xnch: Authoritative, Synchronous, Fully Logged**

Every xnch interaction is a contract call. Nexi makes the following calls in a fixed order within a session:

1. `GET /system/state` — session initialization, pins state version
2. `POST /memory/read` — context manifest load
3. `GET /policy/check` × N — parallel dry-run per option (Policy Alignment Filter)
4. `POST /verdict` — final decision submission
5. `POST /memory/write` — outcome registration after execution completes (post-execution callback)

Nexi never skips step 1. It never submits to step 4 with a different state version than it pinned in step 1. It never writes memory (step 5) without a completed verdict reference.

xnch is the **source of truth**. If xnch returns a BLOCK on the final verdict, Nexi does not retry with a modified action — it escalates. Attempting to route around a xnch BLOCK is a design violation, not a retry strategy.

---

### 3. Input/Output Contract

**Input to Nexi**

```
{
  session_id: uuid,
  actor: {
    id: string,
    type: enum(HUMAN | AGENT | SYSTEM),
    auth_token: signed_jwt             // verified by xnch, not Nexi
  },
  request: {
    raw_input: string | structured_object,
    input_type: enum(TEXT | EVENT | SCHEDULED | API_CALL),
    priority: LOW | NORMAL | HIGH | CRITICAL,
    idempotency_key: uuid              // required — prevents duplicate decision sessions
  },
  metadata: {
    source_system: string,
    trace_id: uuid,                    // propagated through entire call chain
    parent_decision_id: uuid | null    // for chained decisions
  }
}
```

No field is optional. Nexi rejects incomplete input at the boundary — partial inputs produce partial reasoning, which produces unreliable decisions.

**Output from Nexi**

Nexi does not return a raw answer. It returns a **decision package**:

```
{
  session_id: uuid,
  decision_id: uuid,
  trace_id: uuid,
  status: enum(DECIDED | ESCALATED | CLARIFICATION_REQUIRED | DEGRADED),
  verdict_ref: uuid,                   // xnch verdict ID for this decision
  selected_action: {
    action_type: enum,
    action_spec: structured_object,
    execution_token: signed_jwt,       // issued by xnch, consumed by execution layer
    token_ttl_ms: int
  } | null,
  decision_record_ref: uuid,           // full reasoning artifact in xnch audit log
  escalation: {
    reason: string,
    required_actor: string,
    hold_id: uuid
  } | null,
  clarification: {
    question: structured_object,
    ambiguity_ref: uuid
  } | null
}
```

The execution layer receives `selected_action.execution_token`. It does not receive the decision record. It does not need to know how the decision was made — only that it was authorized.

Callers who need the reasoning artifact access it through `POST /audit/query` on xnch, not through Nexi's output. Nexi's output is operational. The audit trail is forensic.

---

### 4. How Nexi Improves Over Time Without Training Models

This is the architectural commitment that separates xnch/Nexi from standard AI pipelines. The system improves through **structured outcome feedback**, not model weight updates.

**Mechanism: Outcome Registration**

After every execution, the execution layer reports the outcome back to xnch via a structured outcome record:

```
{
  decision_id: uuid,
  execution_token_ref: uuid,
  outcome_status: SUCCESS | PARTIAL | FAILURE | ROLLED_BACK,
  observed_state_delta: structured_diff,
  side_effects_observed: [string],
  duration_ms: int,
  anomalies: [string]
}
```

Nexi receives this outcome (via xnch callback) and writes a **decision outcome record** to xnch memory through `POST /memory/write`. This record links: the original intent → the options generated → the selected option → the execution outcome.

**How This Feeds Back Into Future Decisions**

The Option Evaluator's **outcome prediction score** is computed by querying this history. When evaluating a new candidate option, Nexi asks xnch: "For structurally similar actions against similar entity classes, what was the historical outcome distribution?"

Similarity is computed structurally — action type, target entity class, constraint profile, actor role — not semantically. This keeps the lookup deterministic and reproducible.

Over time, the outcome history accumulates signal:
- Actions that consistently produce `FAILURE` outcomes get lower outcome prediction scores
- Actions that produce `PARTIAL` outcomes in specific entity contexts get flagged with context-conditional scoring
- Actions that were `MODIFY`-ed by xnch but then succeeded teach Nexi to pre-apply those modifications as option variants in future generation requests

**Scoring Weight Evolution**

The evaluator weight configurations (per intent class) are themselves updated based on outcome data. If `HIGH` urgency decisions consistently produce better outcomes when reversibility is weighted higher, that weight is adjusted. This adjustment is:
- Proposed by an analytics process running against outcome history
- Submitted as a `POLICY_CHECK` to xnch before being applied
- Versioned and logged — every weight change has an audit trail and a causative outcome batch reference

This is not gradient descent. It is **governed parameter adjustment** with full traceability.

**Option Generator Prompt Evolution**

The constrained generation prompts sent to the model layer are versioned templates. When outcome history shows that a specific option generation prompt consistently produces options that get blocked or score poorly, the template is flagged for review. An operator reviews the flag, revises the template, and deploys a new version through xnch's policy path. The model doesn't change — the instructions to the model change.

This means Nexi improves by becoming **better at asking**, not by the model becoming smarter.

**The Compounding Effect**

After sufficient operational history, Nexi's Option Evaluator is not speculating — it is pattern-matching against a rich, structured, outcome-validated decision corpus. The model layer is still generating raw options, but the filter + scoring pipeline is increasingly well-calibrated. The system converges toward higher-quality decisions not because the model improved, but because Nexi's evaluation of model output improved.

This is the architectural analog of a trading system that doesn't retrain its pricing model — it refines its execution strategy based on fill quality and market impact data. The alpha is in the execution layer, not the model.