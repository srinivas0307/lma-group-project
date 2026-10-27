---
marp: false
---

# Intent-Gated Data Transformation Agent: Implementation Roadmap

**Team SSSS** · LMA Group Project 2026
**Status:** Draft for team review. No code has been written yet.

This document is the development plan for the project. It is based on two sources:

- `handout.txt`: the official project handout. Cited as **§N** (handout section).
- `proposal.txt`: our submitted proposal. Cited as **p.N** (proposal page).

Anything marked **[Rec]** is a recommendation from the planning stage. It is not in either document and is open for discussion.

> **Process rule (§6):** do not start implementation until the TA has approved the proposal.

---

## Contents

0. [Decisions needed before we start](#0-decisions-needed-before-we-start)
1. [Project understanding and objectives](#1-project-understanding-and-objectives)
2. [Handout requirements checklist](#2-handout-requirements-checklist)
3. [Proposal compared with the handout](#3-proposal-compared-with-the-handout)
4. [System architecture](#4-system-architecture)
5. [Technology stack](#5-technology-stack)
6. [Phase-wise roadmap](#6-phase-wise-roadmap)
7. [Repository structure](#7-repository-structure)
8. [MVP and priorities](#8-mvp-and-priorities)
9. [Testing, evaluation, and validation](#9-testing-evaluation-and-validation)
10. [Risks and mitigations](#10-risks-and-mitigations)
11. [Final implementation checklist](#11-final-implementation-checklist)
12. [Review notes](#12-review-notes)

---

## 0. Decisions needed before we start

| # | Decision | Options | Team answer |
|---|---|---|---|
| D1 | TA approval status | **Decided:** approved after TA feedback. The one open TA comment is that **agent evaluation is missing**; it is added in this plan (Phase 2 scenario suite, Phase 10 agent evaluation). | ✅ |
| D2 | Available GPU | (a) Colab/Kaggle T4 16 GB, (b) lab GPU ≥ 24 GB on Linux, (c) laptop GPU or CPU only | |
| D3 | Deadline and milestones | (a) about 4–6 weeks (drop P2 items), (b) about 8+ weeks (P2 feasible), (c) other dates | |
| D4 | Team size | **Decided:** 3 members. | ✅ |
| D5 | Predicate vocabulary additions | Add `columns_added` and `rows_retained_where`? Decide in Phase 1. | |

Already decided:

- **LLM access:** local open models (Qwen family).
- **Language:** Python, starting from scratch.
- **Proposal changes:** the proposal is a living document. We refine it at the checkpoints in §6 instead of fixing everything up front.

---

## 1. Project understanding and objectives

**The idea.** LLM-written pandas code often runs without errors but does the wrong thing. It might drop valid rows, change unrelated columns, or deduplicate on the wrong key. We turn each request into an **intent contract** built from a small fixed set of checkable statements (predicates). The contract has two halves:

- `must_change`: what the transformation is expected to change.
- `must_preserve`: what it must leave untouched.

After the code runs, a deterministic verifier compares the data before and after against the contract. If it finds a violation, a bounded repair loop gets a few attempts to fix the code.

**Research question (p.1–2).** Does an intent contract that specifies what must change and what must be preserved improve the detection of silent semantic errors in LLM-generated data transformations, compared with conventional data-quality checks?

**Hypothesis (p.2).** Intent-based verification will accept fewer incorrect transformations while keeping the false-rejection rate low.

**What we have to show:**

1. Contract-based verification beats no verification and beats generic data-quality checks, on tasks where the correct answer is known.
2. The fine-tuned parser produces better contracts than a prompted general-purpose model.
3. The repair loop improves final accuracy.
4. Nothing unsafe ever runs.

---

## 2. Handout requirements checklist

| # | Requirement | Source | Type |
|---|---|---|---|
| H1 | A real problem, a new approach, a prototype, and an evaluation, rather than "just a chatbot or RAG app" | §1 | Mandatory |
| H2 | A clear research question, method, evaluation setup, and evidence for or against the hypothesis | §1, §7 | Mandatory |
| H3 | Combine several of the five capabilities, not just one | §2 | Mandatory |
| H4 | For each capability: why the problem needs it, and how it is measured | §2 | Mandatory |
| H5 | Tool calling: right tool chosen, valid arguments, recovery from errors, no unsafe actions | §2.5 | Mandatory (we use it) |
| H6 | Agentic: the system decides its next step over several steps (planning, memory, state, hand-off to a human) | §2.4 | Mandatory (we use it) |
| H7 | Fine-tuning is lightweight only (LoRA/QLoRA on a small model) | §2.1, §5 | Constraint |
| H8 | Narrow scope | §4 | Constraint |
| H9 | Small or open models; small, carefully designed evaluation sets | §5 | Constraint |
| H10 | Compute budget: GPU type and hours, API calls or tokens, **number of experiment runs** | §5 | Strongly expected (TAs may reject without it) |
| H11 | TA approval before implementation | §6 | Mandatory |
| H12 | A working prototype that uses a sandbox or mock tools for actions | §7.5 | Mandatory |
| H13 | At least one baseline | §7 | Mandatory |
| H14 | A small evaluation set with known ground truth | §7 | Mandatory |
| H15 | At least one quantitative metric per capability used | §7 | Mandatory |
| H16 | An ablation of each capability the research question depends on | §7 | Mandatory |
| H17 | Tool-call accuracy and the rate of unsafe or unapproved actions | §7 | Mandatory |
| H18 | Qualitative examples of hard or interesting cases | §7 | Mandatory |
| H19 | Failure analysis | §7 | Mandatory |
| H20 | Don't rely only on a demo or one LLM-judge score; validate any LLM judge against human labels | §7 | Mandatory |
| H21 | Write-up covering problem, related work, method, experiments, results, limitations, future work | §7.8 | Mandatory deliverable |
| H22 | The proposal contents listed in §8 | §8 | Mandatory (proposal) |
| O1 | Using all five capabilities, RAG, or reasoning | §2 | Optional |
| O2 | Extra ablations beyond the required ones | §7 ("encouraged") | Optional / bonus |
| O3 | Good enough for a workshop submission | §6 | Aspirational |

---

## 3. Proposal compared with the handout

### 3.1 Already covered

- **H1, H2, H3:** three capabilities (p.2–3) and a clear research question.
- **H7, H9:** QLoRA on Qwen2.5-1.5B/3B (p.10).
- **H8:** five transformation types, three datasets (p.5–6).
- **H13:** three baselines (p.6–7), more than required.
- **H14:** gold contracts plus reference transformations (p.6).
- **H15:** metrics for each capability (p.9).
- **H17:** safety metrics with a target of zero unsafe actions (p.9–10).
- **H20:** no LLM judge (p.5, p.9).
- Oracle-contract ablation (p.8) and risk register (p.10).
- **H22:** every section listed in §8 is present.

### 3.2 Gaps and weak spots

> These are **reference notes**, not a to-do list. The proposal was approved after TA feedback, and these points are handled inside the phases of §6 as we reach them.

| ID | Gap | Why it matters | [Rec] Fix |
|---|---|---|---|
| G1 | **No source of incorrect transformations is defined** | Detection recall needs known errors. A fixed code generator may be correct ~90% of the time on simple tasks, leaving too few errors to measure. | Build a **candidate pool** per task: the reference solution, **planted faulty variants** (§7 allows "injected scenarios"), and some LLM samples. Label each candidate correct or incorrect by comparing its output with the reference output. |
| G2 | Stage 1 doesn't say what is being judged | Verifiers should judge *the same* transformations so the comparison is paired and fair. | In Stage 1, every system judges the same frozen candidate pool. |
| G3 | The code generator receives the contract in the diagram (p.3) | If only the proposed system's generator sees the contract, gains can't be attributed to verification vs. generation. | Stage 1: the generator sees only the request, in every system. Contract-conditioned generation becomes a separate Stage 2 variant. |
| G4 | **No ablation of the "bidirectional" claim** | The novelty (p.5) is must-change + must-preserve, and §7 requires ablating what the research question depends on. | Add two ablations: must-change only, and must-preserve only. |
| G5 | Tool calling is a fixed pipeline | §2.5 asks to show the *model* picks tools, fills in arguments, and recovers from errors. Hardcoded calls don't show that. | The repair agent calls tools from a registry (inspect, profile, diff, run, verify) with JSON-schema arguments. Measure tool selection and argument validity. |
| G6 | The agentic part is thin | Pass → accept, fail → repair is close to a fixed loop. | The agent decides whether to repair, inspect further, or hand off to a human (the "UNCERTAIN" branch), and keeps a history of past attempts. |
| G7 | "UNCERTAIN / Flag" is undefined | It affects the metrics. | UNCERTAIN = contract failed to parse, references an unknown column, or a predicate can't be evaluated. Report it separately, and count it as a reject in the main metrics. |
| G8 | Silent semantic error rate and recall overlap | If both use "incorrect transformations" as the denominator, the error rate is just 1 − recall. | Define silent error rate = incorrect-and-accepted ÷ all accepted (how polluted accepted outputs are). Report recall and false-rejection rate as well. |
| G9 | No method for matching rows before and after | `column_values_unchanged` can't be checked after filtering or dedup without knowing which row is which. | Add a hidden `__rid__` row-ID column before execution. Instruct the generator to keep it. If it's missing afterwards, return UNCERTAIN. |
| G10 | The predicate vocabulary misses some cases | Column transformations that *add* columns, and "rows not matching the filter must be kept". | Proposed additions: `columns_added`, `rows_retained_where`. Freeze the vocabulary before annotation. |
| G11 | Fine-tuning data could leak into evaluation | 3–5k auto-generated examples over the *same* datasets could copy evaluation patterns. | Generate training data from **different schemas/datasets** and templates. Hold out evaluation datasets, or at least their columns and phrasings. |
| G12 | Who generates the code isn't stated | Qwen 1.5B writes poor pandas code. | Use one fixed generator (e.g. Qwen2.5-Coder-7B-Instruct, 4-bit) for every system. |
| G13 | Number of experiment runs is missing (H10) | TAs may reject the proposal without it. | Add the estimate from §5.3 below. |
| G14 | Qualitative examples are not mentioned (H18) | Required by §7. | Hand-picked case studies in Phase 11. |
| G15 | "0 unsafe actions" is untested | Zero means nothing if nothing unsafe was attempted. | Add a red-team set: unsafe requests and code using `os`, `open`, sockets, `eval`, etc. |
| G16 | No statistical treatment | Small evaluation sets (§5) need confidence intervals. | Wilson or bootstrap confidence intervals; McNemar's test for paired system comparisons. |
| G17 | Reasoning isn't claimed, but diagnosis reasons | Fine — §2 allows "varying amounts". | Don't claim it. Diagnosis-trace analysis is optional (P2). |

### 3.3 Conflicts with the handout

- No direct conflicts. The project fits §5 (small models, local inference) and §4 (narrow scope).
- The only process risk is H11: starting code before TA approval.

---

## 4. System architecture

> **Revision note (v2).** The first version of this section had several issues. They are listed in §4.0 so reviewers can see what changed and why.

### 4.0 Issues fixed in this revision

| # | Issue in v1 | Why it was a problem | Fix in v2 |
|---|---|---|---|
| A1 | One diagram mixed Stage 1 and Stage 2 | Stage 1 judges a **frozen candidate pool** (G2), so the online code generator isn't part of it. v1 even said the generator "sees request only in Stage 1", which contradicts the frozen pool. | Separate diagrams for the Stage 2 runtime (§4.1) and Stage 1 mode (§4.2). The generator builds the pool offline for Stage 1 and runs online only in Stage 2. |
| A2 | Baseline 2 (conventional checks) was a dead-end node with no outputs | You couldn't see how Baseline 2 produces a verdict or triggers repair (Phase 9 runs "Baseline 2 + repair"). | One **verifier slot** with three modes: none / conventional / intent. Every mode produces the same verdict type. |
| A3 | The before snapshot never reached the diff | A diff needs both datasets; v1 only showed the sandbox output going in. | `__rid__` injector → read-only before snapshot, which feeds both the sandbox and the diff. |
| A4 | Several failure paths were missing | Invalid contract, blocked code, sandbox crash or timeout, and missing `__rid__` all just stopped. Blocked code went only to the logger. | Every failure goes to an explicit outcome: a failure report for repair, REJECT, or ESCALATE. |
| A5 | The tool registry wasn't in the diagram | Tool calling is a graded capability (§2.5, H5), yet the agent's tools were invisible. Tool calls could also bypass the safety checker. | Repair agent ↔ tool registry is shown. `run_transformation` always goes through the safety checker and sandbox. The registry enforces a tool allow-list. |
| A6 | No separation between the runtime system and evaluation | Nothing stopped gold contracts or reference outputs from leaking into the runtime. The evaluator was listed as step 7 of the request flow. | A separate **evaluation harness** (§4.3). Ground truth reaches the runtime only through the oracle parser, which is the point of that ablation. |
| A7 | The repair agent could, in principle, change the contract | An agent could "fix" a violation by weakening the checks. That would inflate repair success. | The contract and before snapshot are **read-only** once created. The agent changes only code. |
| A8 | Final outcomes were vague ("Accept", "Flag / hand off") | Metrics need a fixed set of outcomes, and v1 didn't say what happens when repair is off. | Three outcomes: **ACCEPT, REJECT, ESCALATE**. If repair is off (Stage 1), a violation means REJECT. |
| A9 | Ablations and baselines weren't shown as configurations | Hard to check that all systems share the same parts and budgets (a fairness issue). | Systems are defined as configurations of shared slots (§4.4). |
| A10 | The state machine was linear and only one repair counter existed | No cap on tool calls per attempt, no detection of repeated code, and no branch for systems without a contract. | Revised state machine (§4.6) with separate limits (§4.8). |
| A11 | Trace edges were partial (only some nodes logged) | Metrics are computed from traces, so a partial trace means missing metrics. | Every component emits trace events, so the logger edges are left out of the diagrams. |

### 4.1 Runtime pipeline (Stage 2, full system)

```mermaid
flowchart TD
    U[User request]
    D[(Input dataset)]

    D --> RID[Row-ID injector<br/>adds hidden __rid__]
    RID --> BEF[(Before snapshot<br/>read-only)]
    BEF --> SCH[Schema extractor<br/>columns, dtypes, nulls, sample rows]

    U --> P[Intent parser slot<br/>fine-tuned / prompted / oracle / none]
    SCH --> P
    P --> CV{Contract validator<br/>pydantic + columns exist}
    CV -->|invalid| ESC
    CV -->|valid| CF[Contract filter<br/>full / change-only / preserve-only]
    CF --> C[(Intent contract<br/>read-only)]

    U --> G[Code generator<br/>fixed LLM]
    SCH --> G
    C -. optional variant .-> G

    G --> SC{Safety checker<br/>AST allow-list}
    SC -->|allowed| SB[Sandbox executor<br/>separate process, no network,<br/>CPU / memory / time limits]
    BEF --> SB
    SB -->|ok| AFT[(After snapshot)]

    BEF --> DF[Diff + profiler<br/>aligned on __rid__]
    AFT --> DF
    DF --> V{Verifier slot<br/>none / conventional / intent}
    C --> V

    SC -->|blocked| FR
    SB -->|crash / timeout| FR
    DF -->|__rid__ missing| ESC
    V -->|PASS| ACC([ACCEPT])
    V -->|UNCERTAIN| ESC([ESCALATE<br/>hand off to human])
    V -->|VIOLATION| FR[Failure report<br/>violated predicates / failed checks /<br/>error, with example rows]

    FR -->|repair off or budget used up| REJ([REJECT])
    FR -->|repair on| RA[Repair agent<br/>attempt memory]
    RA <-->|validated tool calls| TR[Tool registry<br/>allow-listed tools]
    RA -->|propose_fix: revised code| SC
    RA -->|escalate| ESC
```

`run_transformation` in the tool registry reuses the **same** safety checker and sandbox, and runs on a copy of the before snapshot. No path can execute code without passing the safety checker.

### 4.2 Stage 1 mode (verification only)

```mermaid
flowchart LR
    POOL[(Frozen candidate pool<br/>reference + planted faulty +<br/>harmless variants + LLM samples)] --> SC{Safety checker}
    SC -->|allowed| SB[Sandbox]
    SC -->|blocked| REJ([REJECT])
    SB -->|crash / timeout| REJ
    SB --> DF[Diff + profiler]
    DF --> V{Verifier slot}
    P[Intent parser slot] --> CV{Contract validator} --> C[(Contract)] --> V
    V -->|PASS| ACC([ACCEPT])
    V -->|VIOLATION| REJ
    V -->|UNCERTAIN| ESC([ESCALATE])
```

- Every system judges the **same** candidates, so comparisons are paired (McNemar).
- The parser runs once per task (per seed) and its contract is reused for every candidate of that task.
- LLM candidates in the pool came from the code generator offline (Phase 5), seeing **only the request**, so no system gets an advantage at generation time (G3).

### 4.3 Runtime vs. evaluation harness

```mermaid
flowchart LR
    subgraph RT[Runtime system: never sees ground truth]
        PIPE[Pipeline from 4.1 / 4.2]
    end
    subgraph EH[Evaluation harness: offline]
        GT[(Gold contracts,<br/>reference code and outputs,<br/>pool labels, expected tools)]
        EV[Evaluator<br/>outputs_equal, contract P/R/F1,<br/>tool-call and safety metrics]
    end
    GT -->|oracle parser config only| PIPE
    PIPE -->|final outputs + JSONL traces| EV
    GT --> EV
```

The runtime writes final outputs and traces. The evaluator reads them **afterwards** and compares them with ground truth. The only ground truth allowed into the runtime is the gold contract, and only in the oracle-parser configuration.

### 4.4 Systems as configurations

All systems share the same schema extractor, generator, safety checker, sandbox, diff, decoding settings, and budgets. They differ only in the slots below.

| System | Parser slot | Contract filter | Verifier slot | Repair | Used in |
|---|---|---|---|---|---|
| Baseline 1: Direct LLM | none | — | none (always PASS if it runs) | off | Stage 1 and Stage 2 reference |
| Baseline 2: Conventional | none | — | conventional | off / on | Stage 1 / Stage 2 |
| Baseline 3: General-LLM parser | prompted Qwen2.5-7B | full | intent | off / on | Stage 1 / Stage 2 |
| **Proposed** | fine-tuned Qwen2.5-1.5B / 3B | full | intent | off / on | Stage 1 / Stage 2 |
| Oracle-contract ablation | oracle (gold) | full | intent | off / on | Stage 1 / Stage 2 |
| Change-only ablation | fine-tuned and oracle | change-only | intent | off | Stage 1 |
| Preserve-only ablation | fine-tuned and oracle | preserve-only | intent | off | Stage 1 |
| Repair-without-tools ablation (P1) | fine-tuned | full | intent | on, tools disabled | Stage 2 |
| Contract-conditioned generation (P2) | fine-tuned | full | intent | on | Stage 2 variant |

Each row is one YAML file in `configs/`.

### 4.5 Components

| Component | Responsibility | Uses LLM? |
|---|---|---|
| Row-ID injector | Adds a hidden integer `__rid__` column so before and after rows can be matched. It is removed from the final output; `outputs_equal` ignores it. | No |
| Before snapshot | The input dataset with `__rid__`. Read-only for the rest of the run. | No |
| Schema extractor | Column names, dtypes, null counts, a few sample rows. Feeds the parser and the generator. | No |
| Intent parser slot | Request + schema → contract JSON. Implementations: `FineTunedParser`, `PromptedParser`, `OracleParser`, or none. Same constrained decoding for every LLM parser. | Yes (except oracle / none) |
| Contract validator | Pydantic validation and a check that every target column exists in the schema. Failure → ESCALATE (counted as UNCERTAIN). | No |
| Contract filter | Keeps the full contract, only `must_change`, or only `must_preserve`. Used for the G4 ablations. | No |
| Code generator | Request + schema → `transform(df) -> df`. Instructed to keep `__rid__`. Used **offline** to build the Stage 1 pool and **online** in Stage 2. | Yes (one fixed model for all systems) |
| Safety checker | AST allow-list: only `pandas` and `numpy` imports. Blocks `open`, `exec`, `eval`, `compile`, `__import__`, `os`, `sys`, `subprocess`, `socket`, `pathlib`, `shutil`, and dunder attribute access. Every block is logged as an unsafe attempt. | No |
| Sandbox executor | Runs code in a separate process inside a temp directory, with no network and CPU, memory, and wall-clock limits. Data goes in and out by file. Returns `ok`, `crash`, or `timeout` plus stderr. | No |
| Diff and profiler | Aligns on `__rid__`. Rows removed or added, changed cells per column, columns added or dropped, dtype and null changes. Missing `__rid__` → ESCALATE. | No |
| Verifier slot | **none**: PASS whenever the code ran. **conventional**: Baseline 2's fixed generic suite (schema, nulls, dtypes, uniqueness, ranges, row consistency, distribution shift), thresholds tuned on dev. **intent**: one deterministic function per predicate. All return the same `Verdict(status, failures, evidence_rows)`. | No |
| Failure report | One format for every kind of failure: violated predicates or failed checks, sandbox errors, safety blocks, with example rows. This is what the repair agent sees. | No |
| Repair agent | Inputs: request, schema, current code, failure report, attempt history, and the read-only contract (intent systems only). Actions: a tool call, `propose_fix(code)`, or `escalate(reason)`. It never sees gold contracts, reference code, or reference outputs. | Yes |
| Tool registry | Allow-listed tools with JSON argument schemas: `inspect_dataset`, `profile_column`, `diff_datasets`, `run_transformation` (via safety checker + sandbox, on a copy), `run_verifier` (the system's configured verifier). Validates arguments, returns structured errors, and rejects and logs calls to unknown tools. | No |
| Orchestrator | Runs the state machine (§4.6) and enforces all limits (§4.8). | No |
| Tracer | Every component emits JSONL events: step, inputs, outputs, tokens, latency, verdict, tool call and result. | No |
| Evaluator (offline) | Reads traces and outputs; computes all metrics against ground truth. Not part of the runtime. | No |

### 4.6 Orchestrator state machine

```mermaid
stateDiagram-v2
    [*] --> PREPARE
    PREPARE --> PARSE: system has a parser
    PREPARE --> GENERATE: no parser (Baselines 1, 2)
    PARSE --> ESCALATED: invalid contract
    PARSE --> GENERATE: valid contract
    GENERATE --> SAFETY_CHECK
    SAFETY_CHECK --> EXECUTE: allowed
    SAFETY_CHECK --> FAILED: blocked
    EXECUTE --> VERIFY: ok
    EXECUTE --> FAILED: crash or timeout
    VERIFY --> ACCEPTED: PASS
    VERIFY --> ESCALATED: UNCERTAIN
    VERIFY --> FAILED: VIOLATION
    FAILED --> REPAIR: repair on and budget left
    FAILED --> REJECTED: repair off or budget used up
    REPAIR --> REPAIR: tool call
    REPAIR --> SAFETY_CHECK: propose_fix
    REPAIR --> ESCALATED: escalate or repeated code
    ACCEPTED --> [*]
    REJECTED --> [*]
    ESCALATED --> [*]
```

- **PREPARE** = inject `__rid__`, freeze the before snapshot, extract the schema.
- In Stage 1, **GENERATE** means "load the next candidate from the pool", and repair is always off.
- PARSE and GENERATE don't depend on each other, so they can run in either order (or in parallel).

### 4.7 Request flow (proposed system, Stage 2)

1. **Prepare.** The request and CSV arrive. `__rid__` is added, the before snapshot is frozen, and the schema is extracted.
2. **Parse.** The fine-tuned parser produces a contract. The validator checks it; if it's invalid, the run ends as ESCALATE. Otherwise the contract is frozen.
3. **Generate.** The fixed generator writes `transform(df)` from the request and schema.
4. **Safety check.** Blocked code is logged as an unsafe attempt and becomes a failure report.
5. **Execute.** The sandbox runs the code on the before snapshot. A crash or timeout becomes a failure report.
6. **Diff and verify.** The diff is computed on `__rid__`, and the intent verifier checks every predicate. PASS → ACCEPT; UNCERTAIN → ESCALATE; VIOLATION → failure report.
7. **Repair.** The agent reads the failure report and history, may call tools, then proposes revised code (back to step 4) or escalates. Limits in §4.8.
8. **Finish.** `__rid__` is stripped from the final output, and the outcome and full trace are written. Evaluation happens later in the harness (§4.3).

### 4.8 Limits, error handling, and invariants

**Limits** (all set in config; defaults shown):

| Limit | Default | When exceeded |
|---|---|---|
| Repair attempts (`propose_fix` calls) | 3 (proposal allows 2–3) | REJECT |
| Tool calls per attempt | 8 | The next action must be `propose_fix` or `escalate` |
| Sandbox wall-clock per run | 30 s | Timeout → failure report |
| Sandbox memory | 2 GB | Crash → failure report |
| Token budget per task | Fixed, same for all systems | REJECT |
| Identical code proposed twice (hash match) | — | First time: returned as an observation and counted as an attempt. Second time: ESCALATE. |

**Error handling:**

- Malformed LLM output (contract JSON or tool call): one re-prompt with the validation error, then counted as a failure (invalid contract → ESCALATE; invalid tool call → tool error observation).
- Tool errors (bad arguments, unknown tool) go back to the agent as structured observations. That's how we measure "recovers from errors" (§2.5).
- Blocked code and unknown-tool calls are always counted in the safety metrics, even if the agent recovers afterwards.

**Invariants** (these protect the validity of the results):

1. The before snapshot and the contract are read-only after creation. The repair agent can change only code.
2. Ground truth never reaches the runtime, except the gold contract in the oracle configuration.
3. All verifiers are deterministic and use no LLM.
4. Every code execution goes through the safety checker, including `run_transformation`.
5. The generator, decoding settings, and budgets are identical across systems. Only the slots in §4.4 differ.

**Outcomes and metrics:** each run ends as ACCEPT, REJECT, or ESCALATE. For detection metrics, REJECT and ESCALATE both count as "not accepted"; the ESCALATE rate is also reported separately (G7).

---

## 5. Technology stack

### 5.1 Libraries

| Area | Choice | Build or reuse | Why / trade-offs |
|---|---|---|---|
| Language | Python 3.11 | — | Matches pandas, PEFT, and DS-1000. |
| Data | pandas, numpy, pyarrow | Reuse | This is what is being transformed. |
| Contract schema | pydantic v2 | Reuse library, build schema | Strict validation; exports JSON schema usable for constrained decoding. |
| Verifier, diff, conventional checks | Plain pandas | **Build** | This is the research core. Great Expectations/Pandera would hide the logic. Our own ~7 generic checks also document exactly what Baseline 2 contains. |
| Agent loop, tool registry, orchestrator | Plain Python | **Build** | The course rewards understanding the mechanisms. LangChain/LangGraph would hide exactly what we're graded on. |
| Sandbox | `subprocess` + `resource` limits + timeout + AST checker; network cut via `unshare -n`, or Docker `--network none` if available | Build | Docker is stronger but needs setup. Start with subprocess, add Docker if possible. |
| Fine-tuning | `transformers` + `peft` + `trl` (SFTTrainer) + `bitsandbytes` (QLoRA) | Reuse | Standard; fits a 16 GB GPU. |
| Inference | `vLLM` on a Linux GPU (serves LoRA adapters, JSON-guided decoding), otherwise plain `transformers` | Reuse, behind an `LLMClient` interface | A few hundred calls is fine with `transformers`. Ollama would require converting adapters to GGUF. |
| Structured output | JSON-schema constrained decoding (vLLM guided decoding or `outlines`) + pydantic re-validation | Reuse | Small models break JSON. **Same decoding for every parser** to keep the comparison fair. |
| Config and runs | YAML configs per experiment, fixed seeds | Build | Reproducibility. |
| Tests | `pytest`, `ruff` | Reuse | Unit tests for every predicate and tool. |
| Statistics and plots | `scipy` / `statsmodels` (McNemar, Wilson), `matplotlib` | Reuse | For G16. |

Dependency versions will be pinned in `requirements.txt` in Phase 0.

### 5.2 Models [Rec]

| Role | Model | Notes |
|---|---|---|
| Parser to fine-tune | Qwen2.5-1.5B-Instruct and Qwen2.5-3B-Instruct | Training both gives a "size vs. tuning" result — the handout's own example (§2.1). Qwen3-1.7B/4B are reasonable newer alternatives; pick one family and stick with it. |
| Baseline 3 general parser | Qwen2.5-7B-Instruct, few-shot prompted | A "larger prompted model" the tuned small model should match or beat. |
| Code generator (fixed for all systems) | Qwen2.5-Coder-7B-Instruct, 4-bit | 1.5B/3B are too weak at pandas; we want a realistic error rate. |
| Repair agent | Qwen2.5-7B-Instruct | Supports tool calling; 1.5B calls tools unreliably. |

### 5.3 Compute estimate (to add to the proposal — H10, G13)

- **Fine-tuning:** ~2–4 GPU h per model size × 2 sizes × ~2 hyperparameter runs ≈ **8–16 GPU hours** on a T4 / L4 / A10-class GPU.
- **Stage 1:** ~70 tasks × ~7 candidates ≈ 500 judgments per system. Only the parser calls an LLM: ~70 calls per parser per seed.
- **Stage 2:** ~70 tasks × 4 systems × up to 4 generations × 3 seeds ≈ 3.4k generation calls plus agent calls — a few GPU hours of 7B inference.
- **API cost:** $0 (all local).

---

## 6. Phase-wise roadmap (revised)

### 6.0 How this plan works

**Ordering principle.** The research core (verifier + labelled candidate pool + oracle contracts) is built first, with **no LLM involved**. That tests the research question early: if the oracle-contract verifier can't beat conventional checks, we find out in week 3, not week 8.

**Agent evaluation (TA feedback).** The TA's only open comment was that the agent isn't evaluated. Rather than bolting it on at the end, it is built up across phases:

| Phase | Agent-evaluation work |
|---|---|
| 2 | Design and label the **agent scenario suite** (repairable, should-escalate, already-correct, tool-fault, safety scenarios) |
| 3 | Add **fault-injection hooks** to the tool registry (forced tool errors, timeouts) |
| 5 | Add a **scripted LLM backend** so agent logic can be tested deterministically |
| 9 | Build the agent and test every control path with the scripted LLM |
| 10 | **Full agent evaluation:** outcome, decision, tool-use, efficiency, safety, and trajectory metrics, against agent baselines and ablations |
| 11 | Agent inside the full Stage 2 end-to-end comparison |

**Planning assumptions** (update when D2 and D3 are decided):

- 3 people, called **A**, **B**, **C** below. Suggested ownership:
  - **A**: contracts, verifier, parser, fine-tuning.
  - **B**: tasks, candidate pool, baselines, metrics, evaluation runs.
  - **C**: sandbox, tools, LLM serving, repair agent, agent evaluation.
- About **10 weeks** in total. If the deadline is shorter, cut P2 items first (§8).
- Effort estimates are per owner and include tests.

**Every phase has the same parts:** objective · owner and effort · depends on · components · tasks · tests · deliverables · exit criteria (and a demo where useful).

**Proposal checkpoints.** The proposal is refined at three points, using real results rather than guesses:

- **After Phase 4:** confirm the predicate vocabulary and evaluation-set design.
- **After Phase 10:** add the agent-evaluation section (TA feedback).
- **Before the write-up:** final methods and metrics.

### 6.1 Overview

| Phase | Name | Owner | Est. weeks | Depends on | Milestone / gate |
|---|---|---|---|---|---|
| 0 | Setup and shared interfaces | All | 1 (wk 1) | — | Repo runs; shared types agreed |
| 1 | Contract schema and verifier | A | 1.5 (wk 1–2) | 0 | All predicate tests pass |
| 2 | Task set, candidate pool, agent scenario suite | B (+A for annotation) | 2 (wk 1–3) | 0 (final check needs 1) | Labelled pool and scenarios frozen |
| 3 | Sandbox, safety checker, tools | C | 1.5 (wk 1–2) | 0 | Red-team set 100% blocked |
| 4 | Stage 1 harness with oracle contracts | B | 1 (wk 3) | 1, 2, 3 | **MVP-1 · Gate 1** |
| 5 | LLM backend and code generator | C | 1 (wk 3–4) | 3, 4 | **Gate 2**; pool v2 |
| 6 | Prompted parser and Baseline 3 | A | 1 (wk 4) | 5 | **MVP-2** |
| 7 | Fine-tuning data and QLoRA parser | A | 2 (wk 4–6) | 6 | **Gate 3** |
| 8 | Full Stage 1 evaluation | B | 0.5 (wk 6) | 7 | Main Stage 1 table |
| 9 | Repair agent | C | 2 (wk 5–7) | 3, 5 (and 6 for intent mode) | **Gate 4** |
| 10 | Agent evaluation | C + B | 1.5 (wk 7–8) | 2, 9 | Agent results table |
| 11 | Stage 2 end-to-end and safety | All | 1 (wk 8–9) | 8, 10 | Stage 2 table |
| 12 | Analysis, write-up, demo | All | 1.5 (wk 9–10) | 11 | Submission |

**Critical path:** 0 → 1 → 4 → 5 → 6 → 7 → 8 → 11 → 12, with 9 → 10 running alongside 7–8.

**Decision gates:**

| Gate | When | Question | If the answer is "no" |
|---|---|---|---|
| 1 | End of Phase 4 | Does the oracle-contract verifier beat conventional checks on recall at a similar false-rejection rate? | Inspect missed errors by type; refine predicates or tasks; consider narrowing to the error classes where intent helps (p.10 Risk 4). |
| 2 | End of Phase 5 | Does the real code generator produce enough incorrect outputs to be interesting? | Rely more on planted variants for Stage 1; use a smaller or weaker generator only for Stage 2 diversity. |
| 3 | End of Phase 7 | Does the fine-tuned parser match or beat the prompted 7B parser on contract F1? | Still report it (a negative result is valid); try the 3B model, more data, or better templates if time allows. |
| 4 | End of Phase 9 | Do all agent control paths pass with the scripted LLM? | Do not run real-model agent experiments until they do. |

---

### Phase 0 — Setup and shared interfaces

- **Objective:** everyone can work in parallel from day one.
- **Owner / effort:** all, ~1 week.
- **Depends on:** nothing.
- **Components:**
  - Repository skeleton (§7), `pyproject.toml`, pinned `requirements.txt`, `ruff`, `pytest`.
  - `src/igt/types.py`: shared data classes agreed **before** splitting work: `IntentContract`, `Predicate`, `Condition`, `Verdict`, `FailureReport`, `ExecResult`, `ToolCall`, `ToolResult`, `TraceEvent`, `RunOutcome` (ACCEPT / REJECT / ESCALATE).
  - `scripts/download_data.py`: Airbnb, Adult, Wine Quality into `data/raw/`, with checksums.
  - `configs/base.yaml`: paths, seeds, limits from §4.8.
- **Tasks:**
  1. Create the repo, branch rules, and a pull-request review habit (one reviewer per PR).
  2. Agree and write `types.py` with docstrings; this is the contract between team members.
  3. Download data; write `data/README.md` describing each dataset's columns, size, and licence.
  4. Check GPU access: load a 4-bit 7B model and a 1.5B model, and time one generation (feeds the compute estimate).
- **Tests:** `pytest` runs; datasets load and checksums match; types import cleanly.
- **Deliverables:** working repo, data, `types.py`, GPU timing note.
- **Exit criteria:** all three members can run `pytest` and load data locally.

### Phase 1 — Contract schema and deterministic verifier

- **Objective:** given a contract and before/after data, return per-predicate verdicts.
- **Owner / effort:** A, ~1.5 weeks.
- **Depends on:** Phase 0.
- **Components:**
  - `contracts/schema.py`: pydantic models; JSON-schema export for constrained decoding.
  - `contracts/normalize.py`: canonical form (predicate type + target + normalized condition/value), used for matching and for contract metrics.
  - `verifier/alignment.py`: `inject_rid(df)`, `align(before, after)`.
  - `verifier/diff.py`: rows removed/added, changed cells per column, columns added/dropped, dtype and null changes.
  - `verifier/predicates/`: one module per predicate, each `check(before, after, diff, pred) -> PredicateResult`.
  - `verifier/intent_verifier.py`: combines predicate results into a `Verdict` (PASS / VIOLATION / UNCERTAIN) with evidence rows.
- **Tasks:**
  1. Write `docs/predicates.md`: exact meaning of each predicate, tolerances (float `atol`, null equality, dtype families), and when it returns UNCERTAIN. Decide D5.
  2. Implement schema and normalization.
  3. Implement alignment and diff.
  4. Implement the predicates, one at a time, each with tests.
  5. Implement the verifier and a readable report (`Verdict.explain()`).
- **Tests:** per predicate: pass case, violation case, edge cases (empty frame, all-null column, missing column → UNCERTAIN, dropped `__rid__` → UNCERTAIN). Normalization treats equivalent conditions as equal.
- **Deliverables:** verifier package, predicate docs.
- **Exit criteria:** all tests pass; vocabulary v1 frozen (changes after this need team agreement, because annotations depend on it).

### Phase 2 — Task set, candidate pool, and agent scenario suite

- **Objective:** ground truth for both the verification experiments and the agent evaluation.
- **Owner / effort:** B, ~2 weeks (A helps with annotation and agreement).
- **Depends on:** Phase 0; the final validation step needs Phase 1.
- **Components:**
  - `eval/tasks/*.yaml`: id, dataset, transformation type, request text, reference code, gold contract, dev/test split flag.
  - `eval/mutations.py`: mutation operators that plant faulty variants.
  - `eval/equivalence.py`: `outputs_equal(ref, cand)` ignoring row order, index, and `__rid__`, with dtype tolerance.
  - `eval/build_pool.py` → `eval/pool_v1.jsonl` of (task, candidate code, source, is_correct).
  - `eval/agent_scenarios/*.yaml`: the agent scenario suite (below).
  - `docs/annotation_guide.md`.
- **Tasks:**
  1. Write ~70 tasks: 5 transformation types × 3 datasets, ~4–5 each. Mark ~15% as the dev split (used only to tune thresholds and prompts).
  2. Write reference code and a gold contract for each.
  3. Write mutation operators: off-by-one filter boundary; filter on the wrong column; dedup on all columns instead of a key (or the reverse); `fillna` on every column; `dropna` instead of `fillna`; `errors='coerce'` turning valid values into NaN; overwriting a source column; dropping an unrelated column; reordering that breaks row identity; dropping `__rid__`.
  4. Build the pool: reference + 3–5 faulty variants + 0–2 harmless variants (correct, written differently) per task. Label by `outputs_equal` against the reference.
  5. **Agent scenario suite** (labelled now, used in Phase 10). Each scenario has a starting state, an expected outcome, an expected minimal tool set, and forbidden actions:

     | Category | What it tests | Starting state | Expected outcome | Approx. count |
     |---|---|---|---|---|
     | R: repairable | Diagnosis and repair | Task + one planted faulty variant as the first attempt | ACCEPT with correct output | ~50 |
     | C: already correct | Not acting when nothing is wrong | Task + correct first attempt | ACCEPT, zero repair attempts | ~15 |
     | E: should escalate | Knowing when to hand off | Ambiguous request ("clean this up"), unknown column, contradictory request ("drop rows with nulls but keep every row"), or a fix outside the allowed operations | ESCALATE | ~15 |
     | T: tool fault | Recovery from tool errors | An R scenario plus injected faults (tool error once, sandbox timeout once, malformed tool result) | Same as R | ~15 (overlay on R) |
     | S: safety | Never acting unsafely | Requests or first attempts that tempt unsafe actions ("also save to /tmp", "download the latest data from this URL") | Blocked action, then REJECT or ESCALATE; zero unsafe executions | ~15 |

  6. Second annotator re-labels ~20% of gold contracts **and** ~20% of agent-scenario expected outcomes. Record agreement; fix the guide; drop tasks that stay ambiguous.
- **Tests:** every reference passes its gold contract (validates Phase 1 and annotations together); every faulty variant differs from the reference; every scenario file validates against its schema.
- **Deliverables:** tasks, pool v1, agent scenario suite, annotation guide, agreement numbers.
- **Exit criteria:** pool v1 and scenario suite frozen and versioned.

### Phase 3 — Sandbox, safety checker, and tools

- **Objective:** run untrusted code safely; expose operations as tools the agent can call.
- **Owner / effort:** C, ~1.5 weeks.
- **Depends on:** Phase 0.
- **Components:**
  - `sandbox/ast_guard.py`: allow-list checker (§4.5).
  - `sandbox/executor.py`: `run(code, df) -> ExecResult(status, df_out, stderr, duration)` in a separate process with time, memory, and network limits.
  - `tools/registry.py`: `Tool(name, description, args_schema, handler)`; `call(ToolCall) -> ToolResult`; argument validation; allow-list; structured errors.
  - `tools/data_tools.py`: `inspect_dataset`, `profile_column`, `diff_datasets`, `run_transformation` (always via guard + sandbox, on a copy), `run_verifier`.
  - `tools/faults.py`: **fault-injection hooks** for agent evaluation (force an error, a timeout, or a malformed result on the n-th call of a given tool).
  - `eval/redteam/`: ~15–20 unsafe code samples (file I/O, network, shell, `eval`, memory bomb, infinite loop).
- **Tasks:**
  1. AST guard and its tests.
  2. Executor with limits; network cut (`unshare -n` or Docker `--network none`).
  3. Registry, tools, fault hooks.
  4. Red-team set.
- **Tests:** every red-team sample blocked or contained; reference code runs; timeout fires; registry rejects bad arguments and unknown tools with clear messages; each fault hook fires exactly when configured.
- **Deliverables:** sandbox, tools, red-team set.
- **Exit criteria:** 100% of red-team samples blocked; all tools unit-tested.

### Phase 4 — Stage 1 harness with oracle contracts (MVP-1)

- **Objective:** first research result, with no LLM involved.
- **Owner / effort:** B, ~1 week.
- **Depends on:** Phases 1, 2, 3.
- **Components:**
  - `baselines/conventional_checks.py`: Baseline 2's generic suite, thresholds in config.
  - `verifier/slot.py`: one interface for none / conventional / intent verifiers (§4.4).
  - `eval/runner_stage1.py`: runs any system config over the frozen pool.
  - `eval/metrics.py`: recall, false-rejection rate, silent error rate, ESCALATE rate, with Wilson confidence intervals.
  - `eval/stats.py`: McNemar's test for paired comparisons.
  - `tracing/logger.py`: JSONL events.
- **Tasks:**
  1. Implement conventional checks; tune thresholds on the dev split only.
  2. Run Baseline 1, Baseline 2, and the oracle-contract verifier on pool v1.
  3. Produce the table overall and per transformation type and mutation type.
  4. **Gate 1** review meeting; proposal checkpoint 1.
- **Tests:** metrics hand-checked on a 5-item toy pool; same config gives identical results twice.
- **Demo:** `python -m eval.runner_stage1 --config configs/stage1_oracle.yaml`.
- **Exit criteria:** table produced; Gate 1 decision recorded in §12.

### Phase 5 — LLM backend and code generator

- **Objective:** local models behind one interface; real generated code in the pool.
- **Owner / effort:** C, ~1 week.
- **Depends on:** Phases 3, 4.
- **Components:**
  - `llm/client.py`: `LLMClient.generate(messages, json_schema=None, tools=None, seed=...)`.
  - `llm/backends/`: `hf.py` (transformers), `vllm.py` (if available), **`scripted.py`** (replays pre-written responses; used for agent tests).
  - `llm/cache.py`: cache by (model, prompt, seed) so reruns are cheap and reproducible.
  - `llm/prompts/`: versioned templates.
  - `agents/code_generator.py`.
- **Tasks:**
  1. Client, backends, cache.
  2. Generator prompt (keep `__rid__`, define `transform(df)`); tune on dev tasks only.
  3. Sample several generations per task; record run rate and correctness.
  4. Add generated candidates to the pool (labelled by `outputs_equal`) → pool v2, frozen.
  5. **Gate 2** review.
- **Tests:** schema-constrained output always parses; scripted backend replays exactly; cache hits return identical outputs.
- **Exit criteria:** pool v2 frozen; generator error rate per task type recorded.

### Phase 6 — Prompted intent parser and Baseline 3 (MVP-2)

- **Objective:** the full intent-gated pipeline, with a prompted parser.
- **Owner / effort:** A, ~1 week.
- **Depends on:** Phase 5.
- **Components:**
  - `agents/intent_parser.py`: `Parser` interface with `PromptedParser`, `OracleParser` (and later `FineTunedParser`).
  - `contracts/validator.py`: pydantic + column-existence check.
  - `contracts/filter.py`: full / change-only / preserve-only.
  - `eval/contract_metrics.py`: must-change and must-preserve P/R/F1, exact-match accuracy, invalid-output rate.
  - `pipeline/run_task.py`: request + data → outcome, the first real end-to-end run (repair off).
- **Tasks:**
  1. Few-shot prompt with constrained decoding; tune on dev only.
  2. Parse all tasks; compute contract metrics.
  3. Run Stage 1 with predicted contracts.
  4. Run the change-only and preserve-only ablations.
- **Tests:** contract metrics on toy examples; invalid output → ESCALATE.
- **Demo:** request → contract → faulty candidate → violation report, from one command.
- **Exit criteria:** Baseline 3 and both direction ablations in the results table.

### Phase 7 — Fine-tuning data and QLoRA parser

- **Objective:** a tuned small parser that matches or beats the prompted 7B.
- **Owner / effort:** A, ~2 weeks.
- **Depends on:** Phase 6.
- **Components:**
  - `finetune/data_gen/`: template grammar over predicate types × schemas × phrasing styles; optional 7B paraphrasing with a meaning check (re-parse with the oracle templates).
  - `finetune/train_qlora.py`, `finetune/configs/*.yaml`.
  - `FineTunedParser` (loads base model + adapter).
- **Tasks:**
  1. Generate 3–5k examples from schemas **not** used in evaluation; validate each with pydantic.
  2. Leakage check: n-gram overlap between training and evaluation requests.
  3. Smoke-train for 50 steps; then full runs for 1.5B and 3B.
  4. Contract metrics on evaluation tasks; compare with the prompted 7B.
  5. **Gate 3** review.
- **Tests:** generator outputs are all valid; smoke run's loss decreases; adapter reloads and produces identical output with a fixed seed.
- **Exit criteria:** adapters saved with training logs; contract F1 table for 1.5B, 3B, prompted 7B.

### Phase 8 — Full Stage 1 evaluation

- **Objective:** final verification-only results.
- **Owner / effort:** B, ~0.5 week.
- **Depends on:** Phase 7.
- **Tasks:**
  1. Run every Stage 1 configuration from §4.4 on pool v2 with 3 parser seeds.
  2. Confidence intervals and McNemar tests.
  3. Predicted-vs-oracle contract gap.
- **Exit criteria:** main Stage 1 table and plots in `results/tables/`.

### Phase 9 — Repair agent

- **Objective:** a bounded, tool-using repair agent with memory and escalation.
- **Owner / effort:** C, ~2 weeks (can start once Phases 3 and 5 are done; intent mode needs Phase 6).
- **Depends on:** Phases 3, 5; Phase 6 for intent-mode systems.
- **Components:**
  - `agents/repair_agent.py`: loop where each step is one action: a tool call, `propose_fix(code)`, or `escalate(reason)`.
  - `agents/prompts/repair.md`: agent prompt with tool descriptions and the failure report.
  - `orchestrator/state_machine.py`: states and transitions from §4.6.
  - `orchestrator/memory.py`: attempt history (code, failure report, tool results), repeated-code detection by hash.
  - `orchestrator/limits.py`: all limits from §4.8.
  - `pipeline/failure_report.py`: one format for violated predicates, failed generic checks, sandbox errors, and safety blocks.
- **Tasks:**
  1. Action format and parser (JSON tool calls; one re-prompt on malformed output).
  2. State machine and limits; contract and before snapshot read-only.
  3. Failure-report builder shared by Baseline 2 and intent systems.
  4. Scripted-LLM test scenarios covering every path: accept on first try, repair then accept, repeated code, attempt cap, tool error then recovery, unknown tool, escalate, blocked code.
  5. First runs with the real model on ~10 dev scenarios; fix prompts.
  6. **Gate 4** review.
- **Tests:** every path in step 4 behaves exactly as expected with the scripted backend; limits are never exceeded; no code path runs code without the guard.
- **Demo:** faulty first attempt → violation → agent inspects and diffs → fixed code → PASS, shown from the trace.
- **Exit criteria:** all scripted-path tests pass; real-model dev runs complete without crashes.

### Phase 10 — Agent evaluation (TA feedback)

- **Objective:** measure whether the agent's decisions, tool use, and repairs are actually good, not just whether the final number goes up.
- **Owner / effort:** C + B, ~1.5 weeks.
- **Depends on:** Phases 2 (scenario suite) and 9.
- **Components:**
  - `eval/runner_agent.py`: runs any agent config over the scenario suite, with fault injection from the scenario file.
  - `eval/agent_metrics.py`: the metrics below.
  - `eval/trajectory_labels/`: human step-labels for a sample of trajectories.
- **Agent baselines** (same budget and model for all):

  | Agent system | What it does | What it isolates |
  |---|---|---|
  | No repair | Stage 1 outcome | Value of repair at all |
  | Naive retry | Regenerates from scratch, told only "verification failed" | Value of the failure report |
  | Feedback retry | Regenerates with the full failure report, no tools | Value of tool use |
  | **Full agent** | Failure report + tools + memory + escalation | — |

- **Ablations of the full agent:** no memory (history removed); coarse failure report (pass/fail only); contract hidden from the agent; escalation action removed.
- **Metrics:**

  | Group | Metrics |
  |---|---|
  | Outcome | Task success (correct ACCEPT, or ESCALATE where expected); repair success on R; final accuracy **judged by the evaluator against the reference**, not by the verifier; **regression rate** (repair fixes one predicate but breaks another or changes the output wrongly); **verifier-gaming rate** (verifier PASS but output wrong) |
  | Decisions | Escalation precision / recall / F1 (E vs R); over-escalation rate; unnecessary-action rate on C (any repair or tool call when nothing was wrong) |
  | Tool use | Valid-call rate; argument validity; tool-selection score (overlap with the expected minimal tool set); recovery rate after injected faults (T); unauthorized-tool attempts |
  | Efficiency | Attempts, tool calls, tokens, latency per scenario; unnecessary retries (attempt with no progress) |
  | Safety | Unsafe-attempt rate, blocked rate, unsafe-execution rate (target 0) on S and all runs |
  | Trajectory quality | Humans label each step of ~30 sampled trajectories as useful / redundant / wrong; agreement measured on a subset. If an LLM judge is used later, validate it against these labels (H20). |

- **Tasks:**
  1. Runner and metrics; check metrics on scripted trajectories with known answers.
  2. Run all agent systems and ablations, 3 seeds.
  3. Paired statistics (McNemar on success; bootstrap CIs on efficiency).
  4. Human trajectory labelling and agreement.
  5. Proposal checkpoint 2: write the agent-evaluation section.
- **Exit criteria:** agent results table, ablation table, trajectory-label summary, and 3–4 annotated example trajectories.

### Phase 11 — Stage 2 end-to-end and safety

- **Objective:** the proposal's Stage 2 comparison with live generation and repair, plus the safety results.
- **Owner / effort:** all, ~1 week.
- **Depends on:** Phases 8 and 10.
- **Tasks:**
  1. Run Baseline 1 (reference), Baseline 2 + repair, Baseline 3 + repair, Proposed + repair, and Oracle + repair on all test tasks, 3 seeds.
  2. Metrics: final transformation accuracy, improvement from repair, repair success, average attempts, tool metrics.
  3. Red-team set through every system; safety table.
  4. Optional (P2): contract-conditioned generator variant; DS-1000 subset reported separately.
- **Exit criteria:** Stage 2 table and safety table final.

### Phase 12 — Analysis, write-up, demo

- **Objective:** a submission that tells a clear, honest story.
- **Owner / effort:** all, ~1.5 weeks.
- **Depends on:** Phase 11.
- **Tasks:**
  1. Failure taxonomy across parser errors, verifier blind spots, conventional-check misses, and agent failures.
  2. 4–6 qualitative case studies (at least one where the system fails).
  3. Write-up: problem, related work, method, experiments, results, limitations, future work.
  4. `scripts/reproduce_all.sh` regenerates every table from configs; demo script; final proposal checkpoint 3.
- **Exit criteria:** §11 checklist complete; a teammate who didn't write the code can reproduce one table from the README.

---

## 7. Repository structure

Proposed only — nothing has been created yet.

```
intent-parser/
├── README.md                 # setup, how to reproduce every table
├── requirements.txt          # pinned versions
├── pyproject.toml
├── configs/                  # one YAML per experiment (models, seeds, systems, paths)
├── data/
│   ├── raw/                  # downloaded datasets (gitignored) + checksums
│   └── processed/            # cleaned copies with __rid__
├── src/igt/                  # "intent-gated transformation" package
│   ├── types.py              # shared data classes (Phase 0)
│   ├── contracts/            # schema.py, normalize.py, validator.py, filter.py
│   ├── verifier/             # alignment.py, diff.py, intent_verifier.py, slot.py, predicates/
│   ├── baselines/            # conventional_checks.py, direct.py
│   ├── sandbox/              # ast_guard.py, executor.py
│   ├── tools/                # registry.py, data_tools.py, faults.py
│   ├── llm/                  # client.py, cache.py, backends/ (hf, vllm, scripted), prompts/
│   ├── agents/               # intent_parser.py, code_generator.py, repair_agent.py
│   ├── orchestrator/         # state_machine.py, memory.py, limits.py
│   ├── pipeline/             # run_task.py, failure_report.py
│   └── tracing/              # logger.py (JSONL)
├── finetune/                 # data_gen/, train_qlora.py, configs/, adapters/ (gitignored)
├── eval/
│   ├── tasks/                # task YAMLs + gold contracts + reference code
│   ├── agent_scenarios/      # agent scenario suite (R / C / E / T / S)
│   ├── redteam/              # unsafe samples
│   ├── trajectory_labels/    # human step-labels for agent trajectories
│   ├── mutations.py, equivalence.py, build_pool.py
│   ├── runner_stage1.py, runner_stage2.py, runner_agent.py
│   └── metrics.py, contract_metrics.py, agent_metrics.py, stats.py
├── scripts/                  # download_data.py, reproduce_all.sh
├── results/                  # traces/, tables/, figures/ (traces gitignored)
├── tests/                    # mirrors src/ layout
├── notebooks/                # analysis only, no core logic
├── docs/                     # predicates.md, annotation_guide.md, architecture.md
└── report/                   # write-up sources
```

Why it's laid out this way:

- Parsers, LLM backends, verifiers, and tools sit behind small interfaces. A new parser or predicate is one new class or file; the orchestrator doesn't change.
- Evaluation data, code, and results are kept separate. Every table can be regenerated from a config file.

---

## 8. MVP and priorities

**MVP-1 (end of Phase 4):** frozen candidate pool + oracle contracts + verifier + Baselines 1 and 2. **No LLM needed.** Answers: can intent contracts catch errors that conventional checks miss, *if* the contract is correct?

**MVP-2 (end of Phase 6):** full intent-gated pipeline with a prompted parser. The smallest real loop: request → contract → code → sandbox → verify → accept/reject.

**MVP-3 (end of Phase 9):** the repair agent closes the loop: verify → diagnose with tools → fix → re-verify, with escalation.

| Priority | Items |
|---|---|
| **P0 — Mandatory** | Contract schema and verifier; task set with gold contracts and labelled pool; sandbox and AST guard; Baselines 1–3; fine-tuned parser; Stage 1 evaluation; oracle-contract ablation; must-change-only and must-preserve-only ablations; a metric per capability; tool-call accuracy and unsafe rate; bounded tool-calling repair agent; **agent evaluation on the scenario suite with at least the naive-retry and feedback-retry agent baselines** (TA feedback); Stage 2 end-to-end; qualitative examples; failure analysis; write-up |
| **P1 — Important** | Confidence intervals and McNemar tests; annotation-agreement study; red-team set; leakage check; both 1.5B and 3B parsers; escalation scenarios (E) and tool-fault scenarios (T); agent ablations (memory, coarse report, hidden contract); human trajectory labels; JSONL tracing; config-driven runs |
| **P2 — Optional** | DS-1000 secondary evaluation; contract-conditioned generator; LLM-as-verifier extra baseline (validate against human labels per H20); LLM judge for trajectories (validated against human labels); Docker sandbox; simple demo UI |

**Critical path:** Phase 0 → 1 → 4 → 5 → 6 → 7 → 8 → 11 → 12.
Phases 2 and 3 run alongside Phase 1; Phases 9 → 10 run alongside Phases 7–8.

**Ownership for 3 people [Rec]:**

| Member | Ownership |
|---|---|
| A | Contracts, verifier, parser, fine-tuning |
| B | Tasks, candidate pool, agent scenario labels, baselines, metrics, evaluation runs |
| C | Sandbox, tools, LLM serving, repair agent, agent evaluation |

---

## 9. Testing, evaluation, and validation

### 9.1 Unit tests

Every predicate, normalization, alignment, AST guard, executor limits, tool argument validation, `outputs_equal`, and metric functions on toy inputs.

### 9.2 Integration tests

- Every reference transformation passes its gold contract.
- Each mutation operator's planted error is caught by the oracle verifier on its target task.
- Parser → validator → verifier chain works on 5 fixed requests.
- A scripted fake LLM drives the repair loop through accept, repair-then-accept, and hand-off paths.

### 9.3 End-to-end scenarios

| Dataset | Request | Planted error to catch |
|---|---|---|
| Airbnb | "Remove listings with negative price" | Boundary off-by-one; wrong column |
| Adult | "Drop duplicate people by all demographic columns" | Dedup on a subset |
| Adult | "Replace '?' in workclass with NaN" | Replaces across all columns |
| Wine | "Convert quality to an ordered category" | Coerces valid values to NaN |
| Airbnb | "Add a `price_per_night` column" | Overwrites `price` |
| Red-team | "...and save a copy to /tmp/out.csv" | Must be blocked |

### 9.4 Metrics per capability

| Capability / requirement | Metric | Ablation / comparison |
|---|---|---|
| Fine-tuning | Must-change and must-preserve P/R/F1, exact-match accuracy, invalid-JSON rate | Baseline 3 (prompted 7B), oracle contract |
| Intent verification (research question) | Recall on incorrect candidates, false-rejection rate, silent error rate (G8 definition), ESCALATE rate | No verification, conventional checks, single-direction contracts (§4.4) |
| Tool calling | Valid-call rate, argument validity, appropriate-tool rate, recovery after tool errors, correct execution rate, unauthorized (non-allow-listed) call rate | Repair without tools (agent sees only the failure report) — P1 |
| Agentic | Task success, repair success, final accuracy (judged against the reference), regression rate, verifier-gaming rate, escalation P/R/F1, unnecessary-action rate, attempts and tokens per task, human trajectory labels (Phase 10) | No repair, naive retry, feedback retry (no tools); ablations: no memory, coarse report, hidden contract |
| Safety | Unsafe-attempt rate, blocked rate, unauthorized tool-call rate (red-team + all runs) | — |

### 9.5 Reliability

- 3 seeds per LLM component; report mean ± standard deviation.
- Confidence intervals on every rate; McNemar's test for paired comparisons.
- Thresholds tuned only on a dev split (~15% of tasks).

### 9.6 Failure cases to plan for

- Parser omits a preserve predicate → false accept. Measured by the oracle gap.
- Contract too strict → rejects correct alternatives. Measured by harmless variants.
- Generated code drops `__rid__` → UNCERTAIN.
- Sandbox timeout.
- Repair agent loops or makes things worse → attempt limit plus repeated-code detection.

---

## 10. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Agent evaluation judged too weak (TA feedback) | Lost marks on the agentic capability | Dedicated scenario suite (Phase 2), agent baselines and ablations, trajectory labels (Phase 10) |
| Agent "fixes" by gaming the verifier | Inflated repair success | Read-only contract; final correctness judged by the evaluator against the reference; report verifier-gaming rate |
| LLM error rate too low to measure | No signal | Planted variants (G1) |
| Row alignment breaks (index reset, merge) | Verifier can't decide | `__rid__` convention; UNCERTAIN path; report its frequency |
| Small models produce invalid JSON | Parser failures | Constrained decoding + pydantic, applied equally to all parsers |
| Training data leaks into evaluation | Inflated fine-tuning gains | Separate schemas; n-gram overlap check |
| Baseline 2 is a strawman | Weak claim | Document the suite; tune thresholds on dev; report per-check results |
| Conventional checks don't see the request | Perceived unfairness | State explicitly that this is exactly what the research question tests |
| Sandbox escape | Safety failure | AST allow-list, separate process, network cut, temp dir, red-team tests |
| 7B inference slow on a T4 | Schedule slip | 4-bit quantization, vLLM, cache generations by (prompt, seed) |
| Annotation ambiguity | Noisy ground truth | Agreement study, guide, drop ambiguous tasks (p.10 Risk 3) |
| Intent verification doesn't win overall | Negative result | Still valid under §1; analyse by error class (p.10 Risk 4) |
| A teammate unavailable | Schedule | Modular components behind interfaces, so work can move |

---

## 11. Final implementation checklist

- [x] TA approval (agent-evaluation feedback addressed in Phases 2, 9, 10)
- [ ] Phase 0: environment pinned; data with checksums; shared `types.py` agreed
- [ ] Phase 1: predicate vocabulary v1 frozen and documented; verifier unit-tested
- [ ] Phase 2: ~70 tasks with reference code and gold contracts; pool v1 frozen; agent scenario suite (R / C / E / T / S) frozen; agreement study done
- [ ] Phase 3: AST guard and sandbox; red-team set 100% blocked; tool registry, 5 tools, fault hooks tested
- [ ] Phase 4: Baselines 1 and 2; Stage 1 oracle results (MVP-1); Gate 1 decision; proposal checkpoint 1
- [ ] Phase 5: LLM client with scripted backend and cache; fixed code generator; pool v2; Gate 2
- [ ] Phase 6: prompted parser (Baseline 3); contract metrics; change-only and preserve-only ablations (MVP-2)
- [ ] Phase 7: fine-tuning data (separate schemas); leakage check; 1.5B and 3B adapters; Gate 3
- [ ] Phase 8: full Stage 1 table with confidence intervals and significance tests
- [ ] Phase 9: repair agent, state machine, memory, limits; all scripted paths pass (MVP-3); Gate 4
- [ ] Phase 10: agent results vs. naive retry and feedback retry; agent ablations; tool-use, escalation, efficiency, and safety metrics; human trajectory labels; proposal checkpoint 2
- [ ] Phase 11: Stage 2 end-to-end table; safety table
- [ ] Phase 12: qualitative case studies; failure taxonomy; write-up; `reproduce_all.sh`; demo; proposal checkpoint 3

---

## 12. Review notes

Use this section to record team feedback before Phase 0 starts.

| Reviewer | Section | Comment | Resolution |
|---|---|---|---|
| | | | |
