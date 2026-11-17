# Documentation Index

This documentation package is the implementation handover for PulseForge AI. Read documents in this order when starting from a clean context.

| Order | File | Purpose |
|---:|---|---|
| 1 | `../PROJECT_CONTEXT.md` | Canonical concept, scope, agreed choices, limitations, and project status. |
| 2 | `../AGENTS.md` | Persistent instructions for Antigravity / coding agents. |
| 3 | `PROJECT_OVERVIEW.md` | Product overview, objectives, users, value proposition, deliverables. |
| 4 | `PROBLEM_STATEMENT.md` | Problem framing, use cases, constraints, impact, and claim boundaries. |
| 5 | `REQUIREMENTS.md` | Functional / non-functional requirements and priorities. |
| 6 | `FEATURE_SPECIFICATIONS.md` | Feature workflows, modules, data/API needs, dependencies, acceptance criteria. |
| 7 | `SYSTEM_ARCHITECTURE.md` | Component architecture, Mermaid diagrams, workflows, and data flow. |
| 8 | `DATABASE_SCHEMA.md` | Explicit no-database MVP decision and optional SQLite metadata schema. |
| 9 | `API_DOCUMENTATION.md` | Internal service contract and optional FastAPI endpoints. |
| 10 | `UI_UX_GUIDELINES.md` | Page layout, interactions, states, accessibility, copy, and visual guidance. |
| 11 | `TECH_STACK.md` | Selected technologies, rationale, versions, setup, environment, and run commands. |
| 12 | `DATASET_AND_MODEL_GUIDE.md` | MIMII acquisition, data preparation, model approach, thresholds, evaluation, attribution. |
| 13 | `IMPLEMENTATION_PLAN.md` | Sequential milestones, files, tasks, testing, outputs, acceptance gates, 48-hour plan. |
| 14 | `TESTING_PLAN.md` | Unit, integration, API, workflow, security, performance, and demo QA. |
| 15 | `DEMO_GUIDE.md` | Demo script, test scenarios, backup plan, pitch structure, technical highlights. |
| 16 | `DECISIONS_AND_OPEN_QUESTIONS.md` | Confirmed decisions, assumptions, unresolved information, and decision log. |
| 17 | `HANDOVER_AUDIT.md` | Consistency audit and agent handoff checklist. |

## Source-of-truth rule

If documents appear to conflict, use this order:

1. `PROJECT_CONTEXT.md` for the product scope and settled decisions.
2. `AGENTS.md` for agent behavior and coding restrictions.
3. The feature / architecture / model documents for implementation detail.
4. The implementation plan for sequencing.

If an implementation change becomes necessary, update all affected documents and record the reason in `DECISIONS_AND_OPEN_QUESTIONS.md`. Do not silently alter the project scope.
