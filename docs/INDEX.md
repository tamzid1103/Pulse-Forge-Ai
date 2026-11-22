# Documentation Index

This documentation package is the implementation handover for PulseForge AI. Read documents in this order when starting from a clean context.

| Order | File | Purpose |
|---:|---|---|
| 1 | `../PROJECT_CONTEXT.md` | Canonical concept, scope, agreed choices, limitations, and project status. |
| 2 | `../AGENTS.md` | Persistent instructions for Antigravity / coding agents. |
| 3 | `PROJECT_MASTER_GUIDE.md` | Definitive project guide, website walkthrough, and pitch manual. |
| 4 | `PROJECT_OVERVIEW.md` | Product overview, objectives, users, value proposition, deliverables. |
| 5 | `PROBLEM_STATEMENT.md` | Problem framing, use cases, constraints, impact, and claim boundaries. |
| 6 | `REQUIREMENTS.md` | Functional / non-functional requirements and priorities. |
| 7 | `FEATURE_SPECIFICATIONS.md` | Feature workflows, modules, data/API needs, dependencies, acceptance criteria. |
| 8 | `SYSTEM_ARCHITECTURE.md` | Component architecture, Mermaid diagrams, workflows, and data flow. |
| 9 | `DATABASE_SCHEMA.md` | SQLite metadata audit schema and CSV export specification. |
| 10 | `API_DOCUMENTATION.md` | Internal service contract, Copilot interface, and FastAPI endpoints. |
| 11 | `UI_UX_GUIDELINES.md` | Page layout, interactions, states, accessibility, copy, and visual guidance. |
| 12 | `TECH_STACK.md` | Selected technologies, rationale, versions, setup, environment, and run commands. |
| 13 | `DATASET_AND_MODEL_GUIDE.md` | MIMII acquisition, data preparation, model approach, thresholds, evaluation, attribution. |
| 14 | `IMPLEMENTATION_PLAN.md` | Sequential milestones, tasks, testing, acceptance gates, and Phase 2 Finals roadmap. |
| 15 | `TESTING_PLAN.md` | Unit, integration, API, workflow, security, performance, and QA. |
| 16 | `DEMO_GUIDE.md` | Demo script, test scenarios, backup plan, pitch structure, technical highlights. |
| 17 | `PRELIMINARY_SUBMISSION_DOSSIER.md` | Official submission text and answers for CloudCamp portal. |
| 18 | `DECISIONS_AND_OPEN_QUESTIONS.md` | Confirmed decisions, assumptions, resolution status, and DEC-001 through DEC-012 log. |
| 19 | `CLOUDCAMP_EVENT_GUIDE.md` | Competition rubric mapping, module scoring, and rules guide. |
| 20 | `HANDOVER_AUDIT.md` | Consistency audit and agent handoff checklist. |

## Source-of-truth rule

If documents appear to conflict, use this order:

1. `PROJECT_CONTEXT.md` for the product scope and settled decisions.
2. `AGENTS.md` for agent behavior and coding restrictions.
3. The feature / architecture / model documents for implementation detail.
4. The implementation plan for sequencing.

If an implementation change becomes necessary, update all affected documents and record the reason in `DECISIONS_AND_OPEN_QUESTIONS.md`. Do not silently alter the project scope.
