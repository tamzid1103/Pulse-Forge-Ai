# PulseForge AI

**Hear the Warning. Trust the Signal. Prevent the Downtime.**

PulseForge AI is a hackathon prototype for trustworthy acoustic predictive maintenance. It analyzes industrial-machine audio, detects behavior that differs from learned normal operation, checks whether the recording is suitable and familiar enough to support the result, and presents an explainable inspection recommendation.

> **Project status:** Documentation and implementation plan only. The application, model, API, dashboard, database, and evaluation results have **not** been implemented or verified yet.

## Selected competition direction

- **Domain:** IndustrySphere AI
- **Primary sub-track:** Predictive Plant
- **Supporting sub-track:** Factory Brain (dashboard / operational layer only)
- **First supported machine:** Fan, using MIMII audio
- **Future machine extensions:** Pump, motor, and other rotating equipment after separate data/model validation

## MVP in one sentence

A user selects the fan machine type, uploads a valid WAV recording, and receives an anomaly assessment, signal-quality checks, an uncertainty/review flag when appropriate, visual evidence, and a practical inspection recommendation.

## Documentation index

Start here, in order:

1. [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) — canonical project brief, decisions, assumptions, and constraints.
2. [`AGENTS.md`](AGENTS.md) — persistent instructions for Google Antigravity or another coding agent.
3. [`docs/INDEX.md`](docs/INDEX.md) — map of all documentation.
4. [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) — sequential build plan and milestone gates.
5. [`docs/SYSTEM_ARCHITECTURE.md`](docs/SYSTEM_ARCHITECTURE.md) — architecture and Mermaid diagrams.
6. [`docs/FEATURE_SPECIFICATIONS.md`](docs/FEATURE_SPECIFICATIONS.md) — feature-level requirements and acceptance criteria.
7. [`docs/DATASET_AND_MODEL_GUIDE.md`](docs/DATASET_AND_MODEL_GUIDE.md) — data handling, model strategy, evaluation, and attribution.
8. [`docs/TESTING_PLAN.md`](docs/TESTING_PLAN.md) and [`docs/DEMO_GUIDE.md`](docs/DEMO_GUIDE.md) — quality and live-demo preparation.

## Quick start for the development agent

1. Read `PROJECT_CONTEXT.md` and `AGENTS.md` completely.
2. Read `docs/DECISIONS_AND_OPEN_QUESTIONS.md` and resolve only decisions that block the MVP.
3. Follow `docs/IMPLEMENTATION_PLAN.md` milestone by milestone. Do not skip the data audit or baseline evaluation.
4. Start with a single fan subset. Do not download every MIMII archive by default; the official public dataset is large and individual fan archives can be around 10 GB.
5. Build a working local Streamlit prototype first. Add a separate FastAPI service only after the model pipeline works end to end or when a clear integration need exists.
6. Never invent performance numbers, machine fault diagnoses, production deployments, partners, or NRB collaborations.

## Recommended local environment

- Python 3.11 (recommended starting point; verify library compatibility on the selected OS).
- Linux or Windows with a supported Python installation.
- CPU is enough for an initial small experiment; a compatible GPU can reduce training time but is not an MVP requirement.
- Use a virtual environment.
- Install packages from `requirements.txt`. For PyTorch, use the official installation selector if a CUDA-specific build is required.

### Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `.env.example` to `.env` only if you need to override local configuration; no external API key is required. Never commit `.env`.

If you later enable the optional FastAPI adapter, install its extra dependencies with `pip install -r requirements-api.txt`. This is not needed for the Streamlit MVP.

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

## Dataset

Use the official [MIMII dataset record on Zenodo](https://zenodo.org/records/3384388). Begin with a small fan subset. The public dataset covers fans, pumps, valves, and slide rails; the official record describes normal and anomalous recordings. Review the dataset license and citation requirements before redistribution or public release.

Do not put the full dataset, raw audio archives, trained model weights, or user-uploaded recordings in Git unless there is an explicit reason and the repository policy permits it. Keep the data local and document the exact subset used.

## Running the application

The commands below become valid after Antigravity implements the app. They are the intended interface, not evidence that these files already exist:

```bash
streamlit run app/dashboard.py
```

Optional API service, only if implemented:

```bash
uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

## Core product safeguards

- The MVP is **fan-only**. Do not imply that a fan model is validated for pumps or motors.
- A high anomaly score is not a calibrated probability of failure.
- Uncertainty and input-quality flags must be distinguishable from anomaly status.
- A poor-quality or unfamiliar recording should lead to a review / inconclusive result, not an overconfident diagnosis.
- The app is decision support only. A qualified operator or engineer makes maintenance decisions.
- No automatic machine shutdown or control of industrial equipment is in scope.

## Useful external references

- [MIMII official Zenodo record](https://zenodo.org/records/3384388)
- [MIMII official baseline repository](https://github.com/MIMII-hitachi/mimii_baseline)
- [MIMII DUE: domain shifts due to operational/environmental changes](https://zenodo.org/records/4740355)
- [MIMII DG: domain generalization dataset](https://zenodo.org/records/6529888)
- [Librosa documentation](https://librosa.org/doc/latest/)
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
- [Streamlit documentation](https://docs.streamlit.io/)
- [FastAPI documentation](https://fastapi.tiangolo.com/)

See `docs/DATASET_AND_MODEL_GUIDE.md` for dataset details and attribution guidance.
