# Technology Stack, Environment, and Setup

## 1. Selected stack

| Layer | Technology | Why selected | Status |
|---|---|---|---|
| Language | Python 3.12.3 (with 3.11 compatibility) | High performance, modern typing, audio & ML ecosystem | Core Runtime |
| Audio processing | Librosa + SoundFile + SciPy | Decodes WAV, extracts log-Mel spectrograms, audio checks | Core Engine |
| Numeric/data | NumPy + Pandas | Array operations, manifests, feature scaling, telemetry | Core Engine |
| Primary ML | PyTorch (CPU-optimized) | Symmetric Dense Autoencoder with early stopping | Core Anomaly Model |
| Baseline ML | Scikit-learn | Isolation Forest benchmark and classification metrics | Core Baseline |
| Cloud Hosting | Streamlit Community Cloud + Debian Linux | Live, public, 24/7 web access: `https://pulse-forge-ai.streamlit.app/` | Live Web Demo |
| Presentation Site | HTML5 + SVG + Tailwind (GitHub Pages) | Executive machinery blueprint & keynote showcase: `https://tamzid1103.github.io/Pulse-Forge-Ai/presentation/` | Live Presentation |
| Primary UI | Streamlit (Multi-View Industrial Suite) | 5 specialized views: Diagnostic Studio, Fleet, Benchmarks, Audit, Standards | Operational Dashboard |
| Metadata Store | SQLite 3 | Embedded, zero-configuration audit trail with CSV export (`src/persistence.py`) | Audit History |
| Factory Brain Copilot | Groq LPU (Llama 3.1/3.3) + Offline SOP Engine | 500+ tok/s OSHA work order generation with zero-crash fallback (`src/copilot.py`) | Operational Copilot |
| Testing | Pytest | 138 automated unit, integration, and regression tests (100% pass rate) | Quality Gate |
| Version control | Git + GitHub | Reproducible collaboration and release tracking | Version Control |
| Protocol / Tools | Model Context Protocol (MCP) | Exposes PulseForge tools & resources to agent ecosystems (`src/mcp_server.py`) | Phase 2 (M9) |
| Application API | FastAPI + Uvicorn + WebSockets | Headless REST & WebSocket audio streaming backend (`src/api.py`) | Phase 2 (M10) |
| Web UI | Lovable (React / Vite / Tailwind) | Enterprise plant floor dashboard for multi-machine control | Phase 2 (M11) |
| Multi-Machine | ModelRegistry + MIMII Extensions | Extends detection to Centrifugal Pumps, Valves, Slide Rails | Phase 2 (M12) |
| Edge Runtime | ONNX Runtime CPU | Sub-25ms inference on Raspberry Pi 5 / NVIDIA Jetson Orin Nano | Phase 2 (M13) |

Do not introduce React, PostgreSQL, Redis, Docker, cloud storage, Kubernetes, or a hosted AI API for the MVP unless the project owner approves a specific need.

## 2. Runtime and hardware

- Start with Python 3.11 and verify dependency compatibility in the selected operating system.
- CPU inference/training on a small subset is acceptable for the first prototype.
- A compatible NVIDIA GPU is optional; install a PyTorch build matching the actual CUDA environment using the official PyTorch selector.
- Keep a CPU fallback and ensure the app can run on the presentation laptop.
- Disk capacity matters: official MIMII archives are large. Download a small fan subset first.

## 3. Suggested requirements file

The root `requirements.txt` contains the initial package names. It is intentionally not locked to exact versions. After the first successful clean install, record the tested versions in a lock file or pinned requirements file.

Packages:

- `numpy`, `pandas`, `scipy`, `matplotlib`
- `librosa`, `soundfile`
- `scikit-learn`, `torch`, `torchaudio`
- `streamlit`, `plotly`
- `fastapi`, `uvicorn`, `python-multipart` (optional API)
- `pytest`

PyTorch installation can depend on operating system and CUDA version. If the generic `pip install -r requirements.txt` selects an unsuitable build, install PyTorch separately from its official instructions, then install the remaining requirements.

## 4. Project setup

### Linux

```bash
mkdir PulseForgeAI
cd PulseForgeAI
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
mkdir PulseForgeAI
cd PulseForgeAI
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked by execution policy, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

## 5. Environment variables

The local MVP should not need secrets or external service credentials. Keep configuration in a local config file or defaults. If environment variables are used, provide a `.env.example` and never commit `.env`.

Suggested variables only if implemented:

```dotenv
PULSEFORGE_ENV=development
PULSEFORGE_MODEL_DIR=./model_artifacts
PULSEFORGE_DEMO_AUDIO_DIR=./demo_audio
PULSEFORGE_MAX_UPLOAD_MB=25
PULSEFORGE_LOG_LEVEL=INFO
PULSEFORGE_ENABLE_HISTORY=false
```

These are suggested names/defaults, not currently required values. No external API key should be necessary for the MVP.

## 6. Directory structure after application implementation

```text
PulseForgeAI/
├── app/
│   ├── dashboard.py
│   └── components/              # optional UI helpers
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── schemas.py
│   ├── audio_io.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── quality.py
│   ├── models/
│   │   ├── autoencoder.py
│   │   └── baseline.py
│   ├── thresholding.py
│   ├── familiarity.py
│   ├── decision.py
│   ├── service.py
│   ├── persistence.py            # optional SQLite
│   └── api.py                    # optional FastAPI adapter
├── scripts/
│   ├── inspect_dataset.py
│   ├── build_manifest.py
│   ├── train_autoencoder.py
│   └── evaluate.py
├── notebooks/
│   ├── 01_audio_exploration.ipynb
│   └── 02_experiment_review.ipynb
├── data/                         # ignored by Git; local only
├── demo_audio/                   # small approved samples only
├── model_artifacts/              # ignored by Git unless approved
├── results/                      # ignored by Git; export selected reports
├── tests/
├── docs/
├── requirements.txt
├── README.md
├── AGENTS.md
└── .gitignore
```

## 7. Configuration and model artifact files

The model bundle should include:

- model checkpoint (e.g. `fan_autoencoder.pt`);
- preprocessing configuration (sample rate, channel policy, segment length, Mel bins, hop length, normalization);
- anomaly threshold and the selection rule used;
- model version / training run ID;
- dataset manifest hash or version reference;
- training seed and environment/package versions;
- validation and test metrics.

A model must refuse to load if required config fields are missing or incompatible.

## 8. Run commands (after implementation)

Streamlit MVP:

```bash
streamlit run app/dashboard.py
```

Optional FastAPI:

```bash
uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

Tests:

```bash
pytest -q
```

Dataset inspection / training commands will be defined by Antigravity under `scripts/` and then documented in the root README. Do not invent command names before the scripts exist.

## 9. Database initialization

No database initialization is required for the MVP. If SQLite history is approved, create the database automatically on app startup using the schema in `DATABASE_SCHEMA.md`; store it under a local ignored data directory.

## 10. Common setup troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `python3.11` not found | Python not installed or not on PATH | Install Python 3.11 and confirm `python --version` / `py -3.11 --version`. |
| Cannot activate venv on PowerShell | Execution policy | Use Command Prompt activation or an approved PowerShell policy. |
| `soundfile` cannot read file | Unsupported/corrupt audio or audio backend issue | Verify WAV decodes with SoundFile; inspect file header and install supported system audio libraries if needed. |
| Librosa / NumPy incompatibility | Package versions conflict | Recreate venv and install a known-compatible set; record pinned versions. |
| PyTorch CUDA unavailable | Installed build does not match CUDA/driver | Use CPU initially or install matching PyTorch build from official selector. |
| Out-of-memory during feature extraction | Too many/long audio segments loaded together | Process in chunks, reduce batch size, use a small subset. |
| No model artifact found | Training not run or wrong path | Run the documented training step and verify configured artifact path. |
| Slow first launch | Model initialization / JIT / file scan | Cache model after load and avoid scanning the full dataset at every app launch. |
| Dataset download too large | Full archive selected | Download one fan archive / smaller documented subset and verify storage before extraction. |

## 11. Official technical references

- [Python downloads](https://www.python.org/downloads/)
- [PyTorch installation](https://pytorch.org/get-started/locally/)
- [Librosa documentation](https://librosa.org/doc/latest/)
- [Streamlit documentation](https://docs.streamlit.io/)
- [FastAPI documentation](https://fastapi.tiangolo.com/)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
