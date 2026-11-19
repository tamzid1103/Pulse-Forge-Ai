"""
PulseForge AI — Operator Dashboard (Milestone M6).

Production Streamlit interface for Industrial Acoustic Anomaly Detection.
Provides single-click curated demo scenarios, live WAV file analysis,
dual metric visualization (Autoencoder + Baseline), latent familiarity checking,
and complete ISO 13374-1 compliant operational decision cards.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any

# Ensure matplotlib uses writable cache directory in restricted environments
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
os.makedirs("/tmp/matplotlib_cache", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st

from src.config import AppConfig, setup_logging
from src.decision import DecisionStatus
from src.persistence import AnalysisStore
from src.schemas import AnalysisResult
from src.service import PulseForgeService

logger = logging.getLogger("pulseforge.dashboard")


# ---------------------------------------------------------------------------
# Streamlit Page Setup & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PulseForge AI | Acoustic Anomaly Detector",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Base typography and background styling */
.main {
    background-color: #0b0f19;
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Header badge styling */
.pulse-header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.25rem 1.75rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
}

.pulse-title {
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    color: #f8fafc;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.pulse-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-top: 0.35rem;
    margin-bottom: 0;
}

.badge-tag {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-right: 0.5rem;
}
.badge-plant { background-color: #1e3a8a; color: #93c5fd; border: 1px solid #2563eb; }
.badge-machine { background-color: #312e81; color: #c7d2fe; border: 1px solid #4f46e5; }
.badge-edge { background-color: #064e3b; color: #6ee7b7; border: 1px solid #059669; }

/* Decision Cards */
.decision-card {
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1.2rem 0;
    border-width: 2px;
    border-style: solid;
}

.card-normal {
    background: linear-gradient(180deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 78, 59, 0.15) 100%);
    border-color: #10b981;
}

.card-anomaly {
    background: linear-gradient(180deg, rgba(239, 68, 68, 0.15) 0%, rgba(127, 29, 29, 0.2) 100%);
    border-color: #ef4444;
}

.card-review {
    background: linear-gradient(180deg, rgba(245, 158, 11, 0.15) 0%, rgba(120, 53, 15, 0.2) 100%);
    border-color: #f59e0b;
}

.card-invalid {
    background: linear-gradient(180deg, rgba(107, 114, 128, 0.15) 0%, rgba(31, 41, 55, 0.2) 100%);
    border-color: #6b7280;
}

.status-title {
    font-size: 1.5rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.status-normal { color: #34d399; }
.status-anomaly { color: #f87171; }
.status-review { color: #fbbf24; }
.status-invalid { color: #9ca3af; }

.action-box {
    background-color: rgba(15, 23, 42, 0.7);
    border-left: 4px solid #38bdf8;
    padding: 0.75rem 1rem;
    border-radius: 4px;
    margin-top: 1rem;
    font-size: 0.92rem;
}

/* Metric tile */
.metric-box {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

.metric-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    color: #94a3b8;
    letter-spacing: 0.05em;
    font-weight: 600;
}

.metric-val {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 0.25rem 0;
}

.metric-sub {
    font-size: 0.75rem;
    color: #64748b;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Service Initialization & Caching
# ---------------------------------------------------------------------------
@st.cache_resource
def get_service() -> PulseForgeService:
    """Initialize and cache the PulseForgeService pipeline singleton."""
    cfg = AppConfig()
    db_path = cfg.paths.project_root / "data" / "analysis_history.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = AnalysisStore(db_path)
    return PulseForgeService(config=cfg, store=store)


service = get_service()
config = service.config


# ---------------------------------------------------------------------------
# Visualizations: Waveform & Spectrogram
# ---------------------------------------------------------------------------
def plot_acoustic_features(audio_data: np.ndarray, sample_rate: int) -> plt.Figure:
    """Render high-resolution dual-panel waveform and log-mel spectrogram."""
    import librosa
    import librosa.display

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.5), dpi=100)
    fig.patch.set_facecolor("#0f172a")

    # 1. Waveform Plot
    t = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
    ax1.set_facecolor("#1e293b")
    ax1.plot(t, audio_data, color="#38bdf8", linewidth=0.7, alpha=0.9)
    ax1.set_xlim(0, t[-1] if len(t) > 0 else 1.0)
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_ylabel("Amplitude", color="#94a3b8", fontsize=8)
    ax1.tick_params(colors="#94a3b8", labelsize=7)
    ax1.grid(True, linestyle="--", alpha=0.2, color="#64748b")
    ax1.set_title("Acoustic Time-Domain Waveform", color="#f8fafc", fontsize=9, loc="left", pad=4)

    # 2. Log-Mel Spectrogram
    ax2.set_facecolor("#1e293b")
    try:
        s_mel = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sample_rate,
            n_fft=config.features.n_fft,
            hop_length=config.features.hop_length,
            n_mels=config.features.n_mels,
            fmin=config.features.fmin,
            fmax=config.features.fmax,
        )
        s_db = librosa.power_to_db(s_mel, ref=np.max)
        img = librosa.display.specshow(
            s_db,
            sr=sample_rate,
            hop_length=config.features.hop_length,
            x_axis="time",
            y_axis="mel",
            fmin=config.features.fmin,
            fmax=config.features.fmax,
            ax=ax2,
            cmap="viridis",
        )
        cbar = fig.colorbar(img, ax=ax2, format="%+2.0f dB", pad=0.015, shrink=0.9)
        cbar.ax.tick_params(colors="#94a3b8", labelsize=6)
    except Exception as e:
        logger.warning("Failed to render spectrogram: %s", e)
        ax2.text(0.5, 0.5, "Spectrogram unavailable for this signal", color="#94a3b8", ha="center")

    ax2.set_xlabel("Time (seconds)", color="#94a3b8", fontsize=8)
    ax2.set_ylabel("Freq (Hz)", color="#94a3b8", fontsize=8)
    ax2.tick_params(colors="#94a3b8", labelsize=7)
    ax2.set_title("Log-Mel Spectrogram (40 filter banks)", color="#f8fafc", fontsize=9, loc="left", pad=4)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar: Equipment Telemetry & Machine Profile
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Monitored Equipment")
    st.markdown(
        """
        **Asset:** Industrial Centrifugal Fan  
        **Machine ID:** `fan_id_00`  
        **Location:** Plant Alpha — Line 3  
        **Sensor:** Acoustic Array (16 kHz, 16-bit PCM)
        """
    )
    st.divider()

    st.markdown("### 🧠 ML Inference Engine")
    thresh = service.detector.threshold if service.detector else 0.8904
    fam_thresh = service.familiarity.threshold if service.familiarity else 9.2549
    st.markdown(
        f"""
        - **Primary Model:** Deep Autoencoder (PyTorch CPU)
        - **Bottleneck Latent Dims:** 16
        - **Calibrated Anomaly Threshold:** `{thresh:.4f}`
        - **Latent Familiarity Threshold:** `{fam_thresh:.4f}`
        - **Baseline Model:** Isolation Forest (50 trees)
        """
    )
    st.divider()

    st.markdown("### 📋 Audit & History")
    recent_runs = service.get_history(limit=5)
    st.caption(f"Showing last {len(recent_runs)} recorded runs:")
    for r in recent_runs:
        s_val = r.get("decision_status", "unknown")
        icon = "🟢" if s_val == "normal" else "🔴" if s_val == "possible_anomaly" else "🟡"
        label = r.get("input_label") or "audio.wav"
        score = r.get("anomaly_score", 0.0)
        ms = r.get("processing_ms", 0)
        created = str(r.get("created_at", ""))[:19]
        st.markdown(
            f"**{icon} {label}**  \n"
            f"<small style='color:#94a3b8'>{created} | Score: {score:.3f} | {ms}ms</small>",
            unsafe_allow_html=True,
        )

    if st.button("🗑️ Clear Audit Log", use_container_width=True):
        service.clear_history()
        st.success("Audit log cleared.")
        st.rerun()

    st.divider()
    st.markdown(
        """
        <div style="font-size: 0.72rem; color: #64748b; line-height: 1.4;">
        <strong>Safety Disclaimer:</strong> PulseForge AI is an edge acoustic decision-support system.
        It does not replace mandatory physical Lockout/Tagout (LOTO) protocols or scheduled manual maintenance inspections.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="pulse-header">
        <div class="pulse-title">
            <span>🏭 PulseForge AI</span>
            <span style="font-size: 0.9rem; font-weight: 500; color: #38bdf8;">v1.0-preliminary</span>
        </div>
        <p class="pulse-subtitle">
            Edge Acoustic Anomaly Detection & Self-Validating Diagnostic Intelligence
        </p>
        <div style="margin-top: 0.75rem;">
            <span class="badge-tag badge-plant">Track: Predictive Plant</span>
            <span class="badge-tag badge-machine">Target: Industrial Fan</span>
            <span class="badge-tag badge-edge">Mode: Deterministic Edge Policy</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Audio Input Selector (1-Click Curated Demos + Upload)
# ---------------------------------------------------------------------------
st.markdown("#### 🎧 Audio Input Selection")

demo_dir = config.paths.project_root / "demo_audio"

SCENARIO_OPTIONS = {
    "🟢 Scenario 1: Normal Operating Fan (Healthy Baseline)": "01_normal_fan.wav",
    "🔴 Scenario 2: Bearing Anomaly (Harmonic Rattling Fault)": "02_bearing_anomaly.wav",
    "🟡 Scenario 3: Silent Sensor Failure (Disconnected / Zero Energy)": "03_silent_sensor_failure.wav",
    "🟡 Scenario 4: Clipped Sensor Saturation (Severe ADC Overflow)": "04_clipped_sensor_saturation.wav",
    "🟡 Scenario 5: Unfamiliar Acoustic Noise (Out-of-Distribution Shift)": "05_unfamiliar_operating_noise.wav",
    "📁 Upload Custom Industrial WAV File": "CUSTOM_UPLOAD",
}

selected_scenario_label = st.selectbox(
    "Choose a curated demonstration scenario or upload custom recording:",
    options=list(SCENARIO_OPTIONS.keys()),
    index=0,
)

selected_file_code = SCENARIO_OPTIONS[selected_scenario_label]

audio_bytes: bytes | None = None
active_filename = ""

if selected_file_code == "CUSTOM_UPLOAD":
    uploaded_file = st.file_uploader(
        "Upload machine audio recording (.wav format, max 30 seconds)",
        type=["wav"],
        help="Upload 16 kHz WAV audio file for automated acoustic analysis.",
    )
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        active_filename = uploaded_file.name
else:
    demo_path = demo_dir / selected_file_code
    if demo_path.is_file():
        audio_bytes = demo_path.read_bytes()
        active_filename = selected_file_code
    else:
        st.error(f"Demo file not found at {demo_path}. Please run `scripts/prepare_demo_audio.py`.")


# ---------------------------------------------------------------------------
# Pipeline Execution & Results Display
# ---------------------------------------------------------------------------
if audio_bytes is not None and len(audio_bytes) > 0:
    # 1. Audio Player
    col_audio, col_meta = st.columns([2, 1])
    with col_audio:
        st.audio(audio_bytes, format="audio/wav")
    with col_meta:
        st.caption(f"**Loaded:** `{active_filename}` ({len(audio_bytes) / 1024:.1f} KB)")

    # 2. Run Analysis through PulseForgeService
    result = None
    audio_sig = None
    run_id = None
    with st.spinner("Analyzing acoustic signature through Autoencoder + Familiarity + Policy..."):
        try:
            result, audio_sig, run_id = service.analyze_bytes(
                audio_bytes, filename=active_filename, record_to_db=True
            )
        except Exception as ex:
            st.error(f"Analysis Error: {ex}")
            st.stop()

    if result is None:
        st.stop()

    # 3. Decision Card Presentation
    st.markdown("---")
    st.markdown("### 📊 Automated Operational Assessment")

    status = result.status
    if status == DecisionStatus.NORMAL:
        card_class = "card-normal"
        status_class = "status-normal"
        status_icon = "🟢"
        headline = "NORMAL OPERATION CONFIRMED"
    elif status == DecisionStatus.POSSIBLE_ANOMALY:
        card_class = "card-anomaly"
        status_class = "status-anomaly"
        status_icon = "🔴"
        headline = "POSSIBLE ACOUSTIC ANOMALY DETECTED"
    elif status == DecisionStatus.NEEDS_REVIEW:
        card_class = "card-review"
        status_class = "status-review"
        status_icon = "🟡"
        headline = "HUMAN OPERATOR REVIEW REQUIRED"
    else:
        card_class = "card-invalid"
        status_class = "status-invalid"
        status_icon = "⚠️"
        headline = "INVALID AUDIO RECORDING"

    # Prominent Decision Hero Box
    st.markdown(
        f"""
        <div class="decision-card {card_class}">
            <div class="status-title {status_class}">
                <span>{status_icon}</span>
                <span>{headline}</span>
            </div>
            <p style="font-size: 1.05rem; color: #f1f5f9; margin-bottom: 0.5rem;">
                {result.explanation}
            </p>
            <div class="action-box">
                <strong style="color: #38bdf8;">Recommended Action:</strong>
                <span style="color: #e2e8f0;"> {result.recommended_action}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4. Diagnostic Metric Gauges
    c1, c2, c3, c4 = st.columns(4)

    score_val = (
        result.model_prediction.anomaly_score
        if result.model_prediction
        else 0.0
    )
    score_thresh = (
        result.model_prediction.threshold
        if result.model_prediction
        else thresh
    )

    with c1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Autoencoder Reconstruction Error</div>
                <div class="metric-val" style="color: {'#f87171' if score_val >= score_thresh else '#34d399'};">
                    {score_val:.4f}
                </div>
                <div class="metric-sub">Decision Threshold: <strong>{score_thresh:.4f}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        fam_dist = (
            result.familiarity.score
            if result.familiarity and result.familiarity.score is not None
            else 0.0
        )
        is_fam = result.familiarity.is_familiar if result.familiarity else True
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Latent Familiarity Distance</div>
                <div class="metric-val" style="color: {'#34d399' if is_fam else '#fbbf24'};">
                    {fam_dist:.3f}
                </div>
                <div class="metric-sub">In-Distribution Limit: <strong>{fam_thresh:.4f}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        peak_seg = (
            max(result.model_prediction.segment_scores)
            if result.model_prediction and result.model_prediction.segment_scores
            else score_val
        )
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Peak Segment Error</div>
                <div class="metric-val" style="color: {'#f87171' if peak_seg >= score_thresh else '#34d399'};">
                    {peak_seg:.4f}
                </div>
                <div class="metric-sub">Max Transient Window Error</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        is_healthy = result.quality.passed
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Signal Quality Health</div>
                <div class="metric-val" style="color: {'#34d399' if is_healthy else '#fbbf24'};">
                    {'PASSED' if is_healthy else 'REJECTED'}
                </div>
                <div class="metric-sub">RMS: {result.quality.rms_level:.4f} | Clip: {result.quality.clipping_ratio * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 5. Dual Waveform & Spectrogram Visualizer
    st.markdown("#### 📈 Acoustic Signal & Spectral Density")
    try:
        audio_stream = io.BytesIO(audio_bytes)
        sig_data, sr = sf.read(audio_stream, dtype="float32")
        if sig_data.ndim > 1:
            sig_data = np.mean(sig_data, axis=1)
        fig = plot_acoustic_features(sig_data, sr)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Unable to render signal visualization: {e}")

    # 6. Detailed Diagnostics & Audit Expander
    with st.expander("🔍 Deep Diagnostics & Sensor Quality Telemetry", expanded=False):
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Sensor Health Telemetry:**")
            st.write(
                {
                    "Quality Passed": result.quality.passed,
                    "RMS Energy Level": f"{result.quality.rms_level:.6f}",
                    "Clipping Ratio": f"{result.quality.clipping_ratio * 100:.2f}%",
                    "DC Offset": f"{result.quality.dc_offset:.6f}",
                    "Is Silent": result.quality.is_silent,
                    "Is Clipped": result.quality.is_clipped,
                    "Duration": f"{result.quality.duration_seconds:.2f} s",
                    "Sample Rate": f"{result.metadata.sample_rate} Hz" if result.metadata else "16000 Hz",
                }
            )
        with d2:
            st.markdown("**Decision State Machine Context:**")
            st.write(
                {
                    "Machine Type": result.machine_type,
                    "Machine ID": result.machine_id,
                    "Final State": result.status.value,
                    "Status Label": result.status_label,
                    "Run ID": run_id,
                    "Autoencoder Anomaly Detected": (
                        result.model_prediction.is_anomaly
                        if result.model_prediction
                        else "N/A"
                    ),
                    "Latent Familiarity Status": (
                        "In-Distribution (Familiar)"
                        if result.familiarity and result.familiarity.is_familiar
                        else "Out-of-Distribution Shift"
                    ),
                }
            )

else:
    st.info("👆 Please select a curated demonstration scenario or upload a WAV recording above to begin analysis.")
