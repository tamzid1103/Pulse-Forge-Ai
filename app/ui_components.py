"""
PulseForge AI — Industrial UI Components & Design System.

Provides reusable presentation helpers, accessible design tokens,
and high-contrast layout components conforming to industrial diagnostic standards.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.schemas import AnalysisResult, DecisionStatus

logger = logging.getLogger("pulseforge.ui")

# ---------------------------------------------------------------------------
# Design System CSS
# ---------------------------------------------------------------------------
CUSTOM_THEME_CSS = """
<style>
/* 1. Global Workspace & Typography */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #F5F7FB;
    color: #172033;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

[data-testid="stHeader"] {
    background-color: rgba(245, 247, 251, 0.9);
}

/* 2. Dark Navy Sidebar (#101B2D) */
[data-testid="stSidebar"] {
    background-color: #101B2D !important;
    border-right: 1px solid #1D2E45;
}

[data-testid="stSidebar"] * {
    color: #CBD5E1;
}

[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: #F8FAFC !important;
    font-weight: 700;
}

[data-testid="stSidebar"] hr {
    border-color: #1E314B !important;
    margin: 1rem 0;
}

/* 3. Application-Style Navigation Menu */
[data-testid="stSidebar"] div[data-testid="stRadio"] > label {
    display: none;
}

[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}

[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label {
    background-color: transparent;
    padding: 0.55rem 0.85rem;
    border-radius: 6px;
    margin: 0;
    cursor: pointer;
    border-left: 3px solid transparent;
    transition: background-color 0.15s ease, border-color 0.15s ease;
}

[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
    background-color: #18283D !important;
    color: #FFFFFF !important;
}

/* Active Navigation Row */
[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
    background-color: #20364D !important;
    border-left: 3px solid #0F766E !important;
    color: #FFFFFF !important;
    font-weight: 600;
}

/* Hide raw radio circles to give application sidebar appearance */
[data-testid="stSidebar"] div[data-testid="stRadio"] input[type="radio"],
[data-testid="stSidebar"] div[data-testid="stRadio"] div[data-testid="stWidgetLabel"] + div svg {
    opacity: 0.15;
}

/* 4. Equipment Context Card (Sidebar) */
.sidebar-context-box {
    background-color: #16253B;
    border: 1px solid #233852;
    border-radius: 6px;
    padding: 0.75rem 0.85rem;
    margin: 0.5rem 0;
}

.context-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.25rem 0;
    font-size: 0.8rem;
    border-bottom: 1px solid #1E314B;
}

.context-row:last-child {
    border-bottom: none;
}

.context-label {
    color: #94A3B8;
    font-weight: 500;
}

.context-val {
    color: #F1F5F9;
    font-weight: 600;
    text-align: right;
}

.context-id {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    color: #5EEAD4;
    background-color: #0E1A2B;
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    font-size: 0.78rem;
}

/* 5. Main Page Header */
.main-header-container {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.main-header-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #172033;
    letter-spacing: -0.02em;
    margin: 0 0 0.2rem 0;
}

.main-header-subtitle {
    color: #64748B;
    font-size: 0.88rem;
    margin: 0;
}

/* 6. Diagnostic Decision Cards (Light Theme, Accessible High-Contrast) */
.decision-card {
    border-radius: 8px;
    padding: 1.25rem 1.4rem;
    margin: 1.2rem 0;
    border-width: 1px;
    border-style: solid;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

.card-normal {
    background-color: #F0FDF4;
    border-color: #86EFAC;
}

.card-anomaly {
    background-color: #FEF2F2;
    border-color: #FCA5A5;
}

.card-review {
    background-color: #FFFBEB;
    border-color: #FCD34D;
}

.card-invalid {
    background-color: #F8FAFC;
    border-color: #CBD5E1;
}

.status-badge {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.badge-normal { background-color: #DCFCE7; color: #15803D; border: 1px solid #86EFAC; }
.badge-anomaly { background-color: #FEE2E2; color: #B91C1C; border: 1px solid #FCA5A5; }
.badge-review { background-color: #FEF3C7; color: #B45309; border: 1px solid #FCD34D; }
.badge-invalid { background-color: #E2E8F0; color: #475569; border: 1px solid #CBD5E1; }

.decision-headline {
    font-size: 1.25rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
}

.headline-normal { color: #166534; }
.headline-anomaly { color: #991B1B; }
.headline-review { color: #92400E; }
.headline-invalid { color: #334155; }

.action-box {
    background-color: #FFFFFF;
    border-left: 3px solid #0F766E;
    padding: 0.65rem 0.9rem;
    border-radius: 4px;
    margin-top: 0.85rem;
    font-size: 0.88rem;
    color: #172033;
    border-top: 1px solid #E2E8F0;
    border-right: 1px solid #E2E8F0;
    border-bottom: 1px solid #E2E8F0;
}

/* 7. Metric Cards */
.metric-tile {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 0.9rem 1rem;
    text-align: center;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
}

.metric-tile-label {
    font-size: 0.74rem;
    text-transform: uppercase;
    color: #64748B;
    letter-spacing: 0.05em;
    font-weight: 600;
    margin-bottom: 0.2rem;
}

.metric-tile-val {
    font-size: 1.45rem;
    font-weight: 700;
    color: #172033;
    margin: 0.2rem 0;
}

.metric-tile-sub {
    font-size: 0.72rem;
    color: #94A3B8;
}

/* 8. Recording Preview Box */
.audio-review-box {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin: 0.8rem 0;
}

.review-meta-item {
    font-size: 0.82rem;
    color: #475569;
}

.review-meta-item strong {
    color: #172033;
}
</style>
"""


def inject_custom_theme() -> None:
    """Inject application-wide CSS theme and typography tokens."""
    st.markdown(CUSTOM_THEME_CSS, unsafe_allow_html=True)


def render_sidebar_brand() -> None:
    """Render clean, compact PulseForge AI brand block without scattered emoji."""
    st.sidebar.markdown(
        """
        <div style="padding: 0.4rem 0 0.8rem 0;">
            <div style="font-size: 1.22rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.025em; line-height: 1.2;">
                PulseForge AI
            </div>
            <div style="font-size: 0.72rem; font-weight: 600; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.15rem;">
                Industrial Acoustic Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_equipment(
    asset: str = "Industrial Centrifugal Fan",
    equipment_id: str = "fan_id_00",
    facility: str = "Plant Alpha — Line 1",
    sensor: str = "16 kHz / 16-bit PCM",
) -> None:
    """Render monitored equipment context in tidy key-value pairs."""
    st.sidebar.markdown(
        f"""
        <div style="font-size: 0.72rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.35rem;">
            Monitored Target
        </div>
        <div class="sidebar-context-box">
            <div class="context-row">
                <span class="context-label">Asset</span>
                <span class="context-val">{asset}</span>
            </div>
            <div class="context-row">
                <span class="context-label">Equipment ID</span>
                <span class="context-id">{equipment_id}</span>
            </div>
            <div class="context-row">
                <span class="context-label">Location</span>
                <span class="context-val">{facility}</span>
            </div>
            <div class="context-row">
                <span class="context-label">Sensor Telemetry</span>
                <span class="context-val">{sensor}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_footer() -> None:
    """Render compact, honest footer without fabricated certifications."""
    st.sidebar.markdown(
        """
        <div style="font-size: 0.7rem; color: #64748B; margin-top: 1.2rem; line-height: 1.4;">
            <div><strong>Runtime:</strong> PyTorch CPU Edge</div>
            <div><strong>Release:</strong> v1.0-preliminary</div>
            <div style="margin-top: 0.3rem; color: #475569;">Conforms to ISO 13374-1 condition monitoring architecture.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(
    title: str,
    subtitle: str,
    badge_label: str | None = None,
) -> None:
    """Render compact, restrained header without oversized decorative icons."""
    badge_html = (
        f'<span style="background-color: #F1F5F9; border: 1px solid #CBD5E1; color: #475569; font-size: 0.72rem; font-weight: 600; padding: 0.2rem 0.55rem; border-radius: 4px; text-transform: uppercase;">{badge_label}</span>'
        if badge_label
        else ""
    )
    st.markdown(
        f"""
        <div class="main-header-container">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <h1 class="main-header-title">{title}</h1>
                {badge_html}
            </div>
            <p class="main-header-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_acoustic_plots(
    audio_data: np.ndarray,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> plt.Figure:
    """Render dual-panel waveform and log-mel spectrogram matched to light theme."""
    import librosa
    import librosa.display

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.4), dpi=100)
    fig.patch.set_facecolor("#FFFFFF")

    # 1. Waveform Plot
    t = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
    ax1.set_facecolor("#F8FAFC")
    ax1.plot(t, audio_data, color="#0F766E", linewidth=0.75, alpha=0.9)
    ax1.set_xlim(0, t[-1] if len(t) > 0 else 1.0)
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_ylabel("Amplitude", color="#475569", fontsize=8)
    ax1.tick_params(colors="#64748B", labelsize=7)
    ax1.grid(True, linestyle="--", alpha=0.5, color="#E2E8F0")
    ax1.set_title("Acoustic Time-Domain Waveform", color="#172033", fontsize=9, loc="left", pad=4, fontweight="600")

    # 2. Log-Mel Spectrogram
    ax2.set_facecolor("#F8FAFC")
    try:
        s_mel = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )
        s_db = librosa.power_to_db(s_mel, ref=np.max)
        img = librosa.display.specshow(
            s_db,
            sr=sample_rate,
            hop_length=hop_length,
            x_axis="time",
            y_axis="mel",
            fmin=fmin,
            fmax=fmax,
            ax=ax2,
            cmap="viridis",
        )
        cbar = fig.colorbar(img, ax=ax2, format="%+2.0f dB", pad=0.015, shrink=0.9)
        cbar.ax.tick_params(colors="#64748B", labelsize=6)
    except Exception as e:
        logger.warning("Failed to render spectrogram: %s", e)
        ax2.text(0.5, 0.5, "Spectrogram unavailable for this signal", color="#64748B", ha="center")

    ax2.set_xlabel("Time (seconds)", color="#475569", fontsize=8)
    ax2.set_ylabel("Freq (Hz)", color="#475569", fontsize=8)
    ax2.tick_params(colors="#64748B", labelsize=7)
    ax2.set_title("Log-Mel Spectral Density (128 filter banks)", color="#172033", fontsize=9, loc="left", pad=4, fontweight="600")

    plt.tight_layout()
    return fig
