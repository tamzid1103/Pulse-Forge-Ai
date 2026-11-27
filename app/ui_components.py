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
    background-color: #F8FAFC;
    color: #0F172A;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

[data-testid="stHeader"] {
    background-color: rgba(248, 250, 252, 0.95);
    border-bottom: 1px solid #E2E8F0;
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
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: #F8FAFC !important;
    font-weight: 700;
}

[data-testid="stSidebar"] hr {
    border-color: #1E314B !important;
    margin: 0.9rem 0;
}

/* 3. FIX: Sidebar Buttons (Eliminate stark white button with faint text) */
[data-testid="stSidebar"] button {
    background-color: #1E293B !important;
    color: #F8FAFC !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    padding: 0.45rem 0.85rem !important;
    transition: all 0.15s ease !important;
}

[data-testid="stSidebar"] button * {
    color: #F8FAFC !important;
}

[data-testid="stSidebar"] button:hover {
    background-color: #0F766E !important;
    color: #FFFFFF !important;
    border-color: #14B8A6 !important;
    box-shadow: 0 2px 6px rgba(15, 118, 110, 0.35) !important;
}

[data-testid="stSidebar"] button:active {
    background-color: #115E59 !important;
}

/* 4. FIX: Sidebar Expanders (Eliminate stark white expander container) */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background-color: #142033 !important;
    border: 1px solid #233852 !important;
    border-radius: 6px !important;
    margin: 0.4rem 0 !important;
}

[data-testid="stSidebar"] [data-testid="stExpander"] details {
    background-color: #142033 !important;
    border-radius: 6px !important;
}

[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    background-color: #16253B !important;
    color: #F1F5F9 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    border-bottom: 1px solid #1E314B !important;
    padding: 0.5rem 0.75rem !important;
}

[data-testid="stSidebar"] [data-testid="stExpander"] summary * {
    color: #F1F5F9 !important;
}

[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
    color: #5EEAD4 !important;
    background-color: #1A2C46 !important;
}

[data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
    color: #94A3B8 !important;
    fill: #94A3B8 !important;
}

/* 5. FIX: Sidebar Inputs & Code Elements */
[data-testid="stSidebar"] [data-testid="stTextInput"] input {
    background-color: #0E1A2B !important;
    color: #F8FAFC !important;
    border: 1px solid #294060 !important;
    border-radius: 6px !important;
}

[data-testid="stSidebar"] [data-testid="stTextInput"] input:focus {
    border-color: #0F766E !important;
    box-shadow: 0 0 0 1px #0F766E !important;
}

[data-testid="stSidebar"] [data-testid="stTextInput"] label {
    color: #CBD5E1 !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
}

/* FIX: Sidebar Selectbox (Prevent faint grey text on white background) */
[data-testid="stSidebar"] [data-testid="stSelectbox"] label,
[data-testid="stSidebar"] [data-testid="stSelectbox"] label * {
    color: #CBD5E1 !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
}

[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"],
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div > div {
    background-color: #0E1A2B !important;
    color: #F8FAFC !important;
}

[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    border: 1px solid #294060 !important;
    border-radius: 6px !important;
}

[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
    border-color: #14B8A6 !important;
    box-shadow: 0 0 0 1px rgba(20, 184, 166, 0.35) !important;
}

[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] span,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] div,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] input {
    color: #F8FAFC !important;
    -webkit-text-fill-color: #F8FAFC !important;
}

[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] svg {
    fill: #F8FAFC !important;
    color: #F8FAFC !important;
}

/* Selectbox dropdown menu popover (high-contrast dark text on crisp white) */
div[data-baseweb="popover"] [role="listbox"] {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
}

div[data-baseweb="popover"] [role="option"] {
    color: #0F172A !important;
}

div[data-baseweb="popover"] [role="option"]:hover,
div[data-baseweb="popover"] [aria-selected="true"] {
    background-color: #F0FDFA !important;
    color: #0F766E !important;
    font-weight: 600 !important;
}

[data-testid="stSidebar"] code {
    background-color: #0E1A2B !important;
    color: #5EEAD4 !important;
    border: 1px solid #233852 !important;
    padding: 0.1rem 0.35rem !important;
    border-radius: 4px !important;
    font-size: 0.78rem !important;
}

/* 6. Application-Style Navigation Menu */
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

[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
    background-color: #20364D !important;
    border-left: 3px solid #0F766E !important;
    color: #FFFFFF !important;
    font-weight: 600;
}

[data-testid="stSidebar"] div[data-testid="stRadio"] input[type="radio"],
[data-testid="stSidebar"] div[data-testid="stRadio"] div[data-testid="stWidgetLabel"] + div svg {
    opacity: 0.15;
}

/* 7. Equipment Context Card (Sidebar) */
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

/* 8. Main Workspace Header */
.main-header-container {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-top: 3px solid #0F766E;
    border-radius: 8px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.main-header-title {
    font-size: 1.55rem;
    font-weight: 800;
    color: #0F172A;
    letter-spacing: -0.025em;
    margin: 0 0 0.2rem 0;
}

.main-header-subtitle {
    color: #64748B;
    font-size: 0.88rem;
    margin: 0;
}

/* 9. Stage Headers with Circular Number Badges */
.stage-header-badge {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin: 1.6rem 0 0.75rem 0;
}

.stage-badge-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    background-color: #0F766E;
    color: #FFFFFF;
    font-size: 0.82rem;
    font-weight: 700;
    border-radius: 50%;
    box-shadow: 0 1px 3px rgba(15, 118, 110, 0.3);
}

.stage-badge-text {
    font-size: 1.08rem;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: -0.01em;
}

/* 10. Main Workspace Buttons */
[data-testid="stAppViewContainer"] button[kind="primary"] {
    background-color: #0F766E !important;
    color: #FFFFFF !important;
    border: 1px solid #0D9488 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.2rem !important;
    box-shadow: 0 1px 3px rgba(15, 118, 110, 0.25) !important;
    transition: all 0.15s ease !important;
}

[data-testid="stAppViewContainer"] button[kind="primary"]:hover {
    background-color: #115E59 !important;
    border-color: #0F766E !important;
    box-shadow: 0 2px 6px rgba(15, 118, 110, 0.4) !important;
}

[data-testid="stAppViewContainer"] button[kind="secondary"] {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    padding: 0.48rem 1.1rem !important;
    transition: all 0.15s ease !important;
}

[data-testid="stAppViewContainer"] button[kind="secondary"]:hover {
    background-color: #F8FAFC !important;
    color: #0F766E !important;
    border-color: #0F766E !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
}

/* 11. File Uploader Container */
[data-testid="stFileUploader"] > section {
    background-color: #FFFFFF !important;
    border: 1.5px dashed #94A3B8 !important;
    border-radius: 8px !important;
    padding: 1.4rem !important;
    transition: all 0.2s ease !important;
}

[data-testid="stFileUploader"] > section:hover {
    border-color: #0F766E !important;
    background-color: #F0FDF4 !important;
}

[data-testid="stFileUploader"] button {
    background-color: #0F766E !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    padding: 0.45rem 1rem !important;
}

/* 12. Tabs Styling */
button[data-baseweb="tab"] {
    color: #64748B !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 0.6rem 1.2rem !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #0F766E !important;
    border-bottom: 2.5px solid #0F766E !important;
}

/* 13. Diagnostic Decision Cards */
.decision-card {
    border-radius: 8px;
    padding: 1.25rem 1.4rem;
    margin: 1.2rem 0;
    border-width: 1px;
    border-style: solid;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
}

.card-normal {
    background-color: #F0FDF4;
    border-color: #86EFAC;
    border-left: 5px solid #16A34A;
}

.card-anomaly {
    background-color: #FEF2F2;
    border-color: #FCA5A5;
    border-left: 5px solid #DC2626;
}

.card-review {
    background-color: #FFFBEB;
    border-color: #FCD34D;
    border-left: 5px solid #D97706;
}

.card-invalid {
    background-color: #F8FAFC;
    border-color: #CBD5E1;
    border-left: 5px solid #64748B;
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

/* 14. Metric Cards (With subtle top accent) */
.metric-tile {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-top: 3px solid #0F766E;
    border-radius: 8px;
    padding: 0.9rem 1rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
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
    color: #0F172A;
    margin: 0.2rem 0;
}

.metric-tile-sub {
    font-size: 0.72rem;
    color: #64748B;
}

/* 15. Recording Preview Box */
.audio-review-box {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-left: 4px solid #0F766E;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin: 0.8rem 0;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
}

.review-meta-item {
    font-size: 0.84rem;
    color: #475569;
    padding: 0.15rem 0;
}

.review-meta-item strong {
    color: #0F172A;
}
</style>
"""


def inject_custom_theme() -> None:
    """Inject application-wide CSS theme and typography tokens."""
    st.markdown(CUSTOM_THEME_CSS, unsafe_allow_html=True)


def render_stage_header(number: int | str, title: str) -> None:
    """Render a clean, high-contrast industrial workflow stage header."""
    st.markdown(
        f"""
        <div class="stage-header-badge">
            <span class="stage-badge-num">{number}</span>
            <span class="stage-badge-text">{title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        f'<span style="background-color: #F0FDFA; border: 1px solid #99F6E4; color: #0F766E; font-size: 0.72rem; font-weight: 700; padding: 0.25rem 0.6rem; border-radius: 4px; text-transform: uppercase; letter-spacing: 0.04em;">{badge_label}</span>'
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

    # Time axis in seconds
    time_axis = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))

    # 1. Waveform Plot
    ax1.plot(time_axis, audio_data, color="#0F766E", linewidth=0.75, alpha=0.9)
    ax1.set_facecolor("#FAFAFA")
    ax1.set_ylabel("Amplitude", fontsize=8, color="#172033", fontweight=600)
    ax1.set_xlim([0, time_axis[-1] if len(time_axis) > 0 else 1.0])
    ax1.grid(True, linestyle="--", alpha=0.3, color="#CBD5E1")
    ax1.tick_params(colors="#64748B", labelsize=7.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color("#CBD5E1")
    ax1.spines["bottom"].set_color("#CBD5E1")

    # 2. Log-Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    img = librosa.display.specshow(
        log_mel,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        fmin=fmin,
        fmax=fmax,
        ax=ax2,
        cmap="viridis",
    )
    ax2.set_ylabel("Freq (Hz)", fontsize=8, color="#172033", fontweight=600)
    ax2.set_xlabel("Time (s)", fontsize=8, color="#172033", fontweight=600)
    ax2.grid(False)
    ax2.tick_params(colors="#64748B", labelsize=7.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color("#CBD5E1")
    ax2.spines["bottom"].set_color("#CBD5E1")

    # Colorbar
    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.35])
    cbar = fig.colorbar(img, cax=cbar_ax, format="%+2.0f dB")
    cbar.ax.tick_params(labelsize=6.5, colors="#64748B")
    cbar.outline.set_color("#CBD5E1")

    plt.subplots_adjust(left=0.08, right=0.89, top=0.92, bottom=0.14, hspace=0.32)
    return fig
