"""
Factory Brain AI Maintenance Copilot for PulseForge AI (Milestone M8).

Integrates Groq LPU inference (llama-3.3-70b-versatile / llama-3.1-8b-instant)
with domain-grounded industrial Standard Operating Procedures (SOPs) and ISO 10816
standards to generate actionable maintenance work orders and answer technician queries.

Features:
- Deterministic telemetry grounding (telemetry from PyTorch Autoencoder + Quality checks).
- 1-Click Maintenance Work Order generator (structured JSON / Markdown).
- Interactive technician diagnostic Q&A assistant.
- Zero-crash offline fallback: when no API key or network is available, provides
  high-value deterministic SOP-based recommendations.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
import re
from typing import Any

from src.config import MACHINE_FAMILY_METADATA
from src.schemas import AnalysisResult, DecisionStatus

logger = logging.getLogger("pulseforge.copilot")


def sanitize_sensitive_tokens(message: str | Exception) -> str:
    """
    Sanitize sensitive API keys, authorization tokens, and credentials from
    exception representations and error strings (CWE-209).
    """
    text = str(message)
    # Mask Groq API keys (gsk_...)
    text = re.sub(r"gsk_[A-Za-z0-9_-]{8,}", "gsk_********************", text)
    # Mask Bearer tokens
    text = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]{8,}", "Bearer [REDACTED]", text, flags=re.IGNORECASE)
    # Mask basic auth credentials in URLs
    text = re.sub(r"://([^:]+):([^@]+)@", r"://\1:[REDACTED]@", text)
    return text

# Try importing groq safely
try:
    from groq import Groq

    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None  # type: ignore

# Try importing streamlit safely for secrets access
try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None  # type: ignore

# Try loading .env if python-dotenv is present
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class IndustrialCopilot:
    """
    Industrial maintenance advisor powered by Groq LPU inference and grounded
    in industrial ventilation fan SOPs and ISO 10816 condition monitoring standards.
    """

    DEFAULT_MODEL = "llama-3.1-8b-instant"
    FALLBACK_MODELS = ["llama-3.1-8b-instant", "llama3-8b-8192", "llama-3.3-70b-versatile"]

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        knowledge_base_dir: Path | None = None,
    ) -> None:
        self.manual_api_key = api_key
        self.model = model
        self.project_root = Path(__file__).resolve().parent.parent
        self.kb_dir = (
            knowledge_base_dir
            if knowledge_base_dir is not None
            else self.project_root / "data" / "knowledge_base"
        )
        self._cached_knowledge: str | None = None

    def get_api_key(self, override_key: str | None = None) -> str | None:
        """
        Resolve the Groq API key with hierarchical priority:
        1. Explicit function override argument
        2. Instance manual_api_key
        3. Streamlit Cloud Secrets (st.secrets["GROQ_API_KEY"])
        4. Environment variable (os.getenv("GROQ_API_KEY"))
        """
        if override_key and override_key.strip():
            return override_key.strip()
        if self.manual_api_key and self.manual_api_key.strip():
            return self.manual_api_key.strip()
        if STREAMLIT_AVAILABLE:
            try:
                sec_key = st.secrets.get("GROQ_API_KEY")
                if sec_key and str(sec_key).strip():
                    return str(sec_key).strip()
            except Exception:
                pass
        env_key = os.getenv("GROQ_API_KEY")
        if env_key and env_key.strip():
            return env_key.strip()
        return None

    def is_configured(self, override_key: str | None = None) -> bool:
        """Check if a valid Groq API key is available and Groq library is installed."""
        if not GROQ_AVAILABLE:
            return False
        key = self.get_api_key(override_key)
        return bool(key and len(key) > 5)

    def load_knowledge_base(self) -> str:
        """Load industrial fan SOP and ISO documentation from disk."""
        if self._cached_knowledge is not None:
            return self._cached_knowledge

        knowledge_parts: list[str] = []
        if self.kb_dir.is_dir():
            for md_file in sorted(self.kb_dir.glob("*.md")):
                try:
                    content = md_file.read_text(encoding="utf-8")
                    knowledge_parts.append(f"--- DOCUMENT: {md_file.name} ---\n{content}\n")
                except Exception as e:
                    logger.warning("Could not read knowledge file %s: %s", md_file, e)

        if not knowledge_parts:
            # Fallback baseline knowledge if files are missing
            self._cached_knowledge = (
                "Industrial Centrifugal Fan Standard SOP: ISO 10816 Zone C/D requires Lockout/Tagout "
                "prior to inspection. Bearing defects exhibit high-frequency acoustic anomalies (2-8 kHz). "
                "Unbalance exhibits 1X rotational vibration. Quality failures require sensor harness check."
            )
        else:
            self._cached_knowledge = "\n".join(knowledge_parts)

        return self._cached_knowledge

    def build_telemetry_prompt(
        self,
        result: AnalysisResult,
        machine_id: str = "fan_id_00",
    ) -> str:
        """
        Format deterministic model telemetry and quality measurements into an
        unambiguous prompt context for the LLM.
        """
        status_str = result.status.value.upper()
        ae_score = (
            result.model_prediction.anomaly_score
            if result.model_prediction
            else 0.0
        )
        ae_thresh = (
            result.model_prediction.threshold
            if result.model_prediction
            else 0.0032
        )
        is_ae_anomaly = (
            result.model_prediction.is_anomaly
            if result.model_prediction
            else False
        )
        fam_dist = (
            result.familiarity.score
            if result.familiarity and result.familiarity.score is not None
            else 0.0
        )
        fam_thresh = (
            result.familiarity.threshold
            if result.familiarity and result.familiarity.threshold is not None
            else 9.25
        )
        is_familiar = (
            result.familiarity.is_familiar
            if result.familiarity
            else True
        )
        q_pass = result.quality.passed
        rms = result.quality.rms_level
        clipping_pct = result.quality.clipping_ratio * 100
        duration = result.quality.duration_seconds
        m_type = (result.machine_type or "fan").lower()
        m_meta = MACHINE_FAMILY_METADATA.get(m_type, {})
        m_display = m_meta.get("display_name", f"Industrial {m_type.capitalize()}")

        context = f"""
TARGET EQUIPMENT TELEMETRY:
- Equipment Asset: {m_display} ({machine_id}, type={m_type})
- Diagnostic Status: {status_str}
- Autoencoder Reconstruction Error (MSE): {ae_score:.6f}
- Autoencoder Baseline Threshold: {ae_thresh:.6f}
- Autoencoder Anomaly Flag: {is_ae_anomaly}
- Latent Space Distance (Mahalanobis): {fam_dist:.4f} (Limit: {fam_thresh:.4f})
- Operating Regime Familiarity: {'In-Distribution (Familiar)' if is_familiar else 'Domain Shift / Unfamiliar'}
- Signal Quality Checks: {'Passed' if q_pass else 'Degraded / Failed'}
- Acoustic RMS Energy: {rms:.6f}
- Waveform Clipping Ratio: {clipping_pct:.2f}%
- Audio Duration: {duration:.2f} seconds
- Pipeline Explanation: {result.explanation}
- Initial Action Recommendation: {result.recommended_action}
"""
        return context.strip()

    def generate_work_order(
        self,
        result: AnalysisResult,
        machine_id: str = "fan_id_00",
        override_key: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate a structured maintenance work order.
        Uses Groq LLM if available; otherwise falls back to deterministic rule-based SOP.
        """
        api_key = self.get_api_key(override_key)

        if not GROQ_AVAILABLE or not api_key:
            logger.info("Groq API key not configured. Using deterministic SOP engine.")
            return self._deterministic_work_order(result, machine_id)

        try:
            client = Groq(api_key=api_key)
            knowledge = self.load_knowledge_base()
            telemetry = self.build_telemetry_prompt(result, machine_id)

            system_prompt = f"""
You are the PulseForge AI Industrial Plant Copilot and Senior Reliability Engineer.
You specialize in predictive maintenance for industrial rotating machinery (fans, blowers, motors)
complying with ISO 13374, ISO 10816-3, and OSHA 1910.147 (Lockout/Tagout).

Your task: Given the deterministic acoustic diagnostic telemetry and maintenance SOP documentation,
generate a formal, actionable Maintenance Work Order in JSON format.

CRITICAL RULES:
1. Never invent or hallucinate failure dates or percentage probabilities of failure.
2. Ground all actions strictly in the provided telemetry and SOP knowledge.
3. If Status is NORMAL: Urgency is ROUTINE; focus on continuing routine inspection and scheduled lubrication.
4. If Status is POSSIBLE_ANOMALY: Urgency is PRIORITY_INSPECTION; focus on bearing raceway acoustic inspection, belt tension, and vibration checks.
5. If Status is NEEDS_REVIEW: If signal degraded, focus on acoustic microphone/wiring sensor troubleshooting. If domain shift, focus on ambient plant noise verification.
6. Always mandate Lockout/Tagout (LOTO) prior to any mechanical access.
7. Return ONLY valid JSON matching this schema:
{{
    "work_order_id": "WO-FAN-2026-XXXX",
    "asset_id": "{machine_id}",
    "diagnostic_status": "{result.status.value}",
    "urgency": "ROUTINE" | "PRIORITY_INSPECTION" | "IMMEDIATE_ACTION",
    "suspected_mechanism": "short description of mechanical condition",
    "iso_severity_zone": "Zone A" | "Zone B" | "Zone C" | "Zone D",
    "recommended_steps": ["step 1", "step 2", "step 3", "step 4"],
    "safety_checklist": ["LOTO step 1", "LOTO step 2", "zero-speed verify"],
    "required_tools": ["tool 1", "tool 2"],
    "required_ppe": ["ppe 1", "ppe 2"],
    "engineering_summary": "2-3 concise sentences explaining the rationale"
}}

KNOWLEDGE BASE:
{knowledge}
"""

            user_prompt = f"""
Please generate the maintenance work order for the following diagnostic run:

{telemetry}
"""

            candidate_models = [self.model]
            for fb in ["llama-3.1-8b-instant", "llama3-8b-8192", "llama-3.3-70b-versatile"]:
                if fb not in candidate_models:
                    candidate_models.append(fb)

            last_error: Exception | None = None
            for model_name in candidate_models:
                try:
                    response = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        model=model_name,
                        temperature=0.1,
                        max_tokens=1000,
                        response_format={"type": "json_object"},
                    )

                    response_content = response.choices[0].message.content
                    work_order = json.loads(response_content)
                    work_order["ai_engine"] = f"{model_name} (Groq LPU)"
                    work_order["created_at"] = datetime.datetime.now(
                        datetime.timezone.utc
                    ).strftime("%Y-%m-%d %H:%M:%S UTC")
                    return work_order
                except Exception as model_err:
                    last_error = model_err
                    logger.warning(
                        "Groq work order generation with model %s failed (%s). Trying candidate fallback...",
                        model_name,
                        model_err,
                    )
                    continue

            raise last_error or RuntimeError("All candidate Groq models failed.")

        except Exception as e:
            sanitized_err = sanitize_sensitive_tokens(e)
            logger.warning("Groq work order generation failed (%s). Falling back to SOP.", sanitized_err)
            fallback = self._deterministic_work_order(result, machine_id)
            fallback["warning"] = f"Generated via offline SOP fallback (Groq error: {sanitized_err})"
            return fallback

    def chat_response(
        self,
        messages: list[dict[str, str]],
        result: AnalysisResult | None = None,
        machine_id: str = "fan_id_00",
        override_key: str | None = None,
    ) -> str:
        """
        Generate an interactive chat response for plant technicians.
        Grounds answers in telemetry and SOP documentation.
        """
        api_key = self.get_api_key(override_key)

        if not GROQ_AVAILABLE or not api_key:
            return self._deterministic_chat_response(messages, result)

        try:
            client = Groq(api_key=api_key)
            knowledge = self.load_knowledge_base()
            telemetry_str = (
                self.build_telemetry_prompt(result, machine_id)
                if result
                else "No active audio evaluation loaded."
            )

            system_prompt = f"""
You are the PulseForge AI Industrial Maintenance Copilot — an expert rotating machinery reliability engineer.
You are assisting an industrial plant operator or maintenance technician.

CURRENT RUN TELEMETRY:
{telemetry_str}

INDUSTRIAL KNOWLEDGE & SOP CONTEXT:
{knowledge}

RULES FOR YOUR RESPONSES:
1. Be concise, direct, professional, and practical. Speak like a senior reliability technician.
2. Ground explanations strictly in the acoustic reconstruction error, signal quality, and latent familiarity metrics.
3. Never invent failure probabilities or exact remaining useful life (RUL) numbers unless validated.
4. Emphasize physical safety (Lockout/Tagout, eye/hearing PPE) whenever mechanical inspections are discussed.
5. If the user asks about the latest diagnostic result, refer directly to the telemetry above.
"""

            groq_messages = [{"role": "system", "content": system_prompt}]
            for msg in messages:
                groq_messages.append({"role": msg["role"], "content": msg["content"]})

            candidate_models = [self.model]
            for fb in ["llama-3.1-8b-instant", "llama3-8b-8192", "llama-3.3-70b-versatile"]:
                if fb not in candidate_models:
                    candidate_models.append(fb)

            last_error: Exception | None = None
            for model_name in candidate_models:
                try:
                    response = client.chat.completions.create(
                        messages=groq_messages,
                        model=model_name,
                        temperature=0.2,
                        max_tokens=800,
                    )
                    content = response.choices[0].message.content
                    if content and content.strip():
                        return content
                except Exception as model_err:
                    last_error = model_err
                    logger.warning(
                        "Groq chat response with model %s failed (%s). Trying candidate fallback...",
                        model_name,
                        model_err,
                    )
                    continue

            raise last_error or RuntimeError("All candidate Groq models failed.")

        except Exception as e:
            sanitized_err = sanitize_sensitive_tokens(e)
            logger.warning("Groq chat failed (%s). Falling back to rule-based response.", sanitized_err)
            sop_reply = self._deterministic_chat_response(messages, result)
            return (
                f"**[Offline SOP Mode]** Groq API connection could not be established ({sanitized_err}).\n\n"
                f"{sop_reply}"
            )

    def _deterministic_work_order(
        self,
        result: AnalysisResult,
        machine_id: str,
    ) -> dict[str, Any]:
        """
        Deterministic, rule-based industrial work order generator matching exact SOP.
        Supports fan, pump, valve, and slider machinery families.
        Used when offline or no API key is provided.
        """
        m_type = (result.machine_type or "fan").lower().strip()
        prefix_map = {"fan": "FAN", "pump": "PUMP", "valve": "VALVE", "slider": "SLDR"}
        wo_prefix = prefix_map.get(m_type, m_type.upper()[:5])

        now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        hash_suffix = abs(hash(now_str + machine_id + m_type)) % 9000 + 1000
        wo_id = f"WO-{wo_prefix}-2026-{hash_suffix}"

        status = result.status

        if status == DecisionStatus.NORMAL:
            return {
                "work_order_id": wo_id,
                "asset_id": machine_id,
                "machine_type": m_type,
                "diagnostic_status": "normal",
                "urgency": "ROUTINE",
                "suspected_mechanism": (
                    f"Nominal dynamic balance on {m_type}; lubrication and structural acoustic signature intact"
                ),
                "iso_severity_zone": "Zone A / Zone B (Unrestricted Long-Term Operation)",
                "recommended_steps": [
                    "Maintain continuous automated acoustic monitoring at 4-hour intervals.",
                    f"Verify preventive lubrication and seal schedule for {m_type} in CMMS.",
                    "Conduct routine visual and acoustic inspection at next planned maintenance window.",
                ],
                "safety_checklist": [
                    "Standard PPE compliance in plant operations zone.",
                    "Ensure fixed machine guards and covers remain securely fastened.",
                ],
                "required_tools": [
                    "Industrial acoustic stethoscope / shotgun microphone",
                    "Handheld tachometer or IR pyrometer",
                ],
                "required_ppe": [
                    "ANSI Z87.1 approved safety glasses",
                    "Steel-toe protective boots",
                    "Class 1 earplugs (SNR 28dB)",
                ],
                "engineering_summary": (
                    f"Acoustic reconstruction error ({result.model_prediction.anomaly_score:.4f} MSE if available) "
                    f"for {m_type} ({machine_id}) remains comfortably below the validation threshold. "
                    "No anomalous transient frequencies detected."
                ),
                "ai_engine": "Deterministic Industrial SOP Engine (Offline)",
                "created_at": now_str,
            }

        elif status == DecisionStatus.POSSIBLE_ANOMALY:
            anomaly_profiles = {
                "pump": (
                    "Impeller cavitation, mechanical seal face wear, or shaft bearing raceway distress",
                    [
                        "Verify pump suction strainer differential pressure and check for air entrainment / cavitation.",
                        "Inspect mechanical seal gland for weeping and measure seal housing temperature (< 80°C).",
                        "Perform high-frequency vibration sweep (10 Hz - 10 kHz) on pump & motor bearings.",
                        "Verify motor-pump coupling alignment tolerances.",
                    ],
                    [
                        "Lockout/Tagout (LOTO) 480V pump motor breaker at MCC panel before opening guards.",
                        "Close and tag suction and discharge isolation valves; bleed casing pressure safely.",
                        "Verify 0 RPM shaft standstill prior to seal or coupling access.",
                    ],
                    [
                        "Handheld ISO 10816 vibration pen",
                        "Fluke 62 Max+ Infrared Thermometer",
                        "Suction/Discharge differential pressure gauge",
                        "Laser shaft alignment kit",
                    ],
                ),
                "valve": (
                    "Internal valve seat wire-drawing leak, stem packing friction, or solenoid chatter",
                    [
                        "Check upstream/downstream process pressure gauge to assess internal seat bypass leakage.",
                        "Inspect valve stem packing gland and actuator linkage for mechanical sticking.",
                        "Test solenoid coil current draw and inspect pneumatic actuator air supply pressure.",
                        "Inspect valve seat elastomer/trim for erosion if leak persists after recalibration.",
                    ],
                    [
                        "Lockout/Tagout (LOTO) both electrical solenoid supply (24V/120V) and pneumatic supply air.",
                        "Depressurize and isolate upstream/downstream process piping before valve disassembly.",
                        "Verify zero stored spring or pneumatic actuator energy.",
                    ],
                    [
                        "Ultrasonic acoustic leak detector",
                        "Digital multimeter (loop/coil continuity)",
                        "Calibrated pneumatic regulator & test gauge",
                        "OEM stem packing torque wrench",
                    ],
                ),
                "slider": (
                    "Ball-screw pitting, linear carriage bearing spalling, or guide rail lubrication starvation",
                    [
                        "Inspect hardened linear guide rails and ball-screw nut for scoring or particulate debris.",
                        "Measure carriage backlash and rail parallelism across full traverse stroke.",
                        "Check drive belt/coupling tension and stepper/servo motor holding torque.",
                        "Purge and replenish NLGI Grade 2 lithium-complex grease on runner blocks.",
                    ],
                    [
                        "Lockout/Tagout (LOTO) servo/stepper motion drive power at control cabinet.",
                        "Engage mechanical Z-axis/carriage safety block to prevent gravity or spring drift.",
                        "Confirm zero stored kinetic motion before clearing enclosure interlocks.",
                    ],
                    [
                        "Dial test indicator with magnetic base",
                        "NLGI Grade 2 grease gun with precision nozzle",
                        "Feeler gauges & precision straightedge",
                        "IR thermal camera (scan bearing temperature)",
                    ],
                ),
            }

            default_fan = (
                "Early-stage bearing raceway spalling, ball cage fatigue, or excessive V-belt slippage",
                [
                    "Dispatch mechanical technician to perform physical bearing housing inspection within 24-48 hours.",
                    "Measure bearing surface temperature using infrared thermometer (target: < 75°C).",
                    "Perform high-frequency vibration sweep (10 Hz - 10 kHz) on inboard and outboard bearing pillows.",
                    "Inspect V-belt tension using digital belt tension meter; inspect pulleys for misalignment.",
                    "Inspect impeller blades for dust accumulation, particulate erosion, or foreign object damage.",
                ],
                [
                    "Lockout/Tagout (LOTO) 480V motor disconnect at MCC panel before opening inspection hatch.",
                    "Attach safety padlock and dated warning tag; verify zero electrical potential.",
                    "Visually confirm fan impeller has achieved 0 RPM full standstill through sight glass.",
                    "Discharge residual static electricity before touching rotating assembly.",
                ],
                [
                    "Handheld ISO 10816 vibration pen / accelerometer",
                    "Fluke 62 Max+ Infrared Thermometer",
                    "Digital V-belt tension tester",
                    "Grease gun with NLGI Grade 2 Polyurea synthetic grease",
                    "Dial indicator with magnetic base (shaft runout check)",
                ],
            )
            mechanism, steps, safety, tools = anomaly_profiles.get(m_type, default_fan)

            return {
                "work_order_id": wo_id,
                "asset_id": machine_id,
                "machine_type": m_type,
                "diagnostic_status": "possible_anomaly",
                "urgency": "PRIORITY_INSPECTION",
                "suspected_mechanism": mechanism,
                "iso_severity_zone": "Zone C (Restricted Operation / Maintenance Required within 48h)",
                "recommended_steps": steps,
                "safety_checklist": safety,
                "required_tools": tools,
                "required_ppe": [
                    "ANSI Z87.1 safety glasses with side shields",
                    "Class 2 hearing protection (earmuffs + plugs)",
                    "Steel-toe safety footwear",
                    "Cut-resistant mechanical work gloves",
                ],
                "engineering_summary": (
                    f"Acoustic reconstruction error on {m_type} ({machine_id}) exceeded operational threshold. "
                    "Latent familiarity confirms this is authentic machine noise with anomalous spectral signatures, "
                    "suggesting localized mechanical friction or component micro-distress."
                ),
                "ai_engine": "Deterministic Industrial SOP Engine (Offline)",
                "created_at": now_str,
            }

        elif status == DecisionStatus.NEEDS_REVIEW:
            is_quality_issue = not result.quality.passed
            mechanism = (
                "Sensor saturation / acoustic clipping or low SNR"
                if is_quality_issue
                else "Acoustic domain shift (unfamiliar operational noise or transient background interference)"
            )
            return {
                "work_order_id": wo_id,
                "asset_id": machine_id,
                "diagnostic_status": "needs_review",
                "urgency": "PRIORITY_INSPECTION",
                "suspected_mechanism": mechanism,
                "iso_severity_zone": "Unclassified (Telemetry Integrity Alert)",
                "recommended_steps": [
                    "Inspect acoustic sensor mounting, cabling, and preamplifier gain settings.",
                    "Verify microphone diaphragm is free from oil mist, dust clogs, or moisture.",
                    "Check for excessive ambient background noise (e.g. overhead crane, adjacent air compressor).",
                    "Re-record 10-second baseline sample under steady operating conditions.",
                ],
                "safety_checklist": [
                    "Do not condemn machine mechanics until acoustic instrumentation integrity is confirmed.",
                    "Standard factory floor PPE required during sensor harness inspection.",
                ],
                "required_tools": [
                    "Digital multimeter (check 24V DC sensor loop / 48V phantom power)",
                    "Acoustic microphone calibrator (94 dB / 114 dB @ 1 kHz)",
                    "Contact cleaning spray and lint-free wipes",
                ],
                "required_ppe": [
                    "Safety glasses",
                    "Steel-toe boots",
                    "Hearing protection",
                ],
                "engineering_summary": (
                    "Automatic normal/anomaly classification was withheld because signal quality or domain familiarity "
                    "did not meet trustworthiness standards. Sensor re-validation is required."
                ),
                "ai_engine": "Deterministic Industrial SOP Engine (Offline)",
                "created_at": now_str,
            }

        else:  # INVALID_INPUT
            return {
                "work_order_id": wo_id,
                "asset_id": machine_id,
                "diagnostic_status": "invalid_input",
                "urgency": "ROUTINE",
                "suspected_mechanism": "Audio file corruption or unsupported audio encoding",
                "iso_severity_zone": "N/A",
                "recommended_steps": [
                    "Verify audio file is 16-bit PCM uncompressed WAV at 16 kHz sample rate.",
                    "Check file size is under maximum 25 MB limit.",
                    "Re-export recording from data logger.",
                ],
                "safety_checklist": ["No machine intervention required."],
                "required_tools": ["Digital audio workstation or sound logger utility"],
                "required_ppe": ["None"],
                "engineering_summary": "Audio stream failed decoding validation checks.",
                "ai_engine": "Deterministic Industrial SOP Engine (Offline)",
                "created_at": now_str,
            }

    def _deterministic_chat_response(
        self,
        messages: list[dict[str, str]],
        result: AnalysisResult | None,
    ) -> str:
        """Fallback chat response when Groq API key is not yet configured."""
        last_user_msg = messages[-1]["content"].lower() if messages else ""

        if "work order" in last_user_msg or "ticket" in last_user_msg:
            return (
                "**Work Order Guidance:** You can generate a full structured work order ticket "
                "using the **'Generate Maintenance Work Order'** button on the tab above. "
                "It automatically details the required LOTO safety checklist, inspection tools, "
                "and suspected bearing/impeller failure modes."
            )

        if "bearing" in last_user_msg:
            return (
                "**Bearing Inspection Protocol (ISO 10816):**\n"
                "1. **Safety:** Apply Lockout/Tagout (LOTO) to 480V motor disconnect.\n"
                "2. **Acoustic Check:** High-frequency clicks (2-8 kHz) indicate race micro-spalling or lack of grease.\n"
                "3. **Temperature:** Check housing with infrared thermometer (alert if > 75°C).\n"
                "4. **Lubrication:** Re-grease with NLGI Grade 2 Polyurea synthetic grease ($G = 0.005 \\times D \\times B$)."
            )

        if "anomaly" in last_user_msg or "score" in last_user_msg:
            score = (
                result.model_prediction.anomaly_score
                if result and result.model_prediction
                else 0.0039
            )
            return (
                f"**Acoustic Score Explanation:** The PyTorch Autoencoder reconstructs the 128-band log-mel spectrogram. "
                f"Reconstruction MSE error reflects how much the incoming sound deviates from trained healthy baseline sounds. "
                f"Current score is **{score:.4f}**. When this exceeds the validation threshold, localized acoustic transient energy "
                f"is present."
            )

        return (
            "**PulseForge Maintenance Copilot (Offline SOP Mode):**\n\n"
            "I am ready to assist with fan diagnostics, ISO 10816 severity standards, and Lockout/Tagout procedures.\n\n"
            "*Tip: To enable live Llama-3.3-70b generative reasoning, add your free Groq API key to `.env` or Streamlit Secrets.*"
        )
