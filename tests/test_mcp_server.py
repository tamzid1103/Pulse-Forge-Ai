"""
Comprehensive unit and protocol tests for PulseForge AI MCP Server (Milestone M9).

Validates:
- Tool registration, parameter validation, and schema definitions.
- Execution of all 5 MCP tools (`analyze_fan_audio`, `inspect_signal_quality`,
  `get_machine_telemetry`, `generate_maintenance_work_order`, `list_demo_scenarios`).
- Error handling for missing files, corrupted inputs, and invalid statuses.
- Resource URI resolution and content retrieval (`pulseforge://fleet/status`,
  `pulseforge://standards/iso10816`, `pulseforge://standards/sop/fan_maintenance`,
  `pulseforge://scenarios/catalog`).
- Prompt template rendering for agent triage and OSHA compliance.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.mcp_server import create_server

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_tool_data(res):
    """Helper to unwrap tool return content from MCPServer response."""
    if res.content and len(res.content) > 0:
        if len(res.content) == 1 and hasattr(res.content[0], "text"):
            return json.loads(res.content[0].text)
        return [json.loads(c.text) for c in res.content if hasattr(c, "text")]
    if res.structured_content:
        return res.structured_content.get("result", res.structured_content)
    return {}


@pytest.fixture(scope="module")
def mcp_server():
    """Create a fresh MCPServer instance for testing."""
    return create_server()


class TestMCPServerRegistration:
    """Validate protocol registration of tools, resources, and prompts."""

    @pytest.mark.anyio
    async def test_tools_registration(self, mcp_server):
        tools = await mcp_server.list_tools()
        tool_names = {t.name for t in tools}
        expected = {
            "analyze_fan_audio",
            "inspect_signal_quality",
            "get_machine_telemetry",
            "generate_maintenance_work_order",
            "list_demo_scenarios",
        }
        assert expected.issubset(tool_names)
        assert len(tools) == 5

    @pytest.mark.anyio
    async def test_resources_registration(self, mcp_server):
        resources = await mcp_server.list_resources()
        resource_uris = {str(r.uri) for r in resources}
        expected = {
            "pulseforge://fleet/status",
            "pulseforge://standards/iso10816",
            "pulseforge://standards/sop/fan_maintenance",
            "pulseforge://scenarios/catalog",
        }
        assert expected.issubset(resource_uris)
        assert len(resources) == 4

    @pytest.mark.anyio
    async def test_prompts_registration(self, mcp_server):
        prompts = await mcp_server.list_prompts()
        prompt_names = {p.name for p in prompts}
        expected = {
            "triage_acoustic_anomaly",
            "osha_loto_compliance_check",
        }
        assert expected.issubset(prompt_names)
        assert len(prompts) == 2


class TestMCPToolsExecution:
    """Validate tool execution logic and structured payloads."""

    @pytest.mark.anyio
    async def test_list_demo_scenarios(self, mcp_server):
        res = await mcp_server.call_tool("list_demo_scenarios", {})
        assert not res.is_error
        scenarios = _get_tool_data(res)
        assert isinstance(scenarios, list)
        assert len(scenarios) == 5

        scenario_ids = [s["id"] for s in scenarios]
        assert "scenario_01" in scenario_ids
        assert "scenario_02" in scenario_ids

        for s in scenarios:
            assert "file_path" in s
            assert "expected_status" in s
            assert "description" in s

    @pytest.mark.anyio
    async def test_inspect_signal_quality_valid(self, mcp_server):
        demo_wav = PROJECT_ROOT / "demo_audio" / "01_normal_fan.wav"
        if not demo_wav.is_file():
            pytest.skip("Demo audio 01_normal_fan.wav not found")

        res = await mcp_server.call_tool("inspect_signal_quality", {"audio_path": str(demo_wav)})
        assert not res.is_error
        data = _get_tool_data(res)
        assert data["suitable_for_ml"] is True
        assert data["duration_seconds"] > 0.0
        assert data["rms_amplitude"] > 0.001
        assert data["is_silent"] is False
        assert data["is_clipped"] is False

    @pytest.mark.anyio
    async def test_inspect_signal_quality_silent(self, mcp_server):
        silent_wav = PROJECT_ROOT / "demo_audio" / "03_silent_sensor_failure.wav"
        if not silent_wav.is_file():
            pytest.skip("Demo audio 03_silent_sensor_failure.wav not found")

        res = await mcp_server.call_tool("inspect_signal_quality", {"audio_path": str(silent_wav)})
        assert not res.is_error
        data = _get_tool_data(res)
        assert data["suitable_for_ml"] is False
        assert data["is_silent"] is True
        assert "Hardware malfunction" in data["recommendation"]

    @pytest.mark.anyio
    async def test_inspect_signal_quality_missing_file(self, mcp_server):
        res = await mcp_server.call_tool(
            "inspect_signal_quality",
            {"audio_path": "/nonexistent/path/to/missing.wav"}
        )
        assert not res.is_error
        data = _get_tool_data(res)
        assert data["error"] == "file_not_found"
        assert data["suitable_for_ml"] is False

    @pytest.mark.anyio
    async def test_analyze_fan_audio_normal(self, mcp_server):
        demo_wav = PROJECT_ROOT / "demo_audio" / "01_normal_fan.wav"
        if not demo_wav.is_file():
            pytest.skip("Demo audio 01_normal_fan.wav not found")

        res = await mcp_server.call_tool(
            "analyze_fan_audio",
            {"audio_path": str(demo_wav), "machine_id": "fan_00_a"}
        )
        assert not res.is_error
        data = _get_tool_data(res)
        assert data["status"] == "normal"
        assert data["machine_id"] == "fan_00_a"
        assert "anomaly_detection" in data
        assert data["anomaly_detection"]["is_anomaly"] is False
        assert data["quality"]["passed"] is True
        assert data["run_id"] is not None

    @pytest.mark.anyio
    async def test_analyze_fan_audio_anomaly(self, mcp_server):
        anomaly_wav = PROJECT_ROOT / "demo_audio" / "02_bearing_anomaly.wav"
        if not anomaly_wav.is_file():
            pytest.skip("Demo audio 02_bearing_anomaly.wav not found")

        res = await mcp_server.call_tool(
            "analyze_fan_audio",
            {"audio_path": str(anomaly_wav), "machine_id": "fan_00_b"}
        )
        assert not res.is_error
        data = _get_tool_data(res)
        assert data["status"] == "possible_anomaly"
        assert data["anomaly_detection"]["is_anomaly"] is True

    @pytest.mark.anyio
    async def test_analyze_fan_audio_missing_file(self, mcp_server):
        res = await mcp_server.call_tool(
            "analyze_fan_audio",
            {"audio_path": "nonexistent_audio.wav"}
        )
        assert not res.is_error
        data = _get_tool_data(res)
        assert data["error"] == "file_not_found"
        assert data["status"] == "invalid_input"

    @pytest.mark.anyio
    async def test_get_machine_telemetry(self, mcp_server):
        res = await mcp_server.call_tool(
            "get_machine_telemetry",
            {"machine_id": "all", "limit": 5}
        )
        assert not res.is_error
        data = _get_tool_data(res)
        assert "monitored_fleet_summary" in data
        assert "recent_inspections" in data
        assert isinstance(data["recent_inspections"], list)

    @pytest.mark.anyio
    async def test_generate_maintenance_work_order(self, mcp_server):
        res = await mcp_server.call_tool(
            "generate_maintenance_work_order",
            {
                "machine_id": "fan_id_00",
                "decision_status": "possible_anomaly",
                "operator_note": "Unusual high-pitch scraping heard during morning shift",
            }
        )
        assert not res.is_error
        data = _get_tool_data(res)
        assert "urgency" in data
        assert data["operator_note"] == "Unusual high-pitch scraping heard during morning shift"
        # Should contain safety and step guidance
        assert "recommended_steps" in data or "recommended_action" in data


class TestMCPResourcesAndPrompts:
    """Validate resource reading and prompt template generation."""

    @pytest.mark.anyio
    async def test_read_fleet_status_resource(self, mcp_server):
        contents = await mcp_server.read_resource("pulseforge://fleet/status")
        assert len(contents) > 0
        text = contents[0].content
        assert "PulseForge AI — Plant Equipment Fleet Telemetry" in text
        assert "Fan-00-A" in text

    @pytest.mark.anyio
    async def test_read_iso_standards_resource(self, mcp_server):
        contents = await mcp_server.read_resource("pulseforge://standards/iso10816")
        assert len(contents) > 0
        text = contents[0].content
        assert "ISO 10816-3" in text
        assert "Zone A" in text
        assert "Zone D" in text

    @pytest.mark.anyio
    async def test_read_fan_sop_resource(self, mcp_server):
        contents = await mcp_server.read_resource("pulseforge://standards/sop/fan_maintenance")
        assert len(contents) > 0
        text = contents[0].content
        assert "Lockout/Tagout" in text or "LOTO" in text

    @pytest.mark.anyio
    async def test_read_scenarios_catalog_resource(self, mcp_server):
        contents = await mcp_server.read_resource("pulseforge://scenarios/catalog")
        assert len(contents) > 0
        catalog = json.loads(contents[0].content)
        assert isinstance(catalog, list)
        assert len(catalog) == 5

    @pytest.mark.anyio
    async def test_get_triage_prompt(self, mcp_server):
        res = await mcp_server.get_prompt(
            "triage_acoustic_anomaly",
            {"machine_id": "fan_01_a", "anomaly_score": "0.0075", "threshold": "0.0032"}
        )
        messages = res.messages
        assert len(messages) > 0
        prompt_text = messages[0].content.text
        assert "fan_01_a" in prompt_text
        assert "0.0075" in prompt_text
        assert "Exceedance Ratio" in prompt_text

    @pytest.mark.anyio
    async def test_get_osha_loto_prompt(self, mcp_server):
        res = await mcp_server.get_prompt(
            "osha_loto_compliance_check",
            {"machine_id": "fan_id_00", "work_order_id": "WO-2026-001"}
        )
        messages = res.messages
        assert len(messages) > 0
        prompt_text = messages[0].content.text
        assert "fan_id_00" in prompt_text
        assert "WO-2026-001" in prompt_text
        assert "OSHA 1910.147" in prompt_text
