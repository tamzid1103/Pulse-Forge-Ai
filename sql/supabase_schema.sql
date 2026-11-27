-- PulseForge AI — Supabase Postgres Schema & Security Setup
-- Aligns 1:1 with docs/DATABASE_SCHEMA.md (metadata only — never raw audio).
-- Run this script once inside your Supabase Dashboard -> SQL Editor.

CREATE TABLE IF NOT EXISTS public.analysis_runs (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    machine_type TEXT NOT NULL CHECK (machine_type IN ('fan', 'pump', 'valve', 'slider')),
    machine_id TEXT NOT NULL,
    input_label TEXT,
    source_kind TEXT NOT NULL CHECK (source_kind IN ('uploaded', 'demo_sample', 'synthetic')),
    audio_duration_seconds DOUBLE PRECISION NOT NULL CHECK (audio_duration_seconds >= 0),
    sample_rate_hz INTEGER NOT NULL CHECK (sample_rate_hz > 0),
    anomaly_score DOUBLE PRECISION NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    decision_status TEXT NOT NULL CHECK (
        decision_status IN ('normal', 'possible_anomaly', 'needs_review', 'invalid_input')
    ),
    quality_status TEXT NOT NULL CHECK (quality_status IN ('good', 'warning', 'poor')),
    quality_reasons_json TEXT,
    unfamiliarity_score DOUBLE PRECISION,
    uncertainty_reasons_json TEXT,
    model_version TEXT NOT NULL,
    processing_ms INTEGER,
    operator_note TEXT
);

CREATE INDEX IF NOT EXISTS idx_analysis_runs_created_at
    ON public.analysis_runs (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_analysis_runs_machine_type
    ON public.analysis_runs (machine_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_analysis_runs_machine_id
    ON public.analysis_runs (machine_id);

-- Security Hardening (Row-Level Security)
-- Keep RLS ENABLED and leave anonymous browser/public policies disabled.
-- The PulseForge Python backend (Streamlit / FastAPI) authenticates over TLS
-- using your server-side Supabase Secret Key stored strictly in .env / Streamlit Secrets.
ALTER TABLE public.analysis_runs ENABLE ROW LEVEL SECURITY;

-- Explicitly revoke default anonymous/public write or delete grants as defense-in-depth.
REVOKE ALL ON TABLE public.analysis_runs FROM anon;
REVOKE ALL ON TABLE public.analysis_runs FROM authenticated;
GRANT ALL ON TABLE public.analysis_runs TO service_role;
