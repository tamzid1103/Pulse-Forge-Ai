# Supabase Cloud & Offline SQLite Setup Guide (`DEC-017`)

PulseForge AI uses **Supabase Postgres** as its primary metadata store when online, automatically fails over to embedded **local SQLite** (`"Offline Mode : Operating Locally"`) when internet/Supabase is unreachable, and uploads queued records back to Supabase without duplicates as soon as connectivity returns.

---

## Step 1 — Create the Table & Security Rules in Supabase

1. Open your project in the [Supabase Dashboard](https://supabase.com/dashboard).
2. In the left sidebar, click **SQL Editor** → **New query**.
3. Open [`sql/supabase_schema.sql`](../sql/supabase_schema.sql) in this repository, copy its full contents, paste into the SQL Editor, and click **Run**.

What this script does safely:
- Creates `public.analysis_runs` (exact 18 metadata columns — zero raw audio stored) + performance indexes.
- Enables **Row-Level Security (RLS)** (`ALTER TABLE public.analysis_runs ENABLE ROW LEVEL SECURITY`).
- **Revokes all `anon` and `authenticated` table privileges** and grants access exclusively to `service_role`, preventing anyone on the public internet from reading, modifying, or deleting your inspection history directly from a browser.

---

## Step 2 — Copy Your Project URL and Server Secret Key

1. **Find your Project URL:**
   - In the Supabase Dashboard left sidebar, click **`Data API`** (under **INTEGRATIONS**, just below `API Keys`) — your **Project URL** (`https://<project-ref>.supabase.co`) is displayed at the top.
   - *(Alternatively, click the green **`Connect`** button in the top navbar → App Frameworks / API.)*
2. **Copy your Server Secret Key:**
   - Go to **Settings → API Keys**.
   - Under **Secret keys** (`default`), click the eye/copy icon to copy your **`sb_secret_...`** key (or under **View legacy API keys** if using legacy `service_role` JWT).
   - *(PulseForge automatically uses `apikey`-only headers for modern `sb_secret_...` keys and `Bearer` headers for legacy `eyJ...` JWT keys.)*
   - > **Security Warning:** Only use this key on the **Python backend** (`.env` or Streamlit Cloud Secrets). **Never** use the `Publishable key` (`sb_publishable_...`) for server sync while `anon` is revoked, and **never** paste `sb_secret_...` into `frontend/` JS or Git.

---

## Step 3 — Connect Your Environment

### Option A: Local Development (`streamlit run app/dashboard.py` & `uvicorn src.api:app`)
1. Copy `.env.example` to `.env` (already in `.gitignore`):
   ```bash
   cp .env.example .env
   ```
2. Fill in your credentials inside `.env`:
   ```dotenv
   SUPABASE_URL=https://your-actual-project-ref.supabase.co
   SUPABASE_KEY=sb_secret_your_actual_server_secret_key
   PULSEFORGE_CLOUD_SYNC=true
   PULSEFORGE_SUPABASE_TIMEOUT_SEC=2.0
   PULSEFORGE_SUPABASE_PROBE_TTL_SEC=15.0
   ```
3. Launch locally:
   ```bash
   .venv/bin/streamlit run app/dashboard.py
   ```

### Option B: Streamlit Community Cloud (`https://pulse-forge-ai.streamlit.app`)
1. Open your app on `https://share.streamlit.io` → **⋮ Menu → Settings → Secrets**.
2. Paste in TOML format and save:
   ```toml
   SUPABASE_URL = "https://your-actual-project-ref.supabase.co"
   SUPABASE_KEY = "sb_secret_your_actual_server_secret_key"
   ```
3. Reboot/redeploy if needed so the container loads the new secrets.

---

## Step 4 — Verify Online, Offline Failover & Reconnect Auto-Sync

1. **Online check:**
   - Open **Acoustic Diagnostic Studio** or **Inspection History & Audit**.
   - Sidebar **Database Sync** should display `● Supabase Cloud (Online)`.
   - Run any demo scenario → verify the new row appears in Supabase Dashboard **Table Editor → `analysis_runs`**.
2. **Offline failover test (local run):**
   - Temporarily turn off Wi-Fi / internet on your machine and run a scenario or refresh.
   - PulseForge displays the toast popup & banner **`"Offline Mode : Operating Locally"`** and saves the record safely to local SQLite (`data/analysis_history.db`, `sync_status = 'pending'`).
3. **Reconnect auto-sync test:**
   - Turn Wi-Fi back on and refresh (or click **Sync with Supabase** in View 4 / sidebar).
   - PulseForge automatically batch-uploads all `pending` rows to Supabase (`on_conflict=id`, zero duplicates), shows **`Back Online — Uploaded N local record(s) to Supabase`**, and switches back to **`Supabase Cloud (Online)`**.
