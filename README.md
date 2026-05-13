# GeoVision — Visual geolocation (Streamlit)

## Project description

**GeoVision** is a small web demo built with [Streamlit](https://streamlit.io). You upload a street-level or travel photograph; the app sends the image plus instructions to OpenAI’s **Responses API** (multimodal: text + image) and asks the model to infer the **most likely country**. The UI shows:

- Three **ranked** country hypotheses with confidence bars  
- A **primary prediction** (“best guess”) with overall confidence  
- Short **reasoning signals** extracted from the model’s JSON reply  

There is **no bundled image dataset**: you supply your own test images at runtime. The app **does not ship with an OpenAI API key**; each user (or host) supplies credentials.

---

## How to run (step-by-step)

### Prerequisites

- **Python 3.10 or newer** (3.11+ recommended)  
- An **OpenAI API key** with access to a **vision-capable** model usable through the Responses API, and **billing / quota** on that OpenAI account (paid usage or active credits)

### Clone and install

```bash
git clone https://github.com/qcutone-collab/visual-geolocation-inference.git
cd visual-geolocation-inference
python -m venv .venv
```

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux**

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Always install from **`requirements.txt`** in the same environment you use for `streamlit run` so the **`openai`** and **`httpx`** packages are present. If you see “openai package is not available”, run `pip install -r requirements.txt` again and restart Streamlit.

### Start the app

```bash
streamlit run app.py
```

Your terminal will show a **Local URL** (usually `http://localhost:8501`). Open it in Chrome, Edge, or Firefox.

---

## API key and configuration

You can use **either** (or both) of the following; **if you paste a key in the app, that pasted key is used first** over Streamlit secrets.

### Option A — Paste in the app (typical for demos)

1. On the main page, open the **API access** section (shown when no key is stored yet).  
2. Paste your full **Secret key** from [platform.openai.com/api-keys](https://platform.openai.com/api-keys).  
3. Click **Save API key**. The value is stored in **Streamlit session state** for that browser session only.

Use **Show configuration** to reveal the **Model identifier** (default `gpt-4.1-mini`), optional **Clear pasted API key**, and status text.

### Option B — Streamlit secrets (local or Cloud)

1. For local runs, create **`.streamlit/secrets.toml`** (gitignored in this repo) or use Streamlit Cloud **Secrets** in the dashboard.  
2. Set:

   ```toml
   OPENAI_API_KEY = "sk-…your real secret…"
   ```

Secrets are used **only when no pasted session key** is saved, so users can still override with a paste.

---

## Controls / how to use

1. **API access** — paste and save your OpenAI secret if prompted.  
2. **Show configuration** — toggle the panel for model ID, key source hint, and **Clear pasted API key**.  
3. Under **Data input**, **upload** a `.jpg`, `.jpeg`, or `.png` image.  
4. Wait for **Inference output**: ranked countries, primary prediction, and reasoning signals.  
5. Use the **Interpretation guide** expander (visible when no image is loaded) for how to read confidence vs. accuracy.

---

## Tech stack and dependencies

| Piece | Role |
|--------|------|
| **Python** | Runtime |
| **Streamlit** | Web UI and file upload |
| **openai** (Python SDK) | Calls `client.responses.create` with image + text |
| **httpx** | HTTP client (declared in `requirements.txt` for reliable installs) |
| **`.streamlit/config.toml`** | Dark theme / primary color for native Streamlit widgets |

```bash
pip install -r requirements.txt
```

The `src/` folder contains additional **training / dataset** scripts for a separate ML pipeline; they are **not** required to run the Streamlit demo in `app.py`.

---

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub.  
2. In [Streamlit Community Cloud](https://streamlit.io/cloud), connect the repository.  
3. **Main file:** `app.py` (at the **repository root**, next to `requirements.txt`).  
4. **Dependencies:** Cloud installs from **`requirements.txt`** at the repo root on each deploy. It must list at least `streamlit`, `openai`, and `httpx` (see the file in this repo).  
5. After changing dependencies, use **Manage app → Reboot** so the environment rebuilds.  
6. **Secrets (optional):** you may add `OPENAI_API_KEY` in Cloud **Secrets**; users can still paste a key in the app, which **takes priority** when saved.

Other hosts work if they can run `streamlit run app.py` and install `requirements.txt`.

---

## Common OpenAI errors

| Symptom | Meaning |
|--------|--------|
| **401 / invalid_api_key** | The secret is wrong, revoked, or not the full key. Create a new secret at [API keys](https://platform.openai.com/api-keys) and save it again in **API access** (or fix secrets). |
| **429 / insufficient_quota** | The key is valid, but the **organization has no usable budget** (no payment method, credits used up, or usage cap hit). Fix under [Billing](https://platform.openai.com/account/billing) and [Limits](https://platform.openai.com/account/limits). |
| **429 (rate limit)** | Too many requests in a short window; wait and retry. |

Inference is billed by **OpenAI** per your plan; GeoVision does not charge separately.

---

## Known issues and limitations

- **Network and billing**: inference requires internet, a valid key, and **available quota** on the OpenAI account.  
- **Not ground truth**: reported **confidence** is model self-assessment, not measured accuracy on a benchmark.  
- **Privacy**: uploaded images are sent to **OpenAI** as part of the API request; do not upload sensitive or private photos.  
- **Model availability**: model IDs and API behavior can change; if a model name stops working, pick another vision-capable model your key supports.  
- **Localhost**: if the browser cannot open the app, confirm the terminal is still running `streamlit run app.py` and try `http://127.0.0.1:8501`. Firewalls or corporate proxies can block local apps.  
- **Fonts**: the UI loads **Google Fonts** from the public internet; offline use may fall back to system fonts.

---

## Repository layout (demo-relevant files)

```
app.py                 # GeoVision Streamlit application
requirements.txt       # Pip dependencies (required at repo root for Streamlit Cloud)
.streamlit/config.toml # Streamlit theme (safe to commit)
PROJECT_PLAN.md        # Planning notes (optional reading)
src/                   # Optional: separate ML tooling, not needed for Streamlit demo
CHANGELOG.md           # Brief history for submission evidence
```

---

## Credits

- **OpenAI** — API, models, and Python SDK.  
- **Streamlit** — application framework ([streamlit.io](https://streamlit.io)).  
- **Google Fonts** — [Plus Jakarta Sans](https://fonts.google.com/specimen/Plus+Jakarta+Sans) and [IBM Plex Mono](https://fonts.google.com/specimen/IBM+Plex+Mono), loaded via `fonts.googleapis.com` in `app.py` CSS.

---

## Submission checklist (quick)

| Requirement | Where |
|-------------|--------|
| Project link | Your GitHub URL + optional Streamlit Cloud deployed URL |
| Runnable build | `pip install -r requirements.txt` then `streamlit run app.py` |
| Assets | User-provided uploads only; theme in `.streamlit/config.toml`; fonts remote |
| README | This file |
| Evidence of work | Git commit history **or** `CHANGELOG.md` |

Published repo: [github.com/qcutone-collab/visual-geolocation-inference](https://github.com/qcutone-collab/visual-geolocation-inference).
