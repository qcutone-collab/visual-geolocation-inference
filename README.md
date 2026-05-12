# GeoVision — Visual geolocation (Streamlit)

## Project description

**GeoVision** is a small web demo built with [Streamlit](https://streamlit.io). You upload a street-level or travel photograph; the app sends the image plus instructions to OpenAI’s **Responses API** (multimodal: text + image) and asks the model to infer the **most likely country**. The UI shows:

- Three **ranked** country hypotheses with confidence bars  
- A **primary prediction** (“best guess”) with overall confidence  
- Short **reasoning signals** extracted from the model’s JSON reply  

There is **no bundled image dataset**: you supply your own test images at runtime.

---

## How to run (step-by-step)

### Prerequisites

- **Python 3.10 or newer** (3.11+ recommended)  
- An **OpenAI API key** with access to a **vision-capable** model usable through the Responses API  

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

### API key (choose one)

**Option A — Streamlit secrets (local, not committed)**

1. Create `.streamlit/secrets.toml` (this filename is ignored by `.gitignore` in this repo).  
2. Add:

   ```toml
   OPENAI_API_KEY = "sk-..."
   ```

**Option B — Paste in the app**

Open the sidebar in the browser and paste your key when prompted. It is kept in **session state** for that browser session only.

### Start the app

```bash
streamlit run app.py
```

Your terminal will show a **Local URL** (usually `http://localhost:8501`). Open it in Chrome, Edge, or Firefox.

---

## Controls / how to use

1. Open the **menu (☰)** to show the **sidebar** if it is collapsed.  
2. Enter your **OpenAI API key** (if not using secrets) and adjust the **model identifier** if your account uses a different vision model. Default in code: `gpt-4.1-mini`.  
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
| **`.streamlit/config.toml`** | Dark theme / primary color for native Streamlit widgets |

Install everything with:

```bash
pip install -r requirements.txt
```

The `src/` folder contains additional **training / dataset** scripts for a separate ML pipeline; they are **not** required to run the Streamlit demo in `app.py`.

---

## Deployment (optional, for a “project link”)

**Streamlit Community Cloud**

1. Push this repo to GitHub.  
2. Connect the repo in [Streamlit Cloud](https://streamlit.io/cloud).  
3. Main file: `app.py`.  
4. In **Secrets**, add `OPENAI_API_KEY` (same key name the app reads).  

Other hosts work if they can run `streamlit run app.py` and set the same secret / env as needed.

---

## Known issues and limitations

- **Network and billing**: inference requires internet and a valid OpenAI key; usage is billed per your OpenAI plan.  
- **Not ground truth**: reported **confidence** is model self-assessment, not measured accuracy on a benchmark.  
- **Privacy**: uploaded images are sent to **OpenAI** as part of the API request; do not upload sensitive or private photos.  
- **Model availability**: model IDs and API behavior can change; if a model name stops working, pick another vision-capable model your key supports.  
- **Localhost**: if the browser cannot open the app, confirm the terminal is still running `streamlit run app.py` and try `http://127.0.0.1:8501`. Firewalls or corporate proxies can block local apps.  
- **Fonts**: the UI loads **Google Fonts** from the public internet; offline use may fall back to system fonts.

---

## Repository layout (demo-relevant files)

```
app.py                 # GeoVision Streamlit application
requirements.txt       # Pip dependencies for the app
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
| Project link | Your GitHub URL + optional Streamlit Cloud (or similar) deployed URL |
| Runnable build | `pip install -r requirements.txt` then `streamlit run app.py` |
| Assets | User-provided uploads only; theme in `.streamlit/config.toml`; fonts remote |
| README | This file |
| Evidence of work | Git commit history **or** `CHANGELOG.md` |

Published repo: [github.com/qcutone-collab/visual-geolocation-inference](https://github.com/qcutone-collab/visual-geolocation-inference).
