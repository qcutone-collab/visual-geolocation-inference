import base64
import html
import json

import streamlit as st

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from openai import APIStatusError
except ImportError:
    APIStatusError = None


def _openai_key_looks_valid(key: str) -> tuple[bool, str]:
    """Reject empty, tutorial-style, or obviously truncated keys before calling OpenAI."""
    k = (key or "").strip()
    if not k:
        return False, "Key is empty."
    if not k.startswith("sk-"):
        return (
            False,
            "The value should start with **sk-**. Copy the full **Secret key** from "
            "[OpenAI API keys](https://platform.openai.com/api-keys), not a placeholder from docs.",
        )
    # Real OpenAI user keys are long; README / tutorial snippets are usually short.
    if len(k) < 40:
        return (
            False,
            "That value is too short to be a real OpenAI secret. Open the key row on "
            "[platform.openai.com/api-keys](https://platform.openai.com/api-keys), click **Reveal**, "
            "and paste the **entire** string (often 50+ characters).",
        )
    low = k.lower()
    bad_snippets = (
        "sk-your",
        "your_key",
        "your-key",
        "yourapi",
        "xxxx",
        "paste",
        "example",
        "placeholder",
        "changeme",
        "replace_me",
        "sk-...",
        "api_key_here",
    )
    for snip in bad_snippets:
        if snip in low:
            return (
                False,
                f'This still looks like **example text** (contains “{snip}”). Use your real secret from '
                "[platform.openai.com/api-keys](https://platform.openai.com/api-keys).",
            )
    return True, ""


st.set_page_config(
    page_title="GeoVision — Visual geolocation",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

GEO_CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,400;0,500;0,600;0,700;1,500&family=IBM+Plex+Mono:wght@400;500&display=swap");

:root {
    --gv-bg0: #070a0d;
    --gv-bg1: #0c1117;
    --gv-surface: rgba(17, 24, 32, 0.72);
    --gv-border: rgba(148, 163, 184, 0.14);
    --gv-text: #e2e8f0;
    --gv-muted: #94a3b8;
    --gv-accent: #2dd4bf;
    --gv-accent-dim: rgba(45, 212, 191, 0.15);
    --gv-violet: #818cf8;
}

html, body, [class*="stApp"] {
    font-family: "Plus Jakarta Sans", system-ui, sans-serif;
}
.stApp {
    background-color: var(--gv-bg0);
    background-image:
        radial-gradient(ellipse 900px 480px at 15% -8%, rgba(45, 212, 191, 0.09), transparent 55%),
        radial-gradient(ellipse 800px 420px at 92% 5%, rgba(129, 140, 248, 0.1), transparent 50%),
        radial-gradient(ellipse 600px 400px at 50% 100%, rgba(45, 212, 191, 0.05), transparent 60%),
        linear-gradient(180deg, var(--gv-bg0) 0%, var(--gv-bg1) 45%, #0a0e14 100%);
    color: var(--gv-text);
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { visibility: hidden; height: 0; }

.block-container {
    padding-top: 0.75rem;
    padding-bottom: 4rem;
    max-width: 1180px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #0f1419 0%, #0c1016 100%) !important;
    border-right: 1px solid var(--gv-border) !important;
}
[data-testid="stSidebar"] .stMarkdown h3 { color: var(--gv-text) !important; font-weight: 600; letter-spacing: -0.02em; }
[data-testid="stSidebar"] label { color: var(--gv-muted) !important; font-size: 0.8rem !important; }

/* Top bar */
.gv-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
    padding: 1rem 0 1.5rem;
    margin-bottom: 0.25rem;
    border-bottom: 1px solid var(--gv-border);
}
.gv-brand {
    display: flex;
    align-items: center;
    gap: 0.85rem;
}
.gv-logo {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: linear-gradient(145deg, rgba(45,212,191,0.25) 0%, rgba(129,140,248,0.2) 100%);
    border: 1px solid rgba(45, 212, 191, 0.35);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.35rem;
    line-height: 1;
}
.gv-product-name {
    font-weight: 700;
    font-size: 1.2rem;
    letter-spacing: -0.03em;
    color: var(--gv-text);
    margin: 0;
}
.gv-product-tag {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.68rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--gv-muted);
    margin: 0.15rem 0 0 0;
}
.gv-nav-meta {
    font-size: 0.8125rem;
    color: var(--gv-muted);
    text-align: right;
    max-width: 280px;
    line-height: 1.45;
}

/* Hero */
.gv-hero {
    text-align: center;
    padding: 2rem 1rem 2.25rem;
}
.gv-hero h1 {
    font-size: clamp(1.75rem, 4.2vw, 2.5rem);
    font-weight: 700;
    letter-spacing: -0.035em;
    margin: 0 0 0.65rem 0;
    line-height: 1.15;
    background: linear-gradient(110deg, #5eead4 0%, #a5b4fc 42%, #c4b5fd 78%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.gv-hero-lead {
    color: var(--gv-muted);
    font-size: 1.0625rem;
    max-width: 36rem;
    margin: 0 auto 1.35rem;
    line-height: 1.6;
}
.gv-trust-row {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.5rem 0.75rem;
}
.gv-pill {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #cbd5e1;
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid var(--gv-border);
    border-radius: 100px;
    padding: 0.35rem 0.75rem;
}

/* Panels & sections */
.gv-panel {
    background: var(--gv-surface);
    border: 1px solid var(--gv-border);
    border-radius: 18px;
    padding: 1.5rem 1.65rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 0 0 1px rgba(255,255,255,0.02), 0 24px 48px -24px rgba(0,0,0,0.45);
    backdrop-filter: blur(12px);
}

/* Streamlit bordered containers used for widget groups (avoid split raw <div> around widgets) */
[class*="st-key-gv_api_access"],
[class*="st-key-gv_configuration"],
[class*="st-key-gv_data_input"] {
    margin-bottom: 1.5rem;
}
[class*="st-key-gv_api_access"] [data-testid="stVerticalBlockBorder"],
[class*="st-key-gv_configuration"] [data-testid="stVerticalBlockBorder"],
[class*="st-key-gv_data_input"] [data-testid="stVerticalBlockBorder"] {
    background: var(--gv-surface) !important;
    border: 1px solid var(--gv-border) !important;
    border-radius: 18px !important;
    box-shadow: 0 0 0 1px rgba(255,255,255,0.02), 0 24px 48px -24px rgba(0,0,0,0.45);
    backdrop-filter: blur(12px);
}
.gv-section-label {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--gv-muted);
    margin: 0 0 1rem 0;
}

[data-testid="stFileUploader"] section {
    padding: 2rem 1.25rem !important;
    border-radius: 14px !important;
    border: 1px dashed rgba(45, 212, 191, 0.35) !important;
    background: rgba(8, 12, 18, 0.55) !important;
    transition: border-color 0.2s ease, background 0.2s ease;
}
[data-testid="stFileUploader"] section:hover {
    border-color: rgba(45, 212, 191, 0.55) !important;
    background: rgba(8, 12, 18, 0.72) !important;
}
[data-testid="stFileUploader"] section small { color: var(--gv-muted) !important; font-size: 0.8rem !important; }

.stTextInput input,
.stTextArea textarea {
    background-color: rgba(8, 12, 18, 0.85) !important;
    border: 1px solid var(--gv-border) !important;
    border-radius: 10px !important;
    color: var(--gv-text) !important;
    font-size: 0.9rem !important;
    pointer-events: auto !important;
    caret-color: var(--gv-accent) !important;
}

/* Empty state steps */
.gv-steps {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 0.5rem;
}
@media (max-width: 900px) {
    .gv-steps { grid-template-columns: 1fr; }
}
.gv-step {
    background: rgba(15, 23, 42, 0.45);
    border: 1px solid var(--gv-border);
    border-radius: 14px;
    padding: 1.25rem 1.35rem;
    position: relative;
    overflow: hidden;
}
.gv-step::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--gv-accent), var(--gv-violet));
    opacity: 0.55;
}
.gv-step-num {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.65rem;
    color: var(--gv-accent);
    letter-spacing: 0.12em;
    margin-bottom: 0.5rem;
}
.gv-step h3 {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--gv-text);
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.02em;
}
.gv-step p {
    font-size: 0.8125rem;
    color: var(--gv-muted);
    margin: 0;
    line-height: 1.5;
}

/* Results layout */
.gv-col-title {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--gv-muted);
    margin: 0 0 1rem 0;
}
.gv-preview-frame {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid var(--gv-border);
    background: #080c12;
    box-shadow: 0 20px 40px -20px rgba(0,0,0,0.6);
}
.gv-results-shell {
    background: rgba(8, 12, 18, 0.5);
    border: 1px solid var(--gv-border);
    border-radius: 16px;
    padding: 1.35rem 1.5rem 1.5rem;
    min-height: 200px;
}

/* Rank rows */
.gv-rank-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.85rem 1rem;
    margin-bottom: 0.65rem;
    border-radius: 12px;
    background: rgba(30, 41, 59, 0.35);
    border: 1px solid var(--gv-border);
}
.gv-rank-row--1 { border-left: 3px solid #eab308; }
.gv-rank-row--2 { border-left: 3px solid #94a3b8; }
.gv-rank-row--3 { border-left: 3px solid #d97706; }
.gv-rank-idx {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--gv-muted);
    min-width: 1.75rem;
}
.gv-rank-body { flex: 1; min-width: 0; }
.gv-rank-name {
    font-size: 1.02rem;
    font-weight: 600;
    color: var(--gv-text);
    margin-bottom: 0.45rem;
    letter-spacing: -0.02em;
}
.gv-bar-track {
    height: 6px;
    border-radius: 100px;
    background: rgba(148, 163, 184, 0.12);
    overflow: hidden;
}
.gv-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #14b8a6, #6366f1);
    transition: width 0.4s ease;
}
.gv-rank-pct {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--gv-accent);
    min-width: 3rem;
    text-align: right;
}

.gv-final {
    margin-top: 1.25rem;
    padding: 1.5rem 1.35rem;
    border-radius: 14px;
    text-align: center;
    background: linear-gradient(155deg, rgba(45, 212, 191, 0.12) 0%, rgba(99, 102, 241, 0.1) 100%);
    border: 1px solid rgba(45, 212, 191, 0.28);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
}
.gv-final-label {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--gv-muted);
    margin-bottom: 0.5rem;
}
.gv-final-country {
    font-size: clamp(1.35rem, 3vw, 1.85rem);
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #f8fafc;
    margin: 0;
}
.gv-final-conf {
    font-size: 0.95rem;
    color: var(--gv-accent);
    margin-top: 0.45rem;
    font-weight: 500;
}

.gv-clues {
    margin-top: 1.15rem;
    padding: 1rem 1.1rem;
    border-radius: 12px;
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid var(--gv-border);
    color: #cbd5e1;
    font-size: 0.875rem;
    line-height: 1.55;
}
.gv-clues strong {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--gv-muted);
    font-weight: 500;
}

.gv-footer {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
    font-size: 0.75rem;
    color: #64748b;
    letter-spacing: 0.02em;
}
.gv-footer a { color: #94a3b8; text-decoration: none; }
.gv-footer a:hover { color: var(--gv-accent); }

div[data-testid="column"] > div {
    min-height: auto;
}
</style>
"""

st.markdown(GEO_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="gv-nav">
        <div class="gv-brand">
            <div class="gv-logo" aria-hidden="true">🌍</div>
            <div>
                <p class="gv-product-name">GeoVision</p>
                <p class="gv-product-tag">Visual geolocation intelligence</p>
            </div>
        </div>
        <p class="gv-nav-meta">Production-style inference UI · Use the configuration toggle below when you need the model or API session tools</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="gv-hero">
        <h1>Where was this photo taken?</h1>
        <p class="gv-hero-lead">
            Upload a street-level image. A vision-language model evaluates architecture, vegetation,
            signage, vehicles, and road markings—then returns ranked country hypotheses with calibrated confidence.
        </p>
        <div class="gv-trust-row">
            <span class="gv-pill">Multimodal reasoning</span>
            <span class="gv-pill">Structured JSON output</span>
            <span class="gv-pill">Session-only key option</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

secrets_key = None
try:
    secrets_key = st.secrets.get("OPENAI_API_KEY")
except Exception:
    secrets_key = None

_raw_session_key = st.session_state.get("OPENAI_API_KEY") or ""
api_key = (secrets_key or _raw_session_key).strip() or None

st.session_state.setdefault("geovision_model_id", "gpt-4.1-mini")

if not api_key:
    st.markdown('<p class="gv-section-label">API access</p>', unsafe_allow_html=True)
    with st.container(border=True, key="gv_api_access"):
        st.markdown(
            "**OpenAI API key** — type or paste below, then click **Save API key**."
        )
        st.text_area(
            "OpenAI API key",
            height=100,
            placeholder="Paste the full sk-… key from platform.openai.com/api-keys",
            key="gv_openai_key_draft",
            help="Stored only in this Streamlit session unless you use Streamlit secrets.",
        )
        save_api_key = st.button("Save API key", type="primary", key="gv_save_openai_key")
    if save_api_key:
        draft = (st.session_state.get("gv_openai_key_draft") or "").strip()
        cleaned = "".join(draft.split())
        if not cleaned:
            st.warning("Enter your API key in the box, then click Save again.")
        else:
            ok, err = _openai_key_looks_valid(cleaned)
            if ok:
                st.session_state["OPENAI_API_KEY"] = cleaned
                st.session_state.pop("gv_openai_key_draft", None)
                st.rerun()
            else:
                st.error(err)

show_configuration = st.toggle(
    "Show configuration",
    value=True,
    key="gv_show_configuration",
    help="Model ID, API status, and clear key — stays on the page; turn off for a cleaner layout.",
)

if show_configuration:
    st.markdown('<p class="gv-section-label">Configuration</p>', unsafe_allow_html=True)
    with st.container(border=True, key="gv_configuration"):
        if secrets_key:
            _sk_ok, _sk_msg = _openai_key_looks_valid(secrets_key)
            if _sk_ok:
                st.info("Using `OPENAI_API_KEY` from Streamlit secrets.")
            else:
                st.error(f"**Streamlit secrets:** {_sk_msg}")
        elif api_key:
            st.success("Session API key active")
            if st.button("Clear session API key", help="Remove the pasted key from this session only."):
                st.session_state.pop("OPENAI_API_KEY", None)
                st.rerun()
        else:
            st.caption("Paste your key in **API access** above when this section is open.")

        model_name = st.text_input(
            "Model identifier",
            help="Any vision-capable model supported by the Responses API.",
            key="geovision_model_id",
        )
else:
    model_name = st.session_state["geovision_model_id"]

st.markdown('<p class="gv-section-label">Data input</p>', unsafe_allow_html=True)
with st.container(border=True, key="gv_data_input"):
    uploaded_file = st.file_uploader(
        "Image upload",
        type=["jpg", "png", "jpeg"],
        help="JPEG or PNG · Clear daylight shots and readable signage improve accuracy.",
        label_visibility="collapsed",
    )
    st.caption("Drag and drop, or click to browse · Max quality depends on source resolution")

if uploaded_file is None:
    st.markdown(
        """
        <div class="gv-panel">
            <p class="gv-section-label" style="margin-bottom:1.25rem;">Workflow</p>
            <div class="gv-steps">
                <div class="gv-step">
                    <div class="gv-step-num">Step 01</div>
                    <h3>Prepare imagery</h3>
                    <p>Use unedited photos when possible. Cropping to the scene of interest helps the model focus.</p>
                </div>
                <div class="gv-step">
                    <div class="gv-step-num">Step 02</div>
                    <h3>Upload &amp; authorize</h3>
                    <p>Turn on <strong>Show configuration</strong> if you need the model or session tools. Keys stay in this session unless you use secrets.</p>
                </div>
                <div class="gv-step">
                    <div class="gv-step-num">Step 03</div>
                    <h3>Review outputs</h3>
                    <p>Inspect top‑3 ranks, the consolidated best guess, and short reasoning signals for transparency.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Interpretation guide", expanded=False):
        st.markdown(
            """
            - **Confidence** reflects model uncertainty; it is not ground-truth accuracy.
            - **Signals** are short rationales—use them to sanity-check the prediction.
            - **Best guess** duplicates the primary `country_guess` field from the model response.
            """
        )

if uploaded_file is not None:
    col_img, col_out = st.columns([1.05, 1], gap="large")
    with col_img:
        st.markdown('<p class="gv-col-title">Source image</p>', unsafe_allow_html=True)
        st.markdown('<div class="gv-preview-frame">', unsafe_allow_html=True)
        st.image(uploaded_file, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_out:
        st.markdown('<p class="gv-col-title">Inference output</p>', unsafe_allow_html=True)
        st.markdown('<div class="gv-results-shell">', unsafe_allow_html=True)
        if OpenAI is None:
            st.error("The `openai` package is required. Install with `pip install openai`.")
        elif not api_key:
            st.warning("Scroll up to **API access**, paste your OpenAI API key, and click **Save API key**.")
        else:
            _ok_key, _key_err = _openai_key_looks_valid(api_key)
            if not _ok_key:
                st.error(_key_err)
                st.caption(
                    "Turn on **Show configuration** and use **Clear session API key**, "
                    "then save a real key from [OpenAI](https://platform.openai.com/api-keys)."
                )
            else:
                client = OpenAI(api_key=api_key)
                image_bytes = uploaded_file.getvalue()
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                mime_type = uploaded_file.type or "image/jpeg"
                data_url = f"data:{mime_type};base64,{image_b64}"

                prompt = (
                    "You are a geolocation assistant. Based only on visual clues in this image, "
                    "predict the most likely country.\n"
                    "Return strict JSON with keys:\n"
                    "country_guess (string),\n"
                    "top_3_countries (array of exactly 3 objects with country and confidence),\n"
                    "confidence (number 0-100),\n"
                    "reasoning_signals (array of short strings).\n"
                    "Keep confidence realistic and avoid overconfidence."
                )

                with st.spinner("Running multimodal analysis…"):
                    try:
                        response = client.responses.create(
                            model=model_name,
                            input=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "input_text", "text": prompt},
                                        {"type": "input_image", "image_url": data_url},
                                    ],
                                }
                            ],
                        )
                        text = response.output_text.strip()

                        try:
                            result = json.loads(text)
                        except json.JSONDecodeError:
                            start = text.find("{")
                            end = text.rfind("}")
                            if start >= 0 and end > start:
                                result = json.loads(text[start : end + 1])
                            else:
                                raise

                        st.markdown(
                            '<p style="margin:0 0 0.9rem 0;font-size:0.72rem;font-weight:600;letter-spacing:0.14em;text-transform:uppercase;color:#94a3b8;">Ranked hypotheses</p>',
                            unsafe_allow_html=True,
                        )
                        for i, item in enumerate(result.get("top_3_countries", []), start=1):
                            country = item.get("country", "Unknown")
                            country_safe = html.escape(str(country))
                            try:
                                conf = float(item.get("confidence", 0) or 0)
                            except (TypeError, ValueError):
                                conf = 0.0
                            conf = max(0.0, min(100.0, conf))
                            row_cls = f"gv-rank-row gv-rank-row--{min(i, 3)}"
                            pct = f"{conf:.0f}"
                            bar_w = f"{conf:.1f}"
                            st.markdown(
                                f'<div class="{row_cls}">'
                                f'<span class="gv-rank-idx">{i:02d}</span>'
                                f'<div class="gv-rank-body">'
                                f'<div class="gv-rank-name">{country_safe}</div>'
                                f'<div class="gv-bar-track"><div class="gv-bar-fill" style="width:{bar_w}%;"></div></div>'
                                f"</div>"
                                f'<span class="gv-rank-pct">{pct}%</span>'
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                        final_country = result.get("country_guess", "Unknown")
                        final_safe = html.escape(str(final_country))
                        try:
                            final_conf = float(result.get("confidence", 0) or 0)
                        except (TypeError, ValueError):
                            final_conf = 0.0
                        final_conf = max(0.0, min(100.0, final_conf))

                        st.markdown(
                            f'<div class="gv-final">'
                            f'<div class="gv-final-label">Primary prediction</div>'
                            f'<p class="gv-final-country">{final_safe}</p>'
                            f'<div class="gv-final-conf">{final_conf:.0f}% model confidence</div>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                        signals = result.get("reasoning_signals", [])
                        if signals:
                            clues = ", ".join(html.escape(str(s)) for s in signals[:10])
                            st.markdown(
                                f'<div class="gv-clues"><strong>Reasoning signals</strong><br/><span style="display:block;margin-top:0.5rem;">{clues}</span></div>',
                                unsafe_allow_html=True,
                            )
                    except Exception as exc:
                        err_lower = str(exc).lower()
                        code_401 = (
                            "401" in str(exc)
                            or "invalid_api_key" in err_lower
                            or (
                                APIStatusError is not None
                                and isinstance(exc, APIStatusError)
                                and getattr(exc, "status_code", None) == 401
                            )
                        )
                        if code_401:
                            st.error(
                                "**OpenAI rejected the API key (401).** Open "
                                "[API keys](https://platform.openai.com/api-keys), create or copy a **Secret key**, "
                                "then use **Clear session API key** (under **Show configuration**) and paste it again in **API access**."
                            )
                        else:
                            st.error(f"Inference failed: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="gv-footer">GeoVision · Visual geolocation prototype · Built with <a href="https://streamlit.io" target="_blank" rel="noopener">Streamlit</a></div>',
    unsafe_allow_html=True,
)
