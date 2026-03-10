"""
ProteinScope v2.1 — Protein Function Predictor
================================================
• Fixed nested f-string SyntaxError (line 748)
• Gemini 2.0 Flash chatbot for interpreting predictions
• Dark-lab UI, batch prediction, download exports
"""

import streamlit as st
import os
import pickle
import json
import io
import re
import csv
import requests
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from collections import Counter

# ════════════════════════════════════════════════════════
#  Page config  (must be FIRST Streamlit call)
# ════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ProteinScope · Function Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════
#  Global CSS — dark-lab aesthetic
# ════════════════════════════════════════════════════════
DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #090e1a; color: #e2e8f0; }

section[data-testid="stSidebar"] {
    background: #0d1424 !important;
    border-right: 1px solid #1e2d4a;
}
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #38bdf8 !important; }

.hero {
    background: linear-gradient(135deg, #0f1f3d 0%, #0a2a4a 50%, #0d1a35 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
}
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #64748b;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.hero-desc {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-top: 0.8rem;
    max-width: 680px;
    line-height: 1.7;
}

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 0.4rem;
}

textarea {
    background: #0d1424 !important;
    border: 1px solid #1e3a5f !important;
    color: #e2e8f0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
    border-radius: 8px !important;
}
textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.15) !important;
}

.pred-card {
    background: #0d1929;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    transition: border-color 0.2s;
}
.pred-card:hover { border-color: #38bdf8; }
.pred-rank { font-family: 'Space Mono', monospace; font-size: 0.72rem; color: #334155; width: 1.6rem; text-align: center; }
.pred-go { font-family: 'Space Mono', monospace; font-size: 0.84rem; color: #38bdf8; min-width: 100px; }
.pred-label { font-size: 0.9rem; color: #e2e8f0; flex: 1; }
.pred-bar-wrap { flex: 2; background: #0a1120; border-radius: 6px; height: 8px; overflow: hidden; }
.pred-bar { height: 100%; border-radius: 6px; transition: width 0.5s ease; }
.pred-pct { font-family: 'Space Mono', monospace; font-size: 0.78rem; min-width: 50px; text-align: right; }

.pill {
    display: inline-block;
    padding: 0.2rem 0.75rem;
    border-radius: 999px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
}
.pill-blue   { background:#0c2d4e; color:#38bdf8; border:1px solid #1e4a72; }
.pill-green  { background:#0d2e25; color:#34d399; border:1px solid #1a4a3a; }
.pill-amber  { background:#2d1f0a; color:#fbbf24; border:1px solid #4a3410; }
.pill-red    { background:#2d0f0f; color:#f87171; border:1px solid #4a1f1f; }
.pill-gray   { background:#141e2e; color:#64748b; border:1px solid #1e2d4a; }
.pill-purple { background:#1a1040; color:#a78bfa; border:1px solid #2d1f6e; }

.stat-row { display:flex; gap:1rem; margin-bottom:1.5rem; flex-wrap:wrap; }
.stat-box {
    background: #0d1929;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    flex: 1; min-width: 120px;
}
.stat-val { font-family: 'Space Mono', monospace; font-size: 1.6rem; color: #38bdf8; font-weight: 700; }
.stat-key { font-size: 0.75rem; color: #64748b; margin-top: 0.2rem; }

div[data-baseweb="select"] > div {
    background: #0d1424 !important;
    border-color: #1e3a5f !important;
    color: #e2e8f0 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.6rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
hr { border-color: #1e2d4a !important; }

button[data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #64748b !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom-color: #38bdf8 !important;
}

.stDownloadButton > button {
    background: #0d2e25 !important;
    color: #34d399 !important;
    border: 1px solid #1a4a3a !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
}

.chat-user {
    background: #0c2d4e;
    border: 1px solid #1e4a72;
    border-radius: 12px 12px 4px 12px;
    padding: 0.8rem 1.1rem;
    margin: 0.6rem 0 0.6rem 4rem;
    color: #e2e8f0;
    font-size: 0.88rem;
    line-height: 1.6;
}
.chat-bot {
    background: #0d1929;
    border: 1px solid #1e3a5f;
    border-radius: 12px 12px 12px 4px;
    padding: 0.8rem 1.1rem;
    margin: 0.6rem 4rem 0.6rem 0;
    color: #cbd5e1;
    font-size: 0.88rem;
    line-height: 1.6;
}
.chat-label-user {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    color: #38bdf8;
    text-align: right;
    margin-bottom: 0.2rem;
    margin-right: 0.2rem;
}
.chat-label-bot {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    color: #a78bfa;
    margin-bottom: 0.2rem;
    margin-left: 0.2rem;
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  Constants
# ════════════════════════════════════════════════════════
PROJECT_ROOT  = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR    = os.path.join(PROJECT_ROOT, 'results', 'models')
ESM_MODEL     = 'facebook/esm2_t6_8M_UR50D'
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VALID_AA      = set("ACDEFGHIKLMNPQRSTVWY")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "moonshotai/kimi-k2-instruct-0905"

GO_LABELS = {
    "GO:0005634": ("Nucleus",               "#818cf8"),
    "GO:0005737": ("Cytoplasm",             "#38bdf8"),
    "GO:0005829": ("Cytosol",               "#34d399"),
    "GO:0016020": ("Membrane",              "#fb923c"),
    "GO:0005886": ("Plasma Membrane",       "#f472b6"),
    "GO:0005739": ("Mitochondrion",         "#facc15"),
    "GO:0005783": ("Endoplasmic Reticulum", "#a78bfa"),
    "GO:0005768": ("Endosome",              "#22d3ee"),
    "GO:0005794": ("Golgi Apparatus",       "#fb7185"),
    "GO:0005576": ("Extracellular Region",  "#86efac"),
    "GO:0005615": ("Extracellular Space",   "#6ee7b7"),
    "GO:0005764": ("Lysosome",              "#fda4af"),
    "GO:0009986": ("Cell Surface",          "#c4b5fd"),
    "GO:0005654": ("Nucleoplasm",           "#93c5fd"),
    "GO:0005694": ("Chromosome",            "#fcd34d"),
    "GO:0000785": ("Chromatin",                    "#f59e0b"),
    "GO:0000978": ("RNA Pol II Promoter Region",   "#e879f9"),
    "GO:0000139": ("Golgi Membrane",               "#fb7185"),
    "GO:0000122": ("Transcription Repressor Cpx",  "#6ee7b7"),
}

EXAMPLE_SEQS = {
    "— choose an example —": "",
    "Hemoglobin β-chain": (
        "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKK"
        "VLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKV"
        "VAGVANALAHKYH"
    ),
    "Green Fluorescent Protein": (
        "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYG"
        "VQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGN"
        "ILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLST"
        "QSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK"
    ),
    "Human Insulin": (
        "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVEL"
        "GGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
    ),
    "p53 Tumor Suppressor": (
        "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPE"
        "AAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYPQGLNGTVNLFRNLNKVSGDGLFKESIGQLTSTE"
        "VKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCS"
    ),
}


# ════════════════════════════════════════════════════════
#  Neural network (must match training architecture)
# ════════════════════════════════════════════════════════
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.skip = (in_dim == out_dim)
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.block(x)
        return out + x if self.skip else out


class ProteinFunctionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_labels, dropout=0.3):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers.append(ResidualBlock(prev, h, dropout))
            prev = h
        self.features   = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, num_labels)

    def forward(self, x):
        return self.classifier(self.features(x))

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))


# ════════════════════════════════════════════════════════
#  Model loading
# ════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🔬 Loading ESM-2 language model and classifier…")
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)
    esm_model = AutoModel.from_pretrained(ESM_MODEL).to(DEVICE).eval()

    ckpt_path = os.path.join(MODELS_DIR, 'best_model.pt')
    if not os.path.exists(ckpt_path):
        return None, None, None, None, "Checkpoint not found: " + ckpt_path

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg  = ckpt['config']
    clf  = ProteinFunctionClassifier(
        cfg['input_dim'], cfg['hidden_dims'], cfg['num_labels'], cfg['dropout']
    ).to(DEVICE)
    clf.load_state_dict(ckpt['model_state_dict'])
    clf.eval()

    lb_path = os.path.join(PROCESSED_DIR, 'label_binarizer.pkl')
    if not os.path.exists(lb_path):
        return None, None, None, None, "label_binarizer.pkl not found: " + lb_path

    with open(lb_path, 'rb') as f:
        mlb = pickle.load(f)

    return tokenizer, esm_model, clf, mlb.classes_, None


# ════════════════════════════════════════════════════════
#  Utilities
# ════════════════════════════════════════════════════════
def validate_sequence(seq: str):
    seq = seq.strip().upper()
    seq = re.sub(r'\s+', '', seq)
    seq = re.sub(r'^>.*\n?', '', seq)
    if not seq:
        return "", "Sequence is empty."
    bad = set(seq) - VALID_AA
    if bad:
        return seq, "Invalid amino-acid characters: " + ", ".join(sorted(bad))
    if len(seq) < 10:
        return seq, "Sequence too short (minimum 10 amino acids)."
    if len(seq) > 2000:
        return seq, "Sequence too long (maximum 2,000 amino acids for this demo)."
    return seq, None


def aa_composition(seq: str) -> dict:
    cnt   = Counter(seq)
    total = len(seq)
    return {aa: round(cnt.get(aa, 0) / total * 100, 2) for aa in sorted(VALID_AA)}


def confidence_pill(p: float):
    if p >= 0.7:
        return "pill-green", "HIGH"
    if p >= 0.4:
        return "pill-amber", "MED"
    return "pill-red", "LOW"


@torch.inference_mode()
def embed_sequence(tokenizer, esm_model, seq: str):
    enc    = tokenizer(seq, return_tensors='pt', padding=True,
                       truncation=True, max_length=1022)
    enc    = {k: v.to(DEVICE) for k, v in enc.items()}
    hidden = esm_model(**enc).last_hidden_state
    mask   = enc['attention_mask'].unsqueeze(-1)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


@torch.inference_mode()
def predict_sequence(tokenizer, esm_model, clf, go_terms,
                     seq: str, threshold=0.10, top_k=8):
    emb   = embed_sequence(tokenizer, esm_model, seq)
    probs = clf.predict_proba(emb).cpu().numpy()[0]
    pairs = sorted(zip(go_terms, probs), key=lambda x: x[1], reverse=True)[:top_k]
    return [(go, float(p)) for go, p in pairs if p >= threshold]


def results_to_csv(predictions, seq: str) -> str:
    buf = io.StringIO()
    w   = csv.writer(buf)
    w.writerow(["Sequence", "GO_Term", "GO_Label", "Confidence"])
    for go, p in predictions:
        label = GO_LABELS.get(go, ("Unknown", ""))[0]
        w.writerow([seq[:30] + "…", go, label, f"{p:.4f}"])
    return buf.getvalue()


def results_to_json(predictions, seq: str) -> str:
    out = {
        "sequence_preview": seq[:60] + ("…" if len(seq) > 60 else ""),
        "sequence_length": len(seq),
        "predictions": [
            {
                "go_term": go,
                "label": GO_LABELS.get(go, ("Unknown", ""))[0],
                "confidence": round(p, 4),
            }
            for go, p in predictions
        ],
    }
    return json.dumps(out, indent=2)


# ════════════════════════════════════════════════════════
#  Gemini API
# ════════════════════════════════════════════════════════
def call_gemini(api_key: str, system_prompt: str, history: list, user_msg: str) -> str:
    # Convert Gemini-style history to OpenAI-style
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        role    = "assistant" if turn["role"] == "model" else "user"
        content = turn["parts"][0]["text"]
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_msg})

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + api_key,
            },
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response else "?"
        if code == 401:
            return "❌ Invalid Groq API key. Check your .env file."
        if code == 429:
            return "⚠️ Rate limit hit. Wait a moment and try again."
        return "❌ HTTP " + str(code) + ": " + (e.response.text if e.response else str(e))
    except Exception as e:
        return "❌ Error: " + str(e)


def build_system_prompt(predictions, seq: str) -> str:
    comp      = aa_composition(seq) if seq else {}
    charged   = sum(comp.get(aa, 0) for aa in "DEKRH")
    hydrophob = sum(comp.get(aa, 0) for aa in "ILMFVW")

    pred_lines = []
    for go, p in predictions:
        label = GO_LABELS.get(go, ("Unknown",))[0]
        pred_lines.append("  - " + go + " (" + label + "): " + str(round(p * 100, 1)) + "% confidence")
    pred_block = "\n".join(pred_lines) if pred_lines else "  No predictions above threshold."

    return (
        "You are ProteinScope AI, a specialist bioinformatics assistant embedded in a\n"
        "protein function prediction tool. You help scientists understand and interpret\n"
        "machine-learning predictions about protein cellular localisation.\n\n"
        "=== CURRENT PREDICTION CONTEXT ===\n"
        "Sequence length  : " + str(len(seq)) + " amino acids\n"
        "Sequence (first 60 aa): " + seq[:60] + ("…" if len(seq) > 60 else "") + "\n"
        "Charged residues : " + str(round(charged, 1)) + "%\n"
        "Hydrophobic res. : " + str(round(hydrophob, 1)) + "%\n\n"
        "Predicted GO Cellular Component terms:\n" + pred_block + "\n\n"
        "=== YOUR ROLE ===\n"
        "• Explain what each predicted GO term means in plain language.\n"
        "• Discuss whether predictions are biologically plausible given sequence properties.\n"
        "• Flag any surprising or contradictory predictions.\n"
        "• Answer follow-up questions about protein biology, GO terms, or the model.\n"
        "• Be concise, accurate, and use accessible language.\n"
        "• If a question is unrelated to the prediction or protein biology, politely redirect.\n\n"
        "Always ground your answers in the predictions shown above."
    )


# ════════════════════════════════════════════════════════
#  Load models
# ════════════════════════════════════════════════════════
tokenizer, esm_model, clf, go_terms, load_error = load_models()

# Session-state init
for key, val in [
    ("chat_history", []),
    ("chat_display", []),
    ("last_predictions", []),
    ("last_sequence", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = val


# ════════════════════════════════════════════════════════
#  Sidebar
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Prediction Settings")
    threshold = st.slider("Confidence threshold", 0.01, 0.90, 0.10, 0.01)
    top_k     = st.slider("Max predictions", 3, 15, 8)
    
    from dotenv import load_dotenv
    load_dotenv()
    gemini_key = os.getenv("GROQ_API_KEY", "")
    st.markdown("## 🤖 Groq API")
    if gemini_key:
        st.markdown('<span class="pill pill-green">✓ Groq key loaded from .env</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill pill-red">❌ GROQ_API_KEY missing in .env</span>', unsafe_allow_html=True)

    st.divider()
    st.markdown("## 🖥️ Runtime")
    device_name = "CUDA GPU" if torch.cuda.is_available() else "CPU"
    st.markdown('<span class="pill pill-blue">' + device_name + '</span>', unsafe_allow_html=True)
    st.markdown('<span class="pill pill-gray">ESM-2 · 8M params</span>', unsafe_allow_html=True)

    st.divider()
    st.markdown("## 📖 GO Colour Key")
    for go, (label, colour) in GO_LABELS.items():
        st.markdown(
            '<span style="display:inline-block;width:10px;height:10px;background:' + colour
            + ';border-radius:50%;margin-right:6px;vertical-align:middle;"></span>'
            + '<span style="font-size:0.78rem;color:#94a3b8;">' + label + '</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown(
        '<span style="font-size:0.7rem;color:#334155;">ProteinScope v2.1 · ESM-2 + ResNet + Gemini</span>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════
#  Hero banner
# ════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <p class="hero-sub">🧬 Genomic language model · ESM-2 · Gene Ontology · Gemini AI</p>
  <h1 class="hero-title">ProteinScope</h1>
  <p class="hero-desc">
    Paste any amino-acid sequence to predict its <strong>cellular component</strong> annotations
    using Facebook's <strong>ESM-2</strong> language model and a residual neural classifier trained
    on UniProt GO data. Then <strong>chat with Gemini</strong> to interpret what the results mean.
  </p>
</div>
""", unsafe_allow_html=True)

if load_error:
    st.error("❌ **Model loading failed:** " + load_error + "\n\nRun `03_train.ipynb` first.")
    st.stop()


# ════════════════════════════════════════════════════════
#  Tabs
# ════════════════════════════════════════════════════════
tab_single, tab_batch, tab_chat, tab_about = st.tabs([
    "🔬 Single Sequence", "📋 Batch Predict", "💬 AI Interpreter", "📚 About"
])


# ────────────────────────────────────────────────────────
#  TAB 1 — Single sequence
# ────────────────────────────────────────────────────────
with tab_single:
    col_input, col_stats = st.columns([3, 2], gap="large")

    with col_input:
        st.markdown('<p class="section-label">Load an example</p>', unsafe_allow_html=True)
        choice      = st.selectbox("", list(EXAMPLE_SEQS.keys()), label_visibility="collapsed")
        default_seq = EXAMPLE_SEQS[choice]

        st.markdown('<p class="section-label">Amino-acid sequence</p>', unsafe_allow_html=True)
        raw_seq = st.text_area(
            "", value=default_seq, height=180,
            placeholder="Paste raw sequence or FASTA (header stripped automatically)…",
            label_visibility="collapsed",
        )

        clean_seq, val_err = validate_sequence(raw_seq)
        seq_len = len(clean_seq)
        length_colour = "#34d399" if seq_len >= 10 else "#f87171"
        st.markdown(
            '<span class="pill pill-gray">Length: '
            '<span style="color:' + length_colour + ';font-weight:700;">'
            + str(seq_len) + '</span> aa</span>',
            unsafe_allow_html=True,
        )

        if val_err and raw_seq.strip():
            st.warning("⚠️ " + val_err)

        predict_btn = st.button(
            "▶ Predict Function", type="primary",
            disabled=bool(val_err) or not clean_seq,
        )

    with col_stats:
        st.markdown('<p class="section-label">Sequence composition</p>', unsafe_allow_html=True)
        if clean_seq and not val_err:
            comp      = aa_composition(clean_seq)
            top_aa    = dict(sorted(comp.items(), key=lambda x: x[1], reverse=True)[:10])
            st.bar_chart(top_aa, height=200, use_container_width=True)
            charged   = sum(comp.get(aa, 0) for aa in "DEKRH")
            hydrophob = sum(comp.get(aa, 0) for aa in "ILMFVW")
            st.markdown(
                '<span class="pill pill-blue">Charged ' + str(round(charged, 1)) + '%</span>'
                '<span class="pill pill-green">Hydrophobic ' + str(round(hydrophob, 1)) + '%</span>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Sequence composition will appear here.")

    if predict_btn and clean_seq and not val_err:
        with st.spinner("🔬 Embedding sequence and predicting GO terms…"):
            predictions = predict_sequence(
                tokenizer, esm_model, clf, go_terms,
                clean_seq, threshold=threshold, top_k=top_k,
            )

        st.session_state.last_predictions = predictions
        st.session_state.last_sequence    = clean_seq
        st.session_state.chat_history     = []
        st.session_state.chat_display     = []

        st.divider()

        if not predictions:
            st.info(
                "No GO terms exceeded the " + str(round(threshold * 100)) + "% confidence threshold. "
                "Try lowering the threshold in the sidebar."
            )
        else:
            top_go, top_p = predictions[0]
            st.markdown(
                '<div class="stat-row">'
                '<div class="stat-box"><div class="stat-val">' + str(len(predictions)) + '</div>'
                '<div class="stat-key">GO terms predicted</div></div>'
                '<div class="stat-box"><div class="stat-val">' + str(round(top_p * 100)) + '%</div>'
                '<div class="stat-key">Top confidence</div></div>'
                '<div class="stat-box"><div class="stat-val">' + str(seq_len) + '</div>'
                '<div class="stat-key">Sequence length (aa)</div></div>'
                '</div>',
                unsafe_allow_html=True,
            )

            st.markdown('<p class="section-label">Predicted cellular locations</p>',
                        unsafe_allow_html=True)

            for rank, (go, p) in enumerate(predictions, 1):
                label, colour  = GO_LABELS.get(go, ("Unknown", "#64748b"))
                pill_cls, pill_txt = confidence_pill(p)
                bar_w = int(p * 100)
                st.markdown(
                    '<div class="pred-card">'
                    '<span class="pred-rank">#' + str(rank) + '</span>'
                    '<span class="pred-go">' + go + '</span>'
                    '<span class="pred-label">' + label
                    + ' <span class="pill ' + pill_cls + '">' + pill_txt + '</span></span>'
                    '<div class="pred-bar-wrap">'
                    '<div class="pred-bar" style="width:' + str(bar_w) + '%;background:' + colour + ';"></div>'
                    '</div>'
                    '<span class="pred-pct" style="color:' + colour + ';">' + str(round(p * 100, 1)) + '%</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )

            st.divider()
            st.markdown('<p class="section-label">Export results</p>', unsafe_allow_html=True)
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                st.download_button(
                    "⬇ Download CSV",
                    data=results_to_csv(predictions, clean_seq),
                    file_name="protein_predictions.csv", mime="text/csv",
                )
            with dcol2:
                st.download_button(
                    "⬇ Download JSON",
                    data=results_to_json(predictions, clean_seq),
                    file_name="protein_predictions.json", mime="application/json",
                )

            st.info("💬 Head to the **AI Interpreter** tab to chat with Gemini about these results!")


# ────────────────────────────────────────────────────────
#  TAB 2 — Batch prediction
# ────────────────────────────────────────────────────────
with tab_batch:
    st.markdown('<p class="section-label">Batch mode — multi-FASTA or one sequence per line</p>',
                unsafe_allow_html=True)
    st.caption("Headers starting with > are stripped automatically.")

    batch_input = st.text_area(
        "", height=220,
        placeholder=">seq1\nMKTYAILV...\n>seq2\nMVHLTPEEK...",
        label_visibility="collapsed",
    )
    batch_btn = st.button("▶ Run Batch Prediction", type="primary")

    if batch_btn and batch_input.strip():
        raw_lines     = batch_input.strip().splitlines()
        sequences     = []
        current       = []
        current_label = "seq1"

        for line in raw_lines:
            line = line.strip()
            if line.startswith(">"):
                if current:
                    sequences.append(("".join(current), current_label))
                    current = []
                current_label = line[1:].split()[0] or ("seq" + str(len(sequences) + 1))
            elif line:
                current.append(line.upper())

        if current:
            sequences.append(("".join(current), current_label))

        if not sequences:
            for i, line in enumerate(raw_lines):
                line = line.strip().upper()
                if line:
                    sequences.append((line, "seq" + str(i + 1)))

        if not sequences:
            st.warning("No valid sequences found.")
        else:
            st.info("Processing " + str(len(sequences)) + " sequence(s)…")
            all_results = []
            prog = st.progress(0)

            for i, (seq, name) in enumerate(sequences):
                seq_clean, err = validate_sequence(seq)
                if err:
                    all_results.append({"name": name, "error": err, "predictions": []})
                else:
                    with st.spinner("Predicting " + name + "…"):
                        preds = predict_sequence(
                            tokenizer, esm_model, clf, go_terms,
                            seq_clean, threshold=threshold, top_k=top_k,
                        )
                    all_results.append({
                        "name": name, "error": None,
                        "seq": seq_clean, "predictions": preds,
                    })
                prog.progress((i + 1) / len(sequences))

            prog.empty()
            st.divider()
            st.markdown('<p class="section-label">Results</p>', unsafe_allow_html=True)

            for res in all_results:
                n_terms = len(res["predictions"])
                status  = "error" if res["error"] else (str(n_terms) + " GO terms")
                with st.expander("📌 " + res["name"] + " — " + status):
                    if res["error"]:
                        st.error(res["error"])
                    elif not res["predictions"]:
                        st.info("No GO terms above threshold.")
                    else:
                        for go, p in res["predictions"]:
                            label, colour = GO_LABELS.get(go, ("Unknown", "#64748b"))
                            st.markdown(
                                '<span class="pill pill-gray">' + go + '</span>'
                                '<span style="color:' + colour + ';font-weight:600;">' + label + '</span>'
                                '<span style="color:#64748b;font-size:0.8rem;margin-left:0.6rem;">'
                                + str(round(p * 100, 1)) + '%</span>',
                                unsafe_allow_html=True,
                            )

            all_rows = []
            for res in all_results:
                for go, p in res.get("predictions", []):
                    label = GO_LABELS.get(go, ("Unknown",))[0]
                    all_rows.append([res["name"], go, label, str(round(p, 4))])

            if all_rows:
                buf = io.StringIO()
                w   = csv.writer(buf)
                w.writerow(["Sequence_ID", "GO_Term", "GO_Label", "Confidence"])
                w.writerows(all_rows)
                st.download_button(
                    "⬇ Download All Results (CSV)",
                    data=buf.getvalue(),
                    file_name="batch_predictions.csv", mime="text/csv",
                )


# ────────────────────────────────────────────────────────
#  TAB 3 — Gemini AI Interpreter
# ────────────────────────────────────────────────────────
with tab_chat:
    st.markdown('<p class="section-label">AI Interpreter — powered by Gemini 2.0 Flash</p>',
                unsafe_allow_html=True)

    if not gemini_key:
        st.warning(
            "⚠️ Add your **Gemini API key** in the sidebar to enable the chatbot. "
            "Get a free key at https://aistudio.google.com/app/apikey"
        )
        st.stop()

    if not st.session_state.last_predictions:
        st.info(
            "🔬 Run a prediction in the **Single Sequence** tab first. "
            "The chatbot is automatically seeded with your prediction results."
        )
        st.stop()

    preds = st.session_state.last_predictions
    seq   = st.session_state.last_sequence

    # Context summary
    st.markdown(
        '<p style="color:#64748b;font-size:0.8rem;font-family:\'Space Mono\',monospace;">'
        'Context: ' + str(len(seq)) + ' aa sequence · ' + str(len(preds)) + ' GO terms predicted</p>',
        unsafe_allow_html=True,
    )
    for go, p in preds:
        label, colour = GO_LABELS.get(go, ("Unknown", "#64748b"))
        st.markdown(
            '<span class="pill pill-gray" style="border-color:' + colour + ';">'
            '<span style="color:' + colour + ';">' + label + '</span> '
            + str(round(p * 100)) + '%</span>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Suggested questions (only if chat is empty)
    if not st.session_state.chat_display:
        st.markdown('<p class="section-label">Suggested questions</p>', unsafe_allow_html=True)
        suggestions = [
            "What do these GO term predictions mean biologically?",
            "Are these predictions consistent with each other?",
            "What type of protein could this be based on its predicted location?",
            "Which prediction should I trust the most and why?",
        ]
        scols = st.columns(2)
        for i, q in enumerate(suggestions):
            if scols[i % 2].button(q, key="suggest_" + str(i)):
                system_prompt = build_system_prompt(preds, seq)
                with st.spinner("Gemini is thinking…"):
                    reply = call_gemini(gemini_key, system_prompt,
                                        st.session_state.chat_history, q)
                st.session_state.chat_history.append({"role": "user",  "parts": [{"text": q}]})
                st.session_state.chat_history.append({"role": "model", "parts": [{"text": reply}]})
                st.session_state.chat_display.append(("user",  q))
                st.session_state.chat_display.append(("model", reply))
                st.rerun()

    # Chat history
    for role, text in st.session_state.chat_display:
        if role == "user":
            st.markdown(
                '<p class="chat-label-user">YOU</p>'
                '<div class="chat-user">' + text + '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p class="chat-label-bot">🤖 PROTEINSCOPE AI</p>'
                '<div class="chat-bot">' + text + '</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    user_input = st.text_input(
        "", placeholder="Ask about the predictions, GO terms, protein biology…",
        label_visibility="collapsed", key="chat_input",
    )
    send_col, clear_col = st.columns([4, 1])
    with send_col:
        send_btn = st.button("Send ➤", type="primary", disabled=not user_input.strip())
    with clear_col:
        clear_btn = st.button("🗑 Clear")

    if clear_btn:
        st.session_state.chat_history = []
        st.session_state.chat_display = []
        st.rerun()

    if send_btn and user_input.strip():
        system_prompt = build_system_prompt(preds, seq)
        with st.spinner("Gemini is thinking…"):
            reply = call_gemini(
                gemini_key, system_prompt,
                st.session_state.chat_history, user_input.strip(),
            )
        st.session_state.chat_history.append({"role": "user",  "parts": [{"text": user_input.strip()}]})
        st.session_state.chat_history.append({"role": "model", "parts": [{"text": reply}]})
        st.session_state.chat_display.append(("user",  user_input.strip()))
        st.session_state.chat_display.append(("model", reply))
        st.rerun()


# ────────────────────────────────────────────────────────
#  TAB 4 — About
# ────────────────────────────────────────────────────────
with tab_about:
    st.markdown('<p class="section-label">Project overview</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
**ProteinScope** predicts Gene Ontology **Cellular Component** annotations from raw
amino-acid sequences using a two-stage pipeline:

1. **ESM-2 (8M)** — Facebook's protein language model converts each sequence into a
   480-dimensional contextual embedding via mean-pooling over the attention mask.
2. **Residual Neural Classifier** — A stack of residual blocks with BatchNorm, GELU
   activation, and dropout maps the embedding to GO term probabilities via sigmoid output.
3. **Gemini 2.0 Flash** — An AI chatbot seeded with the prediction context to explain
   and interpret results in plain language.

Training data is sourced from **UniProt SwissProt**. Multi-label binary cross-entropy
loss is used for training.
        """)

    with c2:
        st.markdown("""
**Supported GO terms include:**

| GO Term | Location |
|---|---|
| GO:0005634 | Nucleus |
| GO:0005737 | Cytoplasm |
| GO:0005829 | Cytosol |
| GO:0016020 | Membrane |
| GO:0005886 | Plasma Membrane |
| GO:0005739 | Mitochondrion |
| GO:0005783 | Endoplasmic Reticulum |
| GO:0005768 | Endosome |

*Use a threshold ≥ 0.5 for high-precision results.*
        """)

    st.divider()
    st.markdown("""
<p class="section-label">References</p>

- Lin et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*.
- Ashburner et al. (2000). Gene Ontology: tool for the unification of biology. *Nature Genetics*.
- UniProt Consortium (2023). UniProt: the Universal Protein Database. *Nucleic Acids Research*.
- Google (2024). Gemini 2.0 Flash — Multimodal AI model.
    """, unsafe_allow_html=True)