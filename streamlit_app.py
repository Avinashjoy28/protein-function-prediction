"""
Protein Function Predictor — Optimized Streamlit App
=====================================================
Improvements over v1:
  • Rich dark-lab UI with custom CSS (no generic AI aesthetics)
  • Sequence validation with helpful error messages
  • Batch prediction (multiple sequences at once)
  • Per-term confidence colour-coding + GO label lookup
  • Downloadable results as CSV / JSON
  • Amino-acid composition mini-chart (st.bar_chart)
  • Model + device info sidebar
  • Robust error handling throughout
  • All heavy objects cached with @st.cache_resource
  • Spinner scoped only around heavy ops
"""

import streamlit as st
import os
import pickle
import json
import io
import re
import csv
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from collections import Counter

# ════════════════════════════════════════════════════════
#  Page config  (must be first Streamlit call)
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
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* ---------- base ---------- */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background: #090e1a;
    color: #e2e8f0;
}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
    background: #0d1424 !important;
    border-right: 1px solid #1e2d4a;
}
section[data-testid="stSidebar"] * {
    color: #94a3b8 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #38bdf8 !important;
}

/* ---------- hero banner ---------- */
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

/* ---------- section headers ---------- */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 0.4rem;
}

/* ---------- sequence textarea ---------- */
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

/* ---------- prediction cards ---------- */
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
.pred-rank {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #334155;
    width: 1.6rem;
    text-align: center;
}
.pred-go {
    font-family: 'Space Mono', monospace;
    font-size: 0.84rem;
    color: #38bdf8;
    min-width: 100px;
}
.pred-label {
    font-size: 0.9rem;
    color: #e2e8f0;
    flex: 1;
}
.pred-bar-wrap {
    flex: 2;
    background: #0a1120;
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
}
.pred-bar {
    height: 100%;
    border-radius: 6px;
    transition: width 0.5s ease;
}
.pred-pct {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    min-width: 50px;
    text-align: right;
}

/* ---------- info pills ---------- */
.pill {
    display: inline-block;
    padding: 0.2rem 0.75rem;
    border-radius: 999px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
}
.pill-blue  { background:#0c2d4e; color:#38bdf8; border:1px solid #1e4a72; }
.pill-green { background:#0d2e25; color:#34d399; border:1px solid #1a4a3a; }
.pill-amber { background:#2d1f0a; color:#fbbf24; border:1px solid #4a3410; }
.pill-red   { background:#2d0f0f; color:#f87171; border:1px solid #4a1f1f; }
.pill-gray  { background:#141e2e; color:#64748b; border:1px solid #1e2d4a; }

/* ---------- stat boxes ---------- */
.stat-row { display:flex; gap:1rem; margin-bottom:1.5rem; flex-wrap:wrap; }
.stat-box {
    background: #0d1929;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    flex: 1;
    min-width: 120px;
}
.stat-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    color: #38bdf8;
    font-weight: 700;
}
.stat-key {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.2rem;
}

/* ---------- selectbox ---------- */
div[data-baseweb="select"] > div {
    background: #0d1424 !important;
    border-color: #1e3a5f !important;
    color: #e2e8f0 !important;
}

/* ---------- buttons ---------- */
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

/* ---------- divider ---------- */
hr { border-color: #1e2d4a !important; }

/* ---------- tabs ---------- */
button[data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #64748b !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom-color: #38bdf8 !important;
}

/* ---------- metric ---------- */
[data-testid="stMetric"] {
    background: #0d1929;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 0.8rem 1rem;
}

/* ---------- warning / error ---------- */
.stAlert { border-radius: 8px !important; }

/* ---------- download button ---------- */
.stDownloadButton > button {
    background: #0d2e25 !important;
    color: #34d399 !important;
    border: 1px solid #1a4a3a !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  Constants & GO label lookup
# ════════════════════════════════════════════════════════
PROJECT_ROOT  = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR    = os.path.join(PROJECT_ROOT, 'results', 'models')
ESM_MODEL     = 'facebook/esm2_t6_8M_UR50D'
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Curated GO → human-readable label mapping
GO_LABELS = {
    "GO:0005634": ("Nucleus",              "#818cf8"),
    "GO:0005737": ("Cytoplasm",            "#38bdf8"),
    "GO:0005829": ("Cytosol",              "#34d399"),
    "GO:0016020": ("Membrane",             "#fb923c"),
    "GO:0005886": ("Plasma Membrane",      "#f472b6"),
    "GO:0005739": ("Mitochondrion",        "#facc15"),
    "GO:0005783": ("Endoplasmic Reticulum","#a78bfa"),
    "GO:0005768": ("Endosome",             "#22d3ee"),
    "GO:0005794": ("Golgi Apparatus",      "#fb7185"),
    "GO:0005576": ("Extracellular Region", "#86efac"),
    "GO:0005615": ("Extracellular Space",  "#6ee7b7"),
    "GO:0005764": ("Lysosome",             "#fda4af"),
    "GO:0009986": ("Cell Surface",         "#c4b5fd"),
    "GO:0005654": ("Nucleoplasm",          "#93c5fd"),
    "GO:0005694": ("Chromosome",           "#fcd34d"),
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
#  Neural network classes  (must match training code)
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
#  Model loading  (cached across reruns)
# ════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🔬 Loading ESM-2 language model and classifier…")
def load_models():
    tokenizer  = AutoTokenizer.from_pretrained(ESM_MODEL)
    esm_model  = AutoModel.from_pretrained(ESM_MODEL).to(DEVICE).eval()

    ckpt_path = os.path.join(MODELS_DIR, 'best_model.pt')
    if not os.path.exists(ckpt_path):
        return None, None, None, None, f"Checkpoint not found: {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg  = ckpt['config']
    clf  = ProteinFunctionClassifier(
        cfg['input_dim'], cfg['hidden_dims'], cfg['num_labels'], cfg['dropout']
    ).to(DEVICE)
    clf.load_state_dict(ckpt['model_state_dict'])
    clf.eval()

    lb_path = os.path.join(PROCESSED_DIR, 'label_binarizer.pkl')
    if not os.path.exists(lb_path):
        return None, None, None, None, f"label_binarizer.pkl not found: {lb_path}"

    with open(lb_path, 'rb') as f:
        mlb = pickle.load(f)

    return tokenizer, esm_model, clf, mlb.classes_, None


# ════════════════════════════════════════════════════════
#  Utilities
# ════════════════════════════════════════════════════════
def validate_sequence(seq: str):
    """Return (cleaned_seq, error_msg | None)."""
    seq = seq.strip().upper()
    seq = re.sub(r'\s+', '', seq)            # strip whitespace / newlines
    seq = re.sub(r'^>.*\n?', '', seq)        # strip FASTA header if present
    if not seq:
        return "", "Sequence is empty."
    bad = set(seq) - VALID_AA
    if bad:
        return seq, f"Invalid amino-acid characters detected: {', '.join(sorted(bad))}"
    if len(seq) < 10:
        return seq, "Sequence too short (minimum 10 amino acids)."
    if len(seq) > 2000:
        return seq, "Sequence too long (maximum 2,000 amino acids for this demo)."
    return seq, None


def aa_composition(seq: str) -> dict:
    cnt   = Counter(seq)
    total = len(seq)
    return {aa: cnt.get(aa, 0) / total * 100 for aa in sorted(VALID_AA)}


def confidence_colour(p: float) -> str:
    if p >= 0.7: return "#34d399"   # green — high
    if p >= 0.4: return "#fbbf24"   # amber — medium
    return "#f87171"                # red   — low


def confidence_pill(p: float) -> str:
    if p >= 0.7: return "pill-green", "HIGH"
    if p >= 0.4: return "pill-amber", "MED"
    return "pill-red", "LOW"


@torch.inference_mode()
def embed_sequence(tokenizer, esm_model, seq: str):
    enc = tokenizer(seq, return_tensors='pt', padding=True,
                    truncation=True, max_length=1022)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
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


def results_to_csv(predictions: list, seq: str) -> str:
    buf = io.StringIO()
    w   = csv.writer(buf)
    w.writerow(["Sequence", "GO_Term", "GO_Label", "Confidence"])
    for go, p in predictions:
        label = GO_LABELS.get(go, ("Unknown", ""))[0]
        w.writerow([seq[:30] + "…", go, label, f"{p:.4f}"])
    return buf.getvalue()


def results_to_json(predictions: list, seq: str) -> str:
    out = {
        "sequence_preview": seq[:60] + ("…" if len(seq) > 60 else ""),
        "sequence_length": len(seq),
        "predictions": [
            {"go_term": go, "label": GO_LABELS.get(go, ("Unknown", ""))[0],
             "confidence": round(p, 4)}
            for go, p in predictions
        ]
    }
    return json.dumps(out, indent=2)


# ════════════════════════════════════════════════════════
#  Load models once
# ════════════════════════════════════════════════════════
tokenizer, esm_model, clf, go_terms, load_error = load_models()


# ════════════════════════════════════════════════════════
#  Sidebar
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    threshold = st.slider("Confidence threshold", 0.01, 0.90, 0.10, 0.01,
                          help="GO terms below this confidence are hidden.")
    top_k     = st.slider("Max predictions shown", 3, 15, 8,
                          help="Upper cap on returned predictions.")

    st.divider()
    st.markdown("## 🖥️ Runtime")

    device_name = "CUDA GPU" if torch.cuda.is_available() else "CPU"
    st.markdown(f'<span class="pill pill-blue">{device_name}</span>', unsafe_allow_html=True)
    st.markdown(f'<span class="pill pill-gray">ESM-2 · 8M params</span>', unsafe_allow_html=True)
    if torch.cuda.is_available():
        st.markdown(f'<span class="pill pill-green">{torch.cuda.get_device_name(0)}</span>',
                    unsafe_allow_html=True)

    st.divider()
    st.markdown("## 📖 GO Colour Key")
    for go, (label, colour) in GO_LABELS.items():
        st.markdown(
            f'<span style="display:inline-block;width:10px;height:10px;'
            f'background:{colour};border-radius:50%;margin-right:6px;vertical-align:middle;"></span>'
            f'<span style="font-size:0.78rem;color:#94a3b8;">{label}</span>',
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown('<span style="font-size:0.7rem;color:#334155;">ProteinScope v2.0 · ESM-2 + ResNet classifier</span>',
                unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  Hero Banner
# ════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <p class="hero-sub">🧬 Genomic language model · ESM-2 · Gene Ontology</p>
  <h1 class="hero-title">ProteinScope</h1>
  <p class="hero-desc">
    Paste any amino-acid sequence to predict its cellular component annotations using
    Facebook's <strong>ESM-2</strong> protein language model combined with a residual
    neural classifier trained on <strong>UniProt GO annotations</strong>.
  </p>
</div>
""", unsafe_allow_html=True)

# Abort if models failed to load
if load_error:
    st.error(f"❌ **Model loading failed:** {load_error}\n\nRun `03_train.ipynb` first.")
    st.stop()

# ════════════════════════════════════════════════════════
#  Main tabs
# ════════════════════════════════════════════════════════
tab_single, tab_batch, tab_about = st.tabs(["🔬 Single Sequence", "📋 Batch Predict", "📚 About"])


# ────────────────────────────────────────────────────────
#  TAB 1 — Single sequence
# ────────────────────────────────────────────────────────
with tab_single:
    col_input, col_stats = st.columns([3, 2], gap="large")

    with col_input:
        st.markdown('<p class="section-label">Load an example</p>', unsafe_allow_html=True)
        choice = st.selectbox("", list(EXAMPLE_SEQS.keys()), label_visibility="collapsed")
        default_seq = EXAMPLE_SEQS[choice]

        st.markdown('<p class="section-label">Amino-acid sequence</p>', unsafe_allow_html=True)
        raw_seq = st.text_area(
            "", value=default_seq, height=180,
            placeholder="Paste raw sequence or FASTA (header will be stripped)…",
            label_visibility="collapsed"
        )

        clean_seq, val_err = validate_sequence(raw_seq)

        # live length indicator
        seq_len = len(clean_seq)
        length_colour = "#34d399" if seq_len >= 10 else "#f87171"
        st.markdown(
            f'<span class="pill pill-gray">Length: '
            f'<span style="color:{length_colour};font-weight:700;">{seq_len}</span> aa</span>',
            unsafe_allow_html=True
        )

        if val_err and raw_seq.strip():
            st.warning(f"⚠️ {val_err}")

        predict_btn = st.button("▶ Predict Function", type="primary", disabled=bool(val_err) or not clean_seq)

    with col_stats:
        st.markdown('<p class="section-label">Sequence composition</p>', unsafe_allow_html=True)
        if clean_seq and not val_err:
            comp = aa_composition(clean_seq)
            # show top-10 residues
            top_aa = dict(sorted(comp.items(), key=lambda x: x[1], reverse=True)[:10])
            st.bar_chart(top_aa, height=200, use_container_width=True)
            # property pills
            charged   = sum(comp.get(aa, 0) for aa in "DEKRH")
            hydrophob = sum(comp.get(aa, 0) for aa in "ILMFVW")
            st.markdown(
                f'<span class="pill pill-blue">Charged {charged:.1f}%</span>'
                f'<span class="pill pill-green">Hydrophobic {hydrophob:.1f}%</span>',
                unsafe_allow_html=True
            )
        else:
            st.info("Sequence preview will appear here.")

    # ── Prediction results ──────────────────────────────
    if predict_btn and clean_seq and not val_err:
        with st.spinner("🔬 Embedding sequence and predicting GO terms…"):
            predictions = predict_sequence(
                tokenizer, esm_model, clf, go_terms,
                clean_seq, threshold=threshold, top_k=top_k
            )

        st.divider()

        if not predictions:
            st.info(f"No GO terms exceeded the {threshold:.0%} confidence threshold. "
                    "Try lowering the threshold in the sidebar.")
        else:
            n = len(predictions)
            top_go, top_p = predictions[0]
            top_label = GO_LABELS.get(top_go, ("Unknown",))[0]

            # Stat row
            st.markdown(
                f'<div class="stat-row">'
                f'  <div class="stat-box"><div class="stat-val">{n}</div>'
                f'      <div class="stat-key">GO terms predicted</div></div>'
                f'  <div class="stat-box"><div class="stat-val">{top_p:.0%}</div>'
                f'      <div class="stat-key">Top confidence</div></div>'
                f'  <div class="stat-box"><div class="stat-val">{seq_len}</div>'
                f'      <div class="stat-key">Sequence length</div></div>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(f'<p class="section-label">Predicted cellular locations</p>',
                        unsafe_allow_html=True)

            for rank, (go, p) in enumerate(predictions, 1):
                label, colour = GO_LABELS.get(go, ("Unknown", "#64748b"))
                pill_cls, pill_txt = confidence_pill(p)
                bar_w = int(p * 100)

                st.markdown(f"""
                <div class="pred-card">
                  <span class="pred-rank">#{rank}</span>
                  <span class="pred-go">{go}</span>
                  <span class="pred-label">
                    {label}
                    <span class="pill {pill_cls}" style="margin-left:0.4rem;">{pill_txt}</span>
                  </span>
                  <div class="pred-bar-wrap">
                    <div class="pred-bar" style="width:{bar_w}%;background:{colour};"></div>
                  </div>
                  <span class="pred-pct" style="color:{colour};">{p:.1%}</span>
                </div>
                """, unsafe_allow_html=True)

            # Download buttons
            st.markdown("---")
            st.markdown('<p class="section-label">Export results</p>', unsafe_allow_html=True)
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                st.download_button(
                    "⬇ Download CSV", data=results_to_csv(predictions, clean_seq),
                    file_name="protein_predictions.csv", mime="text/csv"
                )
            with dcol2:
                st.download_button(
                    "⬇ Download JSON", data=results_to_json(predictions, clean_seq),
                    file_name="protein_predictions.json", mime="application/json"
                )


# ────────────────────────────────────────────────────────
#  TAB 2 — Batch prediction
# ────────────────────────────────────────────────────────
with tab_batch:
    st.markdown("""
    <p class="section-label">Batch mode — one sequence per line</p>
    """, unsafe_allow_html=True)
    st.caption("Enter multiple sequences (one per line) or paste a multi-FASTA block. "
               "Headers starting with `>` are stripped automatically.")

    batch_input = st.text_area(
        "", height=220,
        placeholder=">seq1\nMKTYAILV...\n>seq2\nMVHLTPEEK...",
        label_visibility="collapsed"
    )
    batch_btn = st.button("▶ Run Batch Prediction", type="primary")

    if batch_btn and batch_input.strip():
        # Parse sequences
        raw_lines  = batch_input.strip().splitlines()
        sequences  = []
        current    = []
        label_list = []
        current_label = "seq1"

        for line in raw_lines:
            line = line.strip()
            if line.startswith(">"):
                if current:
                    sequences.append(("".join(current), current_label))
                    current = []
                current_label = line[1:].split()[0] or f"seq{len(sequences)+1}"
            elif line:
                current.append(line.upper())

        if current:
            sequences.append(("".join(current), current_label))

        # If no FASTA headers, treat each non-empty line as a sequence
        if not sequences:
            for i, line in enumerate(raw_lines):
                line = line.strip().upper()
                if line:
                    sequences.append((line, f"seq{i+1}"))

        if not sequences:
            st.warning("No valid sequences found.")
        else:
            st.info(f"Processing {len(sequences)} sequence(s)…")
            all_results = []
            prog = st.progress(0)

            for i, (seq, name) in enumerate(sequences):
                seq_clean, err = validate_sequence(seq)
                if err:
                    all_results.append({"name": name, "error": err, "predictions": []})
                else:
                    with st.spinner(f"Predicting {name}…"):
                        preds = predict_sequence(
                            tokenizer, esm_model, clf, go_terms,
                            seq_clean, threshold=threshold, top_k=top_k
                        )
                    all_results.append({"name": name, "error": None,
                                        "seq": seq_clean, "predictions": preds})
                prog.progress((i + 1) / len(sequences))

            prog.empty()

            # Display results table
            st.divider()
            st.markdown('<p class="section-label">Batch results</p>', unsafe_allow_html=True)

            for res in all_results:
                status = "error" if res['error'] else f"{len(res['predictions'])} GO terms"

                with st.expander(f"📌 {res['name']} ({status})"):
                    if res['error']:
                        st.error(res['error'])
                    elif not res['predictions']:
                        st.info("No GO terms above threshold.")
                    else:
                        for go, p in res['predictions']:
                            label, colour = GO_LABELS.get(go, ("Unknown", "#64748b"))
                            st.markdown(
                                f'<span class="pill pill-gray">{go}</span>'
                                f'<span style="color:{colour};font-weight:600;">{label}</span>'
                                f'<span style="color:#64748b;font-size:0.8rem;margin-left:0.5rem;">{p:.1%}</span>',
                                unsafe_allow_html=True
                            )

            # Bulk download
            all_rows = []
            for res in all_results:
                for go, p in res.get('predictions', []):
                    label = GO_LABELS.get(go, ("Unknown",))[0]
                    all_rows.append([res['name'], go, label, f"{p:.4f}"])

            if all_rows:
                buf = io.StringIO()
                w = csv.writer(buf)
                w.writerow(["Sequence_ID", "GO_Term", "GO_Label", "Confidence"])
                w.writerows(all_rows)
                st.download_button(
                    "⬇ Download All Results (CSV)", data=buf.getvalue(),
                    file_name="batch_predictions.csv", mime="text/csv"
                )


# ────────────────────────────────────────────────────────
#  TAB 3 — About
# ────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
    <p class="section-label">Project overview</p>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**ProteinScope** predicts Gene Ontology **Cellular Component** annotations from raw
amino-acid sequences using a two-stage pipeline:

1. **ESM-2 (8M)** — Facebook's protein language model converts each sequence into a
   480-dimensional contextual embedding via mean-pooling over the attention mask.

2. **Residual Neural Classifier** — A stack of residual blocks with BatchNorm, GELU
   activation and dropout maps the embedding to GO term probabilities via sigmoid output.

Training data is sourced from **UniProt SwissProt**, filtered to the four most
annotated cellular compartments. Multi-label binary cross-entropy loss is used.
        """)

    with c2:
        st.markdown("""
**Supported GO terms include:**

| GO Term | Location |
|---------|----------|
| GO:0005634 | Nucleus |
| GO:0005737 | Cytoplasm |
| GO:0005829 | Cytosol |
| GO:0016020 | Membrane |
| GO:0005886 | Plasma Membrane |
| GO:0005739 | Mitochondrion |
| GO:0005783 | Endoplasmic Reticulum |
| GO:0005768 | Endosome |

*Predictions are probabilistic — use a threshold of ≥0.5 for high-precision results.*
        """)

    st.divider()
    st.markdown("""
    <p class="section-label">References</p>

- Lin et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*.
- Ashburner et al. (2000). Gene Ontology: tool for the unification of biology. *Nature Genetics*.
- UniProt Consortium (2023). UniProt: the Universal Protein Database. *Nucleic Acids Research*.
    """, unsafe_allow_html=True)