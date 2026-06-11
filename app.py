import streamlit as st

# ── MUST be first Streamlit call ──────────────────────────────────────────────
st.set_page_config(
    page_title="PlantDoc · Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

import numpy as np
import os
from PIL import Image

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header   { visibility: hidden; }
.stApp                       { background: #0a0f0a; color: #d4e8d0; }

.hero {
    background: linear-gradient(135deg, #0d1f0d 0%, #0a150a 50%, #071207 100%);
    border: 1px solid #1e3a1e; border-radius: 20px;
    padding: 2.8rem 3rem 2.4rem; margin-bottom: 2rem;
    position: relative; overflow: hidden;
}
.hero::after {
    content: "🌿"; position: absolute; right: 2.5rem; top: 50%;
    transform: translateY(-50%); font-size: 6rem; opacity: 0.12; pointer-events: none;
}
.hero-tag   { font-size:.68rem; font-weight:700; letter-spacing:.16em;
              text-transform:uppercase; color:#4ade80; margin-bottom:.5rem; }
.hero-title { font-size:2.2rem; font-weight:700; color:#ecfdf5; line-height:1.15; margin:0 0 .6rem; }
.hero-sub   { font-size:.9rem; color:#6b9b6b; margin:0; max-width:480px; }

.slabel { font-size:.68rem; font-weight:700; letter-spacing:.16em;
          text-transform:uppercase; color:#4ade80;
          margin:1.8rem 0 .8rem; padding-bottom:.4rem; border-bottom:1px solid #1a2e1a; }

div[data-testid="stFileUploader"] {
    background: #0d1f0d !important; border: 2px dashed #2d5a2d !important;
    border-radius: 14px !important; padding: 1.5rem !important; }
div[data-testid="stFileUploader"] label { color:#6b9b6b !important; }

div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg,#16a34a,#15803d) !important;
    border: none !important; border-radius: 12px !important;
    color: #fff !important; font-weight: 700 !important;
    font-size: 1rem !important; padding: .8rem 0 !important;
    box-shadow: 0 4px 20px rgba(22,163,74,.35) !important; width: 100% !important; }

.res-wrap {
    background: linear-gradient(135deg,#0d2e12,#071207);
    border: 1px solid #16a34a; border-radius: 16px; padding: 2rem 2.2rem; margin-top: 1.4rem; }
.res-tag  { font-size:.68rem; font-weight:700; letter-spacing:.16em;
            text-transform:uppercase; color:#4ade80; margin-bottom:.4rem; }
.res-name { font-family:'DM Mono',monospace; font-size:1.55rem;
            font-weight:500; color:#bbf7d0; line-height:1.3; word-break:break-word; }
.res-conf { font-size:.85rem; color:#4b7a4b; margin-top:.5rem; }

.bar-track { background:#0f2b0f; border-radius:99px; height:8px; margin:.6rem 0 1.4rem; overflow:hidden; }
.bar-fill  { height:8px; border-radius:99px;
             background: linear-gradient(90deg,#16a34a,#4ade80); }

.info-box   { background: #0d1f0d; border: 1px solid #1e3a1e;
              border-radius: 12px; padding: 1.2rem 1.4rem; margin-top:.8rem; }
.info-box p { margin:0; font-size:.88rem; color:#6b9b6b; line-height:1.6; }

.badge-healthy { display:inline-block; background:#14532d; color:#4ade80;
    font-size:.75rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase;
    padding:.3rem .8rem; border-radius:99px; margin-top:.6rem; }
.badge-disease { display:inline-block; background:#450a0a; color:#fca5a5;
    font-size:.75rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase;
    padding:.3rem .8rem; border-radius:99px; margin-top:.6rem; }

.top5    { width:100%; border-collapse:collapse; margin-top:.4rem; }
.top5 th { font-size:.68rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase;
           color:#4ade80; padding:.4rem .6rem; text-align:left; border-bottom:1px solid #1e3a1e; }
.top5 td { font-size:.84rem; color:#a3c9a3; padding:.45rem .6rem;
           border-bottom:1px solid #111d11; font-family:'DM Mono',monospace; }
.top5 tr:last-child td { border-bottom:none; }
.top5 td.pct { color:#4ade80; text-align:right; }

div[data-testid="stAlert"] { border-radius:10px !important; font-size:.88rem !important; }
hr { border-color:#1a2e1a !important; margin:1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── PlantVillage 38-class list ────────────────────────────────────────────────
PLANT_CLASSES = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
    'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy','Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy',
    'Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy','Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot',
    'Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy',
    'Potato___Early_blight','Potato___Late_blight','Potato___healthy','Raspberry___healthy',
    'Soybean___healthy','Squash___Powdery_mildew','Strawberry___Leaf_scorch',
    'Strawberry___healthy','Tomato___Bacterial_spot','Tomato___Early_blight',
    'Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy',
]

DISEASE_INFO = {
    "Apple_scab":        "Fungal disease — dark scabby lesions on leaves and fruit. Use fungicide and remove fallen leaves.",
    "Black_rot":         "Causes circular brown lesions. Remove infected tissue; apply copper-based fungicide.",
    "Cedar_apple_rust":  "Orange pustules on leaves. Remove nearby cedar trees; apply preventive fungicide.",
    "Powdery_mildew":    "White powdery coating on leaves. Improve air circulation; apply sulfur-based fungicide.",
    "Cercospora":        "Gray leaf spot on corn. Rotate crops and use resistant hybrids.",
    "Common_rust":       "Orange pustules on corn leaves. Plant resistant varieties; apply fungicide early.",
    "Northern_Leaf_Blight":"Long tan lesions on corn. Use resistant varieties and rotate crops.",
    "Esca":              "Internal wood disease in grapevines. Prune infected wood and protect wounds.",
    "Leaf_blight":       "Brown marginal necrosis. Remove infected leaves; apply copper fungicide.",
    "Haunglongbing":     "Incurable citrus greening spread by psyllids. Remove and destroy infected trees.",
    "Bacterial_spot":    "Water-soaked spots turning brown. Use copper bactericide and disease-free seeds.",
    "Early_blight":      "Concentric ring lesions. Rotate crops; apply chlorothalonil fungicide.",
    "Late_blight":       "Lesions turn brown rapidly. Apply fungicide immediately — highly contagious.",
    "Leaf_Mold":         "Olive-green mold on undersides. Increase ventilation and reduce humidity.",
    "Septoria_leaf_spot":"Small circular spots with dark borders. Remove leaves; avoid overhead watering.",
    "Spider_mites":      "Stippling and webbing on leaves. Apply miticide or neem oil spray.",
    "Target_Spot":       "Concentric ring lesions on tomato. Apply fungicide; practice crop rotation.",
    "Yellow_Leaf_Curl":  "Virus spread by whiteflies. Use reflective mulches; control whitefly vectors.",
    "mosaic_virus":      "Mosaic discoloration. Remove infected plants immediately — no cure available.",
    "Leaf_scorch":       "Brown leaf edges on strawberry. Remove infected leaves and improve drainage.",
}

def get_disease_info(cls):
    for key, info in DISEASE_INFO.items():
        if key.lower() in cls.lower():
            return info
    return "Consult your local agricultural extension office for treatment recommendations."

def fmt_class(raw):
    parts   = raw.split("___")
    plant   = parts[0].replace("_", " ")
    disease = parts[1].replace("_", " ") if len(parts) > 1 else ""
    return plant, disease

def is_healthy(cls):
    return "healthy" in cls.lower()


# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_PATHS = [
    "best_plant_model.keras", "final_plant_disease_model.keras",
    "plant_model.keras", "model.keras",
    "best_plant_model.h5", "plant_disease_model.h5",
]

@st.cache_resource(show_spinner="Loading model…")
def load_keras_model():
    try:
        import tensorflow as tf
    except ImportError:
        return None, "TensorFlow not installed. Add `tensorflow` to requirements.txt."
    for path in MODEL_PATHS:
        if os.path.exists(path):
            try:
                return tf.keras.models.load_model(path), None
            except Exception as e:
                return None, f"Found `{path}` but failed to load: `{e}`"
    return None, (
        "No model file found. Expected one of:\n\n"
        + "\n".join(f"- `{p}`" for p in MODEL_PATHS)
        + "\n\nCommit your trained `.keras` or `.h5` file to the repo root."
    )


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">🌿 AI Plant Pathology</div>
  <div class="hero-title">Plant Disease Detector</div>
  <p class="hero-sub">Upload a leaf photo — the model identifies the disease across
  38 PlantVillage categories instantly.</p>
</div>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
model, load_error = load_keras_model()

if load_error:
    st.error("⚠️ Model could not be loaded", icon="🚫")
    st.markdown(load_error)
    with st.expander("📋 Setup guide"):
        st.markdown("""
**Download your model from Kaggle after training:**
```python
from IPython.display import FileLink
FileLink('/kaggle/working/best_plant_model.keras')
```

**Commit it to your GitHub repo root alongside app.py:**
```
your-repo/
├── app.py
├── best_plant_model.keras   ← here
└── requirements.txt
```

**requirements.txt:**
```
streamlit>=1.35.0
tensorflow>=2.13.0
numpy>=1.24.0
Pillow>=9.0.0
```
""")
    st.stop()

st.success(f"✅ Model ready · {len(PLANT_CLASSES)} disease classes", icon="🟢")
st.markdown("---")


# ── Two-column layout ─────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="slabel">📷 Upload Leaf Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drag & drop or click to browse",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        st.image(img_pil, caption="Uploaded leaf", use_container_width=True)
        w, h = img_pil.size
        st.caption(f"{w} × {h} px · {uploaded.type}")
        st.markdown("---")
        _, btn_col, _ = st.columns([0.3, 2, 0.3])
        with btn_col:
            predict_btn = st.button("🔬  Analyse Leaf", type="primary")
    else:
        st.markdown("""
<div style="background:#0d1f0d;border:2px dashed #2d5a2d;border-radius:14px;
padding:3rem;text-align:center;color:#4b7a4b;font-size:.9rem;">
  📂 &nbsp; No image uploaded yet<br>
  <span style="font-size:.8rem;color:#2d5a2d">Supports JPG · PNG · WEBP</span>
</div>""", unsafe_allow_html=True)
        predict_btn = False

with right:
    st.markdown('<div class="slabel">🧬 Diagnosis</div>', unsafe_allow_html=True)

    if not uploaded:
        st.markdown("""
<div style="background:#0d1f0d;border:1px solid #1e3a1e;border-radius:14px;
padding:3rem;text-align:center;color:#4b7a4b;font-size:.9rem;">
  Upload a leaf and click <strong style="color:#4ade80">Analyse Leaf</strong>
  to see the diagnosis here.
</div>""", unsafe_allow_html=True)

    elif predict_btn:
        with st.spinner("Analysing leaf…"):
            try:
                import tensorflow as tf

                img_resized = img_pil.resize((224, 224))
                img_array   = np.array(img_resized, dtype=np.float32)
                img_array   = np.expand_dims(img_array, axis=0)
                # The Rescaling(1./255) layer is INSIDE the model (as trained).
                # If your model does NOT have that layer, uncomment the next line:
                # img_array /= 255.0

                preds     = model.predict(img_array, verbose=0)[0]
                top5_idx  = np.argsort(preds)[::-1][:5]
                top1_idx  = top5_idx[0]
                top1_conf = float(preds[top1_idx]) * 100

                n_cls      = len(preds)
                cls_list   = PLANT_CLASSES[:n_cls] if n_cls <= len(PLANT_CLASSES) else PLANT_CLASSES
                top1_cls   = cls_list[top1_idx] if top1_idx < len(cls_list) else f"Class_{top1_idx}"
                plant, disease = fmt_class(top1_cls)
                healthy    = is_healthy(top1_cls)
                badge      = ('<span class="badge-healthy">✅ Healthy</span>'
                              if healthy else
                              '<span class="badge-disease">⚠️ Disease Detected</span>')

                st.markdown(f"""
<div class="res-wrap">
  <div class="res-tag">Top Prediction</div>
  <div class="res-name">{plant}<br>
    <span style="color:#4ade80">{disease}</span>
  </div>
  {badge}
  <div class="bar-track" style="margin-top:.9rem">
    <div class="bar-fill" style="width:{min(top1_conf,100):.1f}%"></div>
  </div>
  <div class="res-conf">Confidence: <strong style="color:#4ade80">{top1_conf:.1f}%</strong></div>
</div>""", unsafe_allow_html=True)

                if not healthy:
                    st.markdown(f"""
<div class="info-box">
  <p><strong style="color:#d4e8d0">💡 Treatment tip:</strong><br>
  {get_disease_info(top1_cls)}</p>
</div>""", unsafe_allow_html=True)

                # Top-5 table
                st.markdown('<div class="slabel">Top 5 Predictions</div>', unsafe_allow_html=True)
                rows = ""
                for rank, idx in enumerate(top5_idx, 1):
                    name = (cls_list[idx].replace("___", " · ").replace("_", " ")
                            if idx < len(cls_list) else f"Class {idx}")
                    conf = float(preds[idx]) * 100
                    rows += f"<tr><td>#{rank}</td><td>{name}</td><td class='pct'>{conf:.1f}%</td></tr>"

                st.markdown(f"""
<table class="top5">
  <thead><tr><th>#</th><th>Class</th><th style="text-align:right">Conf.</th></tr></thead>
  <tbody>{rows}</tbody>
</table>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: `{e}`", icon="❌")
                st.info("Tip: if you get a shape error, make sure your model output "
                        "matches the number of classes in PLANT_CLASSES.")

    else:
        st.markdown("""
<div style="background:#0d1f0d;border:1px solid #1e3a1e;border-radius:14px;
padding:3rem;text-align:center;color:#4b7a4b;font-size:.9rem;">
  Image ready — click <strong style="color:#4ade80">Analyse Leaf</strong> to run the model.
</div>""", unsafe_allow_html=True)

st.markdown("---")
st.caption("PlantDoc · EfficientNetB0 on PlantVillage · 38 disease classes")
