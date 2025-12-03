# app.py - robust model loader + Streamlit UI
import os
import sys
import joblib
import numpy as np
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoConfig, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Indic Language Similarity Explorer")

# ---- config ----
MODEL_DIR = "./indic-bert"
LANG_CENTROIDS_FILE = "language_centroids.pkl"
KMEANS_FILE = "Kmeans_Cluster_Indic_Language_model.pkl"

# ---- helper to list model folder contents for debugging ----
def list_model_dir(path):
    try:
        files = os.listdir(path)
        return files
    except Exception as e:
        return f"Error listing {path}: {e}"

# ---- load supporting artifacts (fail early if not present) ----
try:
    language_centroids = joblib.load(LANG_CENTROIDS_FILE)
    KMeans = joblib.load(KMEANS_FILE)
except Exception as e:
    st.error(f"Failed to load joblib artifacts: {e}")
    st.stop()

device = torch.device("cpu")  # force cpu

# ---- Model loading with multiple fallbacks and clear error reporting ----
load_exceptions = []

st.write("Model folder contents:", list_model_dir(MODEL_DIR))

# Try tokenizer first (tokenizer errors are easier to diagnose)
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
except Exception as e:
    st.error("Tokenizer load failed. Check that tokenizer files (tokenizer.json, vocab, merges, etc.) exist in the folder.")
    st.exception(e)
    st.stop()

model = None

# Attempt 1: simple from_pretrained onto CPU (no device_map)
try:
    st.info("Attempting AutoModel.from_pretrained(..., local_files_only=True, torch_dtype=torch.float32)")
    model = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.float32)
    st.success("Model loaded with AutoModel.from_pretrained (simple CPU load).")
except Exception as e1:
    load_exceptions.append(("simple_load_failed", e1))
    st.warning("Simple from_pretrained failed. Trying CPU map_location fallback...")

    # Attempt 2: use AutoConfig + from_pretrained with map_location patch (more robust)
    try:
        config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)
        st.info("Loaded config.json — now trying AutoModel.from_config + manual state load if necessary.")

        # Try again with from_pretrained but with fewer kwargs
        model = AutoModel.from_pretrained(MODEL_DIR, config=config, local_files_only=True)
        st.success("Model loaded with AutoModel.from_pretrained using explicit config.")
    except Exception as e2:
        load_exceptions.append(("config_then_load_failed", e2))
        st.warning("AutoModel.from_pretrained with explicit config failed. Trying manual state_dict load...")

        # Attempt 3: manual state dict load (works for many broken cases)
        try:
            # instantiate model class from config
            config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)
            model = AutoModel.from_config(config)
            st.info("Instantiated model from config. Now attempting to load state dict manually (safetensors or pytorch).")

            # try safetensors first if available
            state_loaded = False
            safetensors_path = os.path.join(MODEL_DIR, "model.safetensors")
            pytorch_path = os.path.join(MODEL_DIR, "pytorch_model.bin")
            # prefer safetensors if present
            if os.path.exists(safetensors_path):
                try:
                    from safetensors.torch import load_file as safe_load
                    state = safe_load(safetensors_path, device="cpu")
                    model.load_state_dict(state, strict=False)
                    state_loaded = True
                    st.success("Loaded state from model.safetensors using safetensors.")
                except Exception as e3a:
                    load_exceptions.append(("safetensors_load_failed", e3a))
                    st.warning("safetensors present but failed to load via safetensors library.")
            if not state_loaded and os.path.exists(pytorch_path):
                try:
                    state = torch.load(pytorch_path, map_location="cpu")
                    # sometimes the object is a dict with 'state_dict'
                    if "state_dict" in state and isinstance(state["state_dict"], dict):
                        raw_state = state["state_dict"]
                    else:
                        raw_state = state
                    # Some keys might have 'module.' prefix from DataParallel — handle that
                    new_state = {}
                    for k, v in raw_state.items():
                        new_key = k.replace("module.", "") if k.startswith("module.") else k
                        new_state[new_key] = v
                    model.load_state_dict(new_state, strict=False)
                    state_loaded = True
                    st.success("Loaded state from pytorch_model.bin via torch.load.")
                except Exception as e3b:
                    load_exceptions.append(("pytorch_load_failed", e3b))
                    st.warning("pytorch_model.bin present but failed to load via torch.load.")
            if not state_loaded:
                raise RuntimeError("No supported weight files found or all weight-loading attempts failed.")
        except Exception as e3:
            load_exceptions.append(("manual_state_load_failed", e3))
            st.error("All model loading attempts failed. See exceptions below.")
            # Show all collected exceptions for debugging
            for tag, ex in load_exceptions:
                st.write(f"--- {tag} ---")
                st.exception(ex)
            st.stop()

# If we reached here and model loaded, ensure it's on CPU and eval
if model is None:
    st.error("Model variable is None despite passing load steps. Aborting.")
    st.stop()

model.to(device)
model.eval()
st.success("Model ready on CPU.")

# ---- embedding function ----
def embedding_text(text: str) -> np.ndarray:
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # fallback: some models use last_hidden_state, some use pooler_output
    if hasattr(outputs, "last_hidden_state"):
        emb = outputs.last_hidden_state.mean(dim=1)
    elif hasattr(outputs, "pooler_output"):
        emb = outputs.pooler_output
    else:
        # try first element
        first = outputs[0]
        emb = first.mean(dim=1)
    return emb.cpu().numpy()

# ---- Streamlit UI ----
st.title("Indic Language Similarity Explorer")

text = st.text_area("Enter a sentence in any Indian Language:")

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text")
    else:
        try:
            transformed_text = embedding_text(text)
        except Exception as e:
            st.error("Failed during embedding/inference:")
            st.exception(e)
            st.stop()

        distances = {}
        for lang, centroid in language_centroids.items():
            distances[lang] = cosine_similarity(transformed_text.reshape(1, -1), centroid.reshape(1, -1))[0][0]

        sorted_distance = sorted(distances.items(), key=lambda x: x[1], reverse=True)

        st.subheader("Top 3 similar languages")
        for lang, value in sorted_distance[:3]:
            st.write(f"{lang} : {value:.4f}")

        st.subheader("Language-Family Predicted By Model")
        cluster = KMeans.predict(transformed_text.reshape(1, -1))
        if cluster[0] == 1:
            st.write("Indo-Aryan Language")
        else:
            st.write("Dravidian Language")
