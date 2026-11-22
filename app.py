import os
import pickle as pkl

import numpy as np
import streamlit as st
import keras
from numpy.linalg import norm
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalMaxPool2D

# Hugging Face model deposunuzun adı (kullanici-adiniz/depo-adi). Ortam değişkeniyle de verilebilir.
HF_REPO = os.environ.get("HF_REPO", "tufanyldrm1989/fashion-recommender-model")


def get_file(name):
    """Dosya yerelde varsa onu, yoksa Hugging Face model deposundan indirileni döndürür."""
    if os.path.exists(name):
        return name
    if HF_REPO.startswith("kullanici-adiniz"):
        return name
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=HF_REPO, filename=name)


st.set_page_config(page_title="Fashion Recommender", page_icon="👗", layout="wide")
st.title("👗 Fashion Recommender System")
st.caption("Upload a fashion product photo and get visually similar items (ResNet50 + nearest neighbours).")


@st.cache_resource
def load_model():
    try:
        return keras.models.load_model(get_file("resnet50_feature_extractor.keras"))
    except Exception:
        pass  # saved with a different Keras version (e.g. Colab) -> rebuild below
    base = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    return keras.models.Sequential([base, GlobalMaxPool2D()])


@st.cache_resource
def load_index():
    feats = np.array(pkl.load(open(get_file("Images_features.pkl"), "rb")))
    names = pkl.load(open(get_file("filenames.pkl"), "rb"))
    # Sadece images/ klasöründe gerçekten bulunan görselleri kullan
    n = min(len(feats), len(names))
    keep = [i for i in range(n) if os.path.exists(names[i])]
    if not keep:
        st.error("images/ klasöründe hiçbir ürün görseli bulunamadı. images.zip'i app.py'nin yanına açın.")
        st.stop()
    feats = feats[keep]
    names = [names[i] for i in keep]
    nn = NearestNeighbors(n_neighbors=6, algorithm="brute", metric="euclidean").fit(feats)
    return nn, names


def extract(img: Image.Image, model):
    img = img.convert("RGB").resize((224, 224), Image.NEAREST)  # same as keras load_img default
    arr = preprocess_input(np.expand_dims(np.array(img, dtype="float32"), 0))
    out = model.predict(arr, verbose=0).flatten()
    return out / norm(out)


model = load_model()
nn, filenames = load_index()

n_rec = st.slider("Number of recommendations", 1, 5, 5)
upload = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"])

if upload:
    query = Image.open(upload)
    st.subheader("Your image")
    st.image(query, width=200)

    _, idx = nn.kneighbors([extract(query, model)])
    st.subheader("Recommended for you")
    cols = st.columns(n_rec)
    for col, i in zip(cols, idx[0][:n_rec]):
        path = filenames[i]
        if os.path.exists(path):
            col.image(path, use_container_width=True)
        else:
            col.write(path)
else:
    st.info("Upload an image to get started.")
