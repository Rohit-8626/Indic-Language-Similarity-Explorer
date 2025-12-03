import joblib
import numpy as np
from transformers import AutoTokenizer , AutoModel
import torch
import os
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

language_centroids = joblib.load('language_centroids.pkl')
KMeans = joblib.load("Kmeans_Cluster_Indic_Language_model.pkl")

device = torch.device("cpu")
model_name = 'ai4bharat/indic-bert'

@st.cache_resource
def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)
tokenizer = load_tokenizer(model_name)

@st.cache_resource
def load_model(model_name):
    return AutoModel.from_pretrained(model_name)
model = load_model(model_name)
model.to(device)
model.eval()


def embedding_text(text : str) -> np.ndarray:
    # convert the text into the tokens
    inputs = tokenizer(text , padding = True , truncation = True , return_tensors = 'pt')

    # give tokens to the model
    with torch.no_grad():
        outputs = model(**inputs)

    # using mean pooling for embedding for get whole sentence meaning
    embedding  = outputs.last_hidden_state.mean(dim = 1)
    return embedding.cpu().numpy()

st.title("Indic Language Similarity Explorer")

text = st.text_area("Enter a sentence in any Indian Language : ")

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text")
    else:

        transformed_text = embedding_text(text)
        distances = {}

        for lang , centroid in language_centroids.items():
            distances[lang] = cosine_similarity(transformed_text.reshape(1 , -1) , centroid.reshape(1 , -1))[0][0]

        sorted_distance = sorted(distances.items() , key = lambda x: x[1] , reverse=True)

        st.subheader("Top 3 similar languages")
        for lang , value in sorted_distance[:3]:
            st.write(f"{lang} : {value : .4f}")

        st.subheader("Language-Family Predicted By Model")
        cluster = KMeans.predict(transformed_text.reshape(1 , -1))

        if cluster == 1:
            st.write("Indo-Aryan Lnguage")
        else:

            st.write("Dravidian Language")


