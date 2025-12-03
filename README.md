# 🇮🇳 Automatic Grouping of Indian Languages using Text Embeddings (Unsupervised NLP + IndicBERT + Clustering)

##  Project Overview

This project focuses on the **unsupervised clustering** of Indian languages based purely on the **semantic similarity** derived from sentence embeddings.

By employing the **ai4bharat/indic-bert** transformer model, sentences from eight Indian languages were transformed into high-dimensional vector embeddings. Clustering techniques were then applied to these vectors to automatically discover their natural linguistic groupings without relying on pre-existing labels.

The core objective is to analyze whether established **language families**—specifically **Indo-Aryan** and **Dravidian**—cluster naturally in the high-dimensional embedding space, demonstrating the hidden linguistic structure effectively captured by transformer models.

---

##  Languages and Dataset

The project includes eight major Indian languages, providing a balanced representation of the two main language families.

### Languages Included

| Language Family | Languages |
| :--- | :--- |
| **Indo-Aryan** | Hindi, Gujarati, Marathi, Punjabi |
| **Dravidian** | Tamil, Telugu, Kannada, Malayalam |

### Dataset

* **Source:** Kaggle – Indic Language Identification Dataset
* **Subset:** A balanced sample of **300 sentences per language** was used for training and analysis (Total: 2,400 samples).

---

##  Tech Stack

| Category | Tools |
| :--- | :--- |
| **Embedding Model** | ai4bharat / **Indic-BERT** |
| **Clustering** | **KMeans**, Agglomerative, DBSCAN |
| **Visualization** | t-SNE, PCA, Seaborn Heatmap |
| **Similarity Metric** | **Cosine Similarity** |
| **Deployment** | Streamlit App |
| **Storage** | Joblib (`.pkl` files) |

---

##  Methodology

### 1. Dataset Preparation

* The dataset was loaded and carefully balanced (300 samples/language).
* **Crucial Decision:** **No standard text cleaning** (e.g., no stopword removal, lemmatization) was performed. Transformer models are designed to process **raw, natural sentences** to capture full context.

### 2. Sentence Embeddings

Sentences were passed through the Indic-BERT model. The final layer's hidden states were processed using **Mean Pooling** across all tokens to generate a single, fixed-size vector representation for the entire sentence.
Mean pooling was chosen over CLS token pooling as it often yields a more robust representation of the overall sentence semantics.

### 3. Clustering Algorithms Evaluated

Multiple algorithms were tested to find the best balance between statistical metrics and linguistic interpretability.

| Algorithm | Silhouette Score | Interpretation |
| :--- | :--- | :--- |
| **KMeans** | **0.20** | Selected model. Provided the best visual separation and most linguistically meaningful clusters (Indo-Aryan vs. Dravidian). |
| Agglomerative | 0.16 | Clusters were too mixed and lacked clear linguistic boundaries. |
| DBSCAN | 0.41 | High score due to effective noise elimination, but the resulting clusters were semantically weak. |

### 4. Visualization & Key Findings

* **t-SNE 2D Projection:** The non-linear dimensionality reduction showed a clear separation:
    * **Indo-Aryan languages** (Hindi, Marathi, etc.) group tightly together.
    * **Dravidian languages** (Tamil, Telugu, etc.) form an independent, distinct cluster.
* **Centroid Cosine Similarity Heatmap:**
    * Similarity scores ranged meaningfully from **0.79 to 1.00**.
    * **Highest similarities** were observed *within* the same language family.
    * **Lowest similarity** was consistently found between languages from opposing families (e.g., Hindi vs. Tamil).

**Key Finding:**  The embedding space, derived without any supervision, naturally and correctly separates languages based on their established linguistic families, validating the structure captured by Indic-BERT.

---

##  Streamlit Application

A live web application is provided for real-time demonstration of the language clustering logic.

### Functionality

1.  A user inputs a sentence in any of the supported Indian languages.
2.  The application converts the input into an embedding using the Indic-BERT pipeline.
3.  It computes the **Cosine Similarity** between the input embedding and the pre-computed **centroid** (average embedding) for each of the eight languages.
4.  It outputs the **Top-3 Closest Languages**, effectively performing zero-shot language identification based on semantic space proximity.

### Inference Pipeline

$$\text{User Sentence} \xrightarrow{\text{Indic-BERT}} \text{Input Embedding } (E_{\text{in}}) \xrightarrow{\text{Cosine Similarity}} \text{Similarity}(E_{\text{in}}, C_{\text{lang}}) \rightarrow \text{Ranked Closest Languages}$$

###  Example Results

| Input Sentence | Top-3 Predicted Languages (Closest Centroids) |
| :--- | :--- |
| "હું ઘરે જઈ રહ્યો છું" (Gujarati) | **Gujarati**, Hindi, Marathi |
| "मैं घर जा रहा हूँ" (Hindi) | **Hindi**, Punjabi, Gujarati |
| "நான் வீட்டிற்கு செல்கிறேன்" (Tamil) | **Tamil**, Telugu, Malayalam |

## 🧾 Installation

To run the Streamlit application locally:

1.  **Clone the repository:**
    ```bash
    git clone [Your GitHub Repo URL]
    cd Indic-Language-Clustering
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

---

##  Future Work

* **Visualization Integration:** Add the **t-SNE visualization** directly into the Streamlit application for a more compelling user experience.
* **Language Extension:** Expand the analysis to include **15–20 Indian languages** for broader linguistic coverage.
* **Model Comparison:** Experiment with other multilingual transformers, such as **LaBSE** or **MuRIL**, and benchmark their performance against IndicBERT in this clustering task.
* **Zero-Shot Benchmark:** Establish a formal **zero-shot language identification** benchmark based on centroid proximity.

---

##  Conclusion

This project successfully demonstrates the power of deep transformer embeddings in capturing intrinsic linguistic structures. The unsupervised clustering approach, driven by Indic-BERT embeddings, effectively and correctly separated Indian languages into their established **Indo-Aryan** and **Dravidian** families based purely on semantic similarity.

##  Contact

**Author:** Rohit Vastani
* **Email:** rohitvastani8626@gmail.com
* **GitHub:** [https://github.com/Rohit-8626](https://github.com/Rohit-8626)
