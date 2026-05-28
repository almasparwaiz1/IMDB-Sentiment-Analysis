import streamlit as st
import tensorflow as tf
from tensorflow import keras
import pickle
import json
import re
import os
import zipfile
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Path Configurations ---
BASE_DIR = r"Streamlit_Frontend"

MODEL_PATH = os.path.join(BASE_DIR, 'best_tuned_sentiment_model.keras')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'tokenizer.pickle')
NLP_PARAMS_PATH = os.path.join(BASE_DIR, 'nlp_parameters') 

# --- Custom Professional UI Theme Injection ---
st.markdown("""
    <style>
        /* Sidebar Styling (Deep Navy Blue) */
        [data-testid="stSidebar"] {
            background-color: #0F172A;
            color: #FFFFFF;
        }
        [data-testid="stSidebar"] * {
            color: #FFFFFF !important;
        }
        
        /* Main App Background (Clean White) & Text (True Black) */
        .stApp {
            background-color: #FFFFFF;
            color: #000000;
        }
        
        /* Modern Content Cards (Crisp Slate Gray) */
        div.element-container:has(div.custom-card), 
        .stMetric, .stAlert, div[data-testid="stBlock"] > div {
            background-color: #F1F5F9;
            border-radius: 8px;
            padding: 1.2rem;
            margin-bottom: 1rem;
            border: 1px solid #E2E8F0;
        }
        
        /* Accent Elements & Buttons (Vibrant Cobalt Blue) */
        .stButton>button {
            background-color: #2563EB !important;
            color: #FFFFFF !important;
            border-radius: 6px !important;
            border: none !important;
            font-weight: 600 !important;
            width: 100%;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #1D4ED8 !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
        }
        
        /* Headers and Typography */
        h1, h2, h3 {
            color: #0F172A !important;
            font-weight: 700 !important;
        }
        
        /* Status Badges Overrides for Clean Look */
        .stAlert {
            border-left: 5px solid #2563EB !important;
            color: #000000 !important;
        }
        
        /* Adjust main content block padding */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- NLTK Downloads (Cached for performance) ---
@st.cache_resource
def download_nltk_data():
    """Downloads necessary NLTK data if not already present."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

download_nltk_data()

# --- Model and Resources Loading (Cached for performance) ---
@st.cache_resource
def load_ml_resources(model_path, tokenizer_path, nlp_params_path):
    """Loads the ML model, tokenizer, and NLP parameters from disk."""
    try:
        # Step 1: Open the zipped .keras file and extract config data
        with zipfile.ZipFile(model_path, 'r') as archive:
            config_bytes = archive.read('config.json')
            config_str = config_bytes.decode('utf-8')
            
            # Step 2: Remove the quantization_config parameters causing parsing issues
            config_str = re.sub(r'"quantization_config":\s*null,?', '', config_str)
            
            # Clean up trailing commas that could break JSON parsing if they remain
            config_str = re.sub(r',\s*}', '}', config_str)
            config_str = re.sub(r',\s*\]', ']', config_str)
            
            model_config = json.loads(config_str)

        # Step 3: Rebuild the architecture using the universal deserializer layer mapping
        model = keras.layers.deserialize(model_config)
        
        # Step 4: Populate the built skeleton weights from the saved archive artifact file
        model.load_weights(model_path)
        
        # Step 5: Safely load Tokenizer
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
            
        # Step 6: Smart Parameter File Resolution (Tries file, file.json, then falls back gracefully)
        nlp_params = {}
        resolved_params_path = nlp_params_path
        
        if not os.path.exists(resolved_params_path) and os.path.exists(nlp_params_path + ".json"):
            resolved_params_path = nlp_params_path + ".json"
            
        if os.path.exists(resolved_params_path):
            with open(resolved_params_path, 'r') as fp:
                nlp_params = json.load(fp)
        else:
            st.warning("⚠️ `nlp_parameters` configuration file was not found. Using default architecture metrics (MAX_SEQUENCE_LENGTH = 120).")
            nlp_params = {'MAX_SEQUENCE_LENGTH': 120}
            
        return model, tokenizer, nlp_params
        
    except FileNotFoundError as e:
        st.error(f"**Critical Error:** Required foundational file not found: `{e.filename}`. Please verify your working directory path contents.")
        raise RuntimeError(f"Resource missing: {e.filename}")
    except Exception as e:
        st.error(f"An unexpected error occurred while loading ML resources: {e}")
        raise e

# Try to safely extract resources; if it fails, the script execution halts gracefully
try:
    model, tokenizer, nlp_params = load_ml_resources(MODEL_PATH, TOKENIZER_PATH, NLP_PARAMS_PATH)
    MAX_SEQUENCE_LENGTH = nlp_params.get('MAX_SEQUENCE_LENGTH', 120)
except Exception:
    st.stop()

# --- Text Preprocessing Pipeline ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text_pipeline(text):
    """Applies a series of cleaning steps to a single text string."""
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)  # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text) # Keep only alphabetic characters and spaces
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned_tokens)

def preprocess_for_model(text, tokenizer_obj, max_seq_len):
    """Tokenizes and pads text for input to the ML model."""
    cleaned_text = clean_text_pipeline(text)
    text_sequence = tokenizer_obj.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(text_sequence, maxlen=max_seq_len, padding='post', truncating='post')
    return padded_sequence

# --- Prediction Core Engine ---
def predict_sentiment(text_input, ml_model, tokenizer_obj, max_seq_len):
    """Makes a sentiment prediction using the loaded model and preprocessed text."""
    processed_input = preprocess_for_model(text_input, tokenizer_obj, max_seq_len)
    prediction_prob = ml_model.predict(processed_input, verbose=0)[0][0]
    sentiment = "Positive" if prediction_prob > 0.5 else "Negative"
    return sentiment, prediction_prob


# --- Main Interface Application Engine ---
st.title("🎬 Movie Review Sentiment Architecture")
st.markdown("Submit raw text elements below to evaluate real-time neural sentiment predictions across classification layers.")

# UI Input Element
user_input = st.text_area(
    "Target Text Review Sequence:",
    value="",
    height=250,
    max_chars=5000,
    placeholder="Enter application testing string here (e.g., 'This movie was absolutely fantastic! A must-watch for everyone.')",
    key="review_text_area"
)

results_container = st.empty()

# Control Matrix Buttons
col1, col2 = st.columns(2)
with col1:
    analyze_button = st.button("Execute Vectorized Analysis", use_container_width=True, type="primary")
with col2:
    clear_button = st.button("Clear Buffer State", use_container_width=True)

# Application Logical Evaluation
if analyze_button:
    if not user_input.strip():
        st.warning("Analysis buffer cannot execute on empty string inputs.")
    else:
        with results_container.container():
            with st.spinner("Processing structural sequence arrays..."):
                sentiment, probability = predict_sentiment(user_input, model, tokenizer, MAX_SEQUENCE_LENGTH)

                st.subheader("📈 Model Metrics Matrix")
                
                # Render clean structural outputs
                if sentiment == "Positive":
                    st.success(f"**Evaluated Sequence Classification:** Positive 😊")
                    st.progress(float(probability))
                    st.markdown(f"Layer Confidence Level: **{probability*100:.2f}%**")
                else:
                    st.error(f"**Evaluated Sequence Classification:** Negative 😞")
                    st.progress(float(1 - probability))
                    st.markdown(f"Layer Confidence Level: **{(1-probability)*100:.2f}%**")

if clear_button:
    st.rerun()
