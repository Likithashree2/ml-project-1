# ============================================================================
# GENETIC DISEASE PREDICTION - STREAMLIT FRONTEND APPLICATION
# ============================================================================
# This is the main Streamlit web application for genetic disease prediction
# using machine learning models trained on DNA sequence data.
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Genetic Disease Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD MODELS AND METADATA
# ============================================================================
@st.cache_resource
def load_model():
    """Load the trained ML model and metadata"""
    try:
        model_path = "models/best_model.pkl"
        metadata_path = "models/model_metadata.pkl"

        if not os.path.exists(metadata_path):
            return None, None

        metadata = joblib.load(metadata_path)
        model = None

        if os.path.exists(model_path):
            model = joblib.load(model_path)
        elif 'model' in metadata:
            model = metadata['model']

        if model is None:
            return None, None

        return model, metadata
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Load model and metadata
model, metadata = load_model()

# ============================================================================
# SIDEBAR - PROJECT INFORMATION
# ============================================================================
with st.sidebar:
    st.image("https://via.placeholder.com/300x150/4CAF50/FFFFFF?text=Genetic+Disease+Prediction", 
             use_column_width=True)
    
    st.title("🧬 About This Project")
    
    st.markdown("""
    ### Genetic Disease Prediction using Machine Learning
    
    This application uses advanced machine learning techniques to predict 
    genetic diseases based on DNA sequence data.
    
    **Technology Stack:**
    - Python, scikit-learn, Streamlit
    - K-mer feature extraction
    - PCA dimensionality reduction
    - Multiple ML algorithms
    
    **Dataset:**
    - gene_sequence_dataset.csv
    - 2000 DNA sequences
    - Multiple disease classes
    """)
    
    st.divider()
    
    if metadata:
        st.subheader("📊 Model Information")
        st.metric("Best Model", metadata['model_name'])
        st.metric("Accuracy", f"{metadata['accuracy']:.4f}")
    
    st.divider()
    
    st.subheader("📝 Valid DNA Input")
    st.info("""
    **Allowed Characters:**
    - **A** - Adenine
    - **T** - Thymine
    - **G** - Guanine
    - **C** - Cytosine
    
    Example: `ATGCTAGCTA`
    """)

# ============================================================================
# MAIN CONTENT - TITLE AND DESCRIPTION
# ============================================================================
st.title("🧬 Genetic Disease Prediction System")
st.markdown("---")

# Create two columns for title and description
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### Welcome to the Genetic Disease Prediction Application
    
    This application leverages machine learning to predict genetic diseases 
    from DNA sequences. Simply input your DNA sequence and let our AI model 
    analyze it for disease predictions.
    """)

with col2:
    st.info("""
    **Quick Start:**
    1. Enter a DNA sequence
    2. Click "Predict Disease"
    3. View results
    """)

st.markdown("---")

# ============================================================================
# PREPROCESSING PIPELINE LOADING
# ============================================================================
@st.cache_resource
def load_preprocessor_components():
    """Load the preprocessing artifacts used during training."""
    try:
        # First, try to load from separate .pkl files
        vectorizer_path = "models/vectorizer.pkl"
        scaler_path = "models/scaler.pkl"
        pca_path = "models/pca.pkl"

        if (os.path.exists(vectorizer_path) and
            os.path.exists(scaler_path) and
            os.path.exists(pca_path)):
            vectorizer = joblib.load(vectorizer_path)
            scaler = joblib.load(scaler_path)
            pca = joblib.load(pca_path)
            return vectorizer, scaler, pca

        # Fallback to metadata if separate files not available
        if metadata is not None and all(key in metadata for key in ('vectorizer', 'scaler', 'pca')):
            return metadata['vectorizer'], metadata['scaler'], metadata['pca']

        # Last fallback: reconstruct preprocessing from the training dataset
        dataset_path = "gene_sequence_dataset.csv"
        if not os.path.exists(dataset_path):
            return None, None, None

        df = pd.read_csv(dataset_path, nrows=2000)

        # Auto-detect DNA sequence column
        seq_col = None
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_text = str(df[col].iloc[0]).upper()
                if all(c in 'ACGTN' for c in sample_text if c.isalpha()):
                    seq_col = col
                    break

        if seq_col is None:
            return None, None, None

        sequences = df[seq_col].astype(str).str.upper().str.replace(r'[^ATGC]', '', regex=True)
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))
        X_features = vectorizer.fit_transform(sequences)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_features.toarray())

        pca = PCA(n_components=0.95)
        pca.fit(X_scaled)

        return vectorizer, scaler, pca
    except Exception as e:
        st.error(f"Error loading preprocessing pipeline: {str(e)}")
        return None, None, None

# Load preprocessing components
vectorizer, scaler, pca = load_preprocessor_components()

# CHECK IF MODEL AND PREPROCESSING ARE LOADED
if model is None or metadata is None:
    st.error("❌ Error: Model files not found. Please ensure models/best_model.pkl and models/model_metadata.pkl exist.")
    st.stop()

if vectorizer is None or scaler is None or pca is None:
    st.error(
        "❌ Error: Preprocessing artifacts are unavailable. "
        "Please retrain the model or restore the training artifacts."
    )
    st.stop()

# ============================================================================
# DNA SEQUENCE INPUT SECTION
# ============================================================================
st.header("📋 Enter DNA Sequence")

dna_input = st.text_area(
    label="Paste your full DNA sequence below",
    placeholder="Example: GAGGAGTTTGGCAACGAGACGTGGGGCGTGCATGCCTCTGGCAGGTCAAGTCCTGACACCAATGACAAGGACGAGATGGATGACGACAAAGAGAAGATGAGGAGGATGATGAAGAAGAAGATAAGGAG",
    height=180,
    help="Enter a full DNA sequence containing only A, T, G, C characters. Whitespace and line breaks are ignored."
)

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================
def validate_dna_sequence(sequence):
    """
    Validate DNA sequence contains only valid characters
    
    Parameters:
    sequence (str): DNA sequence to validate
    
    Returns:
    tuple: (is_valid, cleaned_sequence, error_message)
    """
    if not sequence:
        return False, None, "Please enter a DNA sequence"
    
    cleaned = ''.join(sequence.split()).upper()
    
    if not cleaned:
        return False, None, "DNA sequence cannot be empty"
    
    # Check for invalid characters
    valid_chars = set('ATGC')
    invalid_chars = sorted(set(cleaned) - valid_chars)

    if invalid_chars:
        return False, None, f"Invalid characters found: {', '.join(invalid_chars)}. Only A, T, G, C are allowed."

    return True, cleaned, None


def preprocess_sequence(sequence, vectorizer, scaler, pca):
    """
    Preprocess a single DNA sequence for prediction using the training pipeline
    
    Parameters:
    sequence (str): DNA sequence
    vectorizer: fitted CountVectorizer
    scaler: fitted StandardScaler
    pca: fitted PCA
    
    Returns:
    ndarray: Preprocessed features
    """
    try:
        features = vectorizer.transform([sequence]).toarray()
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)
        return features_pca
    except Exception as e:
        st.error(f"Error preprocessing sequence: {str(e)}")
        return None

def predict_disease(sequence):
    """
    Make disease prediction for a given DNA sequence
    
    Parameters:
    sequence (str): DNA sequence
    
    Returns:
    tuple: (predicted_class, confidence, probabilities)
    """
    try:
        # Validate input
        is_valid, cleaned_seq, error_msg = validate_dna_sequence(sequence)
        
        if not is_valid:
            return None, None, error_msg
        
        # Preprocess sequence
        features = preprocess_sequence(cleaned_seq, vectorizer, scaler, pca)
        
        if features is None:
            return None, None, "Error during sequence preprocessing"
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get confidence (probability for the predicted class if available)
        try:
            probabilities = model.predict_proba(features)[0]
            confidence = np.max(probabilities)
        except:
            confidence = None
        
        # Decode prediction to disease name
        label_encoder = metadata['label_encoder']
        predicted_disease = label_encoder.inverse_transform([prediction])[0]
        
        return predicted_disease, confidence, None
    
    except Exception as e:
        return None, None, f"Prediction error: {str(e)}"

# ============================================================================
# PREDICTION BUTTON AND RESULTS
# ============================================================================
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    predict_button = st.button("🔬 Predict Disease", use_container_width=True)

if predict_button:
    if not dna_input.strip():
        st.error("❌ Please enter a DNA sequence before predicting")
    else:
        with st.spinner("Analyzing DNA sequence..."):
            predicted_disease, confidence, error = predict_disease(dna_input)
        
        if error:
            st.error(f"❌ {error}")
        else:
            st.success("✅ Prediction Complete!")
            
            # Display results in professional layout
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                st.metric(
                    label="Predicted Disease",
                    value=predicted_disease,
                    delta="High Confidence" if confidence and confidence > 0.8 else "Normal"
                )
            
            with result_col2:
                if confidence is not None:
                    confidence_percent = confidence * 100
                    st.metric(
                        label="Confidence Score",
                        value=f"{confidence_percent:.2f}%"
                    )
                else:
                    st.metric(label="Confidence Score", value="N/A")
            
            with result_col3:
                st.metric(
                    label="Model Used",
                    value=metadata.get('model_name', 'Unknown')
                )
            
            # Display prediction details
            st.markdown("---")
            st.subheader("📊 Prediction Details")
            
            details_col1, details_col2 = st.columns(2)
            
            cleaned_length = len(''.join(dna_input.split()))
            
            with details_col1:
                st.info(f"""
                **Sequence Analysis:**
                - Input length: {len(dna_input.strip())} characters
                - Cleaned length: {cleaned_length} characters
                - Model: {metadata.get('model_name', 'Unknown')}
                """)
            
            with details_col2:
                confidence_text = f"{confidence * 100:.2f}%" if confidence is not None else "N/A"
                st.success(f"""
                **Prediction Result:**
                - Disease: {predicted_disease}
                - Confidence: {confidence_text}
                - Status: Prediction successful
                """)

# ============================================================================
# VISUALIZATIONS SECTION
# ============================================================================
st.markdown("---")
st.header("📈 Model Performance & Visualizations")

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Accuracy Comparison", "PCA Visualization"])

with tab1:
    st.subheader("Confusion Matrix - Best Model")
    if os.path.exists("outputs/confusion_matrix.png"):
        st.image("outputs/confusion_matrix.png", use_column_width=True, caption="Confusion Matrix of Best Model")
    else:
        st.warning("⚠️ Confusion matrix visualization not found")

with tab2:
    st.subheader("Model Accuracy Comparison")
    if os.path.exists("outputs/accuracy_comparison.png"):
        st.image("outputs/accuracy_comparison.png", use_column_width=True, caption="Accuracy Comparison of All Models")
    else:
        st.warning("⚠️ Accuracy comparison visualization not found")

with tab3:
    st.subheader("Feature Space Visualization")
    if os.path.exists("outputs/visualizations.png"):
        st.image("outputs/visualizations.png", use_column_width=True, caption="PCA and Feature Analysis")
    else:
        st.warning("⚠️ Feature visualization not found")

# ============================================================================
# MODEL COMPARISON TABLE
# ============================================================================
st.markdown("---")
st.header("📋 Model Performance Summary")

if os.path.exists("outputs/model_comparison.csv"):
    results_df = pd.read_csv("outputs/model_comparison.csv")
    
    # Format the dataframe for display
    display_df = results_df.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(display_df, use_container_width=True)
    
    # Show best model highlight
    best_row = results_df.loc[results_df['Accuracy'].idxmax()]
    st.success(f"""
    **Best Performing Model:** {best_row['Model']}
    - Accuracy: {best_row['Accuracy']:.4f}
    - Precision: {best_row['Precision']:.4f}
    - Recall: {best_row['Recall']:.4f}
    - F1-Score: {best_row['F1-Score']:.4f}
    """)
else:
    st.warning("⚠️ Model comparison data not found")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p style='color: gray; font-size: 12px;'>
        Genetic Disease Prediction System | Built with Python, scikit-learn, and Streamlit
    </p>
    <p style='color: gray; font-size: 12px;'>
        ⚠️ This is a demonstration system for educational purposes. Always consult medical professionals for actual disease diagnosis.
    </p>
</div>
""", unsafe_allow_html=True)
