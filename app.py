import os
import numpy as np
import tensorflow as tf
import json
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import streamlit as st
import pandas as pd
import requests
from streamlit_lottie import st_lottie
from io import BytesIO
import base64
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set page configuration
st.set_page_config(
    page_title="Medicinal Herb Classifier",
    page_icon="🌿",
    layout="wide",
)

# Custom CSS for green and white theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #2e7d32;
        --secondary-color: #4caf50;
        --background-color: #ffffff;
        --text-color: #2e7d32;
        --accent-color: #81c784;
    }
    
    /* Header styling */
    .main-header {
        color: var(--primary-color);
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Subheader styling */
    .sub-header {
        color: var(--secondary-color);
        font-size: 1.5rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    
    /* Card styling */
    .card {
        background-color: var(--background-color);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid var(--primary-color);
    }
    
    /* Button styling */
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton>button:hover {
        background-color: var(--secondary-color);
    }
    
    /* Metrics styling */
    .metric-value {
        color: var(--primary-color);
        font-size: 2rem;
        font-weight: 700;
    }
    
    .metric-label {
        color: var(--text-color);
        font-size: 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--background-color);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: var(--secondary-color);
    }
</style>
""", unsafe_allow_html=True)

# Function to load lottie animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load lottie animations
lottie_herb = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_jmejybvu.json")  # Plant growing
lottie_scan = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_w51pcehl.json")  # Scanning
lottie_results = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_nw19osms.json")  # Analysis
lottie_upload = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_DMgKk1.json")  # Upload

# Function to load the model
@st.cache_resource
def load_model(model_path="models/herb_classifier.h5"):
    """Load the trained model with caching"""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to load class labels
@st.cache_data
def load_class_labels():
    """Load class labels from the saved JSON file with caching"""
    labels_path = "data/class_labels.json"
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            class_labels = json.load(f)
        return class_labels
    else:
        st.error(f"Class labels file not found at {labels_path}")
        return None

# Function to preprocess image
def preprocess_image(image, target_size=(224, 224)):
    """Preprocess a single image for prediction"""
    try:
        img = image.resize(target_size)
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0,1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Function to make predictions
def predict_herb(model, image, class_labels, target_size=(224, 224)):
    """Make prediction on an image"""
    try:
        # Preprocess the image
        preprocessed_img = preprocess_image(image, target_size)
        
        if preprocessed_img is None:
            return None, None, None
        
        # Make prediction
        with st.spinner("Analyzing image..."):
            predictions = model.predict(preprocessed_img)
        
        # Get the predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        if predicted_class_idx < len(class_labels):
            predicted_class = class_labels[predicted_class_idx]
        else:
            predicted_class = "Unknown"
        
        # Get top 5 predictions
        top_indices = np.argsort(predictions[0])[-5:][::-1]
        top_classes = [class_labels[i] for i in top_indices]
        top_confidences = [float(predictions[0][i]) for i in top_indices]
        
        return predicted_class, confidence, list(zip(top_classes, top_confidences))
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None

# Function to create an image download link
def get_image_download_link(img, filename, text):
    """Generate a download link for an image"""
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Function to test on a batch of images
def batch_predict(model, uploaded_files, class_labels, target_size=(224, 224)):
    """Process multiple uploaded images"""
    results = []
    
    for uploaded_file in uploaded_files:
        try:
            # Read the image
            image = Image.open(uploaded_file)
            
            # Make prediction
            predicted_class, confidence, _ = predict_herb(
                model, image, class_labels, target_size
            )
            
            # Store results
            results.append({
                "File": uploaded_file.name,
                "Predicted Class": predicted_class,
                "Confidence": confidence
            })
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
    
    return pd.DataFrame(results)

# Function to visualize predictions
def visualize_prediction(image, predicted_class, confidence, top_predictions=None):
    """Create a visualization of the prediction"""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Create figure for visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img_array)
    ax.set_title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2%}", fontsize=16)
    ax.axis('off')
    
    return fig

# Main app function
def main():
    """Main function for the Streamlit app"""
    # Header and introduction
    st.markdown('<h1 class="main-header">🌿 Medicinal Herb Classifier</h1>', unsafe_allow_html=True)
    
    # Show the herb lottie animation in the sidebar
    with st.sidebar:
        st_lottie(lottie_herb, height=200, key="herb_animation")
        st.markdown('<h2 class="sub-header">About</h2>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        This application uses a machine learning model to classify medicinal herbs from images. 
        Upload an image of a medicinal herb to identify it!
        </div>
        """, unsafe_allow_html=True)
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["📸 Single Image", "📁 Batch Processing", "📊 Model Info"])
    
    # Load the model
    model = load_model()
    
    # Load class labels
    class_labels = load_class_labels()
    
    if model is None or class_labels is None:
        st.error("Could not load the model or class labels. Please check the paths.")
        return
    
    # Tab 1: Single Image Classification
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<h2 class="sub-header">Upload an Image</h2>', unsafe_allow_html=True)
            st_lottie(lottie_upload, height=200, key="upload_animation")
            
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                # Display the uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Button to predict
                if st.button("Classify Herb"):
                    # Display scanning animation
                    st_lottie(lottie_scan, height=200, key="scan_animation")
                    
                    # Make prediction
                    predicted_class, confidence, top_predictions = predict_herb(
                        model, image, class_labels
                    )
                    
                    if predicted_class is not None:
                        # Store the results in session state
                        st.session_state.predicted_class = predicted_class
                        st.session_state.confidence = confidence
                        st.session_state.top_predictions = top_predictions
                        st.session_state.image = image
                    else:
                        st.error("Could not make a prediction. Please try another image.")
        
        with col2:
            st.markdown('<h2 class="sub-header">Results</h2>', unsafe_allow_html=True)
            
            # Check if prediction has been made
            if 'predicted_class' in st.session_state:
                # Display results animation
                st_lottie(lottie_results, height=150, key="results_animation")
                
                # Display the results
                st.markdown(f"""
                <div class="card">
                    <h3>Prediction: {st.session_state.predicted_class}</h3>
                    <div class="metric-value">{st.session_state.confidence:.2%}</div>
                    <div class="metric-label">Confidence</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display visualization
                fig = visualize_prediction(
                    st.session_state.image, 
                    st.session_state.predicted_class, 
                    st.session_state.confidence
                )
                st.pyplot(fig)
                
                # Display top 5 predictions
                st.markdown('<h3 class="sub-header">Top 5 Predictions</h3>', unsafe_allow_html=True)
                
                for cls, conf in st.session_state.top_predictions:
                    st.progress(conf)
                    st.write(f"{cls}: {conf:.2%}")
    
    # Tab 2: Batch Processing
    with tab2:
        st.markdown('<h2 class="sub-header">Batch Processing</h2>', unsafe_allow_html=True)
        st_lottie(lottie_upload, height=200, key="batch_upload_animation")
        
        uploaded_files = st.file_uploader("Upload multiple images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} images")
            
            if st.button("Process All Images"):
                # Display scanning animation
                st_lottie(lottie_scan, height=200, key="batch_scan_animation")
                
                # Process all images
                with st.spinner("Processing images..."):
                    results_df = batch_predict(model, uploaded_files, class_labels)
                
                # Display results
                st.markdown('<h3 class="sub-header">Batch Results</h3>', unsafe_allow_html=True)
                st.dataframe(results_df)
                
                # Option to download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="herb_classification_results.csv",
                    mime="text/csv",
                )
    
    # Tab 3: Model Info
    with tab3:
        st.markdown('<h2 class="sub-header">Model Information</h2>', unsafe_allow_html=True)
        
        # Display model summary
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("Model Architecture:")
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        st.code("\n".join(model_summary))
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display class labels
        st.markdown('<h3 class="sub-header">Available Herb Classes</h3>', unsafe_allow_html=True)
        
        # Create a grid of herbs
        cols = st.columns(4)
        for i, herb in enumerate(class_labels):
            cols[i % 4].markdown(f"- {herb}")

if __name__ == "__main__":
    main()
