"""
Simple Streamlit Frontend for Skin Cancer Detection
Clean version without monitoring components
"""
import streamlit as st
import requests
from PIL import Image
import io
import os
import json

st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: white;
    }
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }
    
    /* Header area styling */
    .main .block-container {
        padding-top: 2rem;
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }
    
    /* Top toolbar styling */
    .stApp > header {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(15px) !important;
    }
    
    /* Top navigation bar */
    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(15px) !important;
        height: 60px;
    }
    
    /* Toolbar area */
    .stApp > header, .stApp [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(15px) !important;
    }
    
    /* Enhanced sidebar styling */
    .css-1d391kg, .css-1cypcdb {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(15px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Improved buttons */
    .stButton > button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
        background: linear-gradient(45deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Upload section with better visual appeal */
    .upload-section {
        padding: 0px 30px 30px 30px;
        margin: 0px 0 20px 0;
    }
    
    /* Results section with glassmorphism */
    .results-section {
        padding: 0px 30px 30px 30px;
        margin: 0px 0 20px 0;
    }
    
    /* Enhanced typography */
    h1, h2, h3 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    h1 {
        font-size: 2.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* File uploader styling */
    .uploadedFile {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    
    /* Alert and warning boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    /* Info boxes */
    .stInfo {
        background: linear-gradient(145deg, rgba(52, 152, 219, 0.2), rgba(52, 152, 219, 0.1));
        border: 1px solid rgba(52, 152, 219, 0.3);
        border-radius: 10px;
        color: white;
    }
    
    /* Warning boxes */
    .stWarning {
        background: linear-gradient(145deg, rgba(243, 156, 18, 0.2), rgba(243, 156, 18, 0.1));
        border: 1px solid rgba(243, 156, 18, 0.3);
        border-radius: 10px;
        color: white;
    }
    
    /* Success boxes */
    .stSuccess {
        background: linear-gradient(145deg, rgba(39, 174, 96, 0.2), rgba(39, 174, 96, 0.1));
        border: 1px solid rgba(39, 174, 96, 0.3);
        border-radius: 10px;
        color: white;
    }
    
    /* Confidence indicators */
    .confidence-high {
        color: #2ecc71;
        font-weight: bold;
    }
    .confidence-medium {
        color: #f39c12;
        font-weight: bold;
    }
    .confidence-low {
        color: #e74c3c;
        font-weight: bold;
    }
    
    /* Medical interpretation box */
    .medical-box {
        background: linear-gradient(145deg, rgba(52, 152, 219, 0.15), rgba(52, 152, 219, 0.05));
        border: 1px solid rgba(52, 152, 219, 0.3);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.2);
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header with enhanced styling
    st.markdown("### Advanced AI-Powered")
    st.title("Skin Lesion Detection System")
    st.markdown("*Powered by Dual Ensemble Models & Medical AI Interpretation*")
    st.markdown("---")
    
    # Sidebar with improved structure
    with st.sidebar:
        st.markdown("## 🎛️ **Analysis Controls**")
        st.markdown("*Configure your analysis settings below*")
        
        # Upload prompt with icon
        st.markdown("### 📁 **Upload Requirements**")
        st.info("📤 Upload a clear, high-quality image of the skin lesion for optimal AI analysis")
        
        st.markdown("---")
        
        # Model selection with descriptions
        st.markdown("### 🤖 **Model Selection**")
        model_type = st.selectbox(
            "Choose Analysis Model:",
            ["Ensemble (Recommended)", "Model A Only", "Model B Only"],
            help="Ensemble combines both models for highest accuracy"
        )
        
        # Visual indicator for model selection
        if "Ensemble" in model_type:
            st.success("✅ Best accuracy with dual model ensemble")
        
        st.markdown("---")
        
        # Confidence threshold with better explanation
        st.markdown("### ⚖️ **Confidence Settings**")
        confidence_threshold = st.slider(
            "Minimum Confidence Threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Predictions below this threshold will show a warning"
        )
        
        # Visual confidence guide - simplified without columns
        st.markdown("""
        <div style='text-align: center; margin: 10px 0;'>
        <span class="confidence-low">🔴 Low</span> &nbsp;&nbsp;&nbsp;
        <span class="confidence-medium">🟡 Medium</span> &nbsp;&nbsp;&nbsp;
        <span class="confidence-high">🟢 High</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Advanced options
        st.markdown("### ⚙️ **Advanced Options**")
        use_tta = st.checkbox(
            "🔄 Test Time Augmentation", 
            value=True,
            help="Improves accuracy by analyzing multiple image variations"
        )
        
        if use_tta:
            st.success("🔄 Enhanced accuracy enabled")
        
        st.markdown("---")
        
        # About section with better formatting - NO EMOJIS
        st.markdown("### ℹ️ **About This System**")
        st.markdown("""
        <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;'>
        <b>Classification Categories:</b><br>
        • Actinic Keratoses<br>
        • Basal Cell Carcinoma<br>
        • Benign Keratosis-like Lesions<br>
        • Dermatofibroma<br>
        • Melanocytic Nevi<br>
        • Melanoma<br>
        • Vascular Lesions
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area with improved layout
    col1, col2 = st.columns([1.2, 1.8])
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("## 📤 **Image Upload**")
        st.markdown("*Upload your skin lesion image for analysis*")
        
        uploaded_file = st.file_uploader(
            "📁 Choose an image file:",
            type=['png', 'jpg', 'jpeg'],
            help="Supported formats: PNG, JPG, JPEG • Max size: 200MB"
        )
        
        if uploaded_file is not None:
            # Display uploaded image with better formatting
            image = Image.open(uploaded_file)
            st.markdown("### 🖼️ **Preview**")
            st.image(image, caption=f"📎 {uploaded_file.name}", use_container_width=True)
            
            # File info
            file_size = len(uploaded_file.getvalue()) / 1024  # KB
            st.info(f"📊 **File Info:** {uploaded_file.name} • {file_size:.1f} KB")
            
            # Enhanced predict button with loading states
            if st.button("🚀 **ANALYZE LESION**", type="primary", use_container_width=True):
                with st.spinner("🔬 **AI Analysis in Progress...** \n\n🤖 Running ensemble models... \n🧠 Generating medical insights..."):
                    try:
                        # Prepare the request
                        files = {"file": uploaded_file.getvalue()}
                        data = {
                            "use_tta": use_tta,
                            "model_type": model_type.lower().replace(" ", "_")
                        }
                        
                        # Show progress
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("🔍 Preprocessing image...")
                        progress_bar.progress(25)
                        
                        # Make API request
                        status_text.text("🤖 Running AI models...")
                        progress_bar.progress(50)
                        
                        api_base = os.getenv("API_BASE_URL", "http://localhost:8002")
                        response = requests.post(
                            f"{api_base}/predict",
                            files={"file": ("image.jpg", uploaded_file.getvalue(), "image/jpeg")},
                            data=data,
                            timeout=30
                        )
                        
                        status_text.text("🧠 Generating insights...")
                        progress_bar.progress(75)
                        
                        if response.status_code == 200:
                            result = response.json()
                            progress_bar.progress(100)
                            status_text.text("✅ Analysis complete!")
                            
                            # Display results in the second column
                            with col2:
                                display_results(result, confidence_threshold)
                        else:
                            st.error(f"🚨 **Server Error:** {response.status_code} - {response.text}")
                            
                    except requests.exceptions.ConnectionError:
                        st.error("🚨 **Connection Error:** Cannot connect to AI server. Please ensure the server is running on port 8002.")
                    except requests.exceptions.Timeout:
                        st.error("⏱️ **Timeout Error:** Analysis took too long. Please try with a smaller image.")
                    except Exception as e:
                        st.error(f"❌ **Unexpected Error:** {str(e)}")
        
        else:
            # Upload instructions when no file is selected
            st.markdown("### 📋 **Upload Instructions**")
            st.markdown("""
            <div style='background: rgba(52, 152, 219, 0.1); padding: 20px; border-radius: 10px; border-left: 4px solid #3498db;'>
            <b>For best results:</b><br>
            📸 Use good lighting<br>
            🎯 Center the lesion<br>
            📏 Close-up but not blurry<br>
            🔍 Clear image quality<br>
            📱 Any device camera works
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if not uploaded_file:
            st.markdown('<div class="results-section">', unsafe_allow_html=True)
            st.markdown("## 📊 **Analysis Results**")
            st.markdown("*Your AI-powered analysis will appear here*")
            
            # Simple text placeholder without boxes
            st.markdown("""
            <div style='text-align: center; padding: 30px 20px;'>
            <h3 style='color: #bbb;'>🔬 Ready for Analysis</h3>
            <p style='color: #888; margin-top: 20px;'>Upload an image to begin AI-powered skin lesion analysis with medical interpretation</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

def get_medical_interpretation(predicted_class, confidence):
    """Get medical interpretation from local LM Studio MedGemma model"""
    try:
        # Prepare the prompt for medical interpretation
        prompt = f"""As a medical AI assistant, provide a brief clinical interpretation for a skin lesion classified as "{predicted_class}" with {confidence:.1%} confidence. Include:

1. Brief description of this condition
2. Key characteristics to look for
3. General recommendations for next steps
4. When to seek immediate medical attention

Keep the response concise, professional, and emphasize the importance of professional medical evaluation."""

        lm_endpoint = os.getenv("LM_STUDIO_ENDPOINT", "http://localhost:1234/v1/chat/completions")
        response = requests.post(
            lm_endpoint,
            json={
                "model": "medgemma-4b-it-mlx",
                "messages": [
                    {"role": "system", "content": "You are MedGemma, a medical AI assistant. Provide accurate, professional medical information while always emphasizing the need for professional medical consultation."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 400
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            return "Medical interpretation service temporarily unavailable."
            
    except Exception as e:
        return f"Unable to get medical interpretation: {str(e)}"

def display_results(result, confidence_threshold):
    """Display prediction results with enhanced UI"""
    st.markdown('<div class="results-section">', unsafe_allow_html=True)
    st.markdown("## 📊 **AI Analysis Results**")
    # Normalise prediction structure from backend (supports ensemble_prediction OR final_prediction)
    if 'ensemble_prediction' in result:
        pred_block = result['ensemble_prediction']
    elif 'final_prediction' in result:
        pred_block = result['final_prediction']
    else:
        # Attempt nested structure in /analyze response
        pred_block = result.get('prediction', {}).get('ensemble_prediction', {}) or \
                     result.get('prediction', {}).get('final_prediction', {})

    # Map possible key variants
    predicted_class = (
        pred_block.get('class') or
        pred_block.get('predicted_class') or
        pred_block.get('condition') or
        'Unknown'
    )
    confidence = (
        pred_block.get('confidence') or
        pred_block.get('confidence_score') or
        0.0
    )
    
    # Enhanced confidence visualization
    if confidence >= 0.8:
        confidence_color = "🟢"
        confidence_class = "confidence-high"
        confidence_status = "HIGH CONFIDENCE"
    elif confidence >= 0.6:
        confidence_color = "🟡"
        confidence_class = "confidence-medium"
        confidence_status = "MODERATE CONFIDENCE"
    else:
        confidence_color = "🔴"
        confidence_class = "confidence-low"
        confidence_status = "LOW CONFIDENCE"
    
    # Main prediction display
    st.markdown(f"""
    <div style='background: linear-gradient(145deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2)); 
                padding: 25px; border-radius: 15px; border: 2px solid rgba(102, 126, 234, 0.3); 
                text-align: center; margin: 20px 0; box-shadow: 0 8px 25px rgba(0,0,0,0.3);'>
        <h2 style='margin: 0; color: white;'>{confidence_color} <b>{predicted_class.replace('_', ' ').title()}</b></h2>
        <h4 style='margin: 10px 0; color: #bbb;'>Ensemble Model Prediction</h4>
        <div class='{confidence_class}' style='font-size: 1.2em; margin: 10px 0;'>
            <b>{confidence_status}: {confidence:.1%}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced progress bar - simplified
    st.markdown("### 📈 **Confidence Level**")
    st.progress(confidence)
    st.markdown(f"**Score: {confidence:.1%}**")
    
    # Confidence warning with better styling
    if confidence < confidence_threshold:
        st.markdown("""
        <div style='background: rgba(243, 156, 18, 0.2); border: 2px solid rgba(243, 156, 18, 0.5); 
                    border-radius: 10px; padding: 20px; margin: 15px 0;'>
            <h4 style='color: #f39c12; margin: 0;'>⚠️ <b>Low Confidence Alert</b></h4>
            <p style='margin: 10px 0 0 0; color: white;'>
                This prediction falls below your confidence threshold. 
                <b>Professional medical consultation is strongly recommended.</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Medical interpretation with enhanced styling
    st.markdown("---")
    st.markdown("## 🏥 **Medical AI Interpretation**")
    st.markdown("*Powered by MedGemma 4B Medical AI*")
    
    with st.spinner("🧠 Generating professional medical interpretation..."):
        interpretation = get_medical_interpretation(predicted_class, confidence)
        
        # Display interpretation in a styled box
        st.markdown(f"""
        <div class='medical-box'>
            <div style='display: flex; align-items: center; margin-bottom: 15px;'>
                <span style='font-size: 1.5em; margin-right: 10px;'>🩺</span>
                <h4 style='margin: 0; color: white;'>Clinical Interpretation ({predicted_class.replace('_', ' ').title()} - {confidence:.1%} Confidence)</h4>
            </div>
            <div style='color: #e8e8e8; line-height: 1.6; white-space: pre-line;'>
                {interpretation}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced medical disclaimer
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(145deg, rgba(231, 76, 60, 0.15), rgba(192, 57, 43, 0.15)); 
                border: 2px solid rgba(231, 76, 60, 0.3); border-radius: 15px; padding: 25px; margin: 20px 0;'>
        <div style='display: flex; align-items: center; margin-bottom: 15px;'>
            <span style='font-size: 2em; margin-right: 15px;'>⚠️</span>
            <h3 style='margin: 0; color: #e74c3c;'><b>Important Medical Disclaimer</b></h3>
        </div>
        <p style='color: white; line-height: 1.6; margin: 0;'>
            <b>This AI system is a screening tool for educational purposes only and should never replace professional medical diagnosis.</b> 
            All skin lesions require proper evaluation by qualified dermatologists or healthcare providers. 
            Seek immediate medical attention for any concerning skin changes, regardless of AI predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
