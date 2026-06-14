import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Countries Currency Note Recognition System",
    page_icon="💵",
    layout="wide"
)

# ==========================================
# CUSTOM STYLING (Professional Dashboard Look)
# ==========================================
st.markdown("""
    <style>
        /* Main Theme Accents */
        h1 { color: #0B5394 !important; font-weight: 700; text-align: center; margin-bottom: 5px; }
        h4 { text-align: center; color: #666666; margin-bottom: 25px; }
        
        /* Box containers mimicking the UI cards */
        .dashboard-card {
            background-color: #F8F9FA;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #E0E0E0;
            margin-bottom: 20px;
        }
        
        /* Footer custom alignment */
        .footer-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #F0F4F8;
            padding: 12px 20px;
            border-radius: 6px;
            border: 1px solid #D6E4F0;
            font-size: 0.9rem;
            color: #333333;
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_knn_model():
    return joblib.load("currency_knn_model.pkl")

try:
    model = load_knn_model()
except Exception as e:
    st.error("Error loading model file 'currency_knn_model.pkl'. Please ensure it exists in the project directory.")
    st.stop()

# ==========================================
# HEADER SECTION (Updated Title & Subtitle)
# ==========================================
st.markdown("<h1>Countries Currency Note Recognition System</h1>", unsafe_allow_html=True)
st.markdown("<h4>using image processing and K-Nearest Neighbors(KNN)</h4>", unsafe_allow_html=True)

# ==========================================
# SIDEBAR (Flags Added Beside Supported Currencies)
# ==========================================
with st.sidebar:
    st.title("🌐 CURRENCY RECOGNITION")
    st.markdown("---")
    
    st.header("Supported Currencies")
    # Clean indicator format mimicking the visual tags with flags next to them
    st.info("🇮🇳 **INR** → India")
    st.warning("🇦🇪 **AED** → United Arab Emirates")
    st.success("🇹🇭 **THB** → Thailand")
    
    st.markdown("---")
    st.subheader("Project By")
    st.caption("Developed by:")
    st.write("👤 **Sravani Shetty**")

# ==========================================
# MAIN DASHBOARD LAYOUT
# ==========================================
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 1️⃣ Input Currency Note Image")
    uploaded_file = st.file_uploader(
        "Drag and drop an image here or browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    # Placeholder layout when no file is uploaded yet
    if uploaded_file is None:
        st.info("Please upload an image file (JPG, JPEG, PNG up to 10MB) to start recognition.")
        
        st.markdown("### 2️⃣ Preprocessed Image (Grayscale & Resized)")
        st.markdown(
            '<div class="dashboard-card" style="text-align:center; color:#888; padding:40px;">'
            'Preprocessed image will appear here after loading.'
            '</div>', 
            unsafe_allow_html=True
        )

with col_right:
    st.markdown("### 3️⃣ Recognition Result")
    
    if uploaded_file is None:
        # Standby status layout
        st.markdown(
            '<div class="dashboard-card" style="text-align:center; padding:30px;">'
            '<h3 style="color:#777; margin-bottom:2px;">PREDICTED CURRENCY</h3>'
            '<h1 style="color:#ccc; font-size:48px; margin:10px 0;">--</h1>'
            '<hr style="margin:10px 0;">'
            '<p style="text-align:left; margin:5px 0;">🌐 <b>Country:</b> --</p>'
            '<p style="text-align:left; margin:5px 0;">📊 <b>Confidence Score:</b> -- %</p>'
            '</div>', 
            unsafe_allow_html=True
        )
        
        # Static informational currencies table 
        st.markdown("### 📋 Currency Codes & Countries")
        st.markdown("""
        | Flag | Code | Country Name |
        | :---: | :---: | :--- |
        | 🇮🇳 | **INR** | India |
        | 🇦🇪 | **AED** | United Arab Emirates |
        | 🇹🇭 | **THB** | Thailand |
        """)

# ==========================================
# IMAGE PROCESSING & PREDICTION CORE
# ==========================================
if uploaded_file is not None:
    # Read Image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Render operational content inside columns dynamically
    with col_left:
        st.image(image, caption="Uploaded Currency Note", use_container_width=True)
        
        # Preprocessing operations
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        img_resized = cv2.resize(img_gray, (64, 64))
        
        st.markdown("### 2️⃣ Preprocessed Image")
        st.image(img_resized, caption="Resized Grayscale Input (64x64)", width=200)
        
        # Flatten array for KNN model prediction
        img_flatten = img_resized.flatten().reshape(1, -1)

    with col_right:
        # Trigger Action Button Panel
        st.markdown("#### Action Panel")
        predict_btn = st.button("🔮 Predict Currency", type="primary", use_container_width=True)
        
        if predict_btn:
            # Model inference
            prediction = model.predict(img_flatten)[0]
            
            currency_mapping = {
                0: ("AED", "United Arab Emirates", "🇦🇪"),
                1: ("INR", "India", "🇮🇳"),
                2: ("THB", "Thailand", "🇹🇭")
            }
            
            currency_code, country, flag = currency_mapping.get(prediction, ("Unknown", "Unknown", "❓"))
            
            # Interactive output cards block with Flag included
            st.markdown(f"""
                <div class="dashboard-card" style="border-left: 5px solid #0B5394;">
                    <h3 style="color:#0B5394; margin-top:0;">PREDICTED CURRENCY</h3>
                    <h1 style="font-size:42px; margin:10px 0; color:#333;">{flag} {currency_code}</h1>
                    <hr style="margin:10px 0;">
                    <h4 style="text-align:left; color:#555; margin:5px 0;">🌐 Country: <b>{flag} {country}</b></h4>
                </div>
            """, unsafe_allow_html=True)
            
            
        else:
            st.warning("Click the 'Predict Currency' button above to process the results.")

# ==========================================
# SYSTEM UTILITY BUTTONS
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)
col_reset, _ = st.columns([1, 5])
with col_reset:
    if st.button("🔄 Reset Application", use_container_width=True):
        st.rerun()

# ==========================================
# PROFESSIONAL CLEAN FOOTER
# ==========================================
st.markdown(
    """
    <div class="footer-container">
        <div>🛡️ This application recognizes currency notes of India (INR), United Arab Emirates (AED), and Thailand (THB).</div>
        <div><b>All Rights Reserved</b></div>
    </div>
    """,
    unsafe_allow_html=True
)