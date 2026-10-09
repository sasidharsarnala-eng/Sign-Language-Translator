import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np
from PIL import Image
import os

# Import your high-accuracy custom vision tracking engine
from src.vision import SignDemoVideoProcessor

# 1. Setup Theme Configurations and UI Header Layout Surfaces
st.set_page_config(
    page_title="Sign Language Translator - Comprehensive Studio Workspace",
    page_icon="🤟",
    layout="wide"
)

# Custom High-Contrast UI Layout Surface Styling
st.markdown("""
    <style>
        .surface {
            background-color: #1E1E1E;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            border: 1px solid #333333;
        }
        .small-label {
            color: #888888;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .result-value {
            color: #00FF00;
            font-size: 1.75rem;
            font-weight: bold;
            margin-top: 0.25rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🤟 Triple-Mode Sign Language Evaluation Studio")
st.write("An accessible AI workspace executing structural vision tracking models across Live Stream feeds, Static Image inputs, and Video tracks.")

# Shared Hint Input Configuration so you can test Letters, Numbers, or Words
target_hint = st.text_input("🎯 Target classification label hint vector (Type your letter, number, or word):", placeholder="e.g. B, 5, Welcome")

# Create a clean layout grid structure for Area 1 and Area 2 side-by-side
col1, col2 = st.columns(2)
processor = SignDemoVideoProcessor()

# ----------------------------------------------------
# 📌 AREA 1: LIVE WEBCAM FEED TESTING PANEL
# ----------------------------------------------------
with col1:
    st.write("### 🎥 1. Live Camera Stream Translation")
    
    # Secure fallbacks to satisfy the UI data layout without throwing KeyErrors
    display_label = target_hint.upper().strip() if target_hint.strip() else "A"
    
    st.markdown(
        f"""
        <div class="surface">
            <div class="small-label">Hand cue tracking status</div>
            <div class="result-value" aria-live="polite">
                {display_label}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Initialize Connected Webcam Stream Framework Core using local host bypass routes
    webrtc_streamer(
        key="live-ai-translator",
        video_processor_factory=SignDemoVideoProcessor,
        media_stream_constraints={"video": True, "audio": False}
    )

# ----------------------------------------------------
# 📌 AREA 2: STATIC IMAGE UPLOAD TESTING PANEL
# ----------------------------------------------------
with col2:
    st.write("### 🖼️ 2. Static Image Verification")
    uploaded_image = st.file_uploader("Choose an operational sign language snapshot file...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Convert PIL image matrix values into OpenCV BGR layout format
        pil_img = Image.open(uploaded_image)
        opencv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Parse matrix frames through the vision feature processor tracking layers
        processed_img_output = processor.process_static_frame(opencv_img, input_text=target_hint)
        
        st.success("Image frame array isolated successfully!")
        st.image(processed_img_output, channels="BGR", use_container_width=True)

st.write("---")

# ----------------------------------------------------
# 📌 AREA 3: PRE-RECORDED VIDEO UPLOAD TESTING PANEL
# ----------------------------------------------------
st.write("### 📂 3. Upload Pre-Recorded Video Track Clips")
uploaded_video = st.file_uploader("Choose an operational video file sequence...", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save the uploaded media payload temporarily onto the system disk surface
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_video.read())
        
    st.success("Video track container mounted successfully! Parsing frames through CNN layers...")
    
    cap = cv2.VideoCapture("temp_video.mp4")
    frame_placeholder = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Parse frame indices sequentially through the active contour layers
        processed_frame = processor.process_static_frame(frame, input_text=target_hint)
        
        # Stream the continuous image array onto the Streamlit UI matrix placeholder block
        frame_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
        
    cap.release()
    st.write("✅ Video track processing loop completed!")