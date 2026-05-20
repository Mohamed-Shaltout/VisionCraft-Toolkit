import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
import os

# Setup paths for importing the modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '01_Image_Filtering_Lane_Detection', 'src'))
sys.path.append(os.path.join(BASE_DIR, '02_Panorama_Stitching_Homography', 'src'))
sys.path.append(os.path.join(BASE_DIR, '03_Stereo_Vision_Disparity_Map', 'src'))
sys.path.append(os.path.join(BASE_DIR, '04_Pet_Expression_Classification_CNNs', 'src'))

# Import the modules
try:
    import cartoonify
    import lane_detection
    import stitcher
    import disparity
    import models
except ImportError as e:
    st.error(f"Error loading modules: {e}")

st.set_page_config(page_title="VisionCraft Toolkit", page_icon="👁️", layout="wide")

st.title("👁️ VisionCraft Toolkit")
st.markdown("A unified interface for exploring Computer Vision & Deep Learning algorithms.")

# Sidebar Navigation
lab_selection = st.sidebar.selectbox(
    "Select a Laboratory",
    (
        "Introduction", 
        "Lab 1: Cartoonify Image", 
        "Lab 1: Lane Detection",
        "Lab 2: Panorama Stitching",
        "Lab 3: Disparity Map (Stereo Vision)",
        "Lab 4: Pet Expression CNNs"
    )
)

def load_image(uploaded_file):
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img
    return None

if lab_selection == "Introduction":
    st.header("Welcome to VisionCraft")
    st.write("Use the sidebar to navigate through the interactive computer vision labs.")
    st.info("Upload your own images to test the algorithms in real-time!")

elif lab_selection == "Lab 1: Cartoonify Image":
    st.header("🎨 Cartoonify Filter")
    st.write("Applies bilateral filtering and edge detection to create a cartoon effect.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
        
        if st.button("Generate Cartoon"):
            with st.spinner('Applying filters...'):
                # Save temp file for the module that takes a path
                temp_path = "temp_cartoon_input.jpg"
                cv2.imwrite(temp_path, image)
                try:
                    result = cartoonify.generate_cartoon(temp_path)
                    st.image(result, caption="Cartoonified Image", use_column_width=True)
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

elif lab_selection == "Lab 1: Lane Detection":
    st.header("🛣️ Lane Detection")
    st.write("Uses Canny edges and Hough Transforms to detect road lanes.")
    
    uploaded_file = st.file_uploader("Choose a road image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
        
        if st.button("Detect Lanes"):
            with st.spinner('Detecting...'):
                temp_path = "temp_lane_input.jpg"
                cv2.imwrite(temp_path, image)
                try:
                    orig, result = lane_detection.detect_lanes(temp_path)
                    st.image(result, caption="Lane Detection Result", use_column_width=True)
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

elif lab_selection == "Lab 2: Panorama Stitching":
    st.header("📸 Panorama Stitching")
    st.write("Stitches two images together using SIFT and Homography.")
    
    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader("Upload Left Image", type=["jpg", "jpeg", "png"])
    with col2:
        file2 = st.file_uploader("Upload Right Image", type=["jpg", "jpeg", "png"])
        
    if file1 and file2:
        img1 = load_image(file1)
        img2 = load_image(file2)
        
        col1.image(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), use_column_width=True)
        col2.image(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        if st.button("Stitch Images"):
            with st.spinner('Calculating matches and warping...'):
                try:
                    mosaic, H, inliers = stitcher.stitch(img1, img2)
                    st.image(cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB), caption="Panorama Mosaic", use_column_width=True)
                except Exception as e:
                    st.error(f"Stitching failed: {e}")

elif lab_selection == "Lab 3: Disparity Map (Stereo Vision)":
    st.header("🕶️ Stereo Vision Disparity Map")
    st.write("Computes depth map from stereo image pairs using block matching.")
    
    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader("Upload Left Image (Grayscale)", type=["jpg", "jpeg", "png"])
    with col2:
        file2 = st.file_uploader("Upload Right Image (Grayscale)", type=["jpg", "jpeg", "png"])
        
    if file1 and file2:
        img_left = load_image(file1)
        img_right = load_image(file2)
        
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        
        col1.image(gray_left, caption="Left", use_column_width=True, channels="GRAY")
        col2.image(gray_right, caption="Right", use_column_width=True, channels="GRAY")
        
        method = st.selectbox("Matching Method", ["SAD", "SSD"])
        window_size = st.slider("Window Size", 1, 15, 5, step=2)
        
        if st.button("Compute Disparity"):
            with st.spinner('Computing block matching...'):
                disp = disparity.compute_disparity_block_matching(gray_left, gray_right, window_size=window_size, method=method)
                norm_disp = disparity.normalize_disparity(disp)
                st.image(norm_disp, caption="Disparity Map (Jet Colormap)", use_column_width=True, clamp=True)

elif lab_selection == "Lab 4: Pet Expression CNNs":
    st.header("🐶 Pet Expression CNNs")
    st.write("Browse custom CNN models designed to classify pet expressions.")
    
    model_choice = st.selectbox("Select Model Architecture", ["VGG", "ResNet", "MobileNet"])
    
    st.info(f"You have selected {model_choice}. In a full production environment, this would load the pretrained `.pth` weights for inference. The model architecture has been initialized successfully from `models.py`.")
    
    # Just show that we can init it
    try:
        net = models.get_model(model_choice, num_classes=5)
        st.code(str(net))
    except Exception as e:
        st.error(f"Error loading model: {e}")
