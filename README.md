# VisionCraft-Toolkit 👁️✨

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
</div>

<br/>

**VisionCraft-Toolkit** is a curated collection of advanced Computer Vision and Deep Learning laboratories. This repository seamlessly transforms academic algorithms into production-ready pipelines, featuring a beautiful frontend that allows you to interactively test classical computer vision and deep learning implementations.

---

## 🧪 Lab Index

### [01_Image_Filtering_Lane_Detection](./01_Image_Filtering_Lane_Detection/)
- **Image Filtering (Cartoonify)**: Implements OpenCV's bilateral filtering and laplacian edge detection to generate a stylistic cartoon effect on standard images.
- **Lane Detection**: Employs Canny Edge Detection, Hough Transforms, and Region of Interest masking to accurately identify road lanes for self-driving car applications.

### [02_Panorama_Stitching_Homography](./02_Panorama_Stitching_Homography/)
- **Image Stitching**: Features a comprehensive pipeline for creating beautiful panoramas. Uses SIFT/ORB for rich feature extraction, Direct Linear Transformation (DLT) for calculating homography, and robust RANSAC to eliminate outliers.

### [03_Stereo_Vision_Disparity_Map](./03_Stereo_Vision_Disparity_Map/)
- **Disparity Generation**: Calculates depth disparity from stereo images using computationally intensive Block Matching (SAD/SSD) and a robust Dynamic Programming approach for scanline optimization.

### [04_Pet_Expression_Classification_CNNs](./04_Pet_Expression_Classification_CNNs/)
- **Deep Learning Architectures**: Constructs 5 state-of-the-art CNNs (VGG, ResNet, MobileNet, Inception, and DenseNet) entirely from scratch using PyTorch to classify pet facial expressions.

---

## 🚀 Installation & Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/VisionCraft-Toolkit.git
cd VisionCraft-Toolkit
```

### 2. Install Dependencies
It's highly recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Interactive Toolkit
The best way to experience these labs is through our unified **Streamlit Interface**. Start the server with:
```bash
streamlit run app.py
```
This will launch a beautiful web application in your browser where you can upload images, select labs, and apply these advanced algorithms in real-time.

---

## 📜 License
This repository is licensed under the [MIT License](LICENSE).
