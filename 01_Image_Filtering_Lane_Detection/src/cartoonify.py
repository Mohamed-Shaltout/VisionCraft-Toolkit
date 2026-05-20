import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_cartoon(image_path):
    """
    Applies a cartoon effect to an image using OpenCV.
    
    Args:
        image_path (str): Path to the input image.
        
    Returns:
        np.ndarray: The resulting cartoonified image (RGB format).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Convert to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply median blur to reduce noise
    filtered = cv2.medianBlur(image_gray, 5)
    
    # Edge detection using Laplacian
    laplacian = cv2.Laplacian(filtered, cv2.CV_64F)
    laplacian_abs = np.absolute(laplacian)
    
    # Normalize and threshold edges
    p98 = np.percentile(laplacian_abs, 98)
    laplacian_scaled = np.uint8(255 * np.minimum(laplacian_abs / (p98 + 1e-5), 1.0))
    _, sketch = cv2.threshold(laplacian_scaled, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Bilateral filter for color smoothing
    test_cartoon = image.copy()
    for _ in range(4):
        test_cartoon = cv2.bilateralFilter(test_cartoon, 9, 100, 100)
        
    # Combine sketch with smoothed image
    sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    final_cartoon = cv2.bitwise_and(test_cartoon, sketch_bgr)
    
    # Convert BGR to RGB for consistent visualization
    final_cartoon_rgb = cv2.cvtColor(final_cartoon, cv2.COLOR_BGR2RGB)
    return final_cartoon_rgb

if __name__ == "__main__":
    # Example Usage
    path = "../data/road.jpg" # Using road image as a placeholder for testing
    try:
        result = generate_cartoon(path)
        plt.imshow(result)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(e)
