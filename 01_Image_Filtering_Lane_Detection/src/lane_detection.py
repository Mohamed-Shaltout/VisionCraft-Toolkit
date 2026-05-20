import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import os

def make_coordinates(image, line_parameters):
    """Convert slope and intercept into (x1, y1, x2, y2) coordinates."""
    slope, intercept = line_parameters
    height = image.shape[0]
    y1 = height
    y2 = int(y1 * 0.6)
    
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [x1, y1, x2, y2]

def detect_lanes(image_path="road.jpg"):
    """
    Performs lane detection on an image.
    
    Args:
        image_path (str): Path to the image.
        
    Returns:
        tuple: (original_image_rgb, final_image_with_lanes)
    """
    image_url = 'https://raw.githubusercontent.com/udacity/CarND-LaneLines-P1/master/test_images/solidWhiteRight.jpg'
    
    if not os.path.exists(image_path):
        print("Downloading sample image...")
        urllib.request.urlretrieve(image_url, image_path)
        
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image file not found or could not be read.")
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = np.copy(image)
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred_image = cv2.medianBlur(gray_image, 5)
    canny_image = cv2.Canny(blurred_image, 50, 150)
    
    height, width = canny_image.shape
    roi_vertices = np.array([
        [(int(0.1 * width), height),
         (int(0.45 * width), int(0.6 * height)),
         (int(0.55 * width), int(0.6 * height)),
         (int(0.95 * width), height)]
    ], dtype=np.int32)
    
    mask = np.zeros_like(canny_image)
    cv2.fillPoly(mask, roi_vertices, 255)
    roi_image = cv2.bitwise_and(canny_image, mask)
    
    lines = cv2.HoughLinesP(
        roi_image, rho=2, theta=np.pi/180, threshold=100,
        lines=np.array([]), minLineLength=40, maxLineGap=5
    )
    
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope, intercept = parameters
            if slope < -0.5:
                left_fit.append((slope, intercept))
            elif slope > 0.5:
                right_fit.append((slope, intercept))
                
    averaged_lines = []
    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        averaged_lines.append(make_coordinates(image, left_fit_average))
    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        averaged_lines.append(make_coordinates(image, right_fit_average))
        
    line_image = np.zeros_like(original_image)
    if averaged_lines:
        for x1, y1, x2, y2 in averaged_lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
            
    final_image = cv2.addWeighted(original_image, 0.8, line_image, 1, 0)
    return original_image, final_image

if __name__ == "__main__":
    orig, out = detect_lanes("../data/road.jpg")
    plt.imshow(out)
    plt.show()
