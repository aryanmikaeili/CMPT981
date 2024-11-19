import cv2
import numpy as np
import matplotlib.pyplot as plt

import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_high_frequency(image_path, canny_low=100, canny_high=200, threshold=100):
    """
    Detect high-frequency components in an image using multiple methods including Canny edge detection.
    
    Args:
        image_path (str): Path to the input image
        canny_low (int): Lower threshold for Canny edge detection
        canny_high (int): Upper threshold for Canny edge detection
        threshold (int): Threshold for high-frequency detection (0-255)
    
    Returns:
        tuple: (edge coordinates, frequency magnitude image, gradient magnitude image, canny edges)
    """
    # Read the image
    img = cv2.imread(image_path)
    height, width, channels = img.shape  # for color images
    
    print(f"Image dimensions:")
    print(f"Height: {height} pixels")
    print(f"Width: {width} pixels")
    print(f"Total pixels: {height * width}")
    print(f"Number of channels: {channels}")  # Usually 3 for BGR images
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Fourier Transform
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    
    # Normalize magnitude for visualization
    magnitude_log = np.log1p(magnitude)
    magnitude_normalized = cv2.normalize(magnitude_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Method 2: Sobel Edge Detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize gradient magnitude
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Method 3: Canny Edge Detection
    canny_edges = cv2.Canny(gray, canny_low, canny_high)
    
    # Get coordinates of high-frequency components (using Canny edges)
    high_freq_coords = np.where(canny_edges > 0)
    coords_list = list(zip(high_freq_coords[1], high_freq_coords[0]))  # (x,y) format
    
    return coords_list, magnitude_normalized, gradient_magnitude, canny_edges

def visualize_frequency_analysis(original_image_path, coords_list, magnitude_img, gradient_img, canny_edges):
    """
    Visualize the results of frequency analysis with multiple subplots.
    
    Args:
        original_image_path (str): Path to the original image
        coords_list (list): List of coordinates where high frequencies were detected
        magnitude_img (ndarray): Normalized frequency magnitude image
        gradient_img (ndarray): Gradient magnitude image
        canny_edges (ndarray): Canny edge detection result
    """
    # Read original image
    original = cv2.imread(original_image_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Create visualization of high-frequency points on original image
    high_freq_overlay = original_rgb.copy()
    for x, y in coords_list:
        cv2.circle(high_freq_overlay, (x, y), 1, (255, 0, 0), -1)
    
    # Create figure with subplots
    plt.figure(figsize=(20, 10))
    
    # Original Image
    plt.subplot(231)
    plt.imshow(original_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    # Frequency Magnitude (FFT)
    plt.subplot(232)
    plt.imshow(magnitude_img, cmap='gray')
    plt.title('Frequency Magnitude (FFT)')
    plt.axis('off')
    
    # Gradient Magnitude
    plt.subplot(233)
    plt.imshow(gradient_img, cmap='gray')
    plt.title('Gradient Magnitude')
    plt.axis('off')
    
    # Canny Edges
    plt.subplot(234)
    plt.imshow(canny_edges, cmap='gray')
    plt.title('Canny Edges')
    plt.axis('off')
    
    # High Frequency Points Overlay
    plt.subplot(235)
    plt.imshow(high_freq_overlay)
    plt.title('High Frequency Points')
    plt.axis('off')
    
    # Combined View
    plt.subplot(236)
    combined = cv2.addWeighted(original_rgb, 0.7, 
                             cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2RGB), 0.3, 0)
    plt.imshow(combined)
    plt.title('Combined View')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    image_path = "circles4/circle_000.png"
    coords, magnitude_img, gradient_img, canny_edges = detect_high_frequency(image_path, threshold=100)
    visualize_frequency_analysis(image_path, coords, magnitude_img, gradient_img, canny_edges)

    print(type(coords))
    print(len(coords))
    print(type(coords[0]))
    print(coords[0])