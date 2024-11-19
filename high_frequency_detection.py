import cv2
import numpy as np
import matplotlib.pyplot as plt

import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_high_frequency(image_path, canny_low=0, canny_high=10, blur_sigma=0, threshold=100, dilate = False):
    """
    Detect high-frequency components in an image using multiple methods including Canny edge detection.
    
    Args:
        image_path (str): Path to the input image
        canny_low (int): Lower threshold for Canny edge detection (0-255)
        canny_high (int): Upper threshold for Canny edge detection (0-255)
        blur_sigma (int): Gaussian blur sigma (0 means no blur)
        threshold (int): Threshold for high-frequency detection (0-255)
    
    Returns:
        tuple: (edge coordinates, frequency magnitude image, gradient magnitude image, canny edges)
    """
    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur if sigma > 0
    if blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), blur_sigma)
    
    # Method 1: Fourier Transform (unchanged)
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    magnitude_log = np.log1p(magnitude)
    magnitude_normalized = cv2.normalize(magnitude_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Method 2: Sobel Edge Detection (unchanged)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Method 3: Enhanced Canny Edge Detection
    canny_edges = cv2.Canny(gray, canny_low, canny_high)
    if dilate:
        kernel = np.ones((5,5),np.uint8)
        canny_edges = cv2.dilate(canny_edges, kernel, iterations=1)
    # Get coordinates of high-frequency components (using Canny edges)
    high_freq_coords = np.where(canny_edges > 0)
    coords_list = list(zip(high_freq_coords[1], high_freq_coords[0]))
    
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


def test_thresholds(image_path):
    """
    Test different threshold combinations and display results alongside the original image.
    """
    # Read and convert original image
    original = cv2.imread(image_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Test cases for different scenarios
    params = [
        {'name': 'Default', 'low': 100, 'high': 200, 'blur': 0},
        {'name': 'Fine Details', 'low': 10, 'high': 70, 'blur': 0},
        {'name': 'Noise Reduction', 'low': 120, 'high': 240, 'blur': 1},
        {'name': 'Weak Edges', 'low': 30, 'high': 90, 'blur': 0},
        {'name': 'Strong Edges Only', 'low': 0, 'high': 10, 'blur': 1},
    ]
    
    # Create figure with subplots
    plt.figure(figsize=(20, 12))
    
    # Plot original image
    plt.subplot(2, 3, 1)
    plt.imshow(original_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot edge detection results
    for i, param in enumerate(params, 2):
        _, _, _, edges = detect_high_frequency(
            image_path, 
            canny_low=param['low'],
            canny_high=param['high'],
            blur_sigma=param['blur']
        )
        
        plt.subplot(2, 3, i)
        plt.imshow(edges, cmap='gray')
        plt.title(f"{param['name']}\nLow={param['low']}, High={param['high']}, Blur={param['blur']}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Also show a combined view of original + one edge detection result
    plt.figure(figsize=(20, 10))
    
    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(original_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    # Default edge detection
    _, _, _, edges = detect_high_frequency(
        image_path,
        canny_low=100,
        canny_high=200,
        blur_sigma=0
    )
    plt.subplot(1, 3, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Default Edge Detection\nLow=100, High=200, Blur=0')
    plt.axis('off')
    
    # Combined view
    combined = cv2.addWeighted(
        original_rgb,
        0.7,
        cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB),
        0.3,
        0
    )
    plt.subplot(1, 3, 3)
    plt.imshow(combined)
    plt.title('Combined View\n(Original + Default Edge Detection)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    image_path = "circles4/circle_000.png"
    coords, magnitude_img, gradient_img, canny_edges = detect_high_frequency(image_path, threshold=100, dilate = True)

    #dilate the canny_edges

    visualize_frequency_analysis(image_path, coords, magnitude_img, gradient_img, canny_edges)
    #test_thresholds(image_path)

    print(type(coords))
    print(len(coords))
    print(type(coords[0]))
    print(coords[0])