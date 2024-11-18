import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_high_frequency(image_path, threshold=100):
    """
    Detect high-frequency components in an image using multiple methods.
    
    Args:
        image_path (str): Path to the input image
        threshold (int): Threshold for high-frequency detection (0-255)
    
    Returns:
        tuple: (edge coordinates, frequency magnitude image)
    """
    # Read the image
    img = cv2.imread(image_path)
    height, width, channels = img.shape  # for color images
    # Or just height, width = img.shape for grayscale images

    print(f"Image dimensions:")
    print(f"Height: {height} pixels")
    print(f"Width: {width} pixels")
    print(f"Total pixels: {height * width}")
    print(f"Number of channels: {channels}")  # Usually 3 for BGR images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Fourier Transform
    # Apply FFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    
    # Normalize magnitude for visualization
    magnitude_log = np.log1p(magnitude)
    magnitude_normalized = cv2.normalize(magnitude_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Method 2: Sobel Edge Detection
    # Calculate gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize gradient magnitude
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Get coordinates of high-frequency components
    high_freq_coords = np.where(gradient_magnitude > threshold)
    coords_list = list(zip(high_freq_coords[1], high_freq_coords[0]))  # (x,y) format
    
    return coords_list, magnitude_normalized, gradient_magnitude

def visualize_results(img_path, coords, magnitude_img, gradient_img):
    """
    Visualize the results of high-frequency detection.
    """
    # Read original image
    original = cv2.imread(img_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Create visualization of high-frequency points
    high_freq_vis = original.copy()
    for x, y in coords:
        cv2.circle(high_freq_vis, (x, y), 1, (0, 255, 0), -1)
    high_freq_vis = cv2.cvtColor(high_freq_vis, cv2.COLOR_BGR2RGB)
    
    # Plotting
    plt.figure(figsize=(15, 10))
    
    plt.subplot(221)
    plt.imshow(original_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(222)
    plt.imshow(magnitude_img, cmap='gray')
    plt.title('Frequency Magnitude (FFT)')
    plt.axis('off')
    
    plt.subplot(223)
    plt.imshow(gradient_img, cmap='gray')
    plt.title('Gradient Magnitude')
    plt.axis('off')
    
    plt.subplot(224)
    plt.imshow(high_freq_vis)
    plt.title('High Frequency Points')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    image_path = "circles4/circle_000.png"
    coords, magnitude_img, gradient_img = detect_high_frequency(image_path, threshold=100)
    visualize_results(image_path, coords, magnitude_img, gradient_img)

    print(type(coords))
    print(len(coords))
    print(type(coords[0]))
    print(coords[0])