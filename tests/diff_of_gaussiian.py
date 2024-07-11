import cv2
import numpy as np

TEXT_RESOLUTION = 8
filename = "circle.png"
# Load the image
input_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
if input_image is None:
    raise ValueError(f"Could not open or find the image '{filename}'")

original_h, original_w = input_image.shape

# Calculate the new dimensions
new_w = original_w // TEXT_RESOLUTION
new_h = original_h // TEXT_RESOLUTION

# Resize the image
image = cv2.resize(input_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

# Set sigma values
sigma1 = 3
sigma2 = 1.6 * sigma1  # Larger sigma is typically 1.6 times the smaller sigma

# Calculate kernel sizes (make sure they are odd)
k_factor = 6
k_size1 = int(k_factor * sigma1) | 1  # Bitwise OR with 1 ensures the kernel size is odd
k_size2 = int(k_factor * sigma2) | 1

# Apply GaussianBlur at two different scales
blur1 = cv2.GaussianBlur(image, (k_size1, k_size1), sigma1)
blur2 = cv2.GaussianBlur(image, (k_size2, k_size2), sigma2)

# Compute the Difference of Gaussians
dog = cv2.subtract(blur1, blur2)

# Normalize the DoG image to the range [0, 255]
dog_normalized = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
dog = np.uint8(dog_normalized)

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
dog_clahe = clahe.apply(dog_normalized)

thresh = 25  # Define the threshold value
_, dog = cv2.threshold(dog, thresh, 255, cv2.THRESH_BINARY)

# Display the results

cv2.imwrite("circle_dog.png", dog)