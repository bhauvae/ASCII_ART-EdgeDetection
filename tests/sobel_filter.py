import cv2
import numpy as np
import matplotlib.pyplot as plt
# Define ASCII characters to represent edges
EDGES_GRAY_SCALE = r" |_/\\"

# Load text images for different edge directions
TEXT_IMAGES = [
    cv2.imread(f"./edgechar/{i}.png", cv2.IMREAD_GRAYSCALE) for i in range(5)
]
TEXT_RESOLUTION = 8  # Size of each ASCII character image

# Load the image
filename = "circle.png"
image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError(f"Could not open or find the image '{filename}'")

# Resize the image to a smaller size for processing
image_w, image_h = image.shape[1], image.shape[0]

# Apply Sobel filter in x and y directions
ksize = 7
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)

# Calculate magnitude and angle of gradients
sobel_magnitude = cv2.magnitude(sobelx, sobely)
sobel_angle = np.arctan2(sobely, sobelx) * (180 / np.pi)
# plt.hist(sobel_angle)
# plt.show()
sobel_angle = np.where(sobel_magnitude != 0, sobel_angle, np.nan)


# Function to determine edge direction based on angle
def direction_threshold(theta):
    if np.isnan(theta):
        return 0

    angle_factor = 5
    abstheta = np.abs(theta)
    if 0.0 <= abstheta < 0.0 + angle_factor or 180 - angle_factor <= abstheta <= 180.0:
        return 1  # VERTICAL
    elif 90 - angle_factor <= abstheta < 90 + angle_factor:
        return 2  # HORIZONTAL
    elif 0.0 + angle_factor <= abstheta < 90 - angle_factor:
        return 3 if np.sign(theta) > 0 else 4  # DIAGONAL 1
    elif 90 + angle_factor < abstheta < 180 - angle_factor:
        return 4 if np.sign(theta) > 0 else 3  # DIAGONAL 2


# Vectorize the threshold function for efficient element-wise processing
vectorized_threshold = np.vectorize(direction_threshold)
edge_value = vectorized_threshold(sobel_angle)

# Function to calculate the most frequent direction in an 8x8 grid
HIST_THRESH = 5


def most_frequent_direction(grid):
    hist, _ = np.histogram(grid, bins=np.arange(6))  # bins=[1, 2, 3, 4, 5]

    if hist.sum() > HIST_THRESH:

        return np.argmax(hist) # Matching max frequency with edge
    else:
        return 0


# Initialize the output array
output_array = np.zeros((image_h // 8, image_w // 8), dtype=np.uint8)

# Calculate the histogram for each 8x8 grid and set the most frequent direction
for i in range(0, image_h, 8):
    for j in range(0, image_w, 8):
        grid = edge_value[i : i + 8, j : j + 8]
        direction = most_frequent_direction(grid)
        output_array[i // 8, j // 8] = direction

# Create the ASCII art image
ascii_img = np.zeros((image_h, image_w), dtype=np.uint8)
for y in range(image_h // 8):
    for x in range(image_w // 8):
        start_y = y * TEXT_RESOLUTION
        end_y = (y + 1) * TEXT_RESOLUTION
        start_x = x * TEXT_RESOLUTION
        end_x = (x + 1) * TEXT_RESOLUTION

        char_index = output_array[y, x]
        if char_index < len(TEXT_IMAGES):  # Ensure index is within range
            char_img = TEXT_IMAGES[char_index]
            if char_img is not None and char_img.shape == (
                TEXT_RESOLUTION,
                TEXT_RESOLUTION,
            ):
                ascii_img[start_y:end_y, start_x:end_x] = char_img

# Save the ASCII art image
cv2.imwrite("ascii_circle.png", ascii_img)
