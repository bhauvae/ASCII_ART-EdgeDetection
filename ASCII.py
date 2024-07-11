import cv2
import numpy as np
import os

# Define the grayscale characters from darkest to lightest
GRAY_SCALE = " .;coP0?@█"
EDGES_GRAY_SCALE = r" |_/\\"
GRAY_SCALE_LOOKUP = np.array([i for i in range(10) for _ in range(26)], dtype=np.uint8)[
    4:260
]
FILL_IMAGES = np.array(
    [
        cv2.imread(os.path.join("./fillchar", f"{i}.png"), cv2.IMREAD_GRAYSCALE)
        for i in range(len(GRAY_SCALE))
    ]
)
EDGE_IMAGES = np.array(
    [
        cv2.imread(os.path.join("./edgechar", f"{i}.png"), cv2.IMREAD_GRAYSCALE)
        for i in range(5)
    ]
)
TEXT_RESOLUTION = 8

SIGMA_DOG = 3  # Sigm1 1 for dog
SIGMA_FACTOR = 1.6  # Sigma2 = Sigma_factor * Sigma_Dog
KERNEL_FACTOR_DOG = 6  # Kernel size = Kernel_FACTOR_DOG * Sigma_Dog
CLAHE_CLIP_LIMIT = 2
DOG_CONTRAST_THRESH = 25
KERNEL_SIZE_SOBEL = 7
SOBEL_HIST_THRESHOLD = 5
EDGE_ANGLE_RANGE = 10


def create_ascii_fill_image(image):
    image_h, image_w = image.shape

    # Calculate the new dimensions
    reduced_w = image_w // TEXT_RESOLUTION
    reduced_h = image_h // TEXT_RESOLUTION

    # Maintain aspect ratio
    aspect_ratio = image_w / image_h
    if aspect_ratio > 1:
        reduced_w = reduced_w
        reduced_h = int(reduced_w / aspect_ratio)
    else:
        reduced_h = reduced_h
        reduced_w = int(reduced_h * aspect_ratio)

    # Resize the image
    image = cv2.resize(image, (reduced_w, reduced_h), interpolation=cv2.INTER_AREA)
    # Create a blank black image
    ascii_fill_img = np.zeros(
        (reduced_h * TEXT_RESOLUTION, reduced_w * TEXT_RESOLUTION), dtype=np.uint8
    )

    # Get character indices based on pixel intensities
    char_indices = GRAY_SCALE_LOOKUP[image]

    # Place character images in the ascii_fill_img
    for i in range(len(FILL_IMAGES)):
        # Get mask where the char_index matches the current index
        mask = char_indices == i

        if not np.any(mask):
            continue

        # Get the corresponding character image
        char_img = FILL_IMAGES[i]

        # Place the block in the ascii_fill_img
        y_coords, x_coords = np.where(mask)
        for y, x in zip(y_coords, x_coords):
            start_y, start_x = y * TEXT_RESOLUTION, x * TEXT_RESOLUTION
            end_y, end_x = start_y + TEXT_RESOLUTION, start_x + TEXT_RESOLUTION
            ascii_fill_img[start_y:end_y, start_x:end_x] = char_img

    return ascii_fill_img


def diff_of_gaussiian(
    image, apply_normalize=True, apply_clahe=True, apply_contrastthresh=True
):

    # Set sigma values
    sigma1 = SIGMA_DOG
    sigma2 = (
        SIGMA_FACTOR * sigma1
    )  # Larger sigma is typically 1.6 times the smaller sigma

    # Calculate kernel sizes (make sure they are odd)
    k_factor = 6
    k_size1 = (
        int(k_factor * sigma1) | 1
    )  # Bitwise OR with 1 ensures the kernel size is odd
    k_size2 = int(k_factor * sigma2) | 1

    # Apply GaussianBlur at two different scales
    blur1 = cv2.GaussianBlur(image, (k_size1, k_size1), sigma1)
    blur2 = cv2.GaussianBlur(image, (k_size2, k_size2), sigma2)

    # Compute the Difference of Gaussians
    dog = cv2.subtract(blur1, blur2)

    if apply_normalize:
        # Normalize the DoG image to the range [0, 255]
        dog_normalized = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
        dog = np.uint8(dog_normalized)

    if apply_clahe:
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
        dog = clahe.apply(dog_normalized)

    if apply_contrastthresh:
        thresh = DOG_CONTRAST_THRESH  # Define the threshold value
        _, dog = cv2.threshold(dog, thresh, 255, cv2.THRESH_BINARY)

    # Display the results

    return dog


def sobel_filter(image):

    # Apply Sobel filter in x and y directions
    ksize = int(KERNEL_SIZE_SOBEL) | 1
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)

    # Calculate magnitude and angle of gradients
    sobel_magnitude = cv2.magnitude(sobelx, sobely)
    sobel_angle = np.rad2deg(np.arctan2(sobely, sobelx))

    return sobel_magnitude, sobel_angle


def create_ascii_edge_image(image, sobel_angle, sobel_magnitude):

    sobel_angle = np.where(sobel_magnitude != 0, sobel_angle, np.nan)
    image_h, image_w = image.shape

    # Calculate the new dimensions
    reduced_w = image_w // TEXT_RESOLUTION
    reduced_h = image_h // TEXT_RESOLUTION

    # Maintain aspect ratio
    aspect_ratio = image_w / image_h
    if aspect_ratio > 1:
        reduced_w = reduced_w
        reduced_h = int(reduced_w / aspect_ratio)
    else:
        reduced_h = reduced_h
        reduced_w = int(reduced_h * aspect_ratio)

    # Function to determine edge direction based on angle
    def direction_threshold(theta):
        if np.isnan(theta):
            return 0

        angle_factor = EDGE_ANGLE_RANGE
        abstheta = np.abs(theta)
        if (
            0.0 <= abstheta < 0.0 + angle_factor
            or 180 - angle_factor <= abstheta <= 180.0
        ):
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
    sobel_hist_threshold = SOBEL_HIST_THRESHOLD

    def most_frequent_direction(grid):
        hist, _ = np.histogram(grid, bins=np.arange(6))  # bins=[0, 1, 2, 3, 4, 5]

        if hist.sum() > sobel_hist_threshold:

            return np.argmax(hist)  # Matching max frequency with edge
        else:
            return 0

    # Create the ASCII art image
    ascii_img = np.zeros((image_h, image_w), dtype=np.uint8)
    for y in range(reduced_h):
        for x in range(reduced_w):
            start_y = y * TEXT_RESOLUTION
            end_y = (y + 1) * TEXT_RESOLUTION
            start_x = x * TEXT_RESOLUTION
            end_x = (x + 1) * TEXT_RESOLUTION

            grid = edge_value[start_y:end_y, start_x:end_x]
            direction = most_frequent_direction(grid)

            char_index = direction
            if char_index < len(EDGE_IMAGES):  # Ensure index is within range
                char_img = EDGE_IMAGES[char_index]
                if char_img is not None and char_img.shape == (
                    TEXT_RESOLUTION,
                    TEXT_RESOLUTION,
                ):
                    ascii_img[start_y:end_y, start_x:end_x] = char_img

    # Save the ASCII art image
    return ascii_img


def combine_fill_edges(image, ascii_fill_img, ascii_edge_img):

    image_w, image_h = image.shape
    for y in range(image_h // TEXT_RESOLUTION):
        for x in range(image_w // TEXT_RESOLUTION):
            start_y = y * TEXT_RESOLUTION
            end_y = (y + 1) * TEXT_RESOLUTION
            start_x = x * TEXT_RESOLUTION
            end_x = (x + 1) * TEXT_RESOLUTION

            if np.all(ascii_edge_img[start_y:end_y, start_x:end_x] == 0):
                continue
            else:
                ascii_fill_img[start_y:end_y, start_x:end_x] = ascii_edge_img[
                    start_y:end_y, start_x:end_x
                ]

    return ascii_fill_img


def main(filename):

    output_dir = "./ascii_image"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    # Force the output filename to have a .png extension
    output_ascii_filename = os.path.join(
        output_dir, os.path.splitext(os.path.basename(filename))[0] + ".png"
    )

    # Open the image and convert to grayscale
    input_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    if input_image is None:
        raise ValueError(f"Could not open or find the image '{filename}'")

    # Create ASCII art image
    ascii_fill_img = create_ascii_fill_image(input_image)

    dog = diff_of_gaussiian(input_image)
    sobel_magnitude, sobel_angle = sobel_filter(dog)

    ascii_edge_img = create_ascii_edge_image(input_image, sobel_angle, sobel_magnitude)

    ascii_img = combine_fill_edges(input_image, ascii_fill_img, ascii_edge_img)
    # Save the ASCII art image
    cv2.imwrite(output_ascii_filename, ascii_img)
    print(f"ASCII art image saved as '{output_ascii_filename}'")


if __name__ == "__main__":
    main(filename="images\kh.jpg")
