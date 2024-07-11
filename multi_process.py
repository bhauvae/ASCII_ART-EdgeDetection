import cv2
import numpy as np
import os
from multiprocessing import Pool, cpu_count
import time

# Define the grayscale characters from darkest to lightest
GRAY_SCALE = " .;coP0?@█"
GRAY_SCALE_LOOKUP = np.array([i for i in range(10) for _ in range(26)], dtype=np.uint8)[4:260]
TEXT_RESOLUTION = 8

SIGMA_DOG = 3  # Sigma1 for DoG
SIGMA_FACTOR = 1.6  # Sigma2 = Sigma_factor * Sigma_Dog
KERNEL_FACTOR_DOG = 6  # Kernel size = Kernel_FACTOR_DOG * Sigma_Dog
CLAHE_CLIP_LIMIT = 2
DOG_CONTRAST_THRESH = 25
KERNEL_SIZE_SOBEL = 7
SOBEL_HIST_THRESHOLD = 5
EDGE_ANGLE_RANGE = 10

# Load fill and edge images globally to avoid repetitive I/O
FILL_IMAGES = np.array([
    cv2.imread(os.path.join("./fillchar", f"{i}.png"), cv2.IMREAD_GRAYSCALE)
    for i in range(len(GRAY_SCALE))
])
EDGE_IMAGES = np.array([
    cv2.imread(os.path.join("./edgechar", f"{i}.png"), cv2.IMREAD_GRAYSCALE)
    for i in range(5)
])


def create_ascii_fill_image(image):
    image_h, image_w = image.shape

    reduced_w = image_w // TEXT_RESOLUTION
    reduced_h = image_h // TEXT_RESOLUTION

    aspect_ratio = image_w / image_h
    if aspect_ratio > 1:
        reduced_h = int(reduced_w / aspect_ratio)
    else:
        reduced_w = int(reduced_h * aspect_ratio)

    image_resized = cv2.resize(image, (reduced_w, reduced_h), interpolation=cv2.INTER_AREA)
    ascii_fill_img = np.zeros((reduced_h * TEXT_RESOLUTION, reduced_w * TEXT_RESOLUTION), dtype=np.uint8)

    char_indices = GRAY_SCALE_LOOKUP[image_resized]
    for i, char_img in enumerate(FILL_IMAGES):
        mask = char_indices == i
        if np.any(mask):
            for y, x in zip(*np.where(mask)):
                start_y, start_x = y * TEXT_RESOLUTION, x * TEXT_RESOLUTION
                ascii_fill_img[start_y:start_y + TEXT_RESOLUTION, start_x:start_x + TEXT_RESOLUTION] = char_img

    return ascii_fill_img


def diff_of_gaussiian(image, apply_normalize=True, apply_clahe=True, apply_contrastthresh=True):
    sigma1 = SIGMA_DOG
    sigma2 = SIGMA_FACTOR * sigma1

    k_size1 = int(KERNEL_FACTOR_DOG * sigma1) | 1
    k_size2 = int(KERNEL_FACTOR_DOG * sigma2) | 1

    blur1 = cv2.GaussianBlur(image, (k_size1, k_size1), sigma1)
    blur2 = cv2.GaussianBlur(image, (k_size2, k_size2), sigma2)

    dog = cv2.subtract(blur1, blur2)

    if apply_normalize:
        dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)

    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
        dog = clahe.apply(dog)

    if apply_contrastthresh:
        _, dog = cv2.threshold(dog, DOG_CONTRAST_THRESH, 255, cv2.THRESH_BINARY)

    return dog


def sobel_filter(image):
    ksize = int(KERNEL_SIZE_SOBEL) | 1
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)

    sobel_magnitude = cv2.magnitude(sobelx, sobely)
    sobel_angle = np.rad2deg(np.arctan2(sobely, sobelx))

    return sobel_magnitude, sobel_angle


def create_ascii_edge_image(image, sobel_angle, sobel_magnitude):
    sobel_angle = np.where(sobel_magnitude != 0, sobel_angle, np.nan)
    image_h, image_w = image.shape

    reduced_w = image_w // TEXT_RESOLUTION
    reduced_h = image_h // TEXT_RESOLUTION

    aspect_ratio = image_w / image_h
    if aspect_ratio > 1:
        reduced_h = int(reduced_w / aspect_ratio)
    else:
        reduced_w = int(reduced_h * aspect_ratio)

    def direction_threshold(theta):
        if np.isnan(theta):
            return 0

        abstheta = np.abs(theta)
        if 0.0 <= abstheta < 0.0 + EDGE_ANGLE_RANGE or 180 - EDGE_ANGLE_RANGE <= abstheta <= 180.0:
            return 1  # VERTICAL
        elif 90 - EDGE_ANGLE_RANGE <= abstheta < 90 + EDGE_ANGLE_RANGE:
            return 2  # HORIZONTAL
        elif 0.0 + EDGE_ANGLE_RANGE <= abstheta < 90 - EDGE_ANGLE_RANGE:
            return 3 if np.sign(theta) > 0 else 4  # DIAGONAL 1
        elif 90 + EDGE_ANGLE_RANGE < abstheta < 180 - EDGE_ANGLE_RANGE:
            return 4 if np.sign(theta) > 0 else 3  # DIAGONAL 2

    vectorized_threshold = np.vectorize(direction_threshold)
    edge_value = vectorized_threshold(sobel_angle)

    def most_frequent_direction(grid):
        hist, _ = np.histogram(grid, bins=np.arange(6))
        return np.argmax(hist) if hist.sum() > SOBEL_HIST_THRESHOLD else 0

    ascii_img = np.zeros((image_h, image_w), dtype=np.uint8)
    for y in range(reduced_h):
        for x in range(reduced_w):
            start_y = y * TEXT_RESOLUTION
            start_x = x * TEXT_RESOLUTION
            grid = edge_value[start_y:start_y + TEXT_RESOLUTION, start_x:start_x + TEXT_RESOLUTION]
            char_index = most_frequent_direction(grid)
            if char_index < len(EDGE_IMAGES) and not np.all(grid == 0):
                ascii_img[start_y:start_y + TEXT_RESOLUTION, start_x:start_x + TEXT_RESOLUTION] = EDGE_IMAGES[char_index]

    return ascii_img


def combine_fill_edges(image, ascii_fill_img, ascii_edge_img):
    image_w, image_h = image.shape
    for y in range(image_h // TEXT_RESOLUTION):
        for x in range(image_w // TEXT_RESOLUTION):
            start_y = y * TEXT_RESOLUTION
            end_y = (y + 1) * TEXT_RESOLUTION
            start_x = x * TEXT_RESOLUTION
            end_x = (x + 1) * TEXT_RESOLUTION
            if not np.all(ascii_edge_img[start_y:end_y, start_x:end_x] == 0):
                ascii_fill_img[start_y:end_y, start_x:end_x] = ascii_edge_img[start_y:end_y, start_x:end_x]
    return ascii_fill_img


def process_image(filename):
    input_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        raise ValueError(f"Could not open or find the image '{filename}'")

    start_time = time.time()

    ascii_fill_img = create_ascii_fill_image(input_image)
    dog = diff_of_gaussiian(input_image)
    sobel_magnitude, sobel_angle = sobel_filter(dog)
    ascii_edge_img = create_ascii_edge_image(input_image, sobel_angle, sobel_magnitude)
    ascii_img = combine_fill_edges(input_image, ascii_fill_img, ascii_edge_img)

    end_time = time.time()
    print(f"Processing time for '{filename}': {end_time - start_time:.2f} seconds")

    output_dir = "./ascii_image"
    os.makedirs(output_dir, exist_ok=True)
    output_ascii_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(filename))[0] + ".png")
    cv2.imwrite(output_ascii_filename, ascii_img)
    print(f"ASCII art image saved as '{output_ascii_filename}'")


if __name__ == "__main__":
    filenames = ["images\pexels-anya-juarez-tenorio-227888521-12295663.jpg"]  # Add more filenames as needed

    with Pool(cpu_count()) as pool:
        pool.map(process_image, filenames)
