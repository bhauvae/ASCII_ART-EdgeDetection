import cupy as cp
import cv2
import os

# Define the grayscale characters from darkest to lightest
GRAY_SCALE = " .;coP0?@█"
EDGES_GRAY_SCALE = r" |_/\\"
GRAY_SCALE_LOOKUP = cp.array([i for i in range(10) for _ in range(26)], dtype=cp.uint8)[4:260]
TEXT_RESOLUTION = 8

SIGMA_DOG = 3  # Sigma1 for DoG
SIGMA_FACTOR = 1.6  # Sigma2 = Sigma_factor * Sigma_Dog
KERNEL_FACTOR_DOG = 6  # Kernel size = Kernel_FACTOR_DOG * Sigma_Dog
CLAHE_CLIP_LIMIT = 2
DOG_CONTRAST_THRESH = 25
KERNEL_SIZE_SOBEL = 7
SOBEL_HIST_THRESHOLD = 5
EDGE_ANGLE_RANGE = 10

def load_images(directory, count):
    images = []
    for i in range(count):
        image = cv2.imread(os.path.join(directory, f"{i}.png"), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            images.append(cp.asarray(image))
    return images

# Load images to GPU
FILL_IMAGES = load_images("./fillchar", len(GRAY_SCALE))
EDGE_IMAGES = load_images("./edgechar", 5)

def create_ascii_fill_image(image):
    image_cp = cp.asarray(image)
    image_h, image_w = image_cp.shape

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
    image_resized = cp.array(cv2.resize(cp.asnumpy(image_cp), (reduced_w, reduced_h), interpolation=cv2.INTER_AREA))

    # Create a blank black image
    ascii_fill_img = cp.zeros(
        (reduced_h * TEXT_RESOLUTION, reduced_w * TEXT_RESOLUTION), dtype=cp.uint8
    )

    # Get character indices based on pixel intensities
    char_indices = GRAY_SCALE_LOOKUP[image_resized]

    # Place character images in the ascii_fill_img
    for i in range(len(FILL_IMAGES)):
        mask = char_indices == i

        if not cp.any(mask):
            continue

        # Get the corresponding character image
        char_img = FILL_IMAGES[i]

        # Place the block in the ascii_fill_img
        y_coords, x_coords = cp.where(mask)
        for y, x in zip(y_coords.get(), x_coords.get()):
            start_y, start_x = y * TEXT_RESOLUTION, x * TEXT_RESOLUTION
            end_y, end_x = start_y + TEXT_RESOLUTION, start_x + TEXT_RESOLUTION
            ascii_fill_img[start_y:end_y, start_x:end_x] = char_img

    return ascii_fill_img

def diff_of_gaussian(image, apply_normalize=False, apply_clahe=True, apply_contrastthresh=True):
    image_cp = cp.asarray(image)
    
    # Set sigma values
    sigma1 = SIGMA_DOG
    sigma2 = SIGMA_FACTOR * sigma1

    # Calculate kernel sizes (make sure they are odd)
    k_size1 = int(KERNEL_FACTOR_DOG * sigma1) | 1
    k_size2 = int(KERNEL_FACTOR_DOG * sigma2) | 1

    # Apply GaussianBlur at two different scales
    blur1 = cp.array(cv2.GaussianBlur(cp.asnumpy(image_cp), (k_size1, k_size1), sigma1))
    blur2 = cp.array(cv2.GaussianBlur(cp.asnumpy(image_cp), (k_size2, k_size2), sigma2))

    # Compute the Difference of Gaussians
    dog = cp.subtract(blur1, blur2)

    if apply_normalize:
        # Normalize the DoG image to the range [0, 255]
        dog_normalized = cp.clip(cp.interp(dog, (dog.min(), dog.max()), (0, 255)), 0, 255)
        dog = cp.uint8(dog_normalized)

    if apply_clahe:
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
        dog = cp.array(clahe.apply(cp.asnumpy(dog)))

    if apply_contrastthresh:
        thresh = DOG_CONTRAST_THRESH
        _, dog = cv2.threshold(cp.asnumpy(dog), thresh, 255, cv2.THRESH_BINARY)
        dog = cp.array(dog)

    return dog

def sobel_filter(image):
    image_cp = cp.asarray(image)

    # Apply Sobel filter in x and y directions
    ksize = int(KERNEL_SIZE_SOBEL) | 1
    sobelx = cp.array(cv2.Sobel(cp.asnumpy(image_cp), cv2.CV_64F, 1, 0, ksize=ksize))
    sobely = cp.array(cv2.Sobel(cp.asnumpy(image_cp), cv2.CV_64F, 0, 1, ksize=ksize))

    # Calculate magnitude and angle of gradients
    sobel_magnitude = cp.sqrt(sobelx ** 2 + sobely ** 2)
    sobel_angle = cp.rad2deg(cp.arctan2(sobely, sobelx))

    return sobel_magnitude, sobel_angle

def create_ascii_edge_image(image, sobel_angle, sobel_magnitude):
    sobel_angle = cp.where(sobel_magnitude != 0, sobel_angle, cp.nan)
    image_cp = cp.asarray(image)
    image_h, image_w = image_cp.shape

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

    def direction_threshold(theta):
        if cp.isnan(theta).any():  # Handle NaN values
            return cp.zeros_like(theta)

        angle_factor = EDGE_ANGLE_RANGE
        abstheta = cp.abs(theta)

        # Initialize output with default value (e.g., 0)
        result = cp.zeros_like(theta, dtype=cp.int32)

        # Define conditions using CuPy's element-wise operations
        is_vertical_low = (0.0 <= abstheta) & (abstheta < 0.0 + angle_factor)
        is_vertical_high = (180 - angle_factor <= abstheta) & (abstheta <= 180.0)
        is_horizontal = (90 - angle_factor <= abstheta) & (abstheta < 90 + angle_factor)
        is_diagonal1 = (0.0 + angle_factor <= abstheta) & (abstheta < 90 - angle_factor)
        is_diagonal2 = (90 + angle_factor < abstheta) & (abstheta < 180 - angle_factor)

        result[is_vertical_low | is_vertical_high] = 1  # VERTICAL
        result[is_horizontal] = 2  # HORIZONTAL
        result[is_diagonal1] = cp.where(cp.sign(theta[is_diagonal1]) > 0, 3, 4)  # DIAGONAL 1
        result[is_diagonal2] = cp.where(cp.sign(theta[is_diagonal2]) > 0, 4, 3)  # DIAGONAL 2

        return result

    vectorized_threshold = cp.vectorize(direction_threshold)
    edge_value = vectorized_threshold(sobel_angle)

    def most_frequent_direction(grid):
        hist, _ = cp.histogram(grid, bins=cp.arange(6))

        if cp.sum(hist) > SOBEL_HIST_THRESHOLD:
            return cp.argmax(hist)
        else:
            return 0

    ascii_img = cp.zeros((image_h, image_w), dtype=cp.uint8)
    for y in range(reduced_h):
        for x in range(reduced_w):
            start_y = y * TEXT_RESOLUTION
            end_y = (y + 1) * TEXT_RESOLUTION
            start_x = x * TEXT_RESOLUTION
            end_x = (x + 1) * TEXT_RESOLUTION

            grid = edge_value[start_y:end_y, start_x:end_x]
            direction = most_frequent_direction(grid)

            char_index = direction
            if char_index < len(EDGE_IMAGES):
                char_img = EDGE_IMAGES[char_index]
                if char_img is not None and char_img.shape == (TEXT_RESOLUTION, TEXT_RESOLUTION):
                    ascii_img[start_y:end_y, start_x:end_x] = char_img

    return ascii_img

def combine_fill_edges(image, ascii_fill_img, ascii_edge_img):
    image_w, image_h = image.shape
    for y in range(image_h // TEXT_RESOLUTION):
        for x in range(image_w // TEXT_RESOLUTION):
            start_y = y * TEXT_RESOLUTION
            end_y = (y + 1) * TEXT_RESOLUTION
            start_x = x * TEXT_RESOLUTION
            end_x = (x + 1) * TEXT_RESOLUTION

            if cp.all(ascii_edge_img[start_y:end_y, start_x:end_x] == 0):
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

    dog = diff_of_gaussian(input_image)
    sobel_magnitude, sobel_angle = sobel_filter(dog)

    ascii_edge_img = create_ascii_edge_image(input_image, sobel_angle, sobel_magnitude)

    ascii_img = combine_fill_edges(input_image, ascii_fill_img, ascii_edge_img)

    # Save the ASCII art image
    cv2.imwrite(output_ascii_filename, cp.asnumpy(ascii_img))
    print(f"ASCII art image saved as '{output_ascii_filename}'")

if __name__ == "__main__":
    main(filename="./pexels-athul-k-anand-396770270-15105813.jpg")
