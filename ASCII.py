import cv2
import numpy as np
import os
import time
from PIL import Image, ImageSequence
import subprocess
import shutil
import tempfile
import glob
import multiprocessing


def ascii_filter(filename, **kwargs):
    # Set default values for the function arguments
    max_dimension = kwargs.get("max_dimension", 1920)
    text_resolution = kwargs.get("text_resolution", 8)
    get_fill = kwargs.get("get_fill", True)
    get_edges = kwargs.get("get_edges", True)
    sigma_dog = kwargs.get("sigma_dog", 3)
    sigma_factor = kwargs.get("sigma_factor", 1.6)
    kernel_factor_dog = kwargs.get("kernel_factor_dog", 6)
    tau_dog = kwargs.get("tau_dog", 0)
    clahe_clip_limit = kwargs.get("clahe_clip_limit", 2)
    contrast_threshold_dog = kwargs.get("contrast_threshold_dog", 25)
    apply_normalize = kwargs.get("apply_normalize", True)
    apply_clahe = kwargs.get("apply_clahe", True)
    apply_threshold = kwargs.get("apply_threshold", True)
    kernel_size_sobel = kwargs.get("kernel_size_sobel", 7)
    apply_colour = kwargs.get("apply_colour", True)
    colour_map = kwargs.get("colour_map", "white-black")
    apply_bloom = kwargs.get("apply_bloom", True)
    bloom_blur_value = kwargs.get("bloom_blur_value", 10)
    bloom_gain = kwargs.get("bloom_gain", 1)
    apply_contrast = kwargs.get("apply_contrast", True)
    contrast_gamma = kwargs.get("contrast_gamma", 0.2)
    contrast_saturation = kwargs.get("contrast_saturation", 0.5)
    apply_sharpness = kwargs.get("apply_sharpness", True)
    kernel_size_sharpness = kwargs.get("kernel_size_sharpness", 5)
    sigma_sharpness = kwargs.get("sigma_sharpness", 1.0)
    amount_sharpness = kwargs.get("amount_sharpness", 1.0)
    threshold_sharpness = kwargs.get("threshold_sharpness", 0)
    only_dog = kwargs.get("only_dog", False)
    only_sobel = kwargs.get("only_sobel", False)

    # Open the image and convert to grayscale
    input_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        raise ValueError(f"Could not open or find the image '{filename}'")

    start_time = time.time()

    input_image = downscale_image(input_image, max_dimension)

    ascii_fill_img, ascii_edge_img = None, None

    if get_fill:
        # Create ASCII fill image
        ascii_fill_img = create_ascii_fill_image(input_image, text_resolution)
        image = ascii_fill_img

    if get_edges:
        dog = diff_of_gaussiian(
            input_image,
            sigma_dog,
            sigma_factor,
            kernel_factor_dog,
            tau_dog,
            clahe_clip_limit,
            contrast_threshold_dog,
            apply_normalize,
            apply_clahe,
            apply_threshold,
        )
        sobel_magnitude, sobel_angle = sobel_filter(dog, kernel_size_sobel)

        ascii_edge_img = create_ascii_edge_image(
            input_image, sobel_angle, sobel_magnitude, text_resolution
        )
        image = ascii_edge_img

    if not get_fill and not get_edges:
        if only_dog:
            image = diff_of_gaussiian(
                input_image,
                sigma_dog,
                sigma_factor,
                kernel_factor_dog,
                tau_dog,
                clahe_clip_limit,
                contrast_threshold_dog,
                apply_normalize,
                apply_clahe,
                apply_threshold,
            )
        elif only_sobel:
            image, _ = sobel_filter(input_image, kernel_size_sobel)

        else:
            image = diff_of_gaussiian(
                input_image,
                sigma_dog,
                sigma_factor,
                kernel_factor_dog,
                tau_dog,
                clahe_clip_limit,
                contrast_threshold_dog,
                apply_normalize,
                apply_clahe,
                apply_threshold,
            )
            image, _ = sobel_filter(image, kernel_size_sobel)

    if get_fill and get_edges:
        image = combine_fill_edges(
            ascii_fill_img=ascii_fill_img,
            ascii_edge_img=ascii_edge_img,
            text_resolution=text_resolution,
        )

    if apply_colour:
        image = add_color(image, colour_map)

    if apply_bloom:
        image = add_bloom(image, bloom_blur_value, bloom_gain)

    if apply_contrast:
        image = add_contrast(image, contrast_gamma, contrast_saturation)

    if apply_sharpness:
        image = add_sharpness(
            image,
            kernel_size_sharpness,
            sigma_sharpness,
            amount_sharpness,
            threshold_sharpness,
        )

    end_time = time.time()
    print(f"Processing time for '{'f'}': {end_time - start_time:.2f} seconds")

    return image


def create_ascii_fill_image(image, text_resolution):
    GRAY_SCALE = " .;oP0@█"

    gray_scale_lookup = np.array(
        [i for i in range(len(GRAY_SCALE)) for _ in range(int(256 // len(GRAY_SCALE)))],
        dtype=np.uint8,
    )[0:256]
    fill_images = np.array(
        [
            cv2.imread(os.path.join("./fillchar", f"{i}.png"), cv2.IMREAD_GRAYSCALE)
            for i in range(len(GRAY_SCALE))
        ]
    )

    image_h, image_w = image.shape

    # Calculate the new dimensions
    reduced_w = image_w // text_resolution
    reduced_h = image_h // text_resolution

    # Maintain aspect ratio
    aspect_ratio = image_w / image_h
    if aspect_ratio > 1:
        reduced_w = reduced_w
        reduced_h = int(reduced_w / aspect_ratio)
    else:
        reduced_h = reduced_h
        reduced_w = int(reduced_h * aspect_ratio)

    # Resize the image
    image = cv2.resize(
        image, (int(reduced_w), int(reduced_h)), interpolation=cv2.INTER_AREA
    )
    # Create a blank black image
    ascii_fill_img = np.zeros(
        (reduced_h * text_resolution, reduced_w * text_resolution), dtype=np.uint8
    )

    # Get character indices based on pixel intensities
    char_indices = gray_scale_lookup[image]

    # Place character images in the ascii_fill_img
    for i in range(len(fill_images)):
        # Get mask where the char_index matches the current index
        mask = char_indices == i

        if not np.any(mask):
            continue

        # Get the corresponding character image
        char_img = fill_images[i]

        # Place the block in the ascii_fill_img
        y_coords, x_coords = np.where(mask)
        for y, x in zip(y_coords, x_coords):
            start_y, start_x = y * text_resolution, x * text_resolution
            end_y, end_x = start_y + text_resolution, start_x + text_resolution
            ascii_fill_img[start_y:end_y, start_x:end_x] = char_img

    return ascii_fill_img


def diff_of_gaussiian(
    image,
    sigma_dog,
    sigma_factor,
    kernel_factor_dog,
    tau_dog,
    clahe_clip_limit,
    contrast_threshold_dog,
    apply_normalize,
    apply_clahe,
    apply_threshold,
):
    # Sigm1 1 for dog
    # Sigma2 = Sigma_factor * Sigma_Dog
    # Kernel size = Kernel_FACTOR_DOG * Sigma_Dog

    # Set sigma values
    sigma1 = sigma_dog
    sigma2 = (
        sigma_factor * sigma1
    )  # Larger sigma is typically 1.6 times the smaller sigma

    # Calculate kernel sizes (make sure they are odd)
    k_factor = kernel_factor_dog
    k_size1 = (
        int(k_factor * sigma1) | 1
    )  # Bitwise OR with 1 ensures the kernel size is odd
    k_size2 = int(k_factor * sigma2) | 1

    # Apply GaussianBlur at two different scales
    blur1 = cv2.GaussianBlur(image, (k_size1, k_size1), sigma1)
    blur2 = cv2.multiply(
        np.ones_like(image) * (1 + tau_dog),
        cv2.GaussianBlur(image, (k_size2, k_size2), sigma2),
    )

    # Compute the Difference of Gaussians
    dog = cv2.subtract(blur1, blur2)

    if apply_normalize:
        # Normalize the DoG image to the range [0, 255]
        dog_normalized = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
        dog = np.uint8(dog_normalized)

    if apply_clahe:
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
        dog = clahe.apply(dog)

    if apply_threshold:
        thresh = contrast_threshold_dog  # Define the threshold value
        dog = np.where(dog > thresh, 255.0, 0.0)  # Apply the threshold

    # Display the results

    return dog


def sobel_filter(image, kernel_size_sobel):

    # Apply Sobel filter in x and y directions
    ksize = int(kernel_size_sobel) | 1
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)

    # Calculate magnitude and angle of gradients
    sobel_magnitude = cv2.magnitude(sobelx, sobely)
    sobel_angle = np.rad2deg(np.arctan2(sobely, sobelx))

    return sobel_magnitude, sobel_angle


def create_ascii_edge_image(
    image,
    sobel_angle,
    sobel_magnitude,
    histogram_threshold_sobel=5,
    edge_angle_range=10,
    text_resolution=8,
):
    EDGES_GRAY_SCALE = r" |_/\\"

    edge_char_images = np.array(
        [
            cv2.imread(os.path.join("./edgechar", f"{i}.png"), cv2.IMREAD_GRAYSCALE)
            for i in range(5)
        ]
    )

    sobel_angle = np.where(sobel_magnitude != 0, sobel_angle, np.nan)
    image_h, image_w = image.shape

    # Calculate the new dimensions
    reduced_w = image_w // text_resolution
    reduced_h = image_h // text_resolution

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

        angle_tolerance = edge_angle_range
        abstheta = np.abs(theta)
        if (
            0.0 <= abstheta < 0.0 + angle_tolerance
            or 180 - angle_tolerance <= abstheta <= 180.0
        ):
            return 1  # VERTICAL
        elif 90 - angle_tolerance <= abstheta < 90 + angle_tolerance:
            return 2  # HORIZONTAL
        elif 0.0 + angle_tolerance <= abstheta < 90 - angle_tolerance:
            return 3 if np.sign(theta) > 0 else 4  # DIAGONAL 1
        elif 90 + angle_tolerance < abstheta < 180 - angle_tolerance:
            return 4 if np.sign(theta) > 0 else 3  # DIAGONAL 2

    # Vectorize the threshold function for efficient element-wise processing
    vectorized_threshold = np.vectorize(direction_threshold)
    edge_value = vectorized_threshold(sobel_angle)

    # Function to calculate the most frequent direction in an 8x8 grid

    def most_frequent_direction(grid):
        hist, _ = np.histogram(grid, bins=np.arange(6))  # bins=[0, 1, 2, 3, 4, 5]

        if hist.sum() > histogram_threshold_sobel:

            return np.argmax(hist)  # Matching max frequency with edge
        else:
            return 0

    # Create the ASCII art image
    ascii_img = np.zeros((image_h, image_w), dtype=np.uint8)
    for y in range(reduced_h):
        for x in range(reduced_w):
            start_y = y * text_resolution
            end_y = (y + 1) * text_resolution
            start_x = x * text_resolution
            end_x = (x + 1) * text_resolution

            grid = edge_value[start_y:end_y, start_x:end_x]
            direction = most_frequent_direction(grid)

            char_index = direction
            if char_index < len(edge_char_images):  # Ensure index is within range
                char_img = edge_char_images[char_index]
                if char_img is not None and char_img.shape == (
                    text_resolution,
                    text_resolution,
                ):
                    ascii_img[start_y:end_y, start_x:end_x] = char_img

    # Save the ASCII art image
    return ascii_img


def combine_fill_edges(ascii_fill_img, ascii_edge_img, text_resolution):

    image_w, image_h = ascii_fill_img.shape
    for y in range(image_h // text_resolution):
        for x in range(image_w // text_resolution):
            start_y = y * text_resolution
            end_y = (y + 1) * text_resolution
            start_x = x * text_resolution
            end_x = (x + 1) * text_resolution

            if np.all(ascii_edge_img[start_y:end_y, start_x:end_x] == 0):
                continue
            else:
                ascii_fill_img[start_y:end_y, start_x:end_x] = ascii_edge_img[
                    start_y:end_y, start_x:end_x
                ]

    return ascii_fill_img


def downscale_image(image, max_dim):

    # Resize the image while preserving aspect ratio
    height, width = image.shape
    if max(width, height) > max_dim:
        if width > height:
            new_width = max_dim
            new_height = int(height * max_dim / width)
        else:
            new_height = max_dim
            new_width = int(width * max_dim / height)
        image = cv2.resize(
            image, (int(new_width), int(new_height)), interpolation=cv2.INTER_AREA
        )

    return image


def add_color(image, color_map="purple-salmon"):

    colour_map_hex = {
        "black-white": ("#000000", "#FFFFFF"),
        "white-black": ("#FFFFFF", "#000000"),
        "blue-green": ("#0A174E", "#F5D042"),
        "hacker-man": ("#120d13", "#29ff2a"),
        "purple-salmon": ("#180a1c", "#e05964"),
        "darkblue-white": ("#1f1735", "#fffcff"),
    }
    # Define colors

    hex_to_bgr = lambda hex_color: tuple(
        int(hex_color[i : i + 2], 16) for i in (5, 3, 1)
    )

    # Create an RGB image array with the same height and width as the grayscale image
    color_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # Map black (0) to green and white (255) to pink
    color_image[image == 0] = hex_to_bgr(colour_map_hex[color_map][0])
    color_image[image == 255] = hex_to_bgr(colour_map_hex[color_map][1])

    return color_image


def add_bloom(image, bloom_blur_value, bloom_gain):

    blur_value = bloom_blur_value  # bloom smoothness 10s
    gain = bloom_gain  # bloom gain in intensity 1

    # blur
    blur = cv2.GaussianBlur(image, (0, 0), sigmaX=blur_value, sigmaY=blur_value)
    # blend blur and image using gain on blur
    result = cv2.addWeighted(image, 1, blur, gain, 0)

    return result


def add_contrast(image, contrast_gamma, contrast_saturation):

    # Convert the image to float32 for processing
    hdr_image = image.astype(np.float32) / 255.0

    # Create a tone mapping object (you can choose different algorithms)
    tonemap = cv2.createTonemapDrago(
        contrast_gamma, contrast_saturation
    )  # Parameters: gamma, saturation

    # Apply tone mapping
    ldr_tonemapped = tonemap.process(hdr_image)

    # Check for NaN values and replace them with 0
    ldr_tonemapped = np.nan_to_num(ldr_tonemapped)

    # Convert the tone-mapped image back to 8-bit format
    ldr_tonemapped = np.clip(ldr_tonemapped * 255, 0, 255).astype("uint8")

    return ldr_tonemapped


def add_sharpness(
    image, kernel_size_sharpness, sigma_sharpness, amount_sharpness, threshold_sharpness
):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(
        image,
        (kernel_size_sharpness, kernel_size_sharpness),
        sigma_sharpness,
    )
    sharpened = float(amount_sharpness + 1) * image - float(amount_sharpness) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold_sharpness > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold_sharpness
        np.copyto(sharpened, image, where=low_contrast_mask)

    return sharpened


def add_neon(image):
    pass


def add_depth(image):
    pass


def crt_effect(image):
    pass


def process_gif_file(input_gif_path, output_dir="output", affix="-crt", **kwargs):
    """
    Process a GIF file by extracting its frames, applying effects to each frame with `process_image`,
    and reassembling the frames into a new GIF file.
    """
    gif = Image.open(input_gif_path)
    frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]

    def convert_rgba_to_rgb_with_black_background(frame):
        """
        Convert an RGBA image to an RGB image, filling any transparent areas with black.

        Args:
            frame (PIL.Image): The source image in RGBA mode.

        Returns:
            PIL.Image: The converted image in RGB mode with a black background.
        """
        # Create a new black background image in RGBA mode
        black_background = Image.new("RGBA", frame.size, (0, 0, 0, 255))

        # Composite the RGBA frame onto the black background
        rgb_frame_with_alpha = Image.alpha_composite(black_background, frame)

        return rgb_frame_with_alpha

    processed_frames = []
    for i, frame in enumerate(frames[1:]):
        # Convert RGBA to RGB with a black background if necessary
        if frame.mode == "RGBA":
            frame = convert_rgba_to_rgb_with_black_background(frame)
        else:
            frame = frame.convert("RGB")

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".png", mode="w+b"
        ) as temp_frame_file:
            frame.save(temp_frame_file, format="PNG")
            temp_frame_file.flush()
            os.fsync(temp_frame_file.fileno())
            # Process the frame and save it to the output directory
            processed_frame_path = process_image(
                temp_frame_file.name, output_dir=output_dir, affix=affix, **kwargs
            )
            processed_frames.append(Image.open(processed_frame_path))

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Determine new filename and output path
    base_name = os.path.basename(input_gif_path)
    name, ext = os.path.splitext(base_name)
    new_name = f"{name}{affix}.gif"
    output_gif_path = os.path.join(output_dir, new_name)

    # Save processed frames as a new GIF
    processed_frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=processed_frames[1:],
        loop=0,
        format="GIF",
        duration=gif.info["duration"],
    )

    print(f"Processed GIF saved to {output_gif_path}")

    # Clean up temporary files
    for frame in processed_frames:
        frame.close()
    for temp_frame_file in glob.glob(os.path.join(output_dir, f"*{affix}.png")):
        os.unlink(temp_frame_file)


def get_video_framerate(video_path):
    """
    Use ffprobe to get the framerate of the video.

    Args:
        video_path: Path to the video file.

    Returns:
        Framerate of the video as a float.
    """
    cmd = [
        "ffprobe",
        "-v",
        "0",
        "-of",
        "csv=p=0",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        video_path,
    ]
    output = subprocess.check_output(cmd).decode().strip()
    try:
        num, den = map(int, output.split("/"))
        framerate = num / den
    except ValueError:
        framerate = 30.0  # Fallback framerate
    return framerate


def process_image(image_path, output_dir="output", affix="-ascii", **kwargs):
    """
    Process the image by stretching and applying scanline modulation.
    Args:
        image_path: Path to the input image.
        output_dir: Directory where the processed image will be saved.
        affix: String to be affixed to the output filename.
        **kwargs: Additional keyword arguments for image processing effects.
    Returns:
        Path to the processed image.
    """
    image = ascii_filter(filename=image_path, **kwargs)

    # Determine new filename and output path
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    new_name = f"{name}{affix}{ext}"
    output_path = os.path.join(output_dir, new_name)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert the color space from BGR to RGB

    cv2.imwrite(output_path, image)
    print(f"Processed image saved to {output_path}")

    return output_path


def process_video_file(input_video_path, output_video_filename, **kwargs):
    """
    Process a video file by extracting its frames, applying effects to each frame,
    and reassembling the frames into a new video file using the original video's framerate.

    Args:
        input_video_path: Path to the input video file.
        output_video_path: Path where the processed video will be saved.
        **kwargs: Additional keyword arguments passed to the image processing function.
    """
    temp_frame_dir = "imgs"  # Use the hardcoded directory from extract.py
    os.makedirs(temp_frame_dir, exist_ok=True)

    current_dir = os.getcwd()  # Get the current working directory
    output_video_path = os.path.join(current_dir, output_video_filename)

    # Extract frames using external script
    subprocess.call(["python", "extract.py", input_video_path])

    # Get the list of extracted frame files
    frame_files = sorted(os.listdir(temp_frame_dir))

    pool = multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count() // 2))
    print(f"Processing frames with {pool._processes} worker process(es)...")

    processed_frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(temp_frame_dir, frame_file)
        if frame_path.lower().endswith((".png", ".jpg", ".jpeg")):
            processed_frame = pool.apply_async(
                process_image, (frame_path,), {"output_dir": "output", **kwargs}
            )
            processed_frames.append(processed_frame)

    # Wait for all frames to be processed
    for processed_frame in processed_frames:
        processed_frame.wait()

    pool.close()
    pool.join()

    # Get the original video's framerate
    framerate = get_video_framerate(input_video_path)

    # Reassemble video from processed frames in the "output" directory
    processed_frame_dir = "output"  # Directory where processed frames are saved
    # Assuming processed frames are saved with filenames like "0000001-crt.png", adjust the pattern accordingly
    ffmpeg_command_assemble = [
        "ffmpeg",
        "-framerate",
        str(framerate),
        "-i",
        os.path.join(processed_frame_dir, "%07d-crt.png"),
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "av1_nvenc",
        "-b:v",
        "10M",
        output_video_path.replace(".mkv", "_no_audio.mkv"),
    ]
    subprocess.call(ffmpeg_command_assemble)

    # Add audio back to the processed video
    ffmpeg_command_audio = [
        "ffmpeg",
        "-i",
        output_video_path.replace(".mkv", "_no_audio.mkv"),
        "-i",
        input_video_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-strict",
        "experimental",
        output_video_path,
    ]
    subprocess.call(ffmpeg_command_audio)

    # Cleanup temporary directories and intermediate no-audio video
    dirs_to_delete = [temp_frame_dir, processed_frame_dir]
    file_to_delete = output_video_path.replace(".mkv", "_no_audio.mkv")

    for dir_path in dirs_to_delete:
        shutil.rmtree(dir_path, ignore_errors=True)
        print(f"Deleted directory: {dir_path}")

    if os.path.exists(file_to_delete):
        os.remove(file_to_delete)
        print(f"Deleted file: {file_to_delete}")


def process_input(input_path, save, show, **kwargs):
    """
    Determine if input is an image or video and process accordingly.
    """
    if not save and not show:
        raise ValueError("Either 'save' or 'show' must be set to True")
    if save:
        if input_path.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        ):
            process_image(input_path, **kwargs)
        elif input_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            # Extract directory, base filename, and extension
            dir_name = os.path.dirname(input_path)
            base_name, ext = os.path.splitext(os.path.basename(input_path))
            # Construct new output path with '_processed' suffix before the extension
            output_video_path = os.path.join(dir_name, f"{base_name}_processed{ext}")
            process_video_file(input_path, output_video_path, **kwargs)
        elif input_path.lower().endswith((".gif")):
            process_gif_file(input_path, **kwargs)
        else:
            print("Unsupported file format.")
    if show:
        if input_path.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        ):

            image = ascii_filter(input_path, **kwargs)
            cv2.imshow("ASCII Art", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Unsupported file format. Show only supported for images")

if __name__ == "__main__":

    process_input(
        input_path="images/img.png",
        max_dimension=1920,
        text_resolution=8,
        get_fill=True,
        get_edges=True,
        sigma_dog=3,
        sigma_factor=1.4,
        kernel_factor_dog=2,
        tau_dog=0,
        clahe_clip_limit=2,
        contrast_threshold_dog=5,
        apply_normalize=True,
        apply_clahe=False,
        apply_threshold=True,
        kernel_size_sobel=7,
        apply_colour=True,
        colour_map="black-white",
        apply_bloom=False,
        bloom_blur_value=10,
        bloom_gain=1,
        apply_contrast=False,
        contrast_gamma=0.2,
        contrast_saturation=0.5,
        apply_sharpness=False,
        kernel_size_sharpness=5,
        sigma_sharpness=1.0,
        amount_sharpness=1.0,
        threshold_sharpness=0,
        output_dir="./ascii_output/",
        save=False,
        show=True,
        only_dog=True,  # Set to True to only get the DoG image, must set fill and edge to False
        only_sobel=True,  # Set to True to only get the Sobel image, must set fill and edge to False
    )
