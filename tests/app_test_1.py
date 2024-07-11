import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import threading

# Set initial parameters
params = {
    "filename": "",  # Path to the input image
    "max_dimension": 1920,
    "get_fill": True,
    "get_edges": True,
    "sigma_dog": 3,
    "sigma_factor": 1.6,
    "kernel_factor_dog": 6,
    "clahe_clip_limit": 2,
    "contrast_threshold_dog": 25,
    "apply_normalize": True,
    "apply_clahe": True,
    "apply_contrast_threshold": True,
    "kernel_size_sobel": 7,
    "apply_colour": True,
    "colour_map": "purple-salmon",
    "apply_bloom": True,
    "bloom_blur_value": 10,
    "bloom_gain": 1,
    "apply_contrast": True,
    "contrast_gamma": 0.2,
    "contrast_saturation": 0.5,
    "output_dir": "./ascii_output/",
    "save": False,
    "show": False,
}


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASCII Image Processor")
        self.root.configure(bg="black")

        self.image_label = None
        self.canvas = None

        # Setup frames
        self.controls_frame = tk.Frame(self.root, bg="black", padx=10, pady=10)
        self.controls_frame.grid(row=0, column=0, sticky="nsew")

        self.image_frame = tk.Frame(self.root, bg="white", padx=10, pady=10)
        self.image_frame.grid(row=0, column=1, sticky="nsew")

        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.setup_controls()
        self.setup_image_display()
        self.load_image()

    def setup_controls(self):
        row = 0

        def add_dropdown(label, param_name, options):
            nonlocal row
            tk.Label(self.controls_frame, text=label, fg="white", bg="black").grid(
                row=row, column=0, sticky="w"
            )
            dropdown_var = tk.StringVar(value=params[param_name])
            dropdown = ttk.Combobox(
                self.controls_frame,
                textvariable=dropdown_var,
                values=options,
                state="readonly",
            )
            dropdown.bind(
                "<<ComboboxSelected>>",
                lambda _: self.update_param(param_name, dropdown_var.get()),
            )
            dropdown.grid(row=row, column=1, sticky="ew")
            row += 1

        def add_slider(label, param_name, from_, to, resolution=1):
            nonlocal row
            tk.Label(self.controls_frame, text=label, fg="white", bg="black").grid(
                row=row, column=0, sticky="w"
            )
            scale = tk.Scale(
                self.controls_frame,
                from_=from_,
                to=to,
                resolution=resolution,
                orient="horizontal",
                bg="black",
                fg="white",
                highlightbackground="black",
                command=lambda val: self.update_param(param_name, val),
            )
            scale.set(params[param_name])
            scale.grid(row=row, column=1, sticky="ew")
            row += 1

        def add_checkbox(label, param_name):
            nonlocal row
            var = tk.BooleanVar(value=params[param_name])
            chk = tk.Checkbutton(
                self.controls_frame,
                text=label,
                var=var,
                bg="black",
                fg="white",
                highlightbackground="black",
                command=lambda: self.update_param(param_name, var.get()),
            )
            chk.grid(row=row, column=0, columnspan=2, sticky="w")
            self.checkbox_vars[param_name] = var
            row += 1

        self.checkbox_vars = {}
        add_slider("Max Dimension", "max_dimension", 360, 3840, 360)
        add_checkbox("Get Fill", "get_fill")
        add_checkbox("Get Edges", "get_edges")
        add_slider("Sigma DOG", "sigma_dog", 0, 10, 0.1)
        add_slider("Sigma Factor", "sigma_factor", 0, 3, 0.1)
        add_slider("Kernel Factor DOG", "kernel_factor_dog", 1, 10)
        add_slider("CLAHE Clip Limit", "clahe_clip_limit", 0, 10, 0.1)
        add_slider("Contrast Threshold DOG", "contrast_threshold_dog", 0, 255)
        add_checkbox("Apply Normalize", "apply_normalize")
        add_checkbox("Apply CLAHE", "apply_clahe")
        add_checkbox("Apply Contrast Threshold", "apply_contrast_threshold")
        add_slider("Kernel Size Sobel", "kernel_size_sobel", 1, 31, 2)
        add_checkbox("Apply Colour", "apply_colour")
        add_slider("Bloom Blur Value", "bloom_blur_value", 0, 100)
        add_slider("Bloom Gain", "bloom_gain", 0, 5, 0.1)
        add_checkbox("Apply Contrast", "apply_contrast")
        add_slider("Contrast Gamma", "contrast_gamma", 0, 2, 0.1)
        add_slider("Contrast Saturation", "contrast_saturation", 0, 2, 0.1)

        add_dropdown(
            "Colour Map",
            "colour_map",
            [
                "black-white",
                "white-black",
                "blue-green",
                "hacker-man",
                "purple-salmon",
                "darkblue-white",
            ],
        )

        tk.Label(
            self.controls_frame, text="Output Image Size", fg="white", bg="black"
        ).grid(row=row, column=0, sticky="w")
        self.output_size_scale = tk.Scale(
            self.controls_frame,
            from_=100,
            to=1000,
            resolution=50,
            orient="horizontal",
            bg="black",
            fg="white",
            highlightbackground="black",
        )
        self.output_size_scale.set(400)
        self.output_size_scale.grid(row=row, column=1, sticky="ew")
        row += 1

        self.save_button = tk.Button(
            self.controls_frame,
            text="Save Optimal Params",
            command=self.save_params,
            bg="white",
            fg="black",
        )
        self.save_button.grid(row=row, column=0, columnspan=2, sticky="ew", pady=5)
        row += 1

        self.load_button = tk.Button(
            self.controls_frame,
            text="Load Image",
            command=self.load_image,
            bg="white",
            fg="black",
        )
        self.load_button.grid(row=row, column=0, columnspan=2, sticky="ew", pady=5)

        row += 1
        self.status_label = tk.Label(
            self.controls_frame, text="", fg="white", bg="black"
        )
        self.status_label.grid(row=row, column=0, columnspan=2, sticky="w")

    def setup_image_display(self):
        self.canvas = tk.Canvas(self.image_frame, bg="white", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def load_image(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
        )
        if filename:
            params["filename"] = filename
            self.update_image()

    def update_param(self, key, value):
        if key in [
            "max_dimension",
            "sigma_dog",
            "sigma_factor",
            "kernel_factor_dog",
            "clahe_clip_limit",
            "contrast_threshold_dog",
            "kernel_size_sobel",
            "bloom_blur_value",
            "bloom_gain",
            "contrast_gamma",
            "contrast_saturation",
        ]:
            params[key] = float(value)
        elif key == "colour_map":
            params[key] = value
        elif key == "filename":
            params[key] = value
        else:
            params[key] = bool(value)
            self.checkbox_vars[key].set(value)
        self.update_image()

    def save_params(self):
        with open("optimal_params.txt", "w") as f:
            for key, value in params.items():
                f.write(f"{key}={value}\n")
        self.status_label.config(text="Parameters saved!")

    def update_image(self):
        self.status_label.config(text="Processing...")
        self.canvas.delete("all")

        def process_and_display():
            try:
                ascii_img = process_image(**params)
                if ascii_img is not None:
                    image = Image.fromarray(ascii_img)
                    output_size = int(self.output_size_scale.get())
                    image = image.resize((output_size, output_size))
                    self.image_tk = ImageTk.PhotoImage(image)
                    self.canvas.create_image(
                        output_size // 2, output_size // 2, image=self.image_tk
                    )
            except Exception as e:
                print(f"Error processing image: {e}")
            finally:
                self.status_label.config(text="")

        threading.Thread(target=process_and_display).start()


def process_image(**kwargs):
    # Set default values for the function arguments
    filename = kwargs.get("filename", "")
    max_dimension = kwargs.get("max_dimension", 1920)
    text_resolution = kwargs.get("text_resolution", 8)
    get_fill = kwargs.get("get_fill", True)
    get_edges = kwargs.get("get_edges", True)
    sigma_dog = kwargs.get("sigma_dog", 3)
    sigma_factor = kwargs.get("sigma_factor", 1.6)
    kernel_factor_dog = kwargs.get("kernel_factor_dog", 6)
    clahe_clip_limit = kwargs.get("clahe_clip_limit", 2)
    contrast_threshold_dog = kwargs.get("contrast_threshold_dog", 25)
    apply_normalize = kwargs.get("apply_normalize", True)
    apply_clahe = kwargs.get("apply_clahe", True)
    apply_contrast_threshold = kwargs.get("apply_contrast_threshold", True)
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
    output_dir = kwargs.get("output_dir", "./ascii_output/")
    save = kwargs.get("save", False)
    show = kwargs.get("show", False)

    # Open the image and convert to grayscale
    input_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        raise ValueError(f"Could not open or find the image '{filename}'")

    input_image = downscale_image(input_image, max_dimension)

    ascii_fill_img = None
    ascii_edge_img = None

    if get_fill:
        # Create ASCII fill image
        ascii_fill_img = create_ascii_fill_image(input_image, text_resolution)

    if get_edges:
        dog = diff_of_gaussiian(
            input_image,
            sigma_dog,
            sigma_factor,
            kernel_factor_dog,
            clahe_clip_limit,
            contrast_threshold_dog,
            apply_normalize,
            apply_clahe,
            apply_contrast_threshold,
        )
        sobel_magnitude, sobel_angle = sobel_filter(dog, kernel_size_sobel)

        ascii_edge_img = create_ascii_edge_image(
            input_image, sobel_angle, sobel_magnitude, text_resolution
        )

    ascii_img = combine_fill_edges(ascii_fill_img, ascii_edge_img, text_resolution)

    if apply_colour:
        ascii_img = add_color(ascii_img, colour_map)

    if apply_bloom:
        ascii_img = add_bloom(ascii_img, bloom_blur_value, bloom_gain)

    if apply_contrast:
        ascii_img = add_contrast(ascii_img, contrast_gamma, contrast_saturation)

    if apply_sharpness:
        ascii_img = add_sharpness(
            ascii_img,
            kernel_size_sharpness,
            sigma_sharpness,
            amount_sharpness,
            threshold_sharpness,
        )

    if save:
        os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
        # Force the output filename to have a .png extension
        output_ascii_filename = os.path.join(
            output_dir, os.path.splitext(os.path.basename(filename))[0] + ".png"
        )
        cv2.imwrite(output_ascii_filename, ascii_img)
        print(f"ASCII art image saved as '{output_ascii_filename}'")

    if show:
        cv2.imshow("ASCII Art", ascii_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return ascii_img


def create_ascii_fill_image(image, text_resolution):
    GRAY_SCALE = " .;coP0?@█"

    gray_scale_lookup = np.array(
        [i for i in range(10) for _ in range(26)], dtype=np.uint8
    )[4:260]
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
    image = cv2.resize(image, (int(reduced_w), int(reduced_h)), interpolation=cv2.INTER_AREA)
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
    clahe_clip_limit,
    contrast_threshold_dog,
    apply_normalize,
    apply_clahe,
    apply_contrast_threshold,
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
    blur2 = cv2.GaussianBlur(image, (k_size2, k_size2), sigma2)

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

    if apply_contrast_threshold:
        thresh = contrast_threshold_dog  # Define the threshold value
        _, dog = cv2.threshold(dog, thresh, 255, cv2.THRESH_BINARY)

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
    if ascii_edge_img is None:
        return ascii_fill_img
    elif ascii_fill_img is None:
        return ascii_edge_img

    else:
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
        image = cv2.resize(image, (int(new_width), int(new_height)), interpolation=cv2.INTER_AREA)

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

    hex_to_rgb = lambda hex_color: tuple(
        int(hex_color[i : i + 2], 16) for i in (1, 3, 5)
    )

    # Create an RGB image array with the same height and width as the grayscale image
    color_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # Map black (0) to green and white (255) to pink
    color_image[image == 0] = hex_to_rgb(colour_map_hex[color_map][0])
    color_image[image == 255] = hex_to_rgb(colour_map_hex[color_map][1])

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


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
