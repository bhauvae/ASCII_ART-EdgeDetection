import tkinter as tk
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import cv2
import numpy as np
import os
import time


def process_image(filename, **kwargs):
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
    colour_map = kwargs.get("colour_map", "black-white")
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
    show = kwargs.get("show", True)
    only_dog = kwargs.get("only_dog", False)
    only_sobel = kwargs.get("only_sobel", False)

    if not save and not show:
        raise ValueError("Either 'save' or 'show' must be set to True")

    input_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        raise ValueError(f"Could not open or find the image '{filename}'")

    start_time = time.time()

    input_image = downscale_image(input_image, max_dimension)

    ascii_fill_img, ascii_edge_img = None, None

    if get_fill:
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

    if save:
        os.makedirs(output_dir, exist_ok=True)
        output_ascii_filename = os.path.join(
            output_dir, os.path.splitext(os.path.basename(filename))[0] + ".png"
        )
        cv2.imwrite(output_ascii_filename, image)
        print(f"ASCII art image saved as '{output_ascii_filename}'")

    end_time = time.time()
    print(f"Processing time for '{filename}': {end_time - start_time:.2f} seconds")

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

    reduced_w = image_w // text_resolution
    reduced_h = image_h // text_resolution

    aspect_ratio = image_w / image_h
    if aspect_ratio > 1:
        reduced_w = reduced_w
        reduced_h = int(reduced_w / aspect_ratio)
    else:
        reduced_h = reduced_h
        reduced_w = int(reduced_h * aspect_ratio)

    image = cv2.resize(
        image, (int(reduced_w), int(reduced_h)), interpolation=cv2.INTER_AREA
    )

    ascii_fill_img = np.zeros(
        (reduced_h * text_resolution, reduced_w * text_resolution), dtype=np.uint8
    )

    char_indices = gray_scale_lookup[image]

    for i in range(len(fill_images)):
        mask = char_indices == i

        if not np.any(mask):
            continue

        char_img = fill_images[i]

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
    sigma1 = sigma_dog
    sigma2 = sigma_factor * sigma1

    k_factor = kernel_factor_dog
    k_size1 = int(k_factor * sigma1) | 1
    k_size2 = int(k_factor * sigma2) | 1

    blur1 = cv2.GaussianBlur(image, (k_size1, k_size1), sigma1)
    blur2 = cv2.multiply(np.ones_like(image) * (1 + tau_dog) , cv2.GaussianBlur(image, (k_size2, k_size2), sigma2))

    dog = cv2.subtract(blur1, blur2)

    if apply_normalize:
        dog_normalized = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
        dog = np.uint8(dog_normalized)

    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
        dog = clahe.apply(dog)

    if apply_threshold:
        thresh = contrast_threshold_dog
        dog = np.where(dog > thresh, 255.0, 0.0)

    return dog


def sobel_filter(image, kernel_size_sobel):
    ksize = int(kernel_size_sobel) | 1
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)

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

    reduced_w = image_w // text_resolution
    reduced_h = image_h // text_resolution

    aspect_ratio = image_w / image_h
    if aspect_ratio > 1:
        reduced_w = reduced_w
        reduced_h = int(reduced_w / aspect_ratio)
    else:
        reduced_h = reduced_h
        reduced_w = int(reduced_h * aspect_ratio)

    def direction_threshold(theta):
        if np.isnan(theta):
            return 0

        angle_tolerance = edge_angle_range
        abstheta = np.abs(theta)
        if (
            0.0 <= abstheta < 0.0 + angle_tolerance
            or 180 - angle_tolerance <= abstheta <= 180.0
        ):
            return 1
        elif 90 - angle_tolerance <= abstheta < 90 + angle_tolerance:
            return 2
        elif 0.0 + angle_tolerance <= abstheta < 90 - angle_tolerance:
            return 3 if np.sign(theta) > 0 else 4
        elif 90 + angle_tolerance < abstheta < 180 - angle_tolerance:
            return 4 if np.sign(theta) > 0 else 3

    vectorized_threshold = np.vectorize(direction_threshold)
    edge_value = vectorized_threshold(sobel_angle)

    def most_frequent_direction(grid):
        hist, _ = np.histogram(grid, bins=np.arange(6))

        if hist.sum() > histogram_threshold_sobel:
            return np.argmax(hist)
        else:
            return 0

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
            if char_index < len(edge_char_images):
                char_img = edge_char_images[char_index]
                if char_img is not None and char_img.shape == (
                    text_resolution,
                    text_resolution,
                ):
                    ascii_img[start_y:end_y, start_x:end_x] = char_img

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

    hex_to_bgr = lambda hex_color: tuple(
        int(hex_color[i : i + 2], 16) for i in (5, 3, 1)
    )

    color_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    color_image[image == 0] = hex_to_bgr(colour_map_hex[color_map][0])
    color_image[image == 255] = hex_to_bgr(colour_map_hex[color_map][1])

    return color_image


def add_bloom(image, bloom_blur_value, bloom_gain):
    blur_value = bloom_blur_value
    gain = bloom_gain

    blur = cv2.GaussianBlur(image, (0, 0), sigmaX=blur_value, sigmaY=blur_value)
    result = cv2.addWeighted(image, 1, blur, gain, 0)

    return result


def add_contrast(image, contrast_gamma, contrast_saturation):
    hdr_image = image.astype(np.float32) / 255.0

    tonemap = cv2.createTonemapDrago(contrast_gamma, contrast_saturation)

    ldr_tonemapped = tonemap.process(hdr_image)

    ldr_tonemapped = np.nan_to_num(ldr_tonemapped)

    ldr_tonemapped = np.clip(ldr_tonemapped * 255, 0, 255).astype("uint8")

    return ldr_tonemapped


def add_sharpness(
    image, kernel_size_sharpness, sigma_sharpness, amount_sharpness, threshold_sharpness
):
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


class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")


class ImageProcessingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Processing Application")
        self.geometry("1200x800")

        # Initialize variables
        self.current_image = None
        self.processing_label = None
        self.parameter_entries = {}
        self.image_filename = None

        # Define parameters
        self.PARAMETERS = {
            "max_dimension": 1920,
            "text_resolution": 8,
            "get_fill": True,
            "get_edges": True,
            "sigma_dog": 3,
            "sigma_factor": 1.6,
            "kernel_factor_dog": 6,
            "tau_dog": 0,
            "clahe_clip_limit": 2,
            "contrast_threshold_dog": 25,
            "apply_normalize": True,
            "apply_clahe": True,
            "apply_threshold": True,
            "kernel_size_sobel": 7,
            "apply_colour": True,
            "colour_map": "black-white",
            "apply_bloom": True,
            "bloom_blur_value": 10,
            "bloom_gain": 1,
            "apply_contrast": True,
            "contrast_gamma": 0.2,
            "contrast_saturation": 0.5,
            "apply_sharpness": True,
            "kernel_size_sharpness": 5,
            "sigma_sharpness": 1.0,
            "amount_sharpness": 1.0,
            "threshold_sharpness": 0,
            "output_dir": "./ascii_output/",
            "save": False,
            "show": True,
            "only_dog": False,
            "only_sobel": False,
        }

        # Create main canvas to display image
        self.canvas = tk.Canvas(self, bg="gray")
        self.canvas.pack(padx=10, pady=10, side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Placeholder for the image
        self.image_item = self.canvas.create_image(0, 0, anchor=tk.NW)

        # Frame for parameter controls
        self.parameters_frame = tk.Frame(self)
        self.parameters_frame.pack(padx=10, pady=10, side=tk.LEFT, fill=tk.Y)

        # Label for processing indication

        # Create parameter inputs
        self.create_parameter_inputs()

        # Button to load a new image
        self.load_button = tk.Button(
            self.parameters_frame, text="Load Image", command=self.open_file
        )
        self.load_button.grid(
            row=len(self.PARAMETERS),
            column=0,
            columnspan=2,
            padx=5,
            pady=10,
            sticky=tk.W + tk.E,
        )

        # Button to save optimal parameters
        self.save_button = tk.Button(
            self.parameters_frame,
            text="Save Optimal Parameters",
            command=self.save_parameters,
        )
        self.save_button.grid(
            row=len(self.PARAMETERS) + 1,
            column=0,
            columnspan=2,
            padx=5,
            pady=10,
            sticky=tk.W + tk.E,
        )

        # Button to process image
        self.process_button = tk.Button(
            self.parameters_frame,
            text="Process Image",
            command=self.process_image_with_parameters,
        )
        self.process_button.grid(
            row=len(self.PARAMETERS) + 2,
            column=0,
            columnspan=2,
            padx=5,
            pady=10,
            sticky=tk.W + tk.E,
        )

        # Label for processing indication
        self.processing_label = tk.Label(
            self.parameters_frame,
            text="Processing...",
            font=("Arial", 20),
            fg="red",
            bg="gray",
        )
        self.processing_label.grid(
            row=len(self.PARAMETERS) + 3,
            column=0,
            columnspan=2,
            padx=5,
            pady=10,
            sticky=tk.W + tk.E,
        )
        
    def create_parameter_inputs(self):
        row = 0
        col = 0
        for i, (param, default_value) in enumerate(self.PARAMETERS.items()):
            if i % (len(self.PARAMETERS) // 2) == 0 and i != 0:
                row = 0
                col += 2

            label = tk.Label(
                self.parameters_frame, text=param.replace("_", " ").title()
            )
            label.grid(row=row, column=col, padx=5, pady=5, sticky=tk.W)

            if isinstance(default_value, bool):
                var = tk.IntVar(value=int(default_value))
                entry = tk.Checkbutton(self.parameters_frame, variable=var)
            else:
                var = tk.StringVar(value=str(default_value))
                entry = tk.Entry(self.parameters_frame, textvariable=var)

            entry.grid(row=row, column=col + 1, padx=5, pady=5, sticky=tk.W)
            self.parameter_entries[param] = var
            row += 1

    def open_file(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg")]
        )
        if filename:
            image = cv2.imread(filename)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.update_image_display(Image.fromarray(image))
                self.image_filename = filename
                for param, var in self.parameter_entries.items():
                    if isinstance(var, tk.StringVar):
                        var.set(str(self.PARAMETERS[param]))
                    elif isinstance(var, tk.IntVar):
                        var.set(int(self.PARAMETERS[param]))
            else:
                messagebox.showerror("Error", "Failed to open the image file.")

    def update_image_display(self, image):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = image.size

        scale = min(canvas_width / img_width, canvas_height / img_height)
        resized_image = image.resize((int(img_width * scale), int(img_height * scale)))

        self.current_image = ImageTk.PhotoImage(resized_image)
        self.canvas.itemconfig(self.image_item, image=self.current_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def save_parameters(self):
        optimal_params = {}
        for param, var in self.parameter_entries.items():
            if isinstance(var, tk.StringVar):
                optimal_params[param] = var.get()
            elif isinstance(var, tk.IntVar):
                optimal_params[param] = bool(int(var.get()))

        # Example: Save optimal_params to a file or perform any other operation
        print("Optimal Parameters:", optimal_params)

        with open("optimal_params.txt", "w") as f:
            for key, value in optimal_params.items():
                f.write(f"{key}={value}\n")
        messagebox.showinfo("Saved!")

    def process_image_with_parameters(self):
        try:
            self.processing_label.grid()

            kwargs = {}
            for param, var in self.parameter_entries.items():
                if isinstance(var, tk.StringVar):
                    value = var.get()
                    if value.isdigit():
                        kwargs[param] = int(value)
                    else:
                        try:
                            kwargs[param] = float(value)
                        except ValueError:
                            kwargs[param] = value
                elif isinstance(var, tk.IntVar):
                    kwargs[param] = bool(int(var.get()))

            if self.image_filename:
                processed_image = process_image(self.image_filename, **kwargs)
                self.update_image_display(Image.fromarray(processed_image))

        except Exception as e:
            messagebox.showerror("Error-process img with para", str(e))

        finally:
            self.processing_label.grid_remove()


if __name__ == "__main__":
    app = ImageProcessingApp()
    app.mainloop()
