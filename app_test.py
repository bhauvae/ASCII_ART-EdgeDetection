import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import threading

# Define the grayscale characters from darkest to lightest
GRAY_SCALE = " .;coP0?@█"
EDGES_GRAY_SCALE = r" |_/\\"
GRAY_SCALE_LOOKUP = np.array([i for i in range(10) for _ in range(26)], dtype=np.uint8)[4:260]
FILL_IMAGES = np.array([cv2.imread(os.path.join("./fillchar", f"{i}.png"), cv2.IMREAD_GRAYSCALE) for i in range(len(GRAY_SCALE))])
EDGE_IMAGES = np.array([cv2.imread(os.path.join("./edgechar", f"{i}.png"), cv2.IMREAD_GRAYSCALE) for i in range(5)])
TEXT_RESOLUTION = 8
COLOUR_MAP_HEX = {
    "blue-green": ("#0A174E", "#F5D042"),
    "hacker-man": ("#120d13", "#29ff2a"),
    "purple-salmon": ("#180a1c", "#e05964"),
    "darkblue-white": ("#1f1735", "#fffcff"),
}

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
    ascii_fill_img = np.zeros((reduced_h * TEXT_RESOLUTION, reduced_w * TEXT_RESOLUTION), dtype=np.uint8)

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

def diff_of_gaussian(image, sigma_dog, sigma_factor, kernel_factor_dog, clahe_clip_limit, dog_contrast_thresh, apply_normalize=True, apply_clahe=True, apply_contrastthresh=True):
    # Set sigma values
    sigma1 = sigma_dog
    sigma2 = sigma_factor * sigma1  # Larger sigma is typically 1.6 times the smaller sigma

    # Calculate kernel sizes (make sure they are odd)
    k_size1 = int(kernel_factor_dog * sigma1) | 1  # Bitwise OR with 1 ensures the kernel size is odd
    k_size2 = int(kernel_factor_dog * sigma2) | 1

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
        dog = clahe.apply(dog_normalized)

    if apply_contrastthresh:
        _, dog = cv2.threshold(dog, dog_contrast_thresh, 255, cv2.THRESH_BINARY)

    return dog

def sobel_filter(image, kernel_size_sobel):
    ksize = int(kernel_size_sobel) | 1
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)

    sobel_magnitude = cv2.magnitude(sobelx, sobely)
    sobel_angle = np.rad2deg(np.arctan2(sobely, sobelx))

    return sobel_magnitude, sobel_angle

def create_ascii_edge_image(image, sobel_angle, sobel_magnitude, sobel_hist_threshold, edge_angle_range):
    sobel_angle = np.where(sobel_magnitude != 0, sobel_angle, np.nan)
    image_h, image_w = image.shape

    reduced_w = image_w // TEXT_RESOLUTION
    reduced_h = image_h // TEXT_RESOLUTION

    aspect_ratio = image_w / image_h
    if (aspect_ratio > 1):
        reduced_w = reduced_w
        reduced_h = int(reduced_w / aspect_ratio)
    else:
        reduced_h = reduced_h
        reduced_w = int(reduced_h * aspect_ratio)

    def direction_threshold(theta):
        if np.isnan(theta):
            return 0

        angle_factor = edge_angle_range
        abstheta = np.abs(theta)
        if 0.0 <= abstheta < 0.0 + angle_factor or 180 - angle_factor <= abstheta <= 180.0:
            return 1  # VERTICAL
        elif 90 - angle_factor <= abstheta < 90 + angle_factor:
            return 2  # HORIZONTAL
        elif 0.0 + angle_factor <= abstheta < 90 - angle_factor:
            return 3 if np.sign(theta) > 0 else 4  # DIAGONAL 1
        elif 90 + angle_factor < abstheta < 180 - angle_factor:
            return 4 if np.sign(theta) > 0 else 3  # DIAGONAL 2

    vectorized_threshold = np.vectorize(direction_threshold)
    edge_value = vectorized_threshold(sobel_angle)

    def most_frequent_direction(grid):
        hist, _ = np.histogram(grid, bins=np.arange(6))  # bins=[0, 1, 2, 3, 4, 5]
        if hist.sum() > sobel_hist_threshold:
            return np.argmax(hist)  # Matching max frequency with edge
        else:
            return 0

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

            if np.all(ascii_edge_img[start_y:end_y, start_x:end_x] == 0):
                continue
            else:
                ascii_fill_img[start_y:end_y, start_x:end_x] = ascii_edge_img[start_y:end_y, start_x:end_x]

    return ascii_fill_img

def downscale_image(image, max_dim=1920):
    height, width = image.shape
    if max(width, height) > max_dim:
        if width > height:
            new_width = max_dim
            new_height = int(height * max_dim / width)
        else:
            new_height = max_dim
            new_width = int(width * max_dim / height)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image

def add_color(image, color_map="purple-salmon"):
    colour_map_hex = COLOUR_MAP_HEX

    def hex_to_rgb(hex_color):
        return tuple(int(hex_color[i:i+2], 16) for i in (5, 3, 1))

    color_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    color_image[image == 0] = hex_to_rgb(colour_map_hex[color_map][0])
    color_image[image == 255] = hex_to_rgb(colour_map_hex[color_map][1])

    return color_image

def process_image(filename, sigma_dog, sigma_factor, kernel_factor_dog, clahe_clip_limit, dog_contrast_thresh, kernel_size_sobel, sobel_hist_threshold, edge_angle_range):
    output_dir = "./"
    os.makedirs(output_dir, exist_ok=True)

    output_ascii_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(filename))[0] + "asciibw.png")

    input_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        raise ValueError(f"Could not open or find the image '{filename}'")

    input_image = downscale_image(input_image)

    ascii_fill_img = create_ascii_fill_image(input_image)

    dog = diff_of_gaussian(input_image, sigma_dog, sigma_factor, kernel_factor_dog, clahe_clip_limit, dog_contrast_thresh)
    sobel_magnitude, sobel_angle = sobel_filter(dog, kernel_size_sobel)

    ascii_edge_img = create_ascii_edge_image(input_image, sobel_angle, sobel_magnitude, sobel_hist_threshold, edge_angle_range)

    ascii_img = combine_fill_edges(input_image, ascii_fill_img, ascii_edge_img)

    cv2.imwrite(output_ascii_filename, ascii_img)
    return output_ascii_filename

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import threading

# Define the grayscale characters from darkest to lightest
GRAY_SCALE = " .;coP0?@█"
EDGES_GRAY_SCALE = r" |_/\\"
GRAY_SCALE_LOOKUP = np.array([i for i in range(10) for _ in range(26)], dtype=np.uint8)[4:260]
FILL_IMAGES = np.array([cv2.imread(os.path.join("./fillchar", f"{i}.png"), cv2.IMREAD_GRAYSCALE) for i in range(len(GRAY_SCALE))])
EDGE_IMAGES = np.array([cv2.imread(os.path.join("./edgechar", f"{i}.png"), cv2.IMREAD_GRAYSCALE) for i in range(5)])
TEXT_RESOLUTION = 8
COLOUR_MAP_HEX = {
    "blue-green": ("#0A174E", "#F5D042"),
    "hacker-man": ("#120d13", "#29ff2a"),
    "purple-salmon": ("#180a1c", "#e05964"),
    "darkblue-white": ("#1f1735", "#fffcff"),
}

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
    ascii_fill_img = np.zeros((reduced_h * TEXT_RESOLUTION, reduced_w * TEXT_RESOLUTION), dtype=np.uint8)

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

def diff_of_gaussian(image, sigma_dog, sigma_factor, kernel_factor_dog, clahe_clip_limit, dog_contrast_thresh, apply_normalize=True, apply_clahe=True, apply_contrastthresh=True):
    # Set sigma values
    sigma1 = sigma_dog
    sigma2 = sigma_factor * sigma1  # Larger sigma is typically 1.6 times the smaller sigma

    # Calculate kernel sizes (make sure they are odd)
    k_size1 = int(kernel_factor_dog * sigma1) | 1  # Bitwise OR with 1 ensures the kernel size is odd
    k_size2 = int(kernel_factor_dog * sigma2) | 1

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
        dog = clahe.apply(dog_normalized)

    if apply_contrastthresh:
        _, dog = cv2.threshold(dog, dog_contrast_thresh, 255, cv2.THRESH_BINARY)

    return dog

def sobel_filter(image, kernel_size_sobel):
    ksize = int(kernel_size_sobel) | 1
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)

    sobel_magnitude = cv2.magnitude(sobelx, sobely)
    sobel_angle = np.rad2deg(np.arctan2(sobely, sobelx))

    return sobel_magnitude, sobel_angle

def create_ascii_edge_image(image, sobel_angle, sobel_magnitude, sobel_hist_threshold, edge_angle_range):
    sobel_angle = np.where(sobel_magnitude != 0, sobel_angle, np.nan)
    image_h, image_w = image.shape

    reduced_w = image_w // TEXT_RESOLUTION
    reduced_h = image_h // TEXT_RESOLUTION

    aspect_ratio = image_w / image_h
    if (aspect_ratio > 1):
        reduced_w = reduced_w
        reduced_h = int(reduced_w / aspect_ratio)
    else:
        reduced_h = reduced_h
        reduced_w = int(reduced_h * aspect_ratio)

    def direction_threshold(theta):
        if np.isnan(theta):
            return 0

        angle_factor = edge_angle_range
        abstheta = np.abs(theta)
        if 0.0 <= abstheta < 0.0 + angle_factor or 180 - angle_factor <= abstheta <= 180.0:
            return 1  # VERTICAL
        elif 90 - angle_factor <= abstheta < 90 + angle_factor:
            return 2  # HORIZONTAL
        elif 0.0 + angle_factor <= abstheta < 90 - angle_factor:
            return 3 if np.sign(theta) > 0 else 4  # DIAGONAL 1
        elif 90 + angle_factor < abstheta < 180 - angle_factor:
            return 4 if np.sign(theta) > 0 else 3  # DIAGONAL 2

    vectorized_threshold = np.vectorize(direction_threshold)
    edge_value = vectorized_threshold(sobel_angle)

    def most_frequent_direction(grid):
        hist, _ = np.histogram(grid, bins=np.arange(6))  # bins=[0, 1, 2, 3, 4, 5]
        if hist.sum() > sobel_hist_threshold:
            return np.argmax(hist)  # Matching max frequency with edge
        else:
            return 0

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

            if np.all(ascii_edge_img[start_y:end_y, start_x:end_x] == 0):
                continue
            else:
                ascii_fill_img[start_y:end_y, start_x:end_x] = ascii_edge_img[start_y:end_y, start_x:end_x]

    return ascii_fill_img

def downscale_image(image, max_dim=1920):
    height, width = image.shape
    if max(width, height) > max_dim:
        if width > height:
            new_width = max_dim
            new_height = int(height * max_dim / width)
        else:
            new_height = max_dim
            new_width = int(width * max_dim / height)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image

def add_color(image, color_map="purple-salmon"):
    colour_map_hex = COLOUR_MAP_HEX

    def hex_to_rgb(hex_color):
        return tuple(int(hex_color[i:i+2], 16) for i in (5, 3, 1))

    color_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    color_image[image == 0] = hex_to_rgb(colour_map_hex[color_map][0])
    color_image[image == 255] = hex_to_rgb(colour_map_hex[color_map][1])

    return color_image

def process_image(filename, sigma_dog, sigma_factor, kernel_factor_dog, clahe_clip_limit, dog_contrast_thresh, kernel_size_sobel, sobel_hist_threshold, edge_angle_range):
    output_dir = "./"
    os.makedirs(output_dir, exist_ok=True)

    output_ascii_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(filename))[0] + "asciibw.png")

    input_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        raise ValueError(f"Could not open or find the image '{filename}'")

    input_image = downscale_image(input_image)

    ascii_fill_img = create_ascii_fill_image(input_image)

    dog = diff_of_gaussian(input_image, sigma_dog, sigma_factor, kernel_factor_dog, clahe_clip_limit, dog_contrast_thresh)
    sobel_magnitude, sobel_angle = sobel_filter(dog, kernel_size_sobel)

    ascii_edge_img = create_ascii_edge_image(input_image, sobel_angle, sobel_magnitude, sobel_hist_threshold, edge_angle_range)

    ascii_img = combine_fill_edges(input_image, ascii_fill_img, ascii_edge_img)

    cv2.imwrite(output_ascii_filename, ascii_img)
    return output_ascii_filename

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Processing GUI")

        self.filename = None

        self.sigma_dog = tk.DoubleVar(value=SIGMA_DOG)
        self.sigma_factor = tk.DoubleVar(value=SIGMA_FACTOR)
        self.kernel_factor_dog = tk.DoubleVar(value=KERNEL_FACTOR_DOG)
        self.clahe_clip_limit = tk.DoubleVar(value=CLAHE_CLIP_LIMIT)
        self.dog_contrast_thresh = tk.DoubleVar(value=DOG_CONTRAST_THRESH)
        self.kernel_size_sobel = tk.IntVar(value=KERNEL_SIZE_SOBEL)
        self.sobel_hist_threshold = tk.IntVar(value=SOBEL_HIST_THRESHOLD)
        self.edge_angle_range = tk.IntVar(value=EDGE_ANGLE_RANGE)

        self.create_widgets()

    def create_widgets(self):
        tk.Button(self, text="Open Image", command=self.open_image).pack()

        self.create_slider("Sigma DOG", self.sigma_dog, 0.1, 10, 0.1, "Sigma for the Difference of Gaussians.")
        self.create_slider("Sigma Factor", self.sigma_factor, 0.1, 10, 0.1, "Factor to multiply sigma_dog for the second Gaussian blur.")
        self.create_slider("Kernel Factor DOG", self.kernel_factor_dog, 1, 20, 1, "Factor to determine the kernel size for the Gaussian blur.")
        self.create_slider("CLAHE Clip Limit", self.clahe_clip_limit, 1, 10, 0.1, "Clip limit for CLAHE.")
        self.create_slider("DOG Contrast Threshold", self.dog_contrast_thresh, 1, 255, 1, "Threshold for the DoG contrast.")
        self.create_slider("Kernel Size Sobel", self.kernel_size_sobel, 1, 31, 1, "Kernel size for the Sobel filter.")
        self.create_slider("Sobel Hist Threshold", self.sobel_hist_threshold, 1, 100, 1, "Histogram threshold for Sobel.")
        self.create_slider("Edge Angle Range", self.edge_angle_range, 1, 90, 1, "Angle range for edge detection.")

        tk.Button(self, text="Process Image", command=self.start_processing).pack()

        self.image_label = tk.Label(self)
        self.image_label.pack()

    def create_slider(self, label_text, variable, from_, to, resolution, tooltip):
        frame = tk.Frame(self)
        frame.pack()

        label = tk.Label(frame, text=label_text)
        label.pack(side=tk.LEFT)

        scale = tk.Scale(frame, variable=variable, from_=from_, to=to, resolution=resolution, orient="horizontal")
        scale.pack(side=tk.LEFT)

        value_entry = tk.Entry(frame, textvariable=variable, width=5)
        value_entry.pack(side=tk.LEFT)

        label.bind("<Enter>", lambda event: self.show_tooltip(event, tooltip))
        label.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event, text):
        x = event.widget.winfo_rootx() + 20
        y = event.widget.winfo_rooty() + 20
        self.tooltip = tk.Toplevel(self)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip, text=text, background="yellow", relief="solid", borderwidth=1, wraplength=200)
        label.pack()

    def hide_tooltip(self, event):
        if self.tooltip:
            self.tooltip.destroy()

    def open_image(self):
        self.filename = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if self.filename:
            print(f"Selected file: {self.filename}")

    def start_processing(self):
        if not self.filename:
            messagebox.showerror("Error", "No image selected.")
            return

        threading.Thread(target=self.process_image).start()

    def process_image(self):
        self.loading_screen = tk.Toplevel(self)
        self.loading_screen.title("Processing")
        tk.Label(self.loading_screen, text="Processing image, please wait...").pack()

        try:
            output_file = process_image(
                self.filename,
                self.sigma_dog.get(),
                self.sigma_factor.get(),
                self.kernel_factor_dog.get(),
                self.clahe_clip_limit.get(),
                self.dog_contrast_thresh.get(),
                self.kernel_size_sobel.get(),
                self.sobel_hist_threshold.get(),
                self.edge_angle_range.get()
            )

            self.show_image(output_file)
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.loading_screen.destroy()

    def show_image(self, filename):
        image = Image.open(filename)
        image.thumbnail((800, 800))  # Resize for display purposes
        photo = ImageTk.PhotoImage(image)

        self.image_label.configure(image=photo)
        self.image_label.image = photo  # Keep a reference to avoid garbage collection

if __name__ == "__main__":
    app = Application()
    app.mainloop()

