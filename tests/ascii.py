import cv2
import numpy as np
from multiprocessing import Pool, cpu_count

# Define the grayscale characters from darkest to lightest
gray_scales = [r"$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~i!lI;:,\"^`'. ",r" .;coP0?@█"]
gray_scale = gray_scales[1]

text_resolution = 8
filename = r"img.png"
output_ascii_filename = r"./ascii_art.png"


def process_batch(batch):
    return [process_pixel(pixel_value) for pixel_value in batch]


def process_pixel(pixel_value):
    pixel_value = int(pixel_value)  # Ensure pixel_value is an integer
    index = int(pixel_value * (len(gray_scale) - 1) / 255)
    return gray_scale[index]


def create_ascii_image(img, new_w, new_h, font_scale=0.4, thickness=1):
    # Create a blank white image
    ascii_img = (
        np.ones((new_h * text_resolution, new_w * text_resolution, 3), dtype=np.uint8) * 0
    )

    # Flatten the image array and process it in batches
    flattened_img = img.flatten()
    num_cores = cpu_count()
    batch_size = len(flattened_img) // num_cores

    batches = [
        flattened_img[i : i + batch_size]
        for i in range(0, len(flattened_img), batch_size)
    ]

    with Pool(processes=num_cores) as pool:
        ascii_chars_batches = pool.map(process_batch, batches)

    ascii_chars = [char for batch in ascii_chars_batches for char in batch]
    ascii_chars = np.array(ascii_chars).reshape((new_h, new_w))
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    for y in range(new_h):
        for x in range(new_w):
            char = ascii_chars[y, x]
            cv2.putText(
                ascii_img,
                char,
                (x * text_resolution, y * text_resolution + text_resolution),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    return ascii_img


def main():
    try:
        # Open the image and convert to grayscale
        input_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if input_image is None:
            raise ValueError(f"Could not open or find the image '{filename}'")

        original_h, original_w = input_image.shape

        # Calculate the new dimensions
        new_w = original_w // text_resolution
        new_h = original_h // text_resolution

        # Resize the image
        resized_image = cv2.resize(
            input_image, (new_w, new_h), interpolation=cv2.INTER_AREA
        )

        # Create ASCII art image
        ascii_img = create_ascii_image(resized_image, new_w, new_h)

        # Save the ASCII art image
        cv2.imwrite(output_ascii_filename, ascii_img)
        print(f"ASCII art image saved as '{output_ascii_filename}'")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
