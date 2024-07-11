from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Constants
GRAY_SCALE = " .;coP0?@█"
GRAY_SCALE_LOOKUP = np.array([char for char in GRAY_SCALE for _ in range(26)])[4:260]

TEXT_RESOLUTION = 8


def main():
    # File paths
    filename = "images\IMG-20240613-WA0006.jpg"
    output_ascii_filename = "ascii_" + filename

    # Open the image and convert to grayscale
    input_image = Image.open(filename).convert("L")
    original_w, original_h = input_image.size

    # Calculate the new dimensions
    new_w = original_w // TEXT_RESOLUTION
    new_h = original_h // TEXT_RESOLUTION

    # Resize the image
    resized_image = input_image.resize((new_w, new_h))

    # Convert image to numpy array
    img = np.array(resized_image)

    # Create a new image for the ASCII art
    ascii_img = Image.new(
        "RGB", (new_w * TEXT_RESOLUTION, new_h * TEXT_RESOLUTION), color=(0, 0, 0)
    )
    draw = ImageDraw.Draw(ascii_img)

    # Load a font
    try:
        font = ImageFont.truetype("./ascii-art.ttf", size=TEXT_RESOLUTION)
    except IOError:
        font = ImageFont.load_default()

    # Draw the ASCII art on the image
    for y in range(new_h):
        for x in range(new_w):
            pixel_value = img[y, x]
            gray_scale_value = GRAY_SCALE_LOOKUP[pixel_value]
            draw.text(
                (x * TEXT_RESOLUTION, y * TEXT_RESOLUTION),
                gray_scale_value,
                font=font,
                fill=(255, 255, 255),
            )

    # Save the ASCII art image
    ascii_img.save(output_ascii_filename)
    print(f"ASCII art image saved as '{output_ascii_filename}'")


if __name__ == "__main__":
    main()
