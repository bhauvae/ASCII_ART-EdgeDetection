import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
#TODO TRY IMREAD REDUCED GRAYSCALE
# Define the grayscale characters from darkest to lightest
GRAY_SCALE = " .;coP0?@$"
GRAY_SCALE_LOOKUP = np.array([i for i in range(10) for _ in range(26)], dtype=np.uint8)[4:260]
TEXT_IMAGES = [cv2.imread(f"./char/{i}.png", cv2.IMREAD_GRAYSCALE) for i in range(10)]
TEXT_RESOLUTION = 8
filename = "images\dog.png"
output_ascii_filename = "ascii_" + filename


def create_ascii_block(args):
    img, y, x, TEXT_RESOLUTION, TEXT_IMAGES, GRAY_SCALE_LOOKUP = args

    # Calculate the start and end coordinates for placing the character image
    start_y = y * TEXT_RESOLUTION
    end_y = (y + 1) * TEXT_RESOLUTION
    start_x = x * TEXT_RESOLUTION
    end_x = (x + 1) * TEXT_RESOLUTION

    # Get the index of the character image based on pixel intensity
    char_index = GRAY_SCALE_LOOKUP[img[y, x]]

    # Get the corresponding character image
    char_img = TEXT_IMAGES[char_index]

    # Create a block of ASCII characters
    block = np.zeros((TEXT_RESOLUTION, TEXT_RESOLUTION, 3), dtype=np.uint8)
    block[:, :] = char_img

    return start_y, end_y, start_x, end_x, block


def create_ascii_image(img, new_w, new_h):
    args = [(img, y, x, TEXT_RESOLUTION, TEXT_IMAGES, GRAY_SCALE_LOOKUP)
            for y in range(new_h) for x in range(new_w)]

    # Use multiprocessing Pool to speed up the conversion
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(create_ascii_block, args)

    # Create a blank white image
    ascii_img = np.zeros((new_h * TEXT_RESOLUTION, new_w * TEXT_RESOLUTION, 3), dtype=np.uint8)

    # Merge the results into the ascii_img
    for result in results:
        start_y, end_y, start_x, end_x, block = result
        ascii_img[start_y:end_y, start_x:end_x] = block

    return ascii_img


def main():
    # Open the image and convert to grayscale
    input_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        raise ValueError(f"Could not open or find the image '{filename}'")

    original_h, original_w = input_image.shape

    # Calculate the new dimensions
    new_w = original_w // TEXT_RESOLUTION
    new_h = original_h // TEXT_RESOLUTION

    # Resize the image
    resized_image = cv2.resize(input_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create ASCII art image
    ascii_img = create_ascii_image(resized_image, new_w, new_h)

    # Save the ASCII art image
    cv2.imwrite(output_ascii_filename, ascii_img)
    print(f"ASCII art image saved as '{output_ascii_filename}'")


if __name__ == "__main__":
    main()
