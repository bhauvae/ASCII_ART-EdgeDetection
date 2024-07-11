import cv2
import numpy as np
COLOUR_MAP_HEX = {
    "blue-green": ("#0A174E", "#F5D042"),
    "hacker-man": ("#120d13", "#29ff2a"),
    "purple-salmon": ("#180a1c", "#e05964"),
    "darkblue-white": ("#1f1735", "#fffcff"),
}
color_map = "purple-salmon"
image = cv2.imread('barn_bloom.jpg', cv2.IMREAD_GRAYSCALE)
colour_map_hex = COLOUR_MAP_HEX
# Define colors

hex_to_rgb = lambda hex_color: tuple(
    int(hex_color[i : i + 2], 16) for i in (5, 3, 1)
)

# Create an RGB image array with the same height and width as the grayscale image
color_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

# Map black (0) to green and white (255) to pink
color_image[image == 0] = hex_to_rgb(colour_map_hex[color_map][0])
color_image[image == 255] = hex_to_rgb(colour_map_hex[color_map][1])

cv2.imwrite("testasciibw_colour.png", color_image)

   