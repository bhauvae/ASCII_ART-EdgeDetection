import cv2
import numpy as np

def increase_contrast_tonemapping(image_path, output_path):
    # Read the image
    ldr_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Convert the image to float32 for processing
    hdr_image = ldr_image.astype(np.float32) / 255.0
    
    # Create a tone mapping object (you can choose different algorithms)
    tonemap = cv2.createTonemapDrago(0.2, 0.5)  # Parameters: gamma, saturation

    # Apply tone mapping
    ldr_tonemapped = tonemap.process(hdr_image)

    # Convert the tone-mapped image back to 8-bit format
    ldr_tonemapped = np.clip(ldr_tonemapped * 255, 0, 255).astype('uint8')
    
    # Save the result
    cv2.imwrite(output_path, ldr_tonemapped)

# Usage
increase_contrast_tonemapping('test_bloom.jpg', 'test_contrast.png')
