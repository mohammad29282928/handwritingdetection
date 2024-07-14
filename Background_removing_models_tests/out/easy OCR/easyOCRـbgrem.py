#pip install easyocr opencv-python-headless

import os
import cv2
import numpy as np
import easyocr
from rembg import remove
from PIL import Image

# Initialize EasyOCR reader
reader = easyocr.Reader(['en','fa'])

# Define input and output directories
input_dir = '/content/in'
output_dir = '/content/out'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to create a mask for text areas
def create_text_mask(image):
    """
    Create a mask for the text areas in the image using text detection.

    Args:
        image (numpy.ndarray): The input image on which text detection will be performed.

    Returns:
        numpy.ndarray: A binary mask where the text areas are marked with white (255) and the background with black (0).
    """
    # Perform text detection
    results = reader.readtext(image)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Create a mask for each detected text box
    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        top_right = tuple(map(int, top_right))
        bottom_right = tuple(map(int, bottom_right))
        bottom_left = tuple(map(int, bottom_left))

        # Draw the text box on the mask
        cv2.fillPoly(mask, [np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)], 255)

    return mask

# Iterate through all files in the input directory
for filename in os.listdir(input_dir):
    # Construct full file path
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    try:
        # Open the input image
        input_image = cv2.imread(input_path)

        # Convert the input image to 4-channel (BGRA) if it is not already
        if input_image.shape[2] == 3:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2BGRA)

        # Create text mask
        text_mask = create_text_mask(input_image)

        # Invert the text mask
        inverted_mask = cv2.bitwise_not(text_mask)

        # Remove the background using rembg
        pil_image = Image.open(input_path)
        bg_removed = remove(pil_image)

        # Convert PIL image to OpenCV format
        bg_removed_cv = cv2.cvtColor(np.array(bg_removed), cv2.COLOR_RGBA2BGRA)

        # Combine background removed image with text mask
        bg_removed_cv[text_mask == 255] = input_image[text_mask == 255]

        # Save the final image
        output_pil = Image.fromarray(cv2.cvtColor(bg_removed_cv, cv2.COLOR_BGRA2RGBA))
        output_pil.save(output_path, format="PNG")

        print(f"Processed and saved: {output_path}")

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
