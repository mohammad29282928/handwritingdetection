import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define directories
input_dir = '/content/in'
output_dir = '/content/out'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to remove background using GrabCut
def remove_background(image_path, output_path):
    """
    Remove the background from an image using the GrabCut algorithm.
    Args:
        image_path (str): The file path to the input image.
        output_path (str): The file path to save the output image with the background removed.
    Returns:
        None: The function saves the resulting image to the specified output path.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return
    mask = np.zeros(image.shape[:2], np.uint8)
    # Create models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Define the rectangle
    rect = (50, 50, image.shape[1] - 50, image.shape[0] - 50)

    # Apply GrabCut algorithm
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Modify the mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = image * mask2[:, :, np.newaxis]

    # Save the result
    cv2.imwrite(output_path, result)


# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        remove_background(image_path, output_path)

print("Background removal completed.")
