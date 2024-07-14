import os
from sklearn.cluster import MeanShift, estimate_bandwidth
import cv2
import numpy as np

input_dir = '/content/in'
output_dir = '/content/out'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        # Read the image
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Failed to read {filename}")
            continue
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Reshape the image to a 2D array of pixels
        pixels = image_rgb.reshape(-1, 3)
        
        # Estimate bandwidth for MeanShift
        bandwidth = estimate_bandwidth(pixels, quantile=0.2, n_samples=500)
        # Apply MeanShift clustering
        mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        mean_shift.fit(pixels)
        labels = mean_shift.labels_
        
        # Reshape the labels to the shape of the image
        segmented_image = labels.reshape(image_rgb.shape[:2])
        
        # Create a mask for the background
        background_label = np.bincount(labels).argmax()
        background_mask = segmented_image == background_label
        background_mask = background_mask.astype(np.uint8) * 255
        
        # Apply the mask to the original image
        result = cv2.bitwise_and(image, image, mask=background_mask)
        
        # Save the result
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, result)
        print(f"Processed and saved {filename}")

print("Processing complete.")
