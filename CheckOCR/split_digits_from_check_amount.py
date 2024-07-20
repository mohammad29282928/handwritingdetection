import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_image(img, title='Image', cmap='gray'):
    plt.figure(figsize=(10, 5))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load the image
image_path = '/content/perdigits3.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
plot_image(image, 'Original Image')

# Apply binary thresholding
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
plot_image(binary_image, 'Binary Image')

# Perform morphological operations to clean the image and highlight vertical lines
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
morph_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
plot_image(morph_image, 'Morphological Transformation')

# Detect vertical lines using Hough Line Transform
edges = cv2.Canny(morph_image, 50, 150, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# Sort detected lines by their x-coordinate
if lines is not None:
    lines = sorted(lines, key=lambda x: x[0][0])
    # Extract x-coordinates of detected vertical lines
    x_coords = [line[0][0] for line in lines]
else:
    x_coords = []

# Ensure there are exactly 16 coordinates to define 15 cells (including start and end)
if len(x_coords) != 16:
    x_coords = np.linspace(0, binary_image.shape[1], 16, dtype=int)

# Adjust x-coordinates to ensure we don't include borders in the cells
x_coords = np.array(x_coords)
border_padding = 5  # Adjust padding as necessary to exclude borders

# Crop each cell based on the adjusted x-coordinates
cell_images = []
for i in range(15):
    x1 = x_coords[i] + border_padding
    x2 = x_coords[i + 1] - border_padding
    cell_image = binary_image[:, x1:x2]
    cell_images.append(cell_image)

# Display each cell image
plt.figure(figsize=(15, 5))
for i, cell_image in enumerate(cell_images):
    plt.subplot(1, 15, i + 1)
    plt.imshow(cell_image, cmap='gray')
    plt.axis('off')
plt.show()

# Save each cell image for further processing (OCR)
for i, cell_image in enumerate(cell_images):
    cv2.imwrite(f'cell_{i + 1}.png', cell_image)