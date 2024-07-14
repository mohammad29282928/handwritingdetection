
# Imports
import cv2
import pytesseract
import easyocr
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Function to draw bounding boxes
def draw_boxes(image, boxes, color=(0, 255, 0)):
    for box in boxes:
        if len(box) == 4:  # Ensuring box has the correct format
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
    return image

# Read an image from input
input_image_path = '/content/ocr.png'  # Replace with your image path
image = cv2.imread(input_image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 1. Tesseract OCR
tesseract_data = pytesseract.image_to_boxes(image).splitlines()
tesseract_boxes = []
for line in tesseract_data:
    parts = line.split(' ')
    if len(parts) >= 6:
        x1, y1, x2, y2 = map(int, parts[1:5])
        tesseract_boxes.append((x1, image.shape[0] - y2, x2, image.shape[0] - y1))

# 2. EasyOCR
reader = easyocr.Reader(['en'])
easyocr_results = reader.readtext(image_rgb)
easyocr_boxes = []
for result in easyocr_results:
    box = result[0]
    easyocr_boxes.append([int(min(point[0] for point in box)), int(min(point[1] for point in box)),
                          int(max(point[0] for point in box)), int(max(point[1] for point in box))])

# 3. PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')
paddleocr_results = ocr.ocr(input_image_path, cls=True)
paddleocr_boxes = []
for result in paddleocr_results:
    for line in result:
        box = line[0]
        paddleocr_boxes.append([int(min(point[0] for point in box)), int(min(point[1] for point in box)),
                                int(max(point[0] for point in box)), int(max(point[1] for point in box))])

# Draw bounding boxes and display each result separately
models = {
    "Tesseract OCR": (tesseract_boxes, (255, 0, 0)),  # Red
    "EasyOCR": (easyocr_boxes, (0, 255, 0)),         # Green
    "PaddleOCR": (paddleocr_boxes, (0, 0, 255))     # Blue
}

for model_name, (boxes, color) in models.items():
    output_image = image_rgb.copy()
    output_image = draw_boxes(output_image, boxes, color=color)
    plt.figure(figsize=(12, 12))
    plt.title(model_name)
    plt.imshow(output_image)
    plt.axis('off')
    plt.show()
