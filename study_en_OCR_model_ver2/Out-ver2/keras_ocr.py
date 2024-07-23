import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import keras_ocr
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def read_images(input_folder):
    """
    Reads all images from the input folder.

    Args:
    input_folder (str): Path to the input folder containing images.

    Returns:
    List of image file paths.
    """
    image_files = []
    for file in os.listdir(input_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_files.append(os.path.join(input_folder, file))
    return image_files

def draw_boxes(image, boxes, color=(0, 255, 0)):
    for box in boxes:
        if len(box) == 4:  # Ensuring box has the correct format
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
    return image

def save_image(output_folder, filename, image):
    os.makedirs(output_folder, exist_ok=True)
    Image.fromarray(image).save(os.path.join(output_folder, filename + '_highlighted.png'))

def save_text(output_folder, filename, text):
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, filename + '.txt'), 'w') as f:
        f.write(text)

def read_images(input_folder):
    """
    Reads all images from the input folder.
    Args:
    input_folder (str): Path to the input folder containing images.
    Returns:
    List of image file paths.
    """
    image_files = []
    for file in os.listdir(input_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_files.append(os.path.join(input_folder, file))
    return image_files

def preprocess_image(image):
    """
    Preprocess the image to improve OCR results.
    Args:
    image (numpy array): The input image.
    Returns:
    numpy array: The preprocessed image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    processed_image = cv2.adaptiveThreshold(resized, 255, 
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 31, 2)
    # Convert back to a 3-channel image
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
    return processed_image

def draw_boxes(image, boxes, color=(0, 255, 0)):
    for box in boxes:
        if len(box) == 4:  # Ensuring box has the correct format
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
    return image

def save_image(output_folder, filename, image):
    os.makedirs(output_folder, exist_ok=True)
    Image.fromarray(image).save(os.path.join(output_folder, filename + '_highlighted.png'))

def save_text(output_folder, filename, text):
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, filename + '.txt'), 'w') as f:
        f.write(text)

def keras_ocr_function(input_folder, output_folder):
    """
    Uses Keras-OCR to read text from images in the input folder and save highlighted images in the output folder.
    Args:
    input_folder (str): Path to the input folder containing images.
    output_folder (str): Path to the output folder to save highlighted images.
    """
    pipeline = keras_ocr.pipeline.Pipeline()
    image_files = read_images(input_folder)
    for image_file in image_files:
        image = keras_ocr.tools.read(image_file)
        preprocessed_image = preprocess_image(image)
        predictions = pipeline.recognize([preprocessed_image])[0]
        boxes = []
        text_with_confidence = []

        for prediction in predictions:
            text = prediction[0]
            box = prediction[1]
            text_with_confidence.append(f"{text}")

            x1, y1 = int(box[0][0]), int(box[0][1])
            x2, y2 = int(box[2][0]), int(box[2][1])
            boxes.append([x1, y1, x2, y2])

        output_image = draw_boxes(preprocessed_image.copy(), boxes, color=(0, 255, 0))  # Green
        filename = os.path.splitext(os.path.basename(image_file))[0]
        save_image(output_folder, filename, output_image)
        save_text(output_folder, filename, "\n".join(text_with_confidence))

# Usage example:
keras_ocr_function('/content/In', 'Keras_ocr')
