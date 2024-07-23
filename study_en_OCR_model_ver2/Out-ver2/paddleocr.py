from paddleocr import PaddleOCR

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

def paddleocr_ocr(input_folder, output_folder):
    """
    Uses PaddleOCR to read text from images in the input folder and save highlighted images and text in the output folder.

    Args:
    input_folder (str): Path to the input folder containing images.
    output_folder (str): Path to the output folder to save highlighted images and text.
    """
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    image_files = read_images(input_folder)
    for image_file in image_files:
        image = cv2.imread(image_file)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        paddleocr_results = ocr.ocr(image_file, cls=True)
        paddleocr_boxes = []
        text_content = ""

        for result in paddleocr_results:
            for line in result:
                box = line[0]
                text = line[1][0]
                confidence = line[1][1]
                paddleocr_boxes.append([int(min(point[0] for point in box)), int(min(point[1] for point in box)),
                                        int(max(point[0] for point in box)), int(max(point[1] for point in box))])
                text_content += f"Text: {text}, Confidence: {confidence:.2f}\n"

        output_image = draw_boxes(image_rgb.copy(), paddleocr_boxes, color=(0, 0, 255))  # Blue
        filename = os.path.splitext(os.path.basename(image_file))[0]
        save_image(output_folder, filename, output_image)
        save_text(output_folder, filename, text_content.strip())

paddleocr_ocr('/content/In', '/content/PaddleOCR')