import pytesseract
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

def preprocess_and_tesseract_ocr(input_folder, output_folder):
    """
    Preprocess images using OpenCV and uses Tesseract OCR to read text from images in the input folder and save highlighted images and text in the output folder.

    Args:
    input_folder (str): Path to the input folder containing images.
    output_folder (str): Path to the output folder to save highlighted images and text.
    """
    image_files = read_images(input_folder)
    for image_file in image_files:
        image = cv2.imread(image_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Use Tesseract with confidence level output
        tesseract_data = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT)
        tesseract_boxes = []
        text_content = ""

        for i in range(len(tesseract_data['level'])):
            (x, y, w, h, text, conf) = (tesseract_data['left'][i], tesseract_data['top'][i], 
                                        tesseract_data['width'][i], tesseract_data['height'][i], 
                                        tesseract_data['text'][i], tesseract_data['conf'][i])
            if conf != '-1':
                tesseract_boxes.append((x, y, x + w, y + h))
                text_content += f"Text: {text}, Confidence: {conf}\n"

        output_image = draw_boxes(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB), tesseract_boxes, color=(255, 0, 0))  # Red
        filename = os.path.splitext(os.path.basename(image_file))[0]
        save_image(output_folder, filename, output_image)
        save_text(output_folder, filename, text_content.strip())

preprocess_and_tesseract_ocr('/content/In', '/content/pytesseract_opncv')