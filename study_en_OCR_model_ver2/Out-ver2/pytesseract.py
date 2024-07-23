import os
import cv2
from PIL import Image
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


def tesseract_ocr(input_folder, output_folder):
    """
    Uses Tesseract OCR to read text from images in the input folder and save highlighted images and text in the output folder.

    Args:
    input_folder (str): Path to the input folder containing images.
    output_folder (str): Path to the output folder to save highlighted images and text.
    """
    image_files = read_images(input_folder)
    for image_file in image_files:
        image = cv2.imread(image_file)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        d = pytesseract.image_to_data(image_rgb, output_type=pytesseract.Output.DICT)
        n_boxes = len(d['level'])
        text_content = ""
        tesseract_boxes = []
        
        for i in range(n_boxes):
            if int(d['conf'][i]) > 50:  # Confidence threshold
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                tesseract_boxes.append((x, y, x + w, y + h))
                text_content += d['text'][i] + " "
            if d['level'][i] == 5:  # End of a paragraph
                text_content += "\n"

        output_image = draw_boxes(image_rgb.copy(), tesseract_boxes, color=(255, 0, 0))  # Red
        filename = os.path.splitext(os.path.basename(image_file))[0]
        save_image(output_folder, filename, output_image)
        save_text(output_folder, filename, text_content.strip())

tesseract_ocr('/content/In', '/content/pytesseract')
