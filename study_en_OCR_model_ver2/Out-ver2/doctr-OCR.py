import os
import cv2
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
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
    Reads image files from the specified input folder.
    Args:
    input_folder (str): Path to the folder containing the input images.
    Returns:
    list: A list of file paths for the images in the input folder.
    """
    image_files = []
    for file in os.listdir(input_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Ensure we read only image files
            image_files.append(os.path.join(input_folder, file))
    return image_files

def save_image(output_folder, filename, image):
    """
    Saves the image to the specified output folder.
    Args:
    output_folder (str): Path to the folder where the image will be saved.
    filename (str): The name of the file to save the image as.
    image (numpy.ndarray): The image to save.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, f"{filename}.png")
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def perform_ocr_doctr(image_path):
    """
    Performs OCR on the image using the doctr library.
    Args:
    image_path (str): Path to the image file.
    Returns:
    Document: The OCR result as a doctr Document object.
    """
    model = ocr_predictor(pretrained=True)
    doc = DocumentFile.from_images(image_path)
    result = model(doc)
    return result

def draw_boxes(image_path, result):
    """
    Draws bounding boxes around the recognized text in the image.
    Args:
    image_path (str): Path to the image file.
    result (Document): The OCR result as a doctr Document object.
    Returns:
    numpy.ndarray: The image with bounding boxes drawn around recognized text.
    """
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    # Extract bounding box coordinates
                    (x_min, y_min), (x_max, y_max) = word.geometry
                    # Convert normalized coordinates to pixel coordinates
                    box = [
                        int(x_min * width), int(y_min * height),
                        int(x_max * width), int(y_max * height)
                    ]
                    # Draw rectangle on the image
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    return image

def extract_text(result):
    """
    Extracts text content from the OCR result.
    Args:
    result (Document): The OCR result as a doctr Document object.
    Returns:
    str: The extracted text content.
    """
    text_content = ""
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    text_content += word.value + " "
                text_content += "\n"
            text_content += "\n"
        text_content += "\n"
    return text_content

def apply_doctr_model(input_folder, output_folder):
    """
    Applies the doctr OCR model to images in the input folder and saves the results.
    Args:
    input_folder (str): Path to the folder containing the input images.
    output_folder (str): Path to the folder where the results will be saved.
    """
    image_files = read_images(input_folder)
    for image_file in image_files:
        result = perform_ocr_doctr(image_file)

        # Draw bounding boxes on the image
        annotated_image = draw_boxes(image_file, result)

        # Save the annotated image
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        save_image(output_folder, base_name, annotated_image)

        # Extract and save text content to a text file
        text_content = extract_text(result)
        text_file = os.path.join(output_folder, f"{base_name}.txt")
        with open(text_file, 'w') as f:
            f.write(text_content)

# Paths
input_folder = '/content/In'
output_folder = '/content/OCR_Doctr'

# Apply the OCR model
apply_doctr_model(input_folder, output_folder)