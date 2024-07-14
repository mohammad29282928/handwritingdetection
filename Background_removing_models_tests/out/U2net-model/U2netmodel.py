"""run this commands before running code 
!pip install pytesseract
!git clone https://github.com/xuebinqin/U-2-Net.git
!gdown 1IG3HdpcRiDoWNookbncQjeaPN28t90yW
!gdown 1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ 
!apt-get update
!apt-get install -y tesseract-ocr
"""

import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
import torch
from torchvision import transforms
from torch.nn.functional import interpolate

# Add the U2Net model directory to the Python path
import sys
sys.path.append('/content/U-2-Net/model')

from u2net import U2NET

# Initialize Tesseract OCR for Persian and English
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
tess_config = '--oem 3 --psm 6'
languages = 'eng+fas'

# Load the U^2-Net model
u2net = U2NET(3, 1)
u2net.load_state_dict(torch.load('/content/u2net.pth', map_location=torch.device('cpu')))
u2net.eval()

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
    results = pytesseract.image_to_boxes(image, lang=languages, config=tess_config)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Create a mask for each detected text box
    for result in results.splitlines():
        result = result.split()
        x, y, w, h = int(result[1]), int(result[2]), int(result[3]), int(result[4])
        cv2.rectangle(mask, (x, image.shape[0] - y), (w, image.shape[0] - h), 255, -1)

    return mask


# Function to perform background removal with U^2-Net
def remove_background(image):
    """
    Perform background removal on the input image using the U^2-Net model.
    Args:
        image (PIL.Image.Image): The input image from which the background will be removed.
    Returns:
        PIL.Image.Image: The input image with the background removed, where the background is made transparent.
    """
    # Convert image to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image_tensor = transform(image).unsqueeze(0)

    # Predict the mask with U^2-Net
    with torch.no_grad():
        d1, _, _, _, _, _, _ = u2net(image_tensor)
        pred = d1[:, 0, :, :]
        pred = normalize_pred(pred)

    # Resize the mask to the original image size
    pred = interpolate(pred.unsqueeze(0), size=(image.size[1], image.size[0]), mode='bilinear').squeeze()
    mask = pred.cpu().data.numpy()
    mask = (mask * 255).astype(np.uint8)

    # Apply the mask to remove the background
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGRA)
    image_np[:, :, 3] = mask

    return Image.fromarray(image_np)


# Function to normalize the predicted mask
def normalize_pred(d):
    """
    Normalize the predicted mask to a range between 0 and 1.
    Args:
        d (torch.Tensor): The predicted mask tensor.
    Returns:
        torch.Tensor: The normalized predicted mask tensor.
    """
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


# Iterate through all files in the input directory
for filename in os.listdir(input_dir):
    # Construct full file path
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    try:
        # Open the input image
        input_image = Image.open(input_path).convert('RGBA')

        # Create text mask
        text_mask = create_text_mask(np.array(input_image))

        # Convert RGBA to RGB
        input_image_rgb = input_image.convert('RGB')

        # Remove the background using U^2-Net
        bg_removed = remove_background(input_image_rgb)

        # Convert the bg_removed image to OpenCV format
        bg_removed_cv = cv2.cvtColor(np.array(bg_removed), cv2.COLOR_RGB2BGRA)

        # Combine background removed image with text mask
        bg_removed_cv[text_mask == 255] = np.array(input_image)[text_mask == 255]

        # Save the final image
        output_pil = Image.fromarray(cv2.cvtColor(bg_removed_cv, cv2.COLOR_BGRA2RGBA))
        output_pil.save(output_path, format="PNG")

        print(f"Processed and saved: {output_path}")

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
