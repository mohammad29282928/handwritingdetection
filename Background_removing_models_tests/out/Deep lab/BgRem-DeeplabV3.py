import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np

# Define directories
input_dir = '/content/in'
output_dir = '/content/out'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load pre-trained DeepLabV3 model
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()

# Preprocess transformation
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to remove background using DeepLabV3
def remove_background(image):
    """
    Remove the background from the input image using the DeepLabV3 model.
    Args:
        image (PIL.Image.Image): The input image from which the background will be removed.
    Returns:
        PIL.Image.Image: The input image with the background removed, where the background is replaced with black.
    """
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # Create the mask and apply it
    mask = output_predictions.byte().cpu().numpy()
    mask = (mask == 15).astype(np.uint8)  # Assuming 15 is the index for the person class
    result = np.array(image) * mask[:, :, np.newaxis]
    
    return Image.fromarray(result)

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_dir, filename)
        image = Image.open(image_path)
        
        # Remove background
        result_image = remove_background(image)
        
        # Save the result
        result_path = os.path.join(output_dir, filename)
        result_image.save(result_path)

print("Background removal completed.")
