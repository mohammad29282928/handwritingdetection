import cv2
import numpy as np

def remove_stamp(image_path, bbox):
    # Load the image
    image = cv2.imread(image_path)
    
    # Create a mask with the same dimensions as the image, initialized to zero (black)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Draw a white rectangle on the mask for the bounding box
    x, y, w, h = bbox
    mask[y:y+h, x:x+w] = 255
    
    # Apply inpainting using the Telea method
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=18, flags=cv2.INPAINT_TELEA)
    
    # Save or display the resulting image
    output_path = 'inpainted_image.png'
    cv2.imwrite(output_path, inpainted_image)
    
    return output_path

# Example usage:
image_path = '/content/input.jpg'
bbox = (2052, 1173,1000,500)  # Replace with your bounding box coordinates

output_image_path = remove_stamp(image_path, bbox)
print(f"Stamp removed image saved at: {output_image_path}")