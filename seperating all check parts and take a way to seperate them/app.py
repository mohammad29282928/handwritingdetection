import gradio as gr
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
#link of best.pt file  https://drive.google.com/file/d/1B7ELaPVjxHcWYaawLfG7nRq1vQvg3d0R/view?usp=sharing


# Load the YOLOv8 model
model = YOLO('best.pt')

# Define a list of colors for the 6 classes
colors = [
    (255, 0, 0),     # Red
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255)    # Yellow
]

def detect_objects(image):
    # Resize the input image to 200x200 pixels
    image = image.resize((200, 200))
    
    # Convert the input image to a format YOLO can work with
    image = np.array(image)
    
    # Perform detection
    results = model(image)[0]
    
    # Draw bounding boxes on the image
    for box in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, score, class_id = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Select color for the class id
        color = colors[int(class_id) % len(colors)]
        
        # Draw the bounding box with a thinner line
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)  # Line width set to 1
        
        # Put the class name above the bounding box with smaller font
        class_name = model.model.names[int(class_id)]
        cv2.putText(image, f'{class_name} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Convert back to PIL image
    return Image.fromarray(image)

# Define the Gradio interface
interface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="YOLOv8 Object Detection",
    description="Upload an image and YOLOv8 will detect objects in the image."
)

# Launch the interface
interface.launch()
