#pip install rembg
import os
from rembg import remove
from PIL import Image

# Define input and output directories
input_dir = '/content/in'
output_dir = '/content/out'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate through all files in the input directory
for filename in os.listdir(input_dir):
    # Construct full file path
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    try:
        # Open the input image
        input_image = Image.open(input_path)

        # Remove the background
        output_image = remove(input_image)

        # Ensure the output format is PNG to support transparency
        if output_image.mode in ("RGBA", "LA") or (output_image.mode == "P" and "transparency" in output_image.info):
            output_image.save(output_path, format="PNG")
        else:
            output_image.save(output_path)

        print(f"Processed and saved: {output_path}")

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
