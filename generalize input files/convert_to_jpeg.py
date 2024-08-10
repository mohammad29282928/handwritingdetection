import os
import sys
from PIL import Image
from docx import Document
import io
import fitz  # PyMuPDF
from spire.doc import *
from spire.doc.common import *

def create_output_folder(file_path):
    """
    Create an output folder based on the input file's name if it doesn't already exist.
    
    Args:
    - file_path: The path to the input file.
    
    Returns:
    - output_folder: The path to the created output folder.
    """
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_folder = os.path.join(os.path.dirname(file_path), base_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return output_folder

def convert_image_to_jpeg(input_path, output_folder):
    """
    Convert an image file to JPEG format and save it to the output folder.
    
    Args:
    - input_path: The path to the input image file.
    - output_folder: The path to the output folder.
    """
    image = Image.open(input_path)
    image = image.convert('RGB')  # Convert the image to RGB format
    output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_path))[0]}.jpg")
    image.save(output_path, 'JPEG')  # Save the image as JPEG
def trim_whitespace(image):
    """
    Trim the white space around the image.

    Args:
    - image: The image to be trimmed.

    Returns:
    - trimmed_image: The trimmed image.
    """
    # Convert image to grayscale
    gray_image = ImageOps.grayscale(image)
    # Invert the image
    inverted_image = ImageChops.invert(gray_image)
    # Get bounding box of non-black areas
    bbox = inverted_image.getbbox()
    # Crop the image to the bounding box
    trimmed_image = image.crop(bbox)
    return trimmed_image    



def convert_pdf_to_images(pdf_path, output_folder, zoom=2):
    """
    Convert each page of a PDF to a separate JPEG image and save them to the output folder.
    
    Args:
    - pdf_path: The path to the input PDF file.
    - output_folder: The path to the output folder.
    - zoom: The zoom factor for the PDF pages (default is 2).
    """
    try:
        pdf_document = fitz.open(pdf_path)
        name_with_extension = os.path.basename(pdf_path)
        name = os.path.splitext(name_with_extension)[0]
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)  # Load the specified page
            matrix = fitz.Matrix(zoom, zoom)  # Create a transformation matrix for zooming
            pix = page.get_pixmap(matrix=matrix)  # Render the page to an image
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Trim the white space around the image
            trimmed_image = trim_whitespace(image)
            output_path = os.path.join(output_folder, f"{name}_page_{page_num + 1}.JPEG")
            trimmed_image.save(output_path, 'JPEG') # Save the image as JPEG
        
        print(f'Successfully converted {pdf_path} to images in {output_folder}')
    except Exception as e:
        print(f'Error converting {pdf_path}: {e}')
        print("\nTroubleshooting steps:")
        print("- Ensure the PDF file is not corrupted by opening it in a PDF viewer.")
        print("- Verify that the PDF file is accessible and not locked by another process.")
        print("- Check if other PDF files in the same directory are converted successfully.")

def convert_docx_to_jpeg(docx_path, output_folder):
    """
    Convert each page of a DOCX file to a separate JPEG image and save them to the output folder.
    
    Args:
    - docx_path: The path to the input DOCX file.
    - output_folder: The path to the output folder.
    """
    document = Document()
    document.LoadFromFile(docx_path)
    image_streams = document.SaveImageToStreams(ImageType.Bitmap)  # Save each page as an image stream
    for i, image_stream in enumerate(image_streams, start=1):
        output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(docx_path))[0]}_page_{i}.JPEG")
        with open(output_path, 'wb') as image_file:
            image_file.write(image_stream.ToArray())  # Write the image stream to a file
    document.Close()

def convert_stream_to_jpeg(stream, output_folder):
    """
    Convert an image stream to JPEG format and save it to the output folder.
    
    Args:
    - stream: The image stream to convert.
    - output_folder: The path to the output folder.
    """
    image = Image.open(io.BytesIO(stream))
    image = image.convert('RGB')  # Convert the image to RGB format
    output_path = os.path.join(output_folder, "stream_image.jpg")
    image.save(output_path, 'JPEG')  # Save the image as JPEG

def convert_file(input_path):
    """
    Convert an input file to JPEG format based on its file extension.
    
    Args:
    - input_path: The path to the input file.
    """
    file_extension = os.path.splitext(input_path)[1].lower()
    output_folder = create_output_folder(input_path)

    if file_extension in ['.png', '.jpeg', '.jpg', '.bmp', '.gif']:
        convert_image_to_jpeg(input_path, output_folder)
    elif file_extension == '.pdf':
        convert_pdf_to_images(input_path, output_folder)
    elif file_extension in ['.docx', '.doc']:
        convert_docx_to_jpeg(input_path, output_folder)
    else:
        print(f"File format {file_extension} is not supported.")

if __name__ == "__main__":
    input_files = sys.argv[1:]  # Get file paths from command-line arguments
    for file in input_files:
        convert_file(file)
