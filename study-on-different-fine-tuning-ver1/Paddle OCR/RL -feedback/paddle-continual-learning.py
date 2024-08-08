import os
import shutil
import pandas as pd
import cv2
from pathlib import Path

# Configuration
TRAIN_FOLDER = '/content/train'
CSV_PATH = 'train_data.csv'
THRESHOLD = 5  # Number of samples required to trigger fine-tuning

def apply_paddleocr(image_path):
    command = f'python /content/PaddleOCR/tools/infer/predict_rec.py --image_dir="{image_path}" --rec_model_dir="/content/infer_model/" --rec_image_shape="3, 48, 320" --rec_char_dict_path="/content/fa_dict.txt"'
    os.system(command)

def get_user_feedback():
    ground_truth = input("Please enter the ground truth text for the OCR result: ")
    return ground_truth

def copy_image_and_append_csv(image_path, ground_truth, train_folder, csv_file):
    # Create 'train' folder if it doesn't exist
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    
    # Copy image to 'train' folder
    shutil.copy(image_path, train_folder)
    
    # Get the image name
    image_name = os.path.basename(image_path)
    
    # Append to the CSV file with the image path and ground truth text
    data = {'image': [os.path.join(train_folder, image_name)], 'label': [ground_truth]}
    df = pd.DataFrame(data)
    if not os.path.exists(csv_file):
        df.to_csv(csv_file, index=False)
    else:
        df.to_csv(csv_file, mode='a', header=False, index=False)

def generate_txt_from_csv(csv_file):
    txt_file = csv_file.replace('.csv', '.txt')
    command = f'python /content/PaddleOCR/ppocr/utils/gen_label.py --mode="rec" --input_path={csv_file} --output_label={txt_file}'
    os.system(command)
    return txt_file

def train_model():
    command = 'python3 /content/PaddleOCR/tools/train.py -c /content/en_PP-OCRv3_rec.yml -o Global.pretrained_model=/content/pre-trained/best_accuracy'
    os.system(command)

def main():
    sample_count = 0
    csv_file = os.path.join(TRAIN_FOLDER, CSV_PATH)

    while True:
        # Get image path from user
        image_path = input("Please enter the image path (or type 'exit' to quit): ")
        if image_path.lower() == 'exit':
            break

        # Apply PaddleOCR on the image
        apply_paddleocr(image_path)
        
        # Get ground truth text from user
        ground_truth = get_user_feedback()
        
        # Copy image and append to CSV file
        copy_image_and_append_csv(image_path, ground_truth, TRAIN_FOLDER, csv_file)
        
        sample_count += 1
        print(f"Sample {sample_count} added.")

        # Check if we have reached the threshold
        if sample_count >= THRESHOLD:
            print("Threshold reached. Generating txt file and starting fine-tuning.")
            
            # Generate txt file from csv file
            generate_txt_from_csv(csv_file)
            
            # Train the model with the new data
            train_model()
            
            # Reset the sample count
            sample_count = 0

if __name__ == "__main__":
    main()
