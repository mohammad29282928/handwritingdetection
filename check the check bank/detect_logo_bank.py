

pip install ultralytics
import requests
from ultralytics import YOLO
from google.colab import drive
#drive.mount('/content/drive')

#ROOT_DIR = '/content/drive/MyDrive/DE_logo_bank'

# Commented out IPython magic to ensure Python compatibility.

# %cd {ROOT_DIR}

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="ErfoJCqXb5qFukgQjPmE")
project = rf.workspace("edvin-9naiq").project("logo_bank")
version = project.version(1)
dataset = version.download("yolov8")

"""## YOLO8 MODEL1"""
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8x.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data="/content/drive/MyDrive/DE_logo_bank/logo_bank-1/data.yaml", epochs=100, imgsz=640)




# لینک اشتراک‌گذاری فایل در Google Drive
file_id = '1Q0kc8cGETcY0Wg9X-v0SuyvUK_BGe3m_'
file_url = f'https://drive.google.com/uc?export=download&id={file_id}'

# نام فایل دانلود شده
file_name = 'best.pt'

# درخواست دانلود فایل
response = requests.get(file_url)

# ذخیره فایل در سیستم
with open(file_name, 'wb') as file:
    file.write(response.content)

print(f'File {file_name} has been downloaded successfully.')

# Load the pretrained YOLOv8n model
model = YOLO(file_name)

# Run inference on 'X.jpg' with arguments
model.predict("X.jpg", save=True, imgsz=320, conf=0.5)



