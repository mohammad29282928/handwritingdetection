

pip install ultralytics

from google.colab import drive
drive.mount('/content/drive')

ROOT_DIR = '/content/drive/MyDrive/DE_logo_bank'

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


from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("best.pt")

# Run inference on 'bus.jpg' with arguments
model.predict("X.jpg", save=True, imgsz=320, conf=0.5)




