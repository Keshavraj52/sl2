import torch
import cv2
import numpy as np

# Load a pre-trained YOLOv5 model from Ultralytics
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load an image
img_path = 'your_image.jpg'  # Provide path to your image
img = cv2.imread(img_path)

# Convert BGR to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Inference
results = model(img_rgb)

# Results
results.print()  # Print detected objects
results.show()   # Show image with bounding boxes
results.save()   # Save the image with bounding boxes

# Get pandas dataframe of detection
df = results.pandas().xyxy[0]  # Bounding boxes with scores
print(df)
