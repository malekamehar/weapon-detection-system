import cv2
import torch
from PIL import Image

# Load the AI model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_weapons(image_path):
    img = Image.open(image_path)  # Open the image
    results = model(img)  # Detect objects
    
    # Check for weapons
    weapons = []
    for detection in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = detection.tolist()
        class_name = model.names[int(cls)]
        if class_name in ['gun', 'knife'] and conf > 0.5:
            weapons.append(class_name)
    
    return weapons

# Test the function
image_path = "C:\Weapon detection deepseek\weapon_images\test.jpg"  # Replace with your image
detected_weapons = detect_weapons(image_path)

if detected_weapons:
    print("⚠️ Weapons found:", detected_weapons)
else:
    print("✅ No weapons detected.")