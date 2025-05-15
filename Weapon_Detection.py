import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Load YOLO model
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")
layer_names = net.getLayerNames()
out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Weapon classes to detect
target_classes = ["knife", "gun", "firearm", "pistol"]

# Image detection function
def detect_weapons(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(out_layers)

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in target_classes:
                center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype("int")
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) == 0:
        messagebox.showinfo("Result", "No weapon detected!")
        return

    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Image Weapon Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Live webcam detection function
def detect_weapons_live():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Camera Error", "Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(out_layers)

        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] in target_classes:
                    center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype("int")
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Live Weapon Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI actions
def select_image():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if path:
        detect_weapons(path)

# GUI Design
root = tk.Tk()
root.title("Weapon Detection App")
root.geometry("550x450")
root.configure(bg="#222831")

title = tk.Label(root, text="Weapon Detection System", fg="white", bg="#222831", font=("Helvetica", 22, "bold"))
title.pack(pady=30)

img_btn = tk.Button(root, text="📷 Detect Weapon from Image", font=("Arial", 14), bg="#393E46", fg="white", width=32, command=select_image)
img_btn.pack(pady=20)

live_btn = tk.Button(root, text="🎥 Start Live Detection", font=("Arial", 14), bg="#00ADB5", fg="white", width=32, command=detect_weapons_live)
live_btn.pack(pady=20)

info = tk.Label(root, text="Press 'Q' to quit Live Detection", bg="#222831", fg="#AAAAAA", font=("Arial", 10))
info.pack(side="bottom", pady=15)

root.mainloop()
