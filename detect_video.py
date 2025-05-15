import cv2
import torch

# Load AI model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open webcam
cap = cv2.VideoCapture(0)  # '0' for default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect objects
    results = model(frame)
    
    # Draw boxes around weapons
    for detection in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = detection.tolist()
        class_name = model.names[int(cls)]
        if class_name in ['gun', 'knife'] and conf > 0.5:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Show video feed
    cv2.imshow("Weapon Detection", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()