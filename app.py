import cv2
import torch
import serial
import time
from ultralytics import YOLO

model = YOLO("yolov5s.pt")  

cap = cv2.VideoCapture(0)

arduino = serial.Serial(port='COM5', baudrate=9600, timeout=1)
time.sleep(2)  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    human_detected = False
    for r in results:
        for box in r.boxes:
            confidence = float(box.conf[0])
            cls = int(box.cls[0])  
            if cls == 0 and confidence > 0.50:  
                human_detected = True
                break

    if human_detected:
        arduino.write(b'1')  
        print("Human detected! LED ON")
    else:
        arduino.write(b'0')  
        print("No human detected. LED OFF")

    frame_with_boxes = results[0].plot()  
    cv2.imshow("YOLOv5 Human Detection", frame_with_boxes)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()