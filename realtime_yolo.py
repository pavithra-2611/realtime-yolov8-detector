import cv2
from ultralytics import YOLO
import win32gui
import win32con
import time

# Load the pretrained YOLOv8 model (you can change 'yolov8n.pt' to 'yolov8s.pt' for better accuracy)
model = YOLO("yolov8n.pt")  # YOLOv8 Nano — fastest and smallest

# Start the webcam
cap = cv2.VideoCapture(0)

# Optional: Wait a bit for the camera to initialize
time.sleep(1)

print("✅ Press 'q' in the video window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model.predict(source=frame, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()

    # Show the annotated frame
    window_name = "YOLOv8 Live Detection"
    cv2.imshow(window_name, annotated_frame)

    # ✅ Auto-focus the window (Windows only)
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd:
        win32gui.SetForegroundWindow(hwnd)
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()
