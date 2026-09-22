import cv2
from ultralytics import RTDETR
model = RTDETR("rtdetr-x.pt")
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, conf=0.3, verbose=False) 
    annotated_frame = results[0].plot()
    cv2.imshow("Real-Time Object Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()