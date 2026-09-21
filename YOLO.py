from ultralytics import YOLO
import matplotlib.pyplot as plt 
import cv2 as cv

model = YOLO("yolo26x.pt")
image_path = "./Stock Photos/table1.jpg"

original_bgr = cv.imread(image_path)
original_rgb = cv.cvtColor(original_bgr, cv.COLOR_BGR2RGB)

results = model(image_path, conf=0.25)
result= results[0]

annotated_rgb = cv.cvtColor(result.plot(), cv.COLOR_BGR2RGB)
annotated_bgr = cv.cvtColor(annotated_rgb, cv.COLOR_RGB2BGR)
cv.imwrite("YOLO Result_1.jpg", annotated_bgr)

fig, axes = plt.subplots(1,2, figsize=(16,8))
axes[0].imshow(original_rgb)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(annotated_rgb)
axes[1].set_title("Detected by YOLO")
axes[1].axis("off")

plt.tight_layout()
plt.show()
