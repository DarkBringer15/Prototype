import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

model_id = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
image_path = "./Stock Photos/table3.png"  
image = Image.open(image_path).convert("RGB")

text = "a clock. a camera. glasses. a book. a cup. a donut. a telephone. a plant. a lamp. an eraser. a pen. a pencil. a note. a calculator. a candle. a sharpener. a clip. a ruler."

inputs = processor(images=image, text=text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=0.30,
    text_threshold=0.25,
    target_sizes=[image.size[::-1]]
)

print("=== Detection Results ===")
for score, label, box in zip(results[0]["scores"], results[0]["labels"], results[0]["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(f"{label}: {round(score.item(), 3)} at {box}")

fig, ax = plt.subplots(1, figsize=(14, 10))
ax.imshow(image)
ax.axis("off")

for score, label, box in zip(results[0]["scores"], results[0]["labels"], results[0]["boxes"]):
    x_min, y_min, x_max, y_max = box.tolist()
    width = x_max - x_min
    height = y_max - y_min

    # Buat kotak (bounding box)
    rect = patches.Rectangle(
        (x_min, y_min), width, height,
        linewidth=2, edgecolor="lime", facecolor="none"
    )
    ax.add_patch(rect)

    # Tambahkan label dan skor
    label_text = f"{label} {score:.2f}"
    ax.text(
        x_min, y_min - 5, label_text,
        color="black", fontsize=10, weight="bold",
        bbox=dict(facecolor="lime", alpha=0.7, edgecolor="none", pad=2)
    )

plt.tight_layout()
plt.show()
fig.savefig("Dino.png", bbox_inches="tight", dpi=300)
print("\nResult is saved as Dino.png")