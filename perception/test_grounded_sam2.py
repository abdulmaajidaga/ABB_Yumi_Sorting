"""First smoke test for Grounded-SAM-2 (GroundingDINO + SAM2), run before any
real camera is connected -- validates the open-vocabulary detect+segment
pipeline on a static image and text prompt.

Run:
    source perception/.venv/bin/activate
    cd perception/Grounded-SAM-2   # must run from repo root (relative asset paths)
    python ../test_grounded_sam2.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Grounded-SAM-2"))

import cv2
import torch
import numpy as np
import supervision as sv
from torchvision.ops import box_convert

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

TEXT_PROMPT = "car. paper bag. tire."
IMG_PATH = "notebooks/images/groceries.jpg"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "../outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print("Loading SAM2 ...")
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

print("Loading GroundingDINO ...")
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE,
)

image_source, image = load_image(IMG_PATH)
sam2_predictor.set_image(image_source)

print(f"Running detection for prompt: {TEXT_PROMPT!r}")
boxes, confidences, labels = predict(
    model=grounding_model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
    device=DEVICE,
)

h, w, _ = image_source.shape
boxes = boxes * torch.Tensor([w, h, w, h])
input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

print(f"Detected {len(labels)} object(s):")
for label, conf, box in zip(labels, confidences.tolist(), input_boxes.tolist()):
    print(f"  {label!r}  conf={conf:.2f}  box(xyxy)={[round(v, 1) for v in box]}")

if len(labels) == 0:
    print("No detections -- nothing to segment. Check TEXT_PROMPT / thresholds.")
    raise SystemExit(0)

torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

masks, scores, _ = sam2_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)
if masks.ndim == 4:
    masks = masks.squeeze(1)

print(f"SAM2 mask scores: {[round(float(s), 3) for s in scores]}")

class_ids = np.arange(len(labels))
label_strs = [f"{name} {conf:.2f}" for name, conf in zip(labels, confidences.tolist())]

img = cv2.imread(IMG_PATH)
detections = sv.Detections(xyxy=input_boxes, mask=masks.astype(bool), class_id=class_ids)

annotated = sv.BoxAnnotator().annotate(scene=img.copy(), detections=detections)
annotated = sv.LabelAnnotator().annotate(scene=annotated, detections=detections, labels=label_strs)
annotated = sv.MaskAnnotator().annotate(scene=annotated, detections=detections)

out_path = os.path.join(OUTPUT_DIR, "grounded_sam2_smoke_test.jpg")
cv2.imwrite(out_path, annotated)
print(f"Saved annotated result to {os.path.abspath(out_path)}")
