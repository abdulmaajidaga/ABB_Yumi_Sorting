"""Live webcam Grounded-SAM-2 (GroundingDINO + SAM2) detection+segmentation,
to validate the perception pipeline before the RealSense arrives -- no depth,
just a live view of open-vocabulary detect+segment on whatever's in frame.

Camera capture and model inference run on separate threads: the display
always shows the latest camera frame at full rate, with the latest available
detection overlaid on top (inference is much slower than the camera, so the
overlay updates at whatever rate the model can sustain -- a few times a
second -- while the video itself stays smooth instead of stuttering down to
the model's own frame rate).

Run:
    source perception/.venv/bin/activate
    cd perception/Grounded-SAM-2   # must run from repo root (relative asset paths)
    python ../live_grounded_sam2.py

Press 'q' to quit. Edit TEXT_PROMPT below to change what it looks for.
"""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Grounded-SAM-2"))

import cv2
import torch
import numpy as np
import supervision as sv
from torchvision.ops import box_convert
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, predict
import grounding_dino.groundingdino.datasets.transforms as T

TEXT_PROMPT = "cup. bottle. box. phone. remote. pen. bag."
CAMERA_INDEX = 0
CAPTURE_WIDTH, CAPTURE_HEIGHT = 1280, 720
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_small.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GDINO_TRANSFORM = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def frame_to_gdino_input(frame_bgr):
    """BGR numpy frame -> (image_source RGB numpy, transformed tensor) matching
    what grounding_dino.util.inference.load_image does for a file, but for an
    in-memory frame instead of reading from disk."""
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_transformed, _ = GDINO_TRANSFORM(image_pil, None)
    return image_rgb, image_transformed


class LatestFrame:
    """Thread-safe single-slot mailbox -- always holds only the newest value."""

    def __init__(self):
        self._lock = threading.Lock()
        self._value = None

    def set(self, value):
        with self._lock:
            self._value = value

    def get(self):
        with self._lock:
            return self._value


def camera_loop(cap, box: LatestFrame, stop: threading.Event):
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            continue
        box.set(frame)


def inference_loop(grounding_model, sam2_predictor, frame_box: LatestFrame,
                    result_box: LatestFrame, stop: threading.Event):
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()

    while not stop.is_set():
        frame = frame_box.get()
        if frame is None:
            time.sleep(0.01)
            continue

        t0 = time.monotonic()
        image_source, image = frame_to_gdino_input(frame)

        with torch.inference_mode():
            # GroundingDINO's compiled CUDA kernel doesn't support bfloat16 --
            # autocast must only wrap the SAM2 call below, not this one.
            boxes, confidences, labels = predict(
                model=grounding_model,
                image=image,
                caption=TEXT_PROMPT,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device=DEVICE,
            )

            annotated = frame
            if len(labels) > 0:
                h, w, _ = image_source.shape
                boxes_px = boxes * torch.Tensor([w, h, w, h])
                input_boxes = box_convert(boxes=boxes_px, in_fmt="cxcywh", out_fmt="xyxy").numpy()

                with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                    sam2_predictor.set_image(image_source)
                    masks, scores, _ = sam2_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_boxes,
                        multimask_output=False,
                    )
                if masks.ndim == 4:
                    masks = masks.squeeze(1)

                class_ids = np.arange(len(labels))
                label_strs = [f"{n} {c:.2f}" for n, c in zip(labels, confidences.tolist())]
                detections = sv.Detections(xyxy=input_boxes, mask=masks.astype(bool), class_id=class_ids)

                annotated = mask_annotator.annotate(scene=frame.copy(), detections=detections)
                annotated = box_annotator.annotate(scene=annotated, detections=detections)
                annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=label_strs)

        inference_fps = 1.0 / max(time.monotonic() - t0, 1e-6)
        result_box.set((annotated, inference_fps))


def main():
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

    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {CAMERA_INDEX}")

    frame_box = LatestFrame()
    result_box = LatestFrame()
    stop = threading.Event()

    cam_thread = threading.Thread(target=camera_loop, args=(cap, frame_box, stop), daemon=True)
    inf_thread = threading.Thread(
        target=inference_loop,
        args=(grounding_model, sam2_predictor, frame_box, result_box, stop),
        daemon=True,
    )
    cam_thread.start()
    inf_thread.start()

    print(f"Prompt: {TEXT_PROMPT!r}  -- press 'q' in the window to quit.")

    display_fps_t0 = time.monotonic()
    display_frame_count = 0
    display_fps = 0.0
    inference_fps = 0.0

    try:
        while True:
            frame = frame_box.get()
            if frame is None:
                time.sleep(0.01)
                continue

            result = result_box.get()
            shown = result[0] if result is not None else frame
            if result is not None:
                inference_fps = result[1]
            shown = shown.copy()

            display_frame_count += 1
            elapsed = time.monotonic() - display_fps_t0
            if elapsed >= 0.5:
                display_fps = display_frame_count / elapsed
                display_frame_count = 0
                display_fps_t0 = time.monotonic()

            cv2.putText(shown, f"video {display_fps:.1f} FPS  |  detection {inference_fps:.1f} FPS",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Grounded-SAM-2 live", shown)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stop.set()
        cam_thread.join(timeout=2.0)
        inf_thread.join(timeout=2.0)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
