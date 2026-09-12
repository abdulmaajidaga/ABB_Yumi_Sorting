"""Validates shape_classifier_single_view.py against the same realistic
single-viewpoint capture harness used in test_pcd_extractor.py (which showed
the two earlier classifiers collapsing to ~50% or worse on this exact data).

Run:
    source perception/.venv/bin/activate
    python perception/test_single_view_classifier.py
"""

import numpy as np

import synthetic_shapes as syn
from synthetic_depth_camera import render_top_down, camera_to_world_extrinsic
from pcd_extractor import extract_object_pointcloud
from shape_classifier_single_view import classify_shape

CAMERA_HEIGHT = 0.5
FX = FY = 600.0
CX, CY = 320.0, 240.0
IMG_W, IMG_H = 640, 480
EXTRINSIC = camera_to_world_extrinsic(CAMERA_HEIGHT)

N_TRIALS = 200
rng = np.random.default_rng(42)


def make_full_cloud(shape_name, cxy, yaw, noise_frac):
    if shape_name == "sphere":
        return syn.sphere(radius=rng.uniform(0.02, 0.05), center_xy=cxy, noise_frac=noise_frac, rng=rng)
    if shape_name == "cylinder":
        return syn.cylinder(radius=rng.uniform(0.015, 0.04), height=rng.uniform(0.04, 0.09),
                             center_xy=cxy, noise_frac=noise_frac, rng=rng)
    if shape_name == "cone":
        return syn.cone(base_radius=rng.uniform(0.02, 0.05), height=rng.uniform(0.04, 0.08),
                         center_xy=cxy, apex_up=True, noise_frac=noise_frac, rng=rng)
    if shape_name == "cube":
        return syn.cube(side=rng.uniform(0.03, 0.06), center_xy=cxy, yaw=yaw, noise_frac=noise_frac, rng=rng)
    if shape_name == "cuboid":
        l = rng.uniform(0.03, 0.07)
        w = l * rng.uniform(0.4, 0.75)
        h = rng.uniform(0.02, 0.05)
        return syn.cuboid(l=l, w=w, h=h, center_xy=cxy, yaw=yaw, noise_frac=noise_frac, rng=rng)
    if shape_name == "prism":
        return syn.prism(side=rng.uniform(0.03, 0.06), height=rng.uniform(0.04, 0.08),
                          center_xy=cxy, yaw=yaw, noise_frac=noise_frac, rng=rng)
    raise ValueError(shape_name)


def single_viewpoint_cloud(shape_name):
    cxy = (rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05))
    yaw = rng.uniform(0, 2 * np.pi)
    full_cloud = make_full_cloud(shape_name, cxy, yaw, noise_frac=0.015)
    depth_img, mask_img = render_top_down(full_cloud, CAMERA_HEIGHT, FX, FY, CX, CY, IMG_W, IMG_H)
    return extract_object_pointcloud(depth_img, mask_img, FX, FY, CX, CY, extrinsic=EXTRINSIC, outlier_removal=True)


def main():
    shapes = ["sphere", "cylinder", "cone", "cube", "cuboid", "prism"]
    confusion = {s: {} for s in shapes}
    correct_total = 0
    total = 0
    failures = []

    for s in shapes:
        for _ in range(N_TRIALS):
            extracted = single_viewpoint_cloud(s)
            result = classify_shape(extracted)
            confusion[s].setdefault(result.shape, 0)
            confusion[s][result.shape] += 1
            correct_total += int(result.shape == s)
            total += 1
            if result.shape != s and len(failures) < 5:
                failures.append((s, result))

    print(f"Overall: {correct_total}/{total} correct ({100*correct_total/total:.2f}%)\n")
    for s in shapes:
        row = confusion[s]
        n_correct = row.get(s, 0)
        n_total = sum(row.values())
        row_str = ", ".join(f"{k}:{v}" for k, v in sorted(row.items(), key=lambda kv: -kv[1]))
        print(f"  {s:10s} {100*n_correct/n_total:6.2f}%  ({row_str})")

    if failures:
        print(f"\nSample failures:")
        for true_shape, result in failures:
            print(f"  true={true_shape:10s} predicted={result.shape:10s} debug={result.debug}")


if __name__ == "__main__":
    main()
