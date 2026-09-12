"""End-to-end test of the new pieces together: synthesize a full-surface
object -> render a realistic single-viewpoint overhead depth capture of it
(genuine occlusion, not just noise) -> run it through the real
pcd_extractor.py -> classify the result. This is the first test in the
project that actually checks what happens when the classifier only sees the
visible half of an object, the way a real overhead RealSense will.

Run:
    source perception/.venv/bin/activate
    python perception/test_pcd_extractor.py
"""

import numpy as np

import synthetic_shapes as syn
from synthetic_depth_camera import render_top_down, camera_to_world_extrinsic
from pcd_extractor import extract_object_pointcloud
from shape_classifier_geometric import classify_shape as classify_fast
from shape_classifier import classify_shape as classify_production

CAMERA_HEIGHT = 0.5
FX = FY = 600.0
CX, CY = 320.0, 240.0
IMG_W, IMG_H = 640, 480
EXTRINSIC = camera_to_world_extrinsic(CAMERA_HEIGHT)

N_TRIALS_FAST = 200
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
    extracted = extract_object_pointcloud(
        depth_img, mask_img, FX, FY, CX, CY, extrinsic=EXTRINSIC, outlier_removal=True,
    )
    return full_cloud, extracted


def main():
    shapes = ["sphere", "cylinder", "cone", "cube", "cuboid", "prism"]

    # First, one detailed look: point-count comparison between the full
    # (360-degree) cloud and the realistic single-viewpoint extraction.
    print("Point count, full-surface vs. single-viewpoint extraction:")
    for s in shapes:
        full, extracted = single_viewpoint_cloud(s)
        print(f"  {s:10s} full={full.shape[0]:5d}  single-viewpoint={extracted.shape[0]:5d}  "
              f"({100*extracted.shape[0]/full.shape[0]:.1f}% retained)")

    # Now the real question: how does the FAST classifier do on realistic
    # single-viewpoint clouds, at scale?
    print(f"\nFast (geometric) classifier on {N_TRIALS_FAST} single-viewpoint trials per shape:")
    confusion = {s: {} for s in shapes}
    correct_total = 0
    total = 0
    for s in shapes:
        for _ in range(N_TRIALS_FAST):
            _, extracted = single_viewpoint_cloud(s)
            result = classify_fast(extracted)
            confusion[s].setdefault(result.shape, 0)
            confusion[s][result.shape] += 1
            correct_total += int(result.shape == s)
            total += 1

    print(f"Overall: {correct_total}/{total} correct ({100*correct_total/total:.2f}%)\n")
    for s in shapes:
        row = confusion[s]
        n_correct = row.get(s, 0)
        n_total = sum(row.values())
        row_str = ", ".join(f"{k}:{v}" for k, v in sorted(row.items(), key=lambda kv: -kv[1]))
        print(f"  {s:10s} {100*n_correct/n_total:6.2f}%  ({row_str})")

    # Small spot-check with the production (pyransac3d) classifier -- slow,
    # so just a couple of trials per shape.
    print(f"\nProduction (pyransac3d) classifier, 2 single-viewpoint trials per shape:")
    for s in shapes:
        for _ in range(2):
            _, extracted = single_viewpoint_cloud(s)
            result = classify_production(extracted)
            mark = "OK" if result.shape == s else "MISS"
            print(f"  {s:10s} -> {result.shape:10s} [{mark}]")


if __name__ == "__main__":
    main()
