"""Renders the synthetic single-viewpoint capture pipeline visually, so it
can be eyeballed rather than just trusted from numbers: for each shape,
shows the simulated depth image, the full 360-degree point cloud (what the
object actually is), and the extracted single-viewpoint point cloud (what
the pipeline says a real overhead RealSense would see of it).

Run:
    source perception/.venv/bin/activate
    python perception/visualize_pipeline.py
Output: perception/outputs/pipeline_visualization.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def make_shape(shape_name, rng):
    cxy = (0.0, 0.0)
    yaw = 0.4
    if shape_name == "sphere":
        return syn.sphere(radius=0.035, center_xy=cxy, noise_frac=0.01, rng=rng)
    if shape_name == "cylinder":
        return syn.cylinder(radius=0.025, height=0.07, center_xy=cxy, noise_frac=0.01, rng=rng)
    if shape_name == "cone":
        return syn.cone(base_radius=0.035, height=0.06, apex_up=True, center_xy=cxy, noise_frac=0.01, rng=rng)
    if shape_name == "cube":
        return syn.cube(side=0.05, center_xy=cxy, yaw=yaw, noise_frac=0.01, rng=rng)
    if shape_name == "cuboid":
        return syn.cuboid(l=0.06, w=0.03, h=0.04, center_xy=cxy, yaw=yaw, noise_frac=0.01, rng=rng)
    if shape_name == "prism":
        return syn.prism(side=0.05, height=0.06, center_xy=cxy, yaw=yaw, noise_frac=0.01, rng=rng)
    raise ValueError(shape_name)


def main():
    shapes = ["sphere", "cylinder", "cone", "cube", "cuboid", "prism"]
    rng = np.random.default_rng(7)

    fig = plt.figure(figsize=(16, 4 * len(shapes)))

    for row, shape_name in enumerate(shapes):
        full_cloud = make_shape(shape_name, rng)
        depth_img, mask_img = render_top_down(full_cloud, CAMERA_HEIGHT, FX, FY, CX, CY, IMG_W, IMG_H)
        extracted = extract_object_pointcloud(
            depth_img, mask_img, FX, FY, CX, CY, extrinsic=EXTRINSIC, outlier_removal=True,
        )
        result = classify_shape(extracted)

        # Depth image (masked region only, background as NaN so it's blank)
        ax1 = fig.add_subplot(len(shapes), 3, row * 3 + 1)
        depth_display = np.where(mask_img, depth_img, np.nan)
        im = ax1.imshow(depth_display, cmap="viridis")
        ax1.set_title(f"{shape_name}: simulated depth capture")
        ax1.set_xlabel("pixel u")
        ax1.set_ylabel("pixel v")
        plt.colorbar(im, ax=ax1, label="depth (m)", fraction=0.046)

        # Full 360-degree cloud, colored by height
        ax2 = fig.add_subplot(len(shapes), 3, row * 3 + 2, projection="3d")
        ax2.scatter(full_cloud[:, 0], full_cloud[:, 1], full_cloud[:, 2],
                    c=full_cloud[:, 2], cmap="viridis", s=2)
        ax2.set_title(f"{shape_name}: full object (ground truth)")
        ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
        ax2.set_box_aspect([1, 1, 1])

        # Extracted single-viewpoint cloud, same color scale/axes for comparison
        ax3 = fig.add_subplot(len(shapes), 3, row * 3 + 3, projection="3d")
        ax3.scatter(extracted[:, 0], extracted[:, 1], extracted[:, 2],
                    c=extracted[:, 2], cmap="viridis", s=4)
        ax3.set_title(f"{shape_name}: extracted single-view "
                       f"({extracted.shape[0]}/{full_cloud.shape[0]} pts)\n"
                       f"classified as: {result.shape}")
        ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")
        ax3.set_xlim(ax2.get_xlim()); ax3.set_ylim(ax2.get_ylim()); ax3.set_zlim(ax2.get_zlim())
        ax3.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "pipeline_visualization.png")
    plt.savefig(out_path, dpi=110)
    print(f"Saved to {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
