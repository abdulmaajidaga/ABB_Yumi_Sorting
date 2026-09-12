"""The real end-to-end test: synthetic object -> simulated single-viewpoint
depth capture -> pcd_extractor -> shape_classifier_single_view ->
grasp_generator. Checks each stage handed the next something usable, and
that the final grasp pose is internally consistent (unit vectors, orthogonal
axes, unit quaternion) and physically sane (within the gripper's real max
opening, grasp Z between the table and the visible top).

Run:
    source perception/.venv/bin/activate
    python perception/test_full_pipeline.py
"""

import numpy as np

import synthetic_shapes as syn
from synthetic_depth_camera import render_top_down, camera_to_world_extrinsic
from pcd_extractor import extract_object_pointcloud
from shape_classifier_single_view import classify_shape
from grasp_generator import generate_grasp_single_view

CAMERA_HEIGHT = 0.5
FX = FY = 600.0
CX, CY = 320.0, 240.0
IMG_W, IMG_H = 640, 480
EXTRINSIC = camera_to_world_extrinsic(CAMERA_HEIGHT)
GRIPPER_MAX_OPENING = 0.05  # 2 x 0.025m per finger, from yumi.urdf


def make_shape(shape_name, rng, cxy, yaw):
    if shape_name == "sphere":
        return syn.sphere(radius=rng.uniform(0.02, 0.05), center_xy=cxy, noise_frac=0.015, rng=rng)
    if shape_name == "cylinder":
        return syn.cylinder(radius=rng.uniform(0.015, 0.04), height=rng.uniform(0.05, 0.09),
                             center_xy=cxy, noise_frac=0.015, rng=rng)
    if shape_name == "cone":
        return syn.cone(base_radius=rng.uniform(0.02, 0.05), height=rng.uniform(0.05, 0.08),
                         center_xy=cxy, apex_up=True, noise_frac=0.015, rng=rng)
    if shape_name == "cube":
        return syn.cube(side=rng.uniform(0.03, 0.045), center_xy=cxy, yaw=yaw, noise_frac=0.015, rng=rng)
    if shape_name == "cuboid":
        l = rng.uniform(0.04, 0.06)
        w = l * rng.uniform(0.4, 0.7)
        h = rng.uniform(0.03, 0.05)
        return syn.cuboid(l=l, w=w, h=h, center_xy=cxy, yaw=yaw, noise_frac=0.015, rng=rng)
    if shape_name == "prism":
        return syn.prism(side=rng.uniform(0.04, 0.06), height=rng.uniform(0.05, 0.08),
                          center_xy=cxy, yaw=yaw, noise_frac=0.015, rng=rng)
    raise ValueError(shape_name)


def run_one(shape_name, rng):
    cxy = (rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.04))
    yaw = rng.uniform(0, 2 * np.pi)
    full_cloud = make_shape(shape_name, rng, cxy, yaw)

    depth_img, mask_img = render_top_down(full_cloud, CAMERA_HEIGHT, FX, FY, CX, CY, IMG_W, IMG_H)
    extracted = extract_object_pointcloud(
        depth_img, mask_img, FX, FY, CX, CY, extrinsic=EXTRINSIC, outlier_removal=True,
    )

    result = classify_shape(extracted)
    print(f"[{shape_name}] classified as: {result.shape}"
          + ("  -- MISCLASSIFIED" if result.shape != shape_name else ""))

    if result.shape == "unknown":
        print("  SKIP -- classifier gave up\n")
        return

    grasp = generate_grasp_single_view(result)

    approach = np.array(grasp.approach_axis)
    jaw = np.array(grasp.jaw_axis)
    quat = np.array(grasp.quaternion_wxyz)
    z_top = result.params.get("z_top")

    print(f"  grasp_width={grasp.grasp_width*1000:.1f}mm  position={[round(v,4) for v in grasp.position]}"
          f"  z_top={z_top}")
    print(f"  approach_axis={[round(v,3) for v in grasp.approach_axis]}  "
          f"jaw_axis={[round(v,3) for v in grasp.jaw_axis]}  debug={grasp.debug}")

    problems = []
    if abs(np.linalg.norm(approach) - 1) > 1e-6:
        problems.append("approach_axis not unit length")
    if abs(np.linalg.norm(jaw) - 1) > 1e-6:
        problems.append("jaw_axis not unit length")
    if abs(float(np.dot(approach, jaw))) > 1e-6:
        problems.append("approach_axis and jaw_axis not orthogonal")
    if abs(np.linalg.norm(quat) - 1) > 1e-6:
        problems.append("quaternion not unit norm")
    if grasp.grasp_width > GRIPPER_MAX_OPENING:
        problems.append(f"grasp_width {grasp.grasp_width*1000:.1f}mm exceeds gripper max "
                         f"{GRIPPER_MAX_OPENING*1000:.0f}mm")
    if z_top is not None and not (0.0 <= grasp.position[2] <= z_top + 1e-6):
        problems.append(f"grasp Z {grasp.position[2]:.4f} is outside [0, z_top={z_top:.4f}] -- "
                         f"below the table or above the visible surface")

    if problems:
        print("  PROBLEMS: " + "; ".join(problems))
    else:
        print("  OK")
    print()


def main():
    rng = np.random.default_rng(3)
    shapes = ["sphere", "cylinder", "cone", "cube", "cuboid", "prism"]
    for s in shapes:
        for _ in range(3):
            run_one(s, rng)


if __name__ == "__main__":
    main()
