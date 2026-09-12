"""Sanity test for grasp_generator.py: classify_shape -> generate_grasp on
synthetic point clouds for all six shapes, checking the grasp pose is
internally consistent (unit vectors, orthogonal axes, unit quaternion) and
physically sane (grasp_width within the real gripper's max opening).

Run:
    source perception/.venv/bin/activate
    python perception/test_grasp_generator.py
"""

import numpy as np

import synthetic_shapes as syn
from shape_classifier import classify_shape
from grasp_generator import generate_grasp_full3d as generate_grasp

GRIPPER_MAX_OPENING = 0.05  # 2 x 0.025m per finger, from yumi.urdf gripper_r_joint limit


def check(label, points):
    result = classify_shape(points)
    print(f"[{label}] classified as: {result.shape}")
    if result.shape == "unknown":
        print("  SKIP -- classifier gave up")
        return

    grasp = generate_grasp(result)

    approach = np.array(grasp.approach_axis)
    jaw = np.array(grasp.jaw_axis)
    quat = np.array(grasp.quaternion_wxyz)

    ortho = float(np.dot(approach, jaw))
    approach_norm = float(np.linalg.norm(approach))
    jaw_norm = float(np.linalg.norm(jaw))
    quat_norm = float(np.linalg.norm(quat))

    print(f"  grasp_width={grasp.grasp_width*1000:.1f}mm  position={[round(v,4) for v in grasp.position]}")
    print(f"  approach_axis={[round(v,3) for v in grasp.approach_axis]}  "
          f"jaw_axis={[round(v,3) for v in grasp.jaw_axis]}")
    print(f"  |approach|={approach_norm:.4f} |jaw|={jaw_norm:.4f} approach.jaw={ortho:.6f} |quat|={quat_norm:.4f}")
    print(f"  debug={grasp.debug}")

    problems = []
    if abs(approach_norm - 1) > 1e-6:
        problems.append("approach_axis not unit length")
    if abs(jaw_norm - 1) > 1e-6:
        problems.append("jaw_axis not unit length")
    if abs(ortho) > 1e-6:
        problems.append("approach_axis and jaw_axis not orthogonal")
    if abs(quat_norm - 1) > 1e-6:
        problems.append("quaternion not unit norm")
    if grasp.grasp_width > GRIPPER_MAX_OPENING:
        problems.append(f"grasp_width {grasp.grasp_width*1000:.1f}mm exceeds gripper max "
                         f"{GRIPPER_MAX_OPENING*1000:.0f}mm -- object too big to grasp this way")
    if problems:
        print("  PROBLEMS: " + "; ".join(problems))
    else:
        print("  OK")
    print()


def make_lying_cylinder(radius, height, center_xy, noise_frac, rng):
    """synthetic_shapes.cylinder() always generates a standing (vertical-axis)
    cylinder -- rotate it 90 degrees onto its side (axis along world Y) and
    re-rest it on the table, to actually exercise the top_down grasp branch."""
    pts = syn.cylinder(radius=radius, height=height, center_xy=(0, 0), noise_frac=noise_frac, rng=rng)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    new_x = x + center_xy[0]
    new_y = z - height / 2 + center_xy[1]
    new_z = y + radius
    return np.stack([new_x, new_y, new_z], axis=1)


def main():
    rng = np.random.default_rng(0)

    check("sphere", syn.sphere(radius=0.02, center_xy=(0.03, -0.02), noise_frac=0.01, rng=rng))

    check("cylinder (lying down)", make_lying_cylinder(radius=0.015, height=0.07, center_xy=(0, 0), noise_frac=0.01, rng=rng))
    check("cylinder (standing, tall+narrow)", syn.cylinder(radius=0.01, height=0.08, center_xy=(0, 0), noise_frac=0.01, rng=rng))

    check("cone (apex up)", syn.cone(base_radius=0.03, height=0.05, apex_up=True, center_xy=(0, 0), noise_frac=0.01, rng=rng))
    check("cone (apex down)", syn.cone(base_radius=0.03, height=0.05, apex_up=False, center_xy=(0, 0), noise_frac=0.01, rng=rng))

    check("cube", syn.cube(side=0.03, center_xy=(0, 0), yaw=0.4, noise_frac=0.01, rng=rng))
    check("cuboid", syn.cuboid(l=0.06, w=0.025, h=0.04, center_xy=(0, 0), yaw=0.4, noise_frac=0.01, rng=rng))

    check("prism", syn.prism(side=0.04, height=0.05, center_xy=(0, 0), yaw=0.4, noise_frac=0.01, rng=rng))


if __name__ == "__main__":
    main()
