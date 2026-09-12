"""Synthetic point cloud generators for the six sorting primitives, used to
validate shape_classifier.py before any real camera/point cloud is available.

Each object is generated resting on a flat surface (lowest point at z=0),
matching the assumption shape_classifier.py makes. Points are sampled over
the FULL surface (360 degrees, all faces) -- i.e. this tests the geometric
classification logic itself, not real-camera partial-view occlusion (a
RealSense only ever sees one side of an object). That's a deliberate,
separate concern to revisit once real depth data is available.
"""

import numpy as np


def _add_noise(points, noise_frac, characteristic_length, rng):
    if noise_frac <= 0:
        return points
    std = noise_frac * characteristic_length
    return points + rng.normal(0, std, points.shape)


def sphere(radius=0.03, center_xy=(0.0, 0.0), n=2000, noise_frac=0.01, rng=None):
    rng = rng or np.random.default_rng()
    u = rng.uniform(0, 1, n)
    v = rng.uniform(0, 1, n)
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    x = radius * np.sin(phi) * np.cos(theta) + center_xy[0]
    y = radius * np.sin(phi) * np.sin(theta) + center_xy[1]
    z = radius * np.cos(phi) + radius  # shift so bottom touches z=0
    pts = np.stack([x, y, z], axis=1)
    return _add_noise(pts, noise_frac, radius, rng)


def cylinder(radius=0.02, height=0.06, center_xy=(0.0, 0.0), n=2000, noise_frac=0.01, rng=None):
    rng = rng or np.random.default_rng()
    n_side = int(n * 0.7)
    n_cap = (n - n_side) // 2

    theta = rng.uniform(0, 2 * np.pi, n_side)
    z_side = rng.uniform(0, height, n_side)
    x_side = radius * np.cos(theta) + center_xy[0]
    y_side = radius * np.sin(theta) + center_xy[1]

    def cap(z_val, count):
        rr = radius * np.sqrt(rng.uniform(0, 1, count))
        th = rng.uniform(0, 2 * np.pi, count)
        return rr * np.cos(th) + center_xy[0], rr * np.sin(th) + center_xy[1], np.full(count, z_val)

    x_bot, y_bot, z_bot = cap(0.0, n_cap)
    x_top, y_top, z_top = cap(height, n - n_side - n_cap)

    x = np.concatenate([x_side, x_bot, x_top])
    y = np.concatenate([y_side, y_bot, y_top])
    z = np.concatenate([z_side, z_bot, z_top])
    pts = np.stack([x, y, z], axis=1)
    return _add_noise(pts, noise_frac, radius, rng)


def cone(base_radius=0.03, height=0.05, center_xy=(0.0, 0.0), apex_up=True, n=2000, noise_frac=0.01, rng=None):
    rng = rng or np.random.default_rng()
    n_side = int(n * 0.8)
    n_base = n - n_side

    z_side = rng.uniform(0, height, n_side)
    frac = 1 - z_side / height  # 1 at base, 0 at apex
    if not apex_up:
        frac = 1 - frac
        z_side = height - z_side
    r_side = base_radius * frac
    theta = rng.uniform(0, 2 * np.pi, n_side)
    x_side = r_side * np.cos(theta) + center_xy[0]
    y_side = r_side * np.sin(theta) + center_xy[1]

    base_z = 0.0 if apex_up else height
    rr = base_radius * np.sqrt(rng.uniform(0, 1, n_base))
    th = rng.uniform(0, 2 * np.pi, n_base)
    x_base = rr * np.cos(th) + center_xy[0]
    y_base = rr * np.sin(th) + center_xy[1]
    z_base = np.full(n_base, base_z)

    x = np.concatenate([x_side, x_base])
    y = np.concatenate([y_side, y_base])
    z = np.concatenate([z_side, z_base])
    pts = np.stack([x, y, z], axis=1)
    return _add_noise(pts, noise_frac, base_radius, rng)


def _box_faces(l, w, h, center_xy, yaw, n, rng):
    """l,w = horizontal extents (before yaw rotation), h = height."""
    faces_n = n // 6
    pts = []
    half_l, half_w = l / 2, w / 2

    def face(u_range, v_range, fixed_axis, fixed_val, count):
        u = rng.uniform(*u_range, count)
        v = rng.uniform(*v_range, count)
        if fixed_axis == "x":
            return np.stack([np.full(count, fixed_val), u, v], axis=1)
        elif fixed_axis == "y":
            return np.stack([u, np.full(count, fixed_val), v], axis=1)
        else:
            return np.stack([u, v, np.full(count, fixed_val)], axis=1)

    pts.append(face((-half_l, half_l), (0, h), "y", -half_w, faces_n))
    pts.append(face((-half_l, half_l), (0, h), "y", half_w, faces_n))
    pts.append(face((-half_w, half_w), (0, h), "x", -half_l, faces_n))
    pts.append(face((-half_w, half_w), (0, h), "x", half_l, faces_n))
    pts.append(face((-half_l, half_l), (-half_w, half_w), "z", 0.0, faces_n))
    pts.append(face((-half_l, half_l), (-half_w, half_w), "z", h, n - 5 * faces_n))

    all_pts = np.concatenate(pts, axis=0)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    xy_rot = all_pts[:, :2] @ rot.T
    all_pts[:, 0] = xy_rot[:, 0] + center_xy[0]
    all_pts[:, 1] = xy_rot[:, 1] + center_xy[1]
    return all_pts


def cuboid(l=0.05, w=0.03, h=0.04, center_xy=(0.0, 0.0), yaw=0.3, n=2000, noise_frac=0.01, rng=None):
    rng = rng or np.random.default_rng()
    pts = _box_faces(l, w, h, center_xy, yaw, n, rng)
    return _add_noise(pts, noise_frac, max(l, w, h), rng)


def cube(side=0.04, center_xy=(0.0, 0.0), yaw=0.3, n=2000, noise_frac=0.01, rng=None):
    return cuboid(l=side, w=side, h=side, center_xy=center_xy, yaw=yaw, n=n, noise_frac=noise_frac, rng=rng)


def prism(side=0.04, height=0.05, center_xy=(0.0, 0.0), yaw=0.3, n=2000, noise_frac=0.01, rng=None):
    """Triangular prism, equilateral cross-section, vertical axis."""
    rng = rng or np.random.default_rng()
    r_circ = side / np.sqrt(3)
    angles = np.array([90, 210, 330]) * np.pi / 180
    verts = np.stack([r_circ * np.cos(angles), r_circ * np.sin(angles)], axis=1)

    n_side_total = int(n * 0.7)
    n_per_side = n_side_total // 3
    n_cap = (n - n_side_total) // 2

    side_pts = []
    for i in range(3):
        p0, p1 = verts[i], verts[(i + 1) % 3]
        t = rng.uniform(0, 1, n_per_side)
        z = rng.uniform(0, height, n_per_side)
        xy = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
        side_pts.append(np.stack([xy[:, 0], xy[:, 1], z], axis=1))
    side_pts = np.concatenate(side_pts, axis=0)

    def cap(z_val, count):
        b = rng.uniform(0, 1, (count, 3))
        b = b / b.sum(axis=1, keepdims=True)
        xy = b @ verts
        return np.stack([xy[:, 0], xy[:, 1], np.full(count, z_val)], axis=1)

    cap_bot = cap(0.0, n_cap)
    cap_top = cap(height, n - n_side_total - n_cap)

    all_pts = np.concatenate([side_pts, cap_bot, cap_top], axis=0)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    xy_rot = all_pts[:, :2] @ rot.T
    all_pts[:, 0] = xy_rot[:, 0] + center_xy[0]
    all_pts[:, 1] = xy_rot[:, 1] + center_xy[1]
    return _add_noise(all_pts, noise_frac, side, rng)
