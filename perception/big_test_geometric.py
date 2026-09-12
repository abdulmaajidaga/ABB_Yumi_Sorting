"""Large-scale stress test for shape_classifier_geometric.py (the fast,
non-RANSAC classifier) -- thousands of randomized trials per shape, with
noise level AND point count also randomized per case (not fixed), to stress
test more realistically than the earlier fixed-noise smoke test.

Run:
    source perception/.venv/bin/activate
    python perception/big_test_geometric.py
"""

import time

import numpy as np

import synthetic_shapes as syn
from shape_classifier_geometric import classify_shape

N_TRIALS_PER_SHAPE = 10000
NOISE_FRAC_RANGE = (0.003, 0.04)
N_POINTS_RANGE = (300, 3000)
rng = np.random.default_rng(123)


def make_case(shape_name):
    cxy = (rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15))
    yaw = rng.uniform(0, 2 * np.pi)
    noise_frac = rng.uniform(*NOISE_FRAC_RANGE)
    n = int(rng.uniform(*N_POINTS_RANGE))

    if shape_name == "sphere":
        return syn.sphere(radius=rng.uniform(0.015, 0.06), center_xy=cxy, n=n, noise_frac=noise_frac, rng=rng)
    if shape_name == "cylinder":
        return syn.cylinder(radius=rng.uniform(0.01, 0.045), height=rng.uniform(0.03, 0.10),
                             center_xy=cxy, n=n, noise_frac=noise_frac, rng=rng)
    if shape_name == "cone":
        return syn.cone(base_radius=rng.uniform(0.015, 0.06), height=rng.uniform(0.03, 0.09),
                         center_xy=cxy, apex_up=bool(rng.integers(0, 2)), n=n, noise_frac=noise_frac, rng=rng)
    if shape_name == "cube":
        return syn.cube(side=rng.uniform(0.02, 0.07), center_xy=cxy, yaw=yaw, n=n, noise_frac=noise_frac, rng=rng)
    if shape_name == "cuboid":
        l = rng.uniform(0.025, 0.08)
        w = l * rng.uniform(0.35, 0.78)
        h = rng.uniform(0.015, 0.06)
        return syn.cuboid(l=l, w=w, h=h, center_xy=cxy, yaw=yaw, n=n, noise_frac=noise_frac, rng=rng)
    if shape_name == "prism":
        return syn.prism(side=rng.uniform(0.025, 0.07), height=rng.uniform(0.03, 0.09),
                          center_xy=cxy, yaw=yaw, n=n, noise_frac=noise_frac, rng=rng)
    raise ValueError(shape_name)


def main():
    shapes = ["sphere", "cylinder", "cone", "cube", "cuboid", "prism"]
    total = N_TRIALS_PER_SHAPE * len(shapes)
    print(f"Running {total} randomized trials ({N_TRIALS_PER_SHAPE} per shape), "
          f"noise_frac in {NOISE_FRAC_RANGE}, n_points in {N_POINTS_RANGE} ...\n")

    confusion = {s: {} for s in shapes}
    correct_total = 0
    t0 = time.perf_counter()

    for shape_name in shapes:
        for _ in range(N_TRIALS_PER_SHAPE):
            points = make_case(shape_name)
            result = classify_shape(points)
            confusion[shape_name].setdefault(result.shape, 0)
            confusion[shape_name][result.shape] += 1
            if result.shape == shape_name:
                correct_total += 1

    elapsed = time.perf_counter() - t0

    print(f"Overall: {correct_total}/{total} correct ({100 * correct_total / total:.2f}%)")
    print(f"Total time: {elapsed:.2f}s  ({1000 * elapsed / total:.3f}ms/object)\n")

    print("Per-shape accuracy and confusion (true -> predicted counts):")
    for s in shapes:
        row = confusion[s]
        n_correct = row.get(s, 0)
        n_total = sum(row.values())
        row_str = ", ".join(f"{k}:{v}" for k, v in sorted(row.items(), key=lambda kv: -kv[1]))
        print(f"  {s:10s} {100*n_correct/n_total:6.2f}%  ({row_str})")


if __name__ == "__main__":
    main()
