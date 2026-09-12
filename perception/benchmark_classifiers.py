"""Direct accuracy + speed comparison: shape_classifier.py (pyransac3d-based,
current production) vs shape_classifier_geometric.py (original height-slicing
method), on the exact same synthetic point clouds.

Run:
    source perception/.venv/bin/activate
    python perception/benchmark_classifiers.py
"""

import time

import numpy as np

import synthetic_shapes as syn
import shape_classifier as pyransac_clf
import shape_classifier_geometric as geo_clf

N_TRIALS = 3
NOISE_FRAC = 0.015
rng = np.random.default_rng(7)


def make_cases():
    cases = []
    for _ in range(N_TRIALS):
        cxy = (rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1))
        yaw = rng.uniform(0, 2 * np.pi)
        cases.append(("sphere", syn.sphere(radius=rng.uniform(0.02, 0.05), center_xy=cxy, noise_frac=NOISE_FRAC, rng=rng)))
        cases.append(("cylinder", syn.cylinder(radius=rng.uniform(0.015, 0.04), height=rng.uniform(0.04, 0.09), center_xy=cxy, noise_frac=NOISE_FRAC, rng=rng)))
        cases.append(("cone", syn.cone(base_radius=rng.uniform(0.02, 0.05), height=rng.uniform(0.04, 0.08), center_xy=cxy, apex_up=bool(rng.integers(0, 2)), noise_frac=NOISE_FRAC, rng=rng)))
        cases.append(("cube", syn.cube(side=rng.uniform(0.03, 0.06), center_xy=cxy, yaw=yaw, noise_frac=NOISE_FRAC, rng=rng)))
        l = rng.uniform(0.03, 0.07)
        w = l * rng.uniform(0.4, 0.75)
        h = rng.uniform(0.02, 0.05)
        cases.append(("cuboid", syn.cuboid(l=l, w=w, h=h, center_xy=cxy, yaw=yaw, noise_frac=NOISE_FRAC, rng=rng)))
        cases.append(("prism", syn.prism(side=rng.uniform(0.03, 0.06), height=rng.uniform(0.04, 0.08), center_xy=cxy, yaw=yaw, noise_frac=NOISE_FRAC, rng=rng)))
    return cases


def run(classify_fn, cases):
    correct = 0
    t0 = time.perf_counter()
    for true_shape, points in cases:
        result = classify_fn(points)
        if result.shape == true_shape:
            correct += 1
    elapsed = time.perf_counter() - t0
    return correct, elapsed


def main():
    cases = make_cases()
    n = len(cases)
    print(f"{n} test cases ({N_TRIALS} trials x 6 shapes), identical for both classifiers\n")

    correct, elapsed = run(geo_clf.classify_shape, cases)
    print(f"geometric (height-slicing, no RANSAC): {correct}/{n} correct "
          f"({100*correct/n:.1f}%), {elapsed:.3f}s total, {1000*elapsed/n:.2f}ms/object")

    correct, elapsed = run(pyransac_clf.classify_shape, cases)
    print(f"pyransac3d (4x RANSAC fit + compare):  {correct}/{n} correct "
          f"({100*correct/n:.1f}%), {elapsed:.3f}s total, {1000*elapsed/n:.2f}ms/object")


if __name__ == "__main__":
    main()
