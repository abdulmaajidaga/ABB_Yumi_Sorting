"""Smoke test for shape_classifier.py against synthetic point clouds for all
six sorting primitives, with noise, run multiple trials each with randomized
size/position/orientation.

Run:
    source perception/.venv/bin/activate
    python perception/test_shape_classifier.py
"""

import numpy as np

import synthetic_shapes as syn
from shape_classifier import classify_shape

N_TRIALS = 3
NOISE_FRAC = 0.015
rng = np.random.default_rng(42)


def make_cases():
    cases = []
    for _ in range(N_TRIALS):
        cxy = (rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1))
        yaw = rng.uniform(0, 2 * np.pi)

        cases.append(("sphere", syn.sphere(
            radius=rng.uniform(0.02, 0.05), center_xy=cxy, noise_frac=NOISE_FRAC, rng=rng)))

        cases.append(("cylinder", syn.cylinder(
            radius=rng.uniform(0.015, 0.04), height=rng.uniform(0.04, 0.09),
            center_xy=cxy, noise_frac=NOISE_FRAC, rng=rng)))

        cases.append(("cone", syn.cone(
            base_radius=rng.uniform(0.02, 0.05), height=rng.uniform(0.04, 0.08),
            center_xy=cxy, apex_up=bool(rng.integers(0, 2)), noise_frac=NOISE_FRAC, rng=rng)))

        cases.append(("cube", syn.cube(
            side=rng.uniform(0.03, 0.06), center_xy=cxy, yaw=yaw,
            noise_frac=NOISE_FRAC, rng=rng)))

        l = rng.uniform(0.03, 0.07)
        w = l * rng.uniform(0.4, 0.75)  # ensure clearly non-cube aspect ratio
        h = rng.uniform(0.02, 0.05)
        cases.append(("cuboid", syn.cuboid(
            l=l, w=w, h=h, center_xy=cxy, yaw=yaw, noise_frac=NOISE_FRAC, rng=rng)))

        cases.append(("prism", syn.prism(
            side=rng.uniform(0.03, 0.06), height=rng.uniform(0.04, 0.08),
            center_xy=cxy, yaw=yaw, noise_frac=NOISE_FRAC, rng=rng)))
    return cases


def main():
    cases = make_cases()
    correct = 0
    confusion = {}
    failures = []

    for true_shape, points in cases:
        result = classify_shape(points)
        confusion.setdefault(true_shape, {}).setdefault(result.shape, 0)
        confusion[true_shape][result.shape] += 1
        if result.shape == true_shape:
            correct += 1
        else:
            failures.append((true_shape, result))

    total = len(cases)
    print(f"Overall: {correct}/{total} correct ({100 * correct / total:.1f}%)\n")

    print("Confusion (true -> predicted counts):")
    for true_shape in sorted(confusion):
        row = confusion[true_shape]
        row_str = ", ".join(f"{k}:{v}" for k, v in sorted(row.items(), key=lambda kv: -kv[1]))
        print(f"  {true_shape:10s} -> {row_str}")

    if failures:
        print(f"\n{len(failures)} failure(s), details:")
        for true_shape, result in failures:
            print(f"  true={true_shape:10s} predicted={result.shape:10s} debug={result.debug}")


if __name__ == "__main__":
    main()
