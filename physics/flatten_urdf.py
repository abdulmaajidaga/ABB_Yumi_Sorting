"""Generates a PyBullet-loadable copy of yumi.urdf.

p.loadURDF() needs a real file path with resolvable mesh URIs -- it doesn't
understand ROS's "package://" scheme (that's resolved by the ROS package
index / xacro at ROS-launch time, which PyBullet knows nothing about) and
can't accept URDF content as an in-memory string the way robot_state_publisher
can (via the ~robot_description param). So this just does a plain text
substitution -- package://yumi_description/... -> the real absolute path --
and writes the result next to this script. Re-run any time yumi.urdf changes.

Usage:
    python3 physics/flatten_urdf.py
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_URDF = REPO_ROOT / "yumi_description" / "urdf" / "yumi.urdf"
DESC_ROOT = REPO_ROOT / "yumi_description"
OUT_URDF = Path(__file__).resolve().parent / "yumi_flat.urdf"


def main():
    text = SRC_URDF.read_text()
    flattened = text.replace("package://yumi_description/", f"{DESC_ROOT}/")
    remaining = flattened.count("package://")
    if remaining:
        raise SystemExit(f"{remaining} package:// reference(s) left unresolved -- "
                          f"check for a different package name in the URDF.")
    OUT_URDF.write_text(flattened)
    print(f"Wrote {OUT_URDF} ({flattened.count('<mesh filename=')} mesh references, all absolute)")


if __name__ == "__main__":
    main()
