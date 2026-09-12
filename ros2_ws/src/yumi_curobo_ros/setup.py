from setuptools import find_packages, setup

package_name = "yumi_curobo_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="abdul",
    maintainer_email="abdul@example.com",
    description="ROS2 FK/IK services for YuMi backed by curobo (unified_planner node).",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "unified_planner_node = yumi_curobo_ros.unified_planner_node:main",
        ],
    },
)
