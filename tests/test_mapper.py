import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mapper import JOINT_LIMITS, _angle_at_b, keypoints_to_joints


class MappingTests(unittest.TestCase):
    def test_right_angle_geometry(self):
        angle = _angle_at_b(np.array([1., 0.]), np.zeros(2), np.array([0., 1.]))
        self.assertAlmostEqual(angle, np.pi / 2)

    def test_hidden_keypoints_return_neutral(self):
        joints = keypoints_to_joints(np.zeros((17, 3)))
        self.assertTrue(all(value == 0 for value in joints.values()))

    def test_visible_and_degenerate_inputs_stay_finite_and_bounded(self):
        rng = np.random.default_rng(42)
        poses = [np.ones((17, 3))]
        poses.extend(np.column_stack((rng.normal(size=(17, 2)), np.ones(17))) for _ in range(100))
        for pose in poses:
            for name, value in keypoints_to_joints(pose).items():
                low, high = JOINT_LIMITS[name]
                self.assertTrue(np.isfinite(value))
                self.assertLessEqual(low, value)
                self.assertLessEqual(value, high)

    def test_mapping_matches_builtin_urdf(self):
        robot = ET.parse(ROOT / "robot" / "humanoid.urdf").getroot()
        active = {joint.attrib["name"] for joint in robot.findall("joint") if joint.attrib["type"] != "fixed"}
        self.assertEqual(set(JOINT_LIMITS), active)
        for joint in robot.findall("joint"):
            if joint.attrib["name"] in JOINT_LIMITS:
                limit = joint.find("limit")
                self.assertEqual(JOINT_LIMITS[joint.attrib["name"]], (float(limit.attrib["lower"]), float(limit.attrib["upper"])))


if __name__ == "__main__":
    unittest.main()
