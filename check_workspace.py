import rclpy
import numpy as np

from sensor_msgs.msg import CameraInfo
from rclpy.node import Node

from robot.coordinates import CompanyCoordinateMapper


class WorkspaceCheck(Node):
    def __init__(self):
        super().__init__("workspace_projection_check")
        self.mapper = CompanyCoordinateMapper()
        self.sub = self.create_subscription(
            CameraInfo,
            "/depth_cam/rgb/camera_info",
            self.camera_info_callback,
            10,
        )
        self.done = False

    def camera_info_callback(self, msg):
        if self.done:
            return

        self.done = True
        K = np.asarray(msg.k, dtype=float).reshape(3, 3)
        D = np.asarray(msg.d, dtype=float)

        print("\nRGB CameraInfo:")
        print("Image size:", msg.width, "x", msg.height)
        print("Frame:", msg.header.frame_id)
        print("Camera matrix:\n", K)
        print("Distortion:", D)

        print("\nProjected company workspace corners:")
        for shift in (0.0, 0.03, 0.04):
            polygon, _, _ = self.mapper._project_workspace(
                K, D, plane_shift_m=shift
            )
            print(f"Plane shift {shift:.2f} m: {polygon.tolist()}")

        self.destroy_node()


def main():
    rclpy.init()
    node = WorkspaceCheck()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=1.0)
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
