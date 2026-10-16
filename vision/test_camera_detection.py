#!/usr/bin/env python3

import sys
import time
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from detector import FruitDetector


MODEL_PATH = "/home/ubuntu/Downloads/yolo_1791978301.pt"
CAMERA_TOPIC = "/depth_cam/rgb/image_raw"
WINDOW_NAME = "JetArm Fruit Detection"


class CameraDetectionTest(Node):
    def __init__(self):
        super().__init__("jetarm_fruit_detection_test")

        self.bridge = CvBridge()
        self.latest_frame = None
        self.frame_number = 0
        self.last_report_time = 0.0

        self.detector = FruitDetector(
            model_path=MODEL_PATH,
            confidence=0.50,
            image_size=640,
        )

        self.subscription = self.create_subscription(
            Image,
            CAMERA_TOPIC,
            self.image_callback,
            10,
        )

        self.get_logger().info(f"Subscribed to: {CAMERA_TOPIC}")
        self.get_logger().info("Waiting for RGB camera frames...")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )
            self.latest_frame = frame
            self.frame_number += 1

        except Exception as exc:
            self.get_logger().error(
                f"Camera image conversion failed: {exc}"
            )


def main():
    rclpy.init()

    node = None

    try:
        node = CameraDetectionTest()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        last_processed_frame = -1

        print("\n========================================")
        print("JETARM FRUIT DETECTION TEST")
        print("Model:", MODEL_PATH)
        print("Camera:", CAMERA_TOPIC)
        print("Press Q or ESC to quit.")
        print("========================================\n")

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

            if node.latest_frame is None:
                cv2.waitKey(1)
                continue

            if node.frame_number == last_processed_frame:
                cv2.waitKey(1)
                continue

            last_processed_frame = node.frame_number
            frame = node.latest_frame.copy()

            try:
                detections = node.detector.detect(frame)

                for detection in detections:
                    x1, y1, x2, y2 = detection["bbox"]
                    label = detection["class_name"]
                    confidence = detection["confidence"]

                    text = f"{label}: {confidence:.2f}"

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        text,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                now = time.time()

                if now - node.last_report_time >= 2.0:
                    if detections:
                        summary = [
                            (
                                f"{d['class_name']} "
                                f"({d['confidence']:.2f})"
                            )
                            for d in detections
                        ]
                        print("Detected:", ", ".join(summary))
                    else:
                        print("No supported fruits detected.")

                    node.last_report_time = now

                cv2.imshow(WINDOW_NAME, frame)

            except Exception as exc:
                node.get_logger().error(
                    f"Detection failed: {type(exc).__name__}: {exc}"
                )
                break

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    except KeyboardInterrupt:
        print("\nDetection test stopped by user.")

    except Exception as exc:
        print(
            f"\nFATAL ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise

    finally:
        if node is not None:
            node.destroy_node()

        cv2.destroyAllWindows()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
