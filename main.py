#!/usr/bin/env python3

import json
import logging
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import cv2
import rclpy

from cv_bridge import CvBridge
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo

from vision.detector import FruitDetector
from robot.coordinates import CompanyCoordinateMapper
from robot.controller import JetArmController
from sdk import common
from app.utils import calculate_grasp_yaw


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
LOG_DIR = PROJECT_DIR / "logs"
LOG_FILE = LOG_DIR / "jetarm_app.log"
SEQUENCE_FILE = PROJECT_DIR / "config" / "sequence.json"

MODEL_PATH = "/home/ubuntu/Downloads/yolo_1791978301.pt"
CAMERA_TOPIC = "/depth_cam/rgb/image_raw"

CONFIDENCE_THRESHOLD = 0.50
IMAGE_SIZE = 640

FRUITS = ["banana", "apple", "orange"]


# ============================================================
# LOGGING
# ============================================================

def configure_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("jetarm_app")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | "
        "%(threadName)s | %(message)s"
    )

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("=" * 60)
    logger.info("JetArm Fruit Pick-and-Place application starting")
    logger.info("Model: %s", MODEL_PATH)
    logger.info("Camera topic: %s", CAMERA_TOPIC)
    logger.info("Log file: %s", LOG_FILE)
    logger.info("Robot integration loaded; movement requires UI confirmation and IK preflight")

    return logger


LOGGER = configure_logging()


# ============================================================
# CAMERA NODE
# ============================================================

class FruitCameraNode(Node):

    def __init__(self):
        super().__init__("jetarm_fruit_camera")

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_frame = None
        self.frame_count = 0
        self.latest_camera_info = None

        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            "/depth_cam/rgb/camera_info",
            self.camera_info_callback,
            10,
        )

        self.subscription = self.create_subscription(
            Image,
            CAMERA_TOPIC,
            self.image_callback,
            10,
        )

        LOGGER.info("Subscribed to camera topic: %s", CAMERA_TOPIC)

    def camera_info_callback(self, msg):
        with self.lock:
            self.latest_camera_info = {
                "width": int(msg.width),
                "height": int(msg.height),
                "k": list(msg.k),
                "d": list(msg.d),
                "frame_id": msg.header.frame_id,
            }

    def get_camera_matrix(self):
        with self.lock:
            if self.latest_camera_info is None:
                return None
            info = dict(self.latest_camera_info)
        import numpy as np
        return np.asarray(info["k"], dtype=float).reshape(3, 3), info

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )

            with self.lock:
                self.latest_frame = frame
                self.frame_count += 1

        except Exception:
            LOGGER.exception("Camera image conversion failed")

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None, self.frame_count

            return self.latest_frame.copy(), self.frame_count


# ============================================================
# CAMERA WORKER
# ============================================================

class CameraWorker(QThread):

    image_ready = pyqtSignal(QImage)
    detections_ready = pyqtSignal(list)
    status_ready = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.node = None

    def stop(self):
        LOGGER.info("Camera worker stop requested")
        self.running = False
        self.wait()
        LOGGER.info("Camera worker stopped")

    def run(self):
        last_frame_id = -1

        try:
            LOGGER.info("Loading trained YOLO model")

            detector = FruitDetector(
                model_path=MODEL_PATH,
                confidence=CONFIDENCE_THRESHOLD,
                image_size=IMAGE_SIZE,
            )

            if not rclpy.ok():
                raise RuntimeError("ROS 2 must be initialized by main() before starting the camera worker")

            self.node = FruitCameraNode()
            self.status_ready.emit("Waiting for camera frames...")

            while self.running and rclpy.ok():
                rclpy.spin_once(self.node, timeout_sec=0.01)

                frame, frame_id = self.node.get_latest_frame()

                if frame is None or frame_id == last_frame_id:
                    time.sleep(0.005)
                    continue

                last_frame_id = frame_id

                # Run inference on alternating frames.
                if frame_id % 2:
                    continue

                try:
                    detections = detector.detect(frame)
                except Exception:
                    LOGGER.exception("YOLO inference failed")
                    self.status_ready.emit("ERROR: YOLO inference failed")
                    continue

                for item in detections:
                    x1, y1, x2, y2 = item["bbox"]

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        f"{item['class_name']}: {item['confidence']:.2f}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channels = rgb.shape

                qimage = QImage(
                    rgb.data,
                    width,
                    height,
                    channels * width,
                    QImage.Format_RGB888,
                ).copy()

                self.image_ready.emit(qimage)
                self.detections_ready.emit(detections)

                self.status_ready.emit(
                    f"Camera active | {len(detections)} fruit detection(s)"
                )

        except Exception:
            LOGGER.exception("Camera worker failed")
            self.status_ready.emit("Camera error; inspect application logs")

        finally:
            if self.node is not None:
                self.node.destroy_node()
                self.node = None



class PickPlaceWorker(QThread):
    status_ready = pyqtSignal(str)
    item_status = pyqtSignal(int, str)
    finished_result = pyqtSignal(bool, str)
    controller_ready = pyqtSignal(object)

    def __init__(self, sequence, get_detection_snapshot, get_camera_matrix):
        super().__init__()
        self.sequence = sequence
        self.get_detection_snapshot = get_detection_snapshot
        self.get_camera_matrix = get_camera_matrix
        self.stop_event = threading.Event()
        self.controller = None

    def request_stop(self):
        self.stop_event.set()

    def _wait_for_fresh_detections(self, timeout=12.0):
        self.status_ready.emit("Waiting for fresh YOLO detections...")
        deadline = time.monotonic() + timeout
        last_stamp = -1.0
        stable = []
        last_centers = None
        while time.monotonic() < deadline and not self.stop_event.is_set():
            detections, stamp = self.get_detection_snapshot()
            if stamp > last_stamp and detections:
                last_stamp = stamp
                centers = [(d["class_name"], ((d["bbox"][0] + d["bbox"][2]) / 2.0, (d["bbox"][1] + d["bbox"][3]) / 2.0)) for d in detections]
                if last_centers is not None:
                    # A basic stability gate: at least one fruit center must persist across fresh inference updates.
                    stable_match = False
                    for cls, center in centers:
                        for old_cls, old_center in last_centers:
                            if cls == old_cls and ((center[0]-old_center[0])**2 + (center[1]-old_center[1])**2)**0.5 <= 18:
                                stable_match = True
                                break
                        if stable_match:
                            break
                    stable = stable + [detections] if stable_match else [detections]
                else:
                    stable = [detections]
                last_centers = centers
                if len(stable) >= 5:
                    return detections
            time.sleep(0.08)
        if self.stop_event.is_set():
            raise InterruptedError("Stop requested before movement")
        raise TimeoutError("No stable fresh YOLO detections received within 12 seconds")

    def _select_distinct_detections(self, detections):
        used = []
        selected = []
        for item in self.sequence:
            candidates = [d for d in detections if d["class_name"].lower() == item["fruit"].lower()]
            candidates.sort(key=lambda d: d.get("confidence", 0.0), reverse=True)
            chosen = None
            for d in candidates:
                x1, y1, x2, y2 = d["bbox"]
                center = ((x1+x2)/2.0, (y1+y2)/2.0)
                if all(((center[0]-ux)**2 + (center[1]-uy)**2)**0.5 > 35 for ux, uy in used):
                    chosen = d
                    used.append(center)
                    break
            if chosen is None:
                raise ValueError(f"No distinct visible {item['fruit']} detection found for destination '{item['destination_name']}'. Keep every queued fruit visible and retry.")
            selected.append(chosen)
        return selected

    @staticmethod
    def _company_place_yaw(position, angle=0):
        import math
        yaw = math.degrees(math.atan2(position[1], position[0]))
        if position[0] < 0 and position[1] < 0:
            yaw += 180
        elif position[0] < 0 and position[1] > 0:
            yaw -= 180
        yaw1 = yaw + angle
        yaw2 = yaw1 + 90 if yaw < 0 else yaw1 - 90
        chosen = yaw2 if abs(yaw2) < abs(yaw1) else yaw1
        return 500 + int(chosen / 240 * 1000)

    def _company_pick_yaw(self, detection, all_detections, world_position, K, projection):
        import math
        target_info = []
        target_for_detection = None
        x1, y1, x2, y2 = detection["bbox"]
        wanted_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        for index, det in enumerate(all_detections, start=1):
            bx1, by1, bx2, by2 = det["bbox"]
            center = (int((bx1 + bx2) / 2), int((by1 + by2) / 2))
            info = [det["class_name"], index, center, (int(bx2 - bx1), int(by2 - by1)), 0]
            target_info.append(info)
            if det is detection or (det["class_name"] == detection["class_name"] and abs(center[0]-wanted_center[0]) < 1 and abs(center[1]-wanted_center[1]) < 1):
                target_for_detection = info
        if target_for_detection is None:
            raise ValueError("Could not build company grasp target information from YOLO detection")
        yaw = math.degrees(math.atan2(world_position[1], world_position[0]))
        if world_position[0] < 0 and world_position[1] < 0:
            yaw += 180
        elif world_position[0] < 0 and world_position[1] > 0:
            yaw -= 180
        intrinsic = __import__("numpy").matrix(K, dtype=float)
        gripper_size = [
            common.calculate_pixel_length(0.09, intrinsic, projection),
            common.calculate_pixel_length(0.015, intrinsic, projection),
        ]
        result = calculate_grasp_yaw.calculate_gripper_yaw_angle(
            target_for_detection, target_info, gripper_size, yaw
        )
        if result is None:
            raise ValueError(f"Company grasp-yaw calculation failed for {detection['class_name']} at {wanted_center}")
        return 500 + int(float(result[0]) / 240 * 1000)

    def _preflight_position(self, mapper, pixel, height, kind):
        K_result = self.get_camera_matrix()
        if K_result is None:
            raise RuntimeError("CameraInfo has not arrived on /depth_cam/rgb/camera_info")
        K, info = K_result
        if info.get("width") != 640 or info.get("height") != 480:
            raise ValueError(f"Unexpected RGB calibration size {info.get('width')}x{info.get('height')}; expected 640x480")
        mapper.ensure_pixel_in_calibrated_roi(pixel, K, info.get("d", []), (info["width"], info["height"]))
        world = mapper.pixel_to_company_world(pixel, K, height)
        robot_position = mapper.pick_kinematics_position(world) if kind == "pick" else mapper.place_kinematics_position(world)
        return world, robot_position

    def run(self):
        try:
            if self.stop_event.is_set():
                raise InterruptedError("Stop requested")
            self.status_ready.emit("Loading company calibration and JetArm services...")
            mapper = CompanyCoordinateMapper()
            self.controller = JetArmController()
            self.controller_ready.emit(self.controller)
            self.status_ready.emit("Waiting for stable fruit detections and mapping all targets...")
            detections = self._wait_for_fresh_detections()
            chosen_detections = self._select_distinct_detections(detections)
            camera_data = self.get_camera_matrix()
            if camera_data is None:
                raise RuntimeError("CameraInfo has not arrived on /depth_cam/rgb/camera_info")
            K_live, camera_info = camera_data
            projection = mapper.projection_matrix()

            plan = []
            for index, (item, detection) in enumerate(zip(self.sequence, chosen_detections)):
                if self.stop_event.is_set():
                    raise InterruptedError("Stop requested during preflight")
                x1, y1, x2, y2 = detection["bbox"]
                fruit_pixel = (int((x1+x2)/2), int((y1+y2)/2))
                destination = item["destination_pixel"]
                pick_world, pick_position = self._preflight_position(mapper, fruit_pixel, 0.03, "pick")
                place_world, place_position = self._preflight_position(mapper, (destination["x"], destination["y"]), 0.015, "place")
                self.status_ready.emit(f"IK preflight {index+1}/{len(self.sequence)}: {item['fruit']}...")
                pick_yaw = self._company_pick_yaw(detection, detections, pick_world, K_live, projection)
                place_yaw = self._company_place_yaw(place_world, 0)
                self.controller.preflight_pick(pick_position, pitch=80, gripper_depth=0.02)
                self.controller.check_pose([0.11, 0.0, 0.09], pitch=73)
                self.controller.preflight_place(place_position, pitch=80)
                plan.append({"item": item, "fruit_pixel": fruit_pixel, "pick_world": pick_world, "pick_position": pick_position, "place_world": place_world, "place_position": place_position, "pick_yaw": pick_yaw, "place_yaw": place_yaw, "confidence": detection.get("confidence", 0.0)})
                self.item_status.emit(index, f"IK preflight passed (fruit confidence {detection.get('confidence', 0.0):.2f})")

            if self.stop_event.is_set():
                raise InterruptedError("Stop requested before movement")
            self.status_ready.emit("All queued targets passed IK checks. Beginning physical pick-and-place sequence...")

            for index, target in enumerate(plan):
                if self.stop_event.is_set():
                    raise InterruptedError("Stop requested before next item")
                item = target["item"]
                self.item_status.emit(index, "Picking")
                self.status_ready.emit(f"Picking {item['fruit']} for {item['destination_name']}...")
                LOGGER.info("PICK_TARGET fruit=%s pixel=%s world=%s robot=%s", item["fruit"], target["fruit_pixel"], target["pick_world"], target["pick_position"])
                self.controller.pick(target["pick_position"], yaw=target["pick_yaw"], pitch=80, gripper_angle=540, gripper_depth=0.02)
                if self.stop_event.is_set():
                    raise InterruptedError("Stop requested after pickup; object may still be held")
                self.item_status.emit(index, "Placing")
                self.status_ready.emit(f"Placing at {item['destination_name']}...")
                LOGGER.info("PLACE_TARGET name=%s pixel=%s world=%s robot=%s", item["destination_name"], item["destination_pixel"], target["place_world"], target["place_position"])
                self.controller.place(target["place_position"], yaw=target["place_yaw"], pitch=80, gripper_angle=200)
                self.item_status.emit(index, "Completed")

            self.finished_result.emit(True, f"Completed {len(plan)} pick-and-place item(s). Verify the physical placement before running another sequence.")
        except Exception as exc:
            LOGGER.exception("Pick-and-place sequence failed")
            if self.controller is not None:
                try:
                    self.controller.request_stop()
                except Exception:
                    pass
            self.finished_result.emit(False, f"Sequence failed/stopped: {exc}\nInspect logs at {LOG_FILE}. If the arm is moving unexpectedly, use the hardware emergency stop.")
        finally:
            if self.controller is not None:
                self.controller.close()


# ============================================================
# MAIN UI
# ============================================================

class JetArmFruitApp(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("JetArm AI Fruit Pick-and-Place")
        self.resize(1200, 900)

        self.sequence = []
        self.destination_mode = False
        self.display_image_size = (0, 0)

        self.camera_label = QLabel("Waiting for camera...")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 400)
        self.camera_label.setStyleSheet(
            "background-color: #202020; color: white;"
            "border: 1px solid #555;"
        )
        self.camera_label.setMouseTracking(True)
        self.camera_label.mousePressEvent = self.camera_clicked
        self.camera_label.mouseMoveEvent = self.camera_moved

        self.status_label = QLabel("Initializing...")
        self.status_label.setWordWrap(True)

        self.fruit_combo = QComboBox()
        self.fruit_combo.addItems(FRUITS)

        self.destination_name = QLineEdit()
        self.destination_name.setPlaceholderText(
            "Destination name, e.g. Banana Zone"
        )

        self.add_button = QPushButton("Add Fruit to Sequence")
        self.set_point_button = QPushButton("Set Destination Point")
        self.cancel_button = QPushButton("Cancel Point Selection")
        self.remove_button = QPushButton("Remove Selected")
        self.clear_button = QPushButton("Clear Sequence")
        self.save_button = QPushButton("Save Sequence")
        self.load_button = QPushButton("Load Sequence")
        self.stop_button = QPushButton("Stop Camera")
        self.sequence_list = QListWidget()

        self.detection_label = QLabel("No detections yet.")
        self.detection_label.setWordWrap(True)

        self.point_label = QLabel(
            "Destination: not selected. "
            "Select a sequence item, then click Set Destination Point."
        )
        self.point_label.setWordWrap(True)

        self.safety_label = QLabel(
            "Robot ready check pending. Start performs coordinate conversion and IK preflight before motion."
        )
        self.safety_label.setWordWrap(True)
        self.safety_label.setStyleSheet(
            "color: #9c3500; font-weight: bold;"
        )

        self.start_button = QPushButton("Start Pick-and-Place")
        self.start_button.setEnabled(False)
        self.request_stop_button = QPushButton("Request Stop")
        self.request_stop_button.setEnabled(False)
        self.request_stop_button.setToolTip("Requests stop between company motion segments. Use the hardware emergency stop for an immediate stop.")
        self.robot_controller = None
        self.sequence_worker = None
        self.detections_lock = threading.Lock()
        self.latest_detections = []
        self.latest_detections_time = 0.0

        # Fruit configuration
        fruit_layout = QHBoxLayout()
        fruit_layout.addWidget(QLabel("Fruit:"))
        fruit_layout.addWidget(self.fruit_combo)
        fruit_layout.addWidget(QLabel("Destination name:"))
        fruit_layout.addWidget(self.destination_name)
        fruit_layout.addWidget(self.add_button)

        # Destination controls
        destination_layout = QHBoxLayout()
        destination_layout.addWidget(self.set_point_button)
        destination_layout.addWidget(self.cancel_button)

        # Queue controls
        queue_layout = QHBoxLayout()
        queue_layout.addWidget(self.remove_button)
        queue_layout.addWidget(self.clear_button)
        queue_layout.addWidget(self.save_button)
        queue_layout.addWidget(self.load_button)
        queue_layout.addStretch()
        queue_layout.addWidget(self.stop_button)
        queue_layout.addWidget(self.request_stop_button)

        layout = QVBoxLayout()
        layout.addWidget(self.camera_label, stretch=4)
        layout.addWidget(self.status_label)
        layout.addWidget(QLabel("Detected fruits:"))
        layout.addWidget(self.detection_label)
        layout.addLayout(fruit_layout)
        layout.addWidget(QLabel("Processing sequence:"))
        layout.addWidget(self.sequence_list, stretch=1)
        layout.addLayout(destination_layout)
        layout.addWidget(self.point_label)
        layout.addLayout(queue_layout)
        layout.addWidget(self.safety_label)
        layout.addWidget(self.start_button)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.add_button.clicked.connect(self.add_to_sequence)
        self.set_point_button.clicked.connect(self.activate_point_selection)
        self.cancel_button.clicked.connect(self.cancel_point_selection)
        self.remove_button.clicked.connect(self.remove_selected)
        self.clear_button.clicked.connect(self.clear_sequence)
        self.save_button.clicked.connect(self.save_sequence)
        self.load_button.clicked.connect(self.load_sequence)
        self.stop_button.clicked.connect(self.stop_camera)
        self.start_button.clicked.connect(self.start_pick_and_place)
        self.request_stop_button.clicked.connect(self.request_robot_stop)

        self.worker = CameraWorker()
        self.worker.image_ready.connect(self.update_camera)
        self.worker.detections_ready.connect(self.update_detections)
        self.worker.status_ready.connect(self.update_status)
        self.worker.start()

        LOGGER.info("Main UI initialized")

    # --------------------------------------------------------
    # IMAGE AND DESTINATION MARKERS
    # --------------------------------------------------------

    def update_camera(self, image):
        self.display_image_size = (image.width(), image.height())

        # Draw markers onto a copy of the current image.
        pixmap = QPixmap.fromImage(image)
        painter = QPainter(pixmap)

        for item in self.sequence:
            point = item.get("destination_pixel")
            if point is None:
                continue

            x, y = point["x"], point["y"]

            painter.setPen(QPen(QColor(255, 60, 60), 3))
            painter.drawLine(x - 10, y, x + 10, y)
            painter.drawLine(x, y - 10, x, y + 10)
            painter.setBrush(QColor(255, 60, 60, 100))
            painter.drawEllipse(x - 6, y - 6, 12, 12)

            painter.setPen(QPen(QColor(255, 255, 0), 2))
            painter.drawText(x + 12, y - 10, item["destination_name"])

        painter.end()

        scaled = pixmap.scaled(
            self.camera_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.camera_label.setPixmap(scaled)

    def camera_clicked(self, event):
        if event.button() != Qt.LeftButton:
            return

        if not self.destination_mode:
            LOGGER.info("Image clicked while point selection inactive")
            return

        row = self.sequence_list.currentRow()

        if row < 0 or row >= len(self.sequence):
            LOGGER.warning("Destination click ignored: no queue item selected")
            return

        pixmap = self.camera_label.pixmap()

        if pixmap is None:
            LOGGER.warning("Destination click ignored: camera image missing")
            return

        # Map the click from the displayed image to the original image.
        label_width = self.camera_label.width()
        label_height = self.camera_label.height()

        image_width, image_height = self.display_image_size
        if image_width <= 0 or image_height <= 0:
            return

        scale = min(
            label_width / image_width,
            label_height / image_height,
        )

        shown_width = image_width * scale
        shown_height = image_height * scale

        x_offset = (label_width - shown_width) / 2
        y_offset = (label_height - shown_height) / 2

        shown_x = event.pos().x() - x_offset
        shown_y = event.pos().y() - y_offset

        if not (
            0 <= shown_x < shown_width
            and 0 <= shown_y < shown_height
        ):
            LOGGER.warning("Destination click outside camera image")
            return

        image_x = int(shown_x / scale)
        image_y = int(shown_y / scale)

        item = self.sequence[row]
        item["destination_pixel"] = {
            "x": image_x,
            "y": image_y,
        }
        item["status"] = "Destination pixel recorded"

        self.destination_mode = False
        self.camera_label.setCursor(Qt.ArrowCursor)

        self.refresh_sequence()
        self.sequence_list.setCurrentRow(row)

        message = (
            f"{item['destination_name']} saved at image pixel "
            f"({image_x}, {image_y}). Robot coordinates not calculated."
        )

        self.point_label.setText(message)
        self.status_label.setText("Destination marker saved.")

        LOGGER.info(
            "DESTINATION_SAVED sequence=%d fruit=%s name=%s "
            "pixel_x=%d pixel_y=%d",
            row + 1,
            item["fruit"],
            item["destination_name"],
            image_x,
            image_y,
        )

    def camera_moved(self, event):
        if self.destination_mode:
            self.camera_label.setCursor(Qt.CrossCursor)

    # --------------------------------------------------------
    # SEQUENCE
    # --------------------------------------------------------

    def add_to_sequence(self):
        fruit = self.fruit_combo.currentText().strip().lower()
        name = self.destination_name.text().strip()

        if not name:
            QMessageBox.warning(
                self,
                "Destination Name Required",
                "Enter a destination name, for example Banana Zone.",
            )
            return

        item = {
            "fruit": fruit,
            "destination_name": name,
            "destination_pixel": None,
            "status": "Waiting for destination",
        }

        self.sequence.append(item)
        self.refresh_sequence()

        LOGGER.info("SEQUENCE_ADDED item=%s", item)

    def activate_point_selection(self):
        row = self.sequence_list.currentRow()

        if row < 0 or row >= len(self.sequence):
            QMessageBox.information(
                self,
                "Select Sequence Item",
                "Select a fruit in the sequence first.",
            )
            return

        self.destination_mode = True
        self.camera_label.setCursor(Qt.CrossCursor)

        item = self.sequence[row]

        self.point_label.setText(
            f"POINT SELECTION ACTIVE: Click the camera image to set "
            f"the destination for {item['destination_name']}."
        )

        self.status_label.setText(
            "Crosshair active: click a point inside the camera image."
        )

        LOGGER.info(
            "DESTINATION_SELECTION_STARTED sequence=%d name=%s",
            row + 1,
            item["destination_name"],
        )

    def cancel_point_selection(self):
        self.destination_mode = False
        self.camera_label.setCursor(Qt.ArrowCursor)
        self.point_label.setText("Destination selection cancelled.")

        LOGGER.info("DESTINATION_SELECTION_CANCELLED")

    def refresh_sequence(self):
        row = self.sequence_list.currentRow()
        self.sequence_list.clear()

        for index, item in enumerate(self.sequence, start=1):
            point = item["destination_pixel"]

            point_text = (
                "not selected"
                if point is None
                else f"pixel=({point['x']}, {point['y']})"
            )

            self.sequence_list.addItem(
                f"{index}. {item['fruit'].capitalize()} -> "
                f"{item['destination_name']} | {point_text} | "
                f"[{item['status']}]"
            )

        if self.sequence:
            self.sequence_list.setCurrentRow(
                min(max(row, 0), len(self.sequence) - 1)
            )

        ready = bool(self.sequence) and all(item.get("destination_pixel") is not None for item in self.sequence)
        ready = ready and self.worker.isRunning() and self.sequence_worker is None
        self.start_button.setEnabled(ready)

    def remove_selected(self):
        row = self.sequence_list.currentRow()

        if row < 0 or row >= len(self.sequence):
            return

        removed = self.sequence.pop(row)
        self.refresh_sequence()

        LOGGER.info("SEQUENCE_REMOVED item=%s", removed)

    def clear_sequence(self):
        self.sequence.clear()
        self.destination_mode = False
        self.camera_label.setCursor(Qt.ArrowCursor)
        self.refresh_sequence()
        self.point_label.setText("Destination: not selected.")

        LOGGER.info("SEQUENCE_CLEARED")

    def save_sequence(self):
        if not self.sequence:
            QMessageBox.information(
                self,
                "Nothing to Save",
                "Add at least one fruit to the sequence first.",
            )
            return

        incomplete = [
            item["destination_name"]
            for item in self.sequence
            if item.get("destination_pixel") is None
        ]

        if incomplete:
            QMessageBox.warning(
                self,
                "Destination Missing",
                "Set a destination point for every sequence item "
                "before saving.\n\nMissing destination(s):\n"
                + "\n".join(incomplete),
            )
            return

        data = {
            "version": 1,
            "coordinate_type": "camera_image_pixels",
            "robot_movement_enabled": True,
            "sequence": self.sequence,
        }

        try:
            SEQUENCE_FILE.parent.mkdir(parents=True, exist_ok=True)

            temporary_file = SEQUENCE_FILE.with_suffix(".json.tmp")

            with temporary_file.open("w", encoding="utf-8") as file:
                json.dump(data, file, indent=4, ensure_ascii=False)
                file.write("\n")

            temporary_file.replace(SEQUENCE_FILE)

            LOGGER.info(
                "SEQUENCE_SAVED file=%s count=%d",
                SEQUENCE_FILE,
                len(self.sequence),
            )

            QMessageBox.information(
                self,
                "Sequence Saved",
                f"Saved {len(self.sequence)} sequence item(s).\n\n"
                f"File:\n{SEQUENCE_FILE}\n\n"
                "Destination points are saved as camera pixels and converted at execution using company calibration.",
            )

        except Exception as exc:
            LOGGER.exception("Failed to save sequence")

            QMessageBox.critical(
                self,
                "Save Failed",
                f"Could not save the sequence:\n{exc}",
            )

    def load_sequence(self):
        if not SEQUENCE_FILE.is_file():
            QMessageBox.information(
                self,
                "No Saved Sequence",
                f"No saved sequence was found at:\n{SEQUENCE_FILE}",
            )
            return

        try:
            with SEQUENCE_FILE.open("r", encoding="utf-8") as file:
                data = json.load(file)

            if not isinstance(data, dict):
                raise ValueError("The saved file must contain a JSON object.")

            if data.get("version") != 1:
                raise ValueError("Unsupported sequence file version.")

            if data.get("coordinate_type") != "camera_image_pixels":
                raise ValueError("Unexpected coordinate type in saved file.")

            saved_sequence = data.get("sequence")

            if not isinstance(saved_sequence, list):
                raise ValueError("'sequence' must be a list.")

            validated_sequence = []

            for index, item in enumerate(saved_sequence, start=1):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Sequence item {index} is not a valid object."
                    )

                fruit = item.get("fruit")
                name = item.get("destination_name")
                point = item.get("destination_pixel")

                if fruit not in FRUITS:
                    raise ValueError(
                        f"Sequence item {index} has an unsupported fruit."
                    )

                if not isinstance(name, str) or not name.strip():
                    raise ValueError(
                        f"Sequence item {index} has no destination name."
                    )

                if point is not None:
                    if not isinstance(point, dict):
                        raise ValueError(
                            f"Sequence item {index} has an invalid point."
                        )

                    x = point.get("x")
                    y = point.get("y")

                    if (
                        type(x) is not int
                        or type(y) is not int
                        or not (
                            0 <= x < 640
                            and 0 <= y < 480
                        )
                    ):
                        raise ValueError(
                            f"Sequence item {index} has invalid pixel "
                            "coordinates. Expected x in 0-639 and y in 0-479."
                        )

                    point = {"x": x, "y": y}

                validated_sequence.append(
                    {
                        "fruit": fruit,
                        "destination_name": name.strip(),
                        "destination_pixel": point,
                        "status": (
                            "Destination pixel recorded"
                            if point is not None
                            else "Waiting for destination"
                        ),
                    }
                )

            if self.sequence:
                answer = QMessageBox.question(
                    self,
                    "Replace Current Sequence?",
                    "Loading will replace the sequence currently in the UI. "
                    "Save the current sequence first if you need it.\n\n"
                    "Continue?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )

                if answer != QMessageBox.Yes:
                    return

            self.destination_mode = False
            self.camera_label.setCursor(Qt.ArrowCursor)
            self.sequence = validated_sequence
            self.refresh_sequence()

            self.point_label.setText(
                "Loaded sequence. Destination points are camera pixels."
            )

            LOGGER.info(
                "SEQUENCE_LOADED file=%s count=%d",
                SEQUENCE_FILE,
                len(self.sequence),
            )

            QMessageBox.information(
                self,
                "Sequence Loaded",
                f"Loaded {len(self.sequence)} sequence item(s).\n\n"
                "Destination markers are restored. The Start button becomes available when every item has a destination.",
            )

        except Exception as exc:
            LOGGER.exception("Failed to load sequence")

            QMessageBox.critical(
                self,
                "Load Failed",
                f"Could not load the saved sequence:\n{exc}",
            )


    # --------------------------------------------------------
    # ROBOT PICK-AND-PLACE
    # --------------------------------------------------------

    def _get_detection_snapshot(self):
        with self.detections_lock:
            return list(self.latest_detections), self.latest_detections_time

    def start_pick_and_place(self):
        if self.sequence_worker is not None and self.sequence_worker.isRunning():
            return
        if not self.sequence:
            QMessageBox.warning(self, "Empty Sequence", "Add at least one fruit first.")
            return
        missing = [item["destination_name"] for item in self.sequence if not item.get("destination_pixel")]
        if missing:
            QMessageBox.warning(self, "Destination Missing", "Set a destination for every item:\n" + "\n".join(missing))
            return
        answer = QMessageBox.warning(
            self,
            "Confirm Physical Robot Movement",
            "This will move the JetArm and operate its gripper.\n\n"
            "Ensure the arm's workspace is clear, fruits and destination surfaces are within the calibrated work area, "
            "and you can reach the hardware emergency stop. Keep hands clear.\n\n"
            "The application will calculate coordinates from the company calibration and check IK targets first. "
            "IK does not guarantee collision-free motion. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        snapshot, _ = self._get_detection_snapshot()
        if not snapshot:
            QMessageBox.warning(self, "No Fruit Detections", "Wait for YOLO to detect fruits, then try again.")
            return

        self.start_button.setEnabled(False)
        self.request_stop_button.setEnabled(True)
        self.status_label.setText("Starting robot integration and checking calibration/services...")
        self.sequence_worker = PickPlaceWorker(
            sequence=[dict(item) for item in self.sequence],
            get_detection_snapshot=self._get_detection_snapshot,
            get_camera_matrix=self._get_camera_matrix,
        )
        self.sequence_worker.status_ready.connect(self.update_status)
        self.sequence_worker.item_status.connect(self.on_robot_item_status)
        self.sequence_worker.finished_result.connect(self.on_robot_finished)
        self.sequence_worker.controller_ready.connect(self.set_robot_controller)
        self.sequence_worker.start()

    def _get_camera_matrix(self):
        if self.worker.node is None:
            return None
        result = self.worker.node.get_camera_matrix()
        return result

    def set_robot_controller(self, controller):
        self.robot_controller = controller

    def request_robot_stop(self):
        if self.sequence_worker is None or not self.sequence_worker.isRunning():
            return
        self.sequence_worker.request_stop()
        if self.robot_controller is not None:
            try:
                self.robot_controller.request_stop()
            except Exception:
                LOGGER.exception("Could not set company stop flag")
        self.status_label.setText("Stop requested. The company routine checks stop between motion segments; use hardware emergency stop for immediate stopping.")

    def on_robot_item_status(self, index, status):
        if 0 <= index < len(self.sequence):
            self.sequence[index]["status"] = status
            self.refresh_sequence()

    def on_robot_finished(self, success, message):
        LOGGER.info("ROBOT_SEQUENCE_FINISHED success=%s message=%s", success, message)
        self.status_label.setText(message)
        self.request_stop_button.setEnabled(False)
        self.robot_controller = None
        self.sequence_worker = None
        self.refresh_sequence()
        if success:
            QMessageBox.information(self, "Pick-and-Place Complete", message)
        else:
            QMessageBox.warning(self, "Pick-and-Place Stopped/Failed", message)

    # --------------------------------------------------------
    # DETECTIONS AND SHUTDOWN
    # --------------------------------------------------------

    def update_detections(self, detections):
        with self.detections_lock:
            self.latest_detections = list(detections)
            self.latest_detections_time = time.monotonic()

        if not detections:
            self.detection_label.setText("No supported fruits detected.")
            return

        self.detection_label.setText(
            "\n".join(
                f"{index}. {item['class_name']} "
                f"confidence={item['confidence']:.2f} "
                f"bbox={item['bbox']}"
                for index, item in enumerate(detections, start=1)
            )
        )

    def update_status(self, message):
        self.status_label.setText(message)

    def stop_camera(self):
        if self.worker.isRunning():
            self.worker.stop()

        self.stop_button.setEnabled(False)
        self.status_label.setText("Camera stopped.")
        self.refresh_sequence()

    def closeEvent(self, event):
        LOGGER.info("Application close requested")

        if self.sequence_worker is not None and self.sequence_worker.isRunning():
            self.request_robot_stop()
            QMessageBox.warning(
                self,
                "Robot Sequence Stopping",
                "A robot sequence is still active. A stop has been requested. "
                "Keep clear of the arm and close this window only after the sequence reports stopped/finished. "
                "Use the hardware emergency stop for an immediate stop.",
            )
            event.ignore()
            return
        if self.worker.isRunning():
            self.worker.stop()
        event.accept()


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    if not rclpy.ok():
        rclpy.init(args=None)
    app = QApplication(sys.argv)
    window = JetArmFruitApp()
    window.show()
    try:
        app.exec_()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
