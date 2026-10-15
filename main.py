import sys
import threading

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLO

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QComboBox, QGroupBox, QMessageBox
)


CAMERA_TOPIC = "/depth_cam/rgb/image_raw"
MODEL_PATH = "yolov8n.pt"
FRUIT_NAMES = {"apple", "banana", "orange"}


class CameraWorker(QThread):
    frame_ready = pyqtSignal(QImage)
    objects_ready = pyqtSignal(list)
    status_ready = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.node = None
        self.bridge = CvBridge()
        self.running = True
        self.frame_count = 0
        self.lock = threading.Lock()
        self.latest_objects = []

    def run(self):
        try:
            self.status_ready.emit("Loading YOLO model...")
            self.model = YOLO(MODEL_PATH)

            rclpy.init(args=None)
            self.node = Node("jetarm_fruit_detection_ui")

            self.node.create_subscription(
                Image, CAMERA_TOPIC, self.image_callback, 10
            )

            self.status_ready.emit(
                "Camera connected. Looking for apples, bananas and oranges."
            )

            while rclpy.ok() and self.running:
                rclpy.spin_once(self.node, timeout_sec=0.05)

        except Exception as exc:
            self.status_ready.emit(f"ERROR: {exc}")

        finally:
            if self.node is not None:
                self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )

            self.frame_count += 1

            # Run detection on every third frame to reduce CPU load.
            if self.frame_count % 3 == 0:
                results = self.model.predict(
                    source=frame,
                    imgsz=416,
                    conf=0.40,
                    verbose=False,
                    device="cpu"
                )

                detected = []

                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls[0].item())
                        name = result.names[class_id]

                        if name not in FRUIT_NAMES:
                            continue

                        confidence = float(box.conf[0].item())
                        x1, y1, x2, y2 = map(
                            int, box.xyxy[0].tolist()
                        )

                        detected.append({
                            "name": name,
                            "confidence": confidence,
                            "box": (x1, y1, x2, y2)
                        })

                with self.lock:
                    self.latest_objects = detected

                self.objects_ready.emit(detected)

            # Draw the latest detections on the displayed frame.
            with self.lock:
                objects = list(self.latest_objects)

            for obj in objects:
                x1, y1, x2, y2 = obj["box"]
                label = (
                    f'{obj["name"]} '
                    f'{obj["confidence"] * 100:.0f}%'
                )

                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), (0, 220, 0), 2
                )
                cv2.putText(
                    frame, label, (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 220, 0), 2
                )

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, channels = rgb.shape

            image = QImage(
                rgb.data, w, h, channels * w,
                QImage.Format_RGB888
            ).copy()

            self.frame_ready.emit(image)

        except Exception as exc:
            self.status_ready.emit(f"Camera/detection error: {exc}")

    def stop(self):
        self.running = False
        self.wait(3000)


class FruitPickPlaceUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("JetArm Fruit Detection & Pick-and-Place")
        self.resize(1150, 760)

        self.detected_objects = []
        self.selected_object = None

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)

        # Left: camera
        camera_group = QGroupBox("Live Camera and Fruit Detection")
        camera_layout = QVBoxLayout(camera_group)

        self.camera_label = QLabel("Waiting for camera...")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 400)
        self.camera_label.setStyleSheet(
            "background-color: #202020; color: white;"
        )

        camera_layout.addWidget(self.camera_label)

        self.status_label = QLabel("Starting application...")
        self.status_label.setWordWrap(True)
        camera_layout.addWidget(self.status_label)

        main_layout.addWidget(camera_group, 3)

        # Right: controls
        control_group = QGroupBox("Pick-and-Place Controls")
        controls = QVBoxLayout(control_group)

        controls.addWidget(QLabel("1. Select the detected fruit"))

        self.fruit_combo = QComboBox()
        self.fruit_combo.addItem("Waiting for detection...")
        self.fruit_combo.currentIndexChanged.connect(
            self.on_fruit_selected
        )
        controls.addWidget(self.fruit_combo)

        self.selection_label = QLabel("Selected fruit: None")
        self.selection_label.setWordWrap(True)
        controls.addWidget(self.selection_label)

        controls.addSpacing(15)
        controls.addWidget(QLabel("2. Select a placement destination"))

        self.destination_combo = QComboBox()
        self.destination_combo.addItems([
            "Position A",
            "Position B",
            "Position C"
        ])
        controls.addWidget(self.destination_combo)

        self.destination_note = QLabel(
            "Destinations are UI selections only. "
            "Robot coordinates have not been configured."
        )
        self.destination_note.setWordWrap(True)
        controls.addWidget(self.destination_note)

        controls.addSpacing(15)

        self.pick_button = QPushButton("PICK & PLACE")
        self.pick_button.setMinimumHeight(48)
        self.pick_button.setEnabled(False)
        self.pick_button.clicked.connect(self.pick_place_clicked)
        controls.addWidget(self.pick_button)

        self.refresh_button = QPushButton("Refresh Fruit Selection")
        self.refresh_button.clicked.connect(self.update_fruit_list)
        controls.addWidget(self.refresh_button)

        controls.addSpacing(15)

        controls.addWidget(QLabel("Workflow status"))

        self.workflow_label = QLabel(
            "Detection starting. Physical robot motion is disabled."
        )
        self.workflow_label.setWordWrap(True)
        controls.addWidget(self.workflow_label)

        controls.addStretch()

        safety_note = QLabel(
            "SAFETY: Detection and UI selection only. "
            "No robot or gripper commands are sent."
        )
        safety_note.setWordWrap(True)
        safety_note.setStyleSheet("color: #a04000;")
        controls.addWidget(safety_note)

        main_layout.addWidget(control_group, 1)

        self.worker = CameraWorker()
        self.worker.frame_ready.connect(self.show_frame)
        self.worker.objects_ready.connect(self.set_objects)
        self.worker.status_ready.connect(self.set_status)
        self.worker.start()

    def show_frame(self, image):
        pixmap = QPixmap.fromImage(image)
        self.camera_label.setPixmap(
            pixmap.scaled(
                self.camera_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    def set_status(self, text):
        self.status_label.setText(text)

    def set_objects(self, objects):
        self.detected_objects = objects
        self.update_fruit_list()

    def update_fruit_list(self):
        previous = self.fruit_combo.currentData()

        self.fruit_combo.blockSignals(True)
        self.fruit_combo.clear()

        if not self.detected_objects:
            self.fruit_combo.addItem("No fruit detected", None)
        else:
            for index, obj in enumerate(self.detected_objects):
                label = (
                    f'{index + 1}. {obj["name"].title()} '
                    f'({obj["confidence"] * 100:.0f}%)'
                )
                self.fruit_combo.addItem(label, index)

            if isinstance(previous, int):
                if 0 <= previous < len(self.detected_objects):
                    self.fruit_combo.setCurrentIndex(previous)

        self.fruit_combo.blockSignals(False)
        self.on_fruit_selected()

    def on_fruit_selected(self, *_):
        index = self.fruit_combo.currentData()

        if isinstance(index, int) and 0 <= index < len(
            self.detected_objects
        ):
            obj = self.detected_objects[index]
            self.selected_object = obj

            self.selection_label.setText(
                f'Selected fruit: {obj["name"].title()}\n'
                f'Confidence: {obj["confidence"] * 100:.1f}%\n'
                f'Image box: {obj["box"]}'
            )

            self.workflow_label.setText(
                "Fruit selected. Choose a destination. "
                "Physical movement is not yet enabled."
            )
        else:
            self.selected_object = None
            self.selection_label.setText("Selected fruit: None")
            self.workflow_label.setText(
                "Point the camera at an apple, banana or orange."
            )

    def pick_place_clicked(self):
        # Intentionally no physical movement in this first version.
        QMessageBox.information(
            self,
            "Robot motion not enabled",
            "Fruit and destination selection are working as a UI "
            "prototype. Before picking, we must configure and verify "
            "camera-to-robot calibration, safe coordinates, and "
            "the JetArm gripper interface."
        )

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = FruitPickPlaceUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
