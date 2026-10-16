from pathlib import Path
from typing import Any

from ultralytics import YOLO


class FruitDetector:
    """YOLO detector for apple, banana, and orange."""

    SUPPORTED_CLASSES = {"apple", "banana", "orange"}

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.50,
        image_size: int = 640,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.confidence = float(confidence)
        self.image_size = int(image_size)

        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("confidence must be greater than 0 and at most 1.")

        if self.image_size <= 0:
            raise ValueError("image_size must be greater than zero.")

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"YOLO model not found: {self.model_path}\n"
                "Provide the full path to your trained .pt model."
            )

        self.model = YOLO(str(self.model_path))
        self.class_names = self._normalise_class_names(self.model.names)

        available = set(self.class_names.values())
        self.supported_class_ids = {
            class_id
            for class_id, name in self.class_names.items()
            if name in self.SUPPORTED_CLASSES
        }

        if not self.supported_class_ids:
            raise ValueError(
                "The loaded model has no supported fruit classes.\n"
                f"Model classes: {sorted(available)}\n"
                f"Required classes: {sorted(self.SUPPORTED_CLASSES)}"
            )

        print(f"[FruitDetector] Model loaded: {self.model_path}")
        print(f"[FruitDetector] Model classes: {self.class_names}")
        print(
            "[FruitDetector] Supported classes: "
            f"{sorted(self.SUPPORTED_CLASSES.intersection(available))}"
        )

    @staticmethod
    def _normalise_class_names(names: Any) -> dict[int, str]:
        """Convert Ultralytics class names to {class_id: lowercase_name}."""

        if isinstance(names, dict):
            items = names.items()
        elif isinstance(names, (list, tuple)):
            items = enumerate(names)
        else:
            raise TypeError(
                "Unexpected model.names format. Expected a dictionary or list."
            )

        normalised = {}

        for class_id, name in items:
            try:
                numeric_id = int(class_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid model class ID: {class_id!r}"
                ) from exc

            normalised[numeric_id] = str(name).strip().lower()

        return normalised

    def detect(self, frame: Any) -> list[dict[str, Any]]:
        """
        Detect supported fruits in a BGR OpenCV image.

        Returns:
            A list of detections. Each detection contains:
            class_id, class_name, confidence, and bbox.

            bbox is [x1, y1, x2, y2] in image pixels.
        """

        if frame is None:
            raise ValueError("Input frame is None.")

        if not hasattr(frame, "shape") or len(frame.shape) != 3:
            raise ValueError("Expected a three-channel OpenCV image.")

        if frame.shape[0] <= 0 or frame.shape[1] <= 0:
            raise ValueError("Input image has invalid dimensions.")

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        )

        detections: list[dict[str, Any]] = []

        if not results:
            return detections

        result = results[0]
        boxes = result.boxes

        if boxes is None:
            return detections

        for box in boxes:
            class_id = int(box.cls.item())

            if class_id not in self.supported_class_ids:
                continue

            confidence = float(box.conf.item())
            coordinates = box.xyxy[0].tolist()

            x1, y1, x2, y2 = [
                int(round(float(value))) for value in coordinates
            ]

            height, width = frame.shape[:2]

            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": self.class_names[class_id],
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                }
            )

        return detections
