"""
Coordinate conversion using the existing Hiwonder company calibration.

Preserves the original Hiwonder world-coordinate conversion and calibration
corrections. Provides workspace projection diagnostics and rejects pixels
outside the calibrated polygon.

IMPORTANT:
- Uses the original RGB image coordinates.
- Does not modify company calibration files.
- Does not bypass workspace validation.
- Physical movement must only be enabled after verifying calibration.
"""

from pathlib import Path

import cv2
import numpy as np
import yaml

from sdk import common


DEFAULT_COMPANY_CONFIG = Path.home() / "ros2_ws/src/app/config"
DEFAULT_PLANE_SHIFT_M = 0.03


class CompanyCoordinateMapper:
    def __init__(self, config_dir=DEFAULT_COMPANY_CONFIG):
        self.config_dir = Path(config_dir).expanduser()
        self.transform_path = self.config_dir / "transform.yaml"
        self.calibration_path = self.config_dir / "calibration.yaml"

        if not self.transform_path.is_file():
            raise FileNotFoundError(
                f"Company transform not found: {self.transform_path}"
            )

        if not self.calibration_path.is_file():
            raise FileNotFoundError(
                f"Company calibration not found: {self.calibration_path}"
            )

        with self.transform_path.open("r", encoding="utf-8") as f:
            self.transform = yaml.safe_load(f)

        with self.calibration_path.open("r", encoding="utf-8") as f:
            self.calibration = yaml.safe_load(f)

        self._validate_calibration()

    def _validate_calibration(self):
        """Check that the required calibration entries exist and are finite."""
        required_transform = (
            "extristric",
            "corners",
            "white_area_pose_world",
        )

        for key in required_transform:
            if key not in self.transform:
                raise ValueError(
                    f"Missing '{key}' in {self.transform_path}"
                )

        for key in ("pixel", "kinematics"):
            if key not in self.calibration:
                raise ValueError(
                    f"Missing '{key}' in {self.calibration_path}"
                )

        extrinsic = np.asarray(
            self.transform["extristric"], dtype=float
        )
        corners = np.asarray(
            self.transform["corners"], dtype=float
        )
        white_area = np.asarray(
            self.transform["white_area_pose_world"], dtype=float
        )

        if extrinsic.shape != (4, 3):
            raise ValueError(
                f"Expected extristric shape (4, 3), got {extrinsic.shape}"
            )

        if corners.size < 12 or corners.size % 3 != 0:
            raise ValueError(
                "Invalid workspace corners in transform.yaml"
            )

        if white_area.shape != (4, 4):
            raise ValueError(
                "white_area_pose_world must be a 4x4 matrix"
            )

        if not (
            np.all(np.isfinite(extrinsic))
            and np.all(np.isfinite(corners))
            and np.all(np.isfinite(white_area))
        ):
            raise ValueError(
                "Calibration contains non-finite values"
            )

        for section in ("pixel", "kinematics"):
            scale = np.asarray(
                self.calibration[section]["scale"], dtype=float
            )
            offset = np.asarray(
                self.calibration[section]["offset"], dtype=float
            )

            if scale.shape != (3,) or offset.shape != (3,):
                raise ValueError(
                    f"{section} scale and offset must each have 3 values"
                )

            if not (
                np.all(np.isfinite(scale))
                and np.all(np.isfinite(offset))
            ):
                raise ValueError(
                    f"{section} calibration contains non-finite values"
                )

    @staticmethod
    def _camera_parameters(camera_matrix, distortion):
        K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        D = np.asarray(distortion, dtype=np.float64).reshape(-1)

        if not np.all(np.isfinite(K)) or not np.all(np.isfinite(D)):
            raise ValueError(
                "Camera matrix or distortion contains invalid values"
            )

        if abs(np.linalg.det(K)) < 1e-12:
            raise ValueError("Camera matrix is singular")

        return K, D

    def _project_workspace(
        self,
        camera_matrix,
        distortion,
        plane_shift_m=DEFAULT_PLANE_SHIFT_M,
    ):
        """Project the saved Hiwonder workspace corners into RGB pixels."""
        K, D = self._camera_parameters(camera_matrix, distortion)

        extrinsic = np.asarray(
            self.transform["extristric"], dtype=np.float64
        )

        tvec = extrinsic[:1]
        rmat = extrinsic[1:]

        shifted_tvec, shifted_rmat = common.extristric_plane_shift(
            tvec.reshape(3, 1).copy(),
            rmat.copy(),
            float(plane_shift_m),
        )

        corners = np.asarray(
            self.transform["corners"], dtype=np.float64
        ).reshape(-1, 3)

        # The final saved corner represents the workspace center.
        workspace_corners = corners[:-1]

        # Convert the 3x3 rotation matrix to OpenCV's rotation-vector format.
        rvec, _ = cv2.Rodrigues(
            np.asarray(shifted_rmat, dtype=np.float64)
        )

        projected, _ = cv2.projectPoints(
            workspace_corners,
            rvec,
            np.asarray(shifted_tvec, dtype=np.float64).reshape(3, 1),
            K,
            D,
        )

        polygon = np.rint(projected.reshape(-1, 2)).astype(np.int32)

        return polygon, shifted_tvec, shifted_rmat

    def projected_workspace(
        self,
        camera_matrix,
        distortion,
        plane_shift_m=0.0,
    ):
        """
        Return projected workspace corner pixels for diagnostics or drawing.

        This does not establish that the calibration is physically accurate.
        """
        polygon, _, _ = self._project_workspace(
            camera_matrix, distortion, plane_shift_m
        )
        return polygon.tolist()

    def ensure_pixel_in_calibrated_roi(
        self,
        pixel,
        camera_matrix,
        distortion,
        image_size=(640, 480),
    ):
        """
        Reject pixels outside the saved, projected company workspace polygon.

        Returns normally if the point is inside the polygon.
        Raises ValueError otherwise.
        """
        width, height = int(image_size[0]), int(image_size[1])

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid RGB image dimensions: {width}x{height}"
            )

        x, y = float(pixel[0]), float(pixel[1])

        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError(f"Invalid pixel coordinates: {pixel}")

        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(
                f"Pixel ({x:.1f}, {y:.1f}) is outside the RGB image "
                f"bounds {width}x{height}"
            )

        polygon = np.asarray(
            self.projected_workspace(
                camera_matrix, distortion, DEFAULT_PLANE_SHIFT_M
            ),
            dtype=np.int32,
        ).reshape(-1, 2)

        if len(polygon) < 3:
            raise ValueError(
                "The calibrated workspace has fewer than three corners"
            )

        inside = cv2.pointPolygonTest(
            polygon,
            (x, y),
            False,
        )

        if inside < 0:
            visible = [
                (int(px), int(py))
                for px, py in polygon
            ]

            raise ValueError(
                f"Pixel ({x:.1f}, {y:.1f}) is outside the saved company "
                f"workspace polygon. Projected corners: {visible}. "
                "Check the camera-to-workspace calibration before moving "
                "the robot."
            )

    def projection_matrix(self):
        """
        Return the camera projection matrix used by the company code.

        The matrix uses the saved extrinsic transform and the original
        0.03 m plane shift.
        """
        extrinsic = np.asarray(
            self.transform["extristric"], dtype=float
        )

        tvec = extrinsic[:1]
        rmat = extrinsic[1:]

        tvec, rmat = common.extristric_plane_shift(
            tvec.reshape(3, 1).copy(),
            rmat.copy(),
            DEFAULT_PLANE_SHIFT_M,
        )

        return np.row_stack(
            (
                np.column_stack((rmat, tvec)),
                np.array([[0.0, 0.0, 0.0, 1.0]]),
            )
        )

    def pixel_to_company_world(
        self,
        pixel,
        camera_matrix,
        height_m,
    ):
        """
        Convert an original RGB image pixel to the company's robot-world
        coordinates before the kinematics correction.

        The coordinate calculation mirrors the existing Hiwonder
        ObjectSortingNode.get_object_world_position() implementation.
        """
        x, y = float(pixel[0]), float(pixel[1])

        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError(f"Invalid pixel coordinates: {pixel}")

        K = np.matrix(camera_matrix, dtype=float)

        projection = self.projection_matrix()

        world_pose = np.asarray(
            common.pixels_to_world(
                [(x, y)],
                K,
                projection,
            )[0],
            dtype=float,
        ).reshape(-1)

        if world_pose.size < 3:
            raise ValueError(
                "Company pixel-to-world conversion returned fewer than "
                "three coordinates"
            )

        # Preserve the axis sign convention of the original company code.
        world_pose[0] = -world_pose[0]
        world_pose[1] = -world_pose[1]

        white_area = np.asarray(
            self.transform["white_area_pose_world"], dtype=float
        )

        position = white_area[:3, 3] + world_pose[:3]
        position[2] = float(height_m)

        pixel_cal = self.calibration["pixel"]

        scale = np.asarray(pixel_cal["scale"], dtype=float)
        offset = np.asarray(pixel_cal["offset"], dtype=float)

        position = position * scale + offset

        if not np.all(np.isfinite(position)):
            raise ValueError(
                f"Non-finite robot coordinate calculated for pixel "
                f"({x:.1f}, {y:.1f})"
            )

        return position.tolist()

    def pick_kinematics_position(self, world_position):
        """Apply the original company's pickup kinematics correction."""
        pos = np.asarray(world_position, dtype=float).reshape(3)

        cal = self.calibration["kinematics"]
        scale = np.asarray(cal["scale"], dtype=float)
        offset = np.asarray(cal["offset"], dtype=float)

        corrected = pos * scale + offset

        if not np.all(np.isfinite(corrected)):
            raise ValueError(
                "Pickup kinematics correction produced invalid coordinates"
            )

        return corrected.tolist()

    def place_kinematics_position(self, world_position):
        """Apply the company's angle-dependent placement correction."""
        pos = np.asarray(world_position, dtype=float).reshape(3)

        cal = self.calibration["kinematics"]
        offset = np.asarray(cal["offset"], dtype=float)
        scale = np.asarray(cal["scale"], dtype=float)

        angle = float(np.degrees(np.arctan2(pos[1], pos[0])))

        if angle > 45:
            corrected = [
                pos[0] * scale[1] - offset[1],
                pos[1] * scale[0] + offset[0],
                pos[2] * scale[2] + offset[2],
            ]
        elif angle < -45:
            corrected = [
                pos[0] * scale[1] + offset[1],
                pos[1] * scale[0] - offset[0],
                pos[2] * scale[2] + offset[2],
            ]
        else:
            corrected = [
                pos[0] * scale[0] + offset[0],
                pos[1] * scale[1] + offset[1],
                pos[2] * scale[2] + offset[2],
            ]

        if not np.all(np.isfinite(corrected)):
            raise ValueError(
                "Placement kinematics correction produced invalid coordinates"
            )

        return corrected
