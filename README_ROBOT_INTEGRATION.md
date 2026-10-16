# JetArm Fruit Pick-and-Place Integration

This project integrates the existing Hiwonder company `app.utils.pick_and_place`
functions with the fruit YOLO UI. Company scripts are not modified.

## Requirements

- ROS 2 Humble environment sourced.
- `~/ros2_ws/install/setup.bash` sourced, with the Hiwonder SDK packages available.
- SDK and robot launch processes already running, including `/kinematics/set_pose_target`.
- RGB camera and `/depth_cam/rgb/camera_info` available.
- Model file present at `/home/ubuntu/Downloads/yolo_1791978301.pt` (or update `MODEL_PATH` in `main.py`).
- Company calibration at `~/ros2_ws/src/app/config/transform.yaml` and `calibration.yaml`.

## Launch

```bash
cd ~/Desktop/JetArm-Pick-Place
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=5
export CHASSIS_TYPE=JetArm
export need_compile=False
source .venv/bin/activate
python3 -u main.py 2>&1 | tee -a logs/terminal.log
```

## UI workflow

1. Confirm the live camera and YOLO detections are visible.
2. Choose a fruit class and enter a destination name, then Add Fruit to Sequence.
3. Select the queue item, choose Set Destination Point, and click inside the calibrated work area on the camera image.
4. Save the sequence if desired. The sequence file stores destination pixels, not robot coordinates.
5. Keep every queued fruit visible and stationary, and ensure different queued items refer to distinct fruit instances.
6. Click Start Pick-and-Place and read the physical-movement confirmation carefully.
7. The app maps all points through the company calibration, rejects pixels outside the calibrated ROI, and checks IK pulses for pick/place targets before it starts the sequence.

## Important limits

- IK feasibility is not collision checking and does not verify gripper contact.
- The mapping uses the company's calibrated work plane. Fruit height/parallax, camera movement, workspace changes, and calibration drift can still cause position errors.
- The first physical test must be supervised, with the workspace clear and hardware emergency stop accessible. Keep hands clear. Do not use the app's Request Stop as an emergency stop: it requests a stop between company motion segments.
- Pickup yaw now calls the company `calculate_grasp_yaw` helper using YOLO bounding-box width/height with angle set to 0 (YOLO does not provide the rotated rectangle used by the company color contour detector). Validate banana grasp orientation on the first supervised run.
- The company `pick()` and `place()` functions may mutate input coordinates; the adapter always passes fresh lists.
- This integration has been syntax-checked in the build environment, but it has not been run against your physical JetArm here. Verify the first move under direct supervision.

## Source review update (2026-10-15)

Reviewed `main.py`, `robot/controller.py`, `robot/coordinates.py`, `robot/__init__.py`, `vision/detector.py`, `vision/test_camera_detection.py`, and `config/sequence.json`. Python syntax compilation passes for all Python files.

Change made: `robot/controller.py` now sets the IK service `resolution` to `0.01`, matching the fine Cartesian search step used by the Hiwonder service examples. The previous `1.0` value was unnecessarily coarse and could cause reachable targets to fail IK preflight.

No company files were changed. The physical motion functions are already connected in `main.py` through `JetArmController.pick()` and `JetArmController.place()`. However, physical sorting is not yet verified because the saved company `transform.yaml` has not been refreshed by a successful calibration; the earlier calibration logs show the tag detector timing out. Do not bypass the calibrated-workspace polygon check.
