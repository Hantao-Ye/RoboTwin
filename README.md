# Dataset Usage Guide

This guide explains how to configure the data collection process to include RGB, Depth, and Proprioception data, and how to read the collected HDF5 files for training your policy.

## 1. Configuration

To collect **RGB**, **Depth**, and **Proprioception** (Joint Positions/End Effector Pose), you need to modify the task configuration file (e.g., `src/robotwin/task_config/demo_clean.yml`).

Ensure the `data_type` section is configured as follows:

```yaml
data_type:
  rgb: true               # Enable RGB images
  depth: true             # Enable Depth maps (Set to true)
  pointcloud: false
  third_view: true        # Enable Third View (Observer Camera)
  observer: false         # (Legacy/Unused)
  endpose: true           # Enable End Effector Pose (Proprioception)
  qpos: true              # Enable Joint Positions (Proprioception)
  mesh_segmentation: false
  actor_segmentation: true # Enable Actor Segmentation (Object-level labels)
```

## 2. Data Structure

The collected data is saved in HDF5 format under `data/<task_name>/<config_name>/data/episodeX.hdf5`.

The HDF5 file has the following structure:

- **`observation/`**: Contains sensor data.
  - **`<camera_name>/`** (e.g., `cam_left`, `cam_right`, `head_camera`, `wrist_camera`)
    - **`rgb`**: JPEG encoded RGB images. Shape: `(T,)` (Byte strings).
      - **Note**: Images are compressed as JPEG bytes and padded to the maximum length in the sequence to allow storage as fixed-length HDF5 strings.
    - **`depth`**: Depth maps. Shape: `(T, H, W)` (Float32).
    - **`actor_segmentation`**: Actor segmentation masks. Shape: `(T, H, W, 3)` (RGB visualization) or `(T, H, W)` (Labels).
      - **Note**: If enabled, this provides pixel-wise labels where each color/value corresponds to a distinct physical object (Actor) in the scene.
    - **`intrinsic_cv`**: Camera intrinsics.
    - **`extrinsic_cv`**: Camera extrinsics.
- **`third_view_rgb`**: RGB images from the observer camera (if `third_view: true`). Shape: `(T, H, W, 3)`.
- **`joint_action/`**: Contains robot joint states (Proprioception).
  - **`vector`**: Concatenated joint positions (Left Arm + Left Gripper + Right Arm + Right Gripper). Shape: `(T, 14)`.
  - **`left_arm`**, **`right_arm`**: Individual arm joint positions.
  - **`left_gripper`**, **`right_gripper`**: Gripper states.
- **`endpose/`**: Contains end-effector poses (Proprioception).
  - **`left_endpose`**, **`right_endpose`**: 7D pose (Position + Quaternion).

### Metadata (Attributes)

The HDF5 file may contain metadata attributes at the root level:

- **`arm_tag`**: (Optional) Indicates which arm was used for the task (e.g., "left" or "right"). This is especially useful for single-arm tasks collected with a bimanual setup where one arm is hidden.

## 3. Reading the Data (Python Example)

Here is a Python script to load and inspect the data:

```python
import h5py
import numpy as np
import cv2
import os

# Path to a sample episode
file_path = "data/adjust_bottle/single_arm_data_gen/data/episode0.hdf5"

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit()

with h5py.File(file_path, "r") as f:
    # 1. Read RGB Images
    # RGB is stored as JPEG bytes (padded) to save space. You need to decode it.
    # Note: Camera names might be 'cam_left', 'cam_right' depending on config
    cam_name = "cam_left" if "observation/cam_left" in f else "head_camera"
    
    if f"observation/{cam_name}/rgb" in f:
        rgb_bytes = f[f"observation/{cam_name}/rgb"][0] # Get first frame
        # np.frombuffer converts the byte string to a uint8 array
        # cv2.imdecode decodes the JPEG data (ignoring padding)
        rgb_img = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
        print(f"RGB Image Shape: {rgb_img.shape}") # (H, W, 3)

    # 2. Read Depth Maps
    if f"observation/{cam_name}/depth" in f:
        depth_map = f[f"observation/{cam_name}/depth"][0] # Get first frame
        print(f"Depth Map Shape: {depth_map.shape}") # (H, W)
        # Note: Depth is usually stored as float meters or scaled integers.

    # 3. Read Actor Segmentation
    if f"observation/{cam_name}/actor_segmentation" in f:
        seg_mask = f[f"observation/{cam_name}/actor_segmentation"][0]
        print(f"Actor Segmentation Shape: {seg_mask.shape}")
        # This mask helps identify distinct objects (e.g., robot, bottle) in the scene.

    # 4. Read Proprioception (Joint Positions)
    if "joint_action/vector" in f:
        qpos = f["joint_action/vector"][:]
        print(f"Joint Action (qpos) Shape: {qpos.shape}") # (T, 14)
        print(f"First frame qpos: {qpos[0]}")
        # Format: [Left Arm (6), Left Gripper (1), Right Arm (6), Right Gripper (1)]

    # 5. Read Proprioception (End Effector Pose)
    if "endpose/left_endpose" in f:
        ee_pose = f["endpose/left_endpose"][:]
        print(f"Left End Effector Pose Shape: {ee_pose.shape}") # (T, 7)
        # Format: [x, y, z, qw, qx, qy, qz]

    # 6. Read Metadata
    if "arm_tag" in f.attrs:
        print(f"Arm Tag: {f.attrs['arm_tag']}")
```
