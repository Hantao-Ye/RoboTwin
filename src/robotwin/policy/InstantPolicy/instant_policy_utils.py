"""
Instant Policy utilities for RoboTwin integration.

Provides point cloud generation, coordinate transforms, and demo loading.
"""

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from typing import List, Dict, Tuple, Optional
import h5py
from pathlib import Path
import cv2


def depth_to_pointcloud(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    target_ids: List[int],
    depth_scale: float = 1000.0,
) -> np.ndarray:
    """
    Convert depth image to world-frame point cloud, filtered by segmentation mask.

    Args:
        depth: (H, W) depth image in meters
        mask: (H, W) actor segmentation with object IDs
        intrinsic: (3, 3) camera intrinsic matrix
        extrinsic: (3, 4) camera extrinsic [R|t] (cam2world is inverse)
        target_ids: list of object IDs to include

    Returns:
        (N, 3) point cloud in world frame
    """
    H, W = depth.shape
    
    # Create mask for target objects
    object_mask = np.zeros_like(mask, dtype=bool)
    for obj_id in target_ids:
        object_mask |= (mask == obj_id)
    
    # Get valid depth pixels
    valid = (depth > 0) & object_mask
    
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)
    
    # Pixel coordinates
    v, u = np.where(valid)
    z = depth[valid] / depth_scale
    
    # Unproject to camera frame
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    points_cam = np.stack([x, y, z], axis=-1)  # (N, 3)
    
    points_cam = np.stack([x, y, z], axis=-1)  # (N, 3)
    
    # Standard extrinsic [R|t] (World-to-Camera)
    # p_world = R^T @ (p_cam - t)
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    
    R_inv = R.T
    t_inv = -R_inv @ t
    
    points_world = (R_inv @ points_cam.T).T + t_inv
    
    return points_world.astype(np.float32)


def depth_to_pointcloud_all(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    depth_scale: float = 1000.0,
) -> np.ndarray:
    """
    Convert depth image to world-frame point cloud, including ALL valid pixels.
    
    This is a fallback when target_ids is empty.
    
    Args:
        depth: (H, W) depth image in meters
        intrinsic: (3, 3) camera intrinsic matrix
        extrinsic: (3, 4) camera extrinsic [R|t]

    Returns:
        (N, 3) point cloud in world frame
    """
    # Get valid depth pixels (no segmentation filtering)
    valid = depth > 0
    
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)
    
    # Pixel coordinates
    v, u = np.where(valid)
    z = depth[valid] / depth_scale
    
    # Unproject to camera frame
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    points_cam = np.stack([x, y, z], axis=-1)  # (N, 3)
    
    points_cam = np.stack([x, y, z], axis=-1)  # (N, 3)
    
    # Transform to world frame (ignoring cam2world_gl)
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    
    R_inv = R.T
    t_inv = -R_inv @ t
    
    points_world = (R_inv @ points_cam.T).T + t_inv
    
    return points_world.astype(np.float32)


def subsample_pcd(points: np.ndarray, num_points: int = 2048) -> np.ndarray:
    """Subsample point cloud to fixed size."""
    if len(points) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    
    if len(points) < num_points:
        # Repeat points
        indices = np.random.choice(len(points), num_points, replace=True)
    else:
        indices = np.random.choice(len(points), num_points, replace=False)
    
    return points[indices]


def transform_pcd(pcd: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Transform point cloud by 4x4 matrix."""
    if len(pcd) == 0:
        return pcd
    return (T[:3, :3] @ pcd.T).T + T[:3, 3]


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """
    Convert 7D pose [x, y, z, qw, qx, qy, qz] to 4x4 matrix.
    """
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    # RoboTwin uses [qw, qx, qy, qz] order
    quat = pose[3:]  # [qw, qx, qy, qz]
    # scipy uses [qx, qy, qz, qw]
    T[:3, :3] = Rot.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    return T


def matrix_to_delta_ee(T_rel: np.ndarray, gripper: float) -> np.ndarray:
    """
    Convert relative SE3 transform to delta_ee action format.

    Args:
        T_rel: (4, 4) relative transform
        gripper: gripper state (-1=close, 1=open) -> normalized [0, 1]

    Returns:
        (8,) action [dx, dy, dz, dqw, dqx, dqy, dqz, gripper]
    """
    delta_pos = T_rel[:3, 3]
    quat_scipy = Rot.from_matrix(T_rel[:3, :3]).as_quat()  # [qx, qy, qz, qw]
    # Convert to [qw, qx, qy, qz]
    delta_quat = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
    
    # Gripper: BIP uses -1 (close) / 1 (open), RoboTwin expects normalized
    grip_val = (gripper + 1) / 2  # -1 -> 0, 1 -> 1
    
    return np.concatenate([delta_pos, delta_quat, [grip_val]])


def pose_error(T1: np.ndarray, T2: np.ndarray, rot_scale: float = 0.01) -> float:
    """Compute pose error between two transforms."""
    trans_err = np.linalg.norm(T1[:3, 3] - T2[:3, 3])
    rot_err = Rot.from_matrix(T1[:3, :3] @ T2[:3, :3].T).magnitude()
    return trans_err + rot_scale * rot_err


def extract_waypoints(
    T_w_es: List[np.ndarray],
    grips: List[float],
    num_waypoints: int,
) -> List[int]:
    """
    Extract waypoint indices from trajectory.
    Prioritizes gripper state changes and motion pauses.
    """
    n = len(T_w_es)
    if n <= num_waypoints:
        return list(range(n))
    
    waypoints = [0, n - 1]
    
    # Add gripper state change points
    for i in range(1, n - 1):
        if grips[i] != grips[i - 1] and i not in waypoints:
            waypoints.append(i)
    
    # Add motion pause points
    for i in range(1, n):
        err = pose_error(T_w_es[i], T_w_es[i - 1], rot_scale=1)
        closest_dist = min(abs(i - w) for w in waypoints)
        if err < 3.5e-3 and closest_dist > 5:
            waypoints.append(i)
    
    waypoints = sorted(set(waypoints))
    
    # Prune or fill to reach target count
    if len(waypoints) > num_waypoints:
        # Keep evenly spaced
        indices = np.linspace(0, len(waypoints) - 1, num_waypoints).astype(int)
        waypoints = [waypoints[i] for i in indices]
    elif len(waypoints) < num_waypoints:
        # Fill gaps
        while len(waypoints) < num_waypoints:
            # Find largest gap
            gaps = [(waypoints[i + 1] - waypoints[i], i) for i in range(len(waypoints) - 1)]
            gaps.sort(reverse=True)
            gap_size, gap_idx = gaps[0]
            if gap_size <= 1:
                break
            new_wp = (waypoints[gap_idx] + waypoints[gap_idx + 1]) // 2
            waypoints.insert(gap_idx + 1, new_wp)
    
    return sorted(waypoints)


def load_demo_from_hdf5(
    h5_path: str,
    num_waypoints: int = 10,
    num_points: int = 2048,
) -> Dict:
    """
    Load a demonstration from HDF5 file.

    Returns:
        {'pcds': [...], 'T_w_es': [...], 'grips': [...], 'arm_tag': str}
    """
    with h5py.File(h5_path, 'r') as f:
        arm_tag = f.attrs.get('arm_tag', 'right')
        if isinstance(arm_tag, bytes):
            arm_tag = arm_tag.decode('utf-8')
        
        target_ids = list(f.attrs.get('target_object_ids', []))
        
        # Get endpose for the active arm
        endpose_key = f'endpose/{arm_tag}_endpose'
        gripper_key = f'endpose/{arm_tag}_gripper'
        
        endposes = f[endpose_key][:]  # (T, 7)
        grippers = f[gripper_key][:]  # (T,)
        
        n_frames = len(endposes)
        
        # Convert endposes to 4x4 matrices
        T_w_es = [pose_to_matrix(endposes[i]) for i in range(n_frames)]
        
        # Convert gripper to binary (0=open, 1=closed) as BIP expects
        # RoboTwin gripper: 0 = closed, >0.4 = open
        grips = [0 if g > 0.2 else 1 for g in grippers]
        
        # Extract waypoint indices
        wp_indices = extract_waypoints(T_w_es, grips, num_waypoints)
        
        # Get camera names
        obs_group = f['observation']
        cam_names = [k for k in obs_group.keys() if k.startswith('cam_')]
        
        # Extract point clouds at waypoints
        pcds = []
        wp_T_w_es = []
        wp_grips = []
        
        for idx in wp_indices:
            # Merge point clouds from all cameras
            all_points = []
            
            for cam_name in cam_names:
                cam_group = obs_group[cam_name]
                
                depth = cam_group['depth'][idx]
                mask = cam_group['actor_segmentation'][idx]
                intrinsic = cam_group['intrinsic_cv'][idx]
                extrinsic = cam_group['extrinsic_cv'][idx]
                
                points = depth_to_pointcloud(depth, mask, intrinsic, extrinsic, target_ids)
                if len(points) > 0:
                    all_points.append(points)
            
            if all_points:
                merged = np.concatenate(all_points, axis=0)
            else:
                merged = np.zeros((0, 3), dtype=np.float32)
            
            # Subsample and transform to EE frame
            sampled = subsample_pcd(merged, num_points)
            pcd_ee = transform_pcd(sampled, np.linalg.inv(T_w_es[idx]))
            
            pcds.append(pcd_ee)
            wp_T_w_es.append(T_w_es[idx])
            wp_grips.append(grips[idx])
    
    return {
        'pcds': pcds,
        'T_w_es': wp_T_w_es,
        'grips': wp_grips,
        'obs': pcds,  # BIP uses 'obs' key
        'arm_tag': arm_tag,
    }


def load_demo_from_pkl(
    pkl_path: str,
    num_waypoints: int = 10,
    num_points: int = 2048,
) -> Dict:
    """
    Load a demonstration from pre-processed PKL file.
    
    PKL files are generated by BIP's preprepare_data.py and contain:
    - 'pcds': dict with object_id -> list of point clouds per frame (world frame)
    - 'poses': list of 4x4 matrices (EE poses)
    - 'grippers': list of gripper values
    
    Returns:
        {'pcds': [...], 'T_w_es': [...], 'grips': [...], 'arm_tag': str}
    """
    import pickle
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    poses = data.get('poses', [])
    grippers = data.get('grippers', [])
    pcds_dict = data.get('pcds', {})
    
    n_frames = len(poses)
    if n_frames == 0:
        raise ValueError(f"No poses found in {pkl_path}")
    
    # poses are already 4x4 matrices from preprepare_data.py
    T_w_es = poses
    
    # Convert gripper to binary (0=open, 1=closed) as BIP expects
    # RoboTwin gripper: 0 = closed, >0.4 = open
    grips = [0 if g > 0.2 else 1 for g in grippers]
    
    # Extract waypoint indices
    wp_indices = extract_waypoints(T_w_es, grips, num_waypoints)
    
    # Merge point clouds from all object IDs
    # pcds_dict is {object_id: [pcd_frame0, pcd_frame1, ...]}
    pcds_list = []
    wp_T_w_es = []
    wp_grips = []
    
    for idx in wp_indices:
        all_points = []
        for obj_id, obj_pcds in pcds_dict.items():
            if idx < len(obj_pcds) and obj_pcds[idx] is not None:
                pcd = obj_pcds[idx]
                # Point clouds in PKL are already in world frame
                # Only take XYZ (first 3 columns), ignore colors if present
                pcd_xyz = pcd[:, :3] if pcd.ndim == 2 and pcd.shape[1] >= 3 else pcd
                if len(pcd_xyz) > 0:
                    all_points.append(pcd_xyz)
        
        if all_points:
            merged = np.concatenate(all_points, axis=0)
        else:
            merged = np.zeros((0, 3), dtype=np.float32)
        
        # Subsample and transform to EE frame
        sampled = subsample_pcd(merged, num_points)
        pcd_ee = transform_pcd(sampled, np.linalg.inv(T_w_es[idx]))
        
        pcds_list.append(pcd_ee)
        wp_T_w_es.append(T_w_es[idx])
        wp_grips.append(grips[idx])
    
    # Infer arm_tag from filename/path (default to 'right')
    arm_tag = 'right'
    
    return {
        'pcds': pcds_list,
        'T_w_es': wp_T_w_es,
        'grips': wp_grips,
        'obs': pcds_list,  # BIP uses 'obs' key
        'arm_tag': arm_tag,
    }


def find_demo_files(task_name: str, data_root: str = 'data') -> Tuple[List[str], str]:
    """
    Find demo files for a task. Checks both HDF5 and PKL locations.
    
    Returns:
        Tuple of (list of file paths, format string 'hdf5' or 'pkl')
    """
    data_root = Path(data_root)
    
    # First check for pre-processed PKL files
    pkl_dir = data_root / 'processed_pkls' / task_name
    if pkl_dir.exists():
        pkl_files = sorted([str(p) for p in pkl_dir.glob('*_processed.pkl')])
        if pkl_files:
            return pkl_files, 'pkl'
    
    # Fall back to HDF5 files
    hdf5_dir = data_root / task_name / 'single_arm_data_gen' / 'data'
    if hdf5_dir.exists():
        hdf5_files = sorted([str(p) for p in hdf5_dir.glob('episode*.hdf5')])
        if hdf5_files:
            return hdf5_files, 'hdf5'
    
    return [], ''

