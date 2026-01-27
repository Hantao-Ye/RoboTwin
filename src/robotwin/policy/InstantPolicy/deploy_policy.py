"""
Instant Policy deployment for RoboTwin.

Implements the RoboTwin policy interface for BIP's Instant Policy model.
"""

from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot
from typing import Dict, Any, Optional, List

from .instant_policy_utils import (
    depth_to_pointcloud,
    depth_to_pointcloud_all,
    subsample_pcd,
    transform_pcd,
    pose_to_matrix,
    matrix_to_delta_ee,
    load_demo_from_hdf5,
    load_demo_from_pkl,
    find_demo_files,
)

# Global state for model wrapper
_model_state: Dict[str, Any] = {}


def encode_obs(observation: Dict) -> Dict:
    """
    Convert RoboTwin observation to point cloud format.

    Args:
        observation: RoboTwin observation dict with 'observation', 'endpose', etc.

    Returns:
        Dict with 'pcd_world', 'T_w_e', 'grip', 'arm_tag'
    """
    obs = {}
    
    # Get arm tag from model state (set during get_model)
    arm_tag = _model_state.get('arm_tag', 'right')
    obs['arm_tag'] = arm_tag
    
    # Get end-effector pose
    endpose = observation['endpose'][f'{arm_tag}_endpose']
    obs['T_w_e'] = pose_to_matrix(endpose)
    
    # Get gripper state (0=closed, >0.4=open)
    gripper_val = observation['endpose'][f'{arm_tag}_gripper']
    obs['grip'] = 0 if gripper_val > 0.2 else 1  # Binary: 0=open, 1=closed
    
    # Get target object IDs from live observation (set by task environment)
    # This is the critical fix: use runtime IDs, not empty demo metadata
    target_ids = observation.get('target_object_ids', [])
    if not target_ids:
        # Fallback to model state if observation doesn't have it
        target_ids = _model_state.get('target_object_ids', [])
    
    if not target_ids and _model_state.get('step_count', 0) == 0:
        print("[WARNING] No target_object_ids found! Using full point cloud (may include background/table).")
    
    # Debug log: print target_ids once per episode
    # if _model_state.get('step_count', 0) == 0:
    #     print(f"[DEBUG encode_obs] target_object_ids: {target_ids}")
    
    # Generate point cloud from cameras
    all_points = []
    cam_obs = observation.get('observation', {})
    
    for cam_name, cam_data in cam_obs.items():
        if not cam_name.startswith('cam_'):
            continue
        
        depth = cam_data.get('depth')
        mask = cam_data.get('actor_segmentation')
        intrinsic = cam_data.get('intrinsic_cv')
        extrinsic = cam_data.get('extrinsic_cv')
        if target_ids:
            points = depth_to_pointcloud(depth, mask, intrinsic, extrinsic, target_ids)
        else:
            # Fallback: use all valid depth pixels when no target_ids
            points = depth_to_pointcloud_all(depth, intrinsic, extrinsic)
        
        if len(points) > 0:
            all_points.append(points)
    
    if all_points:
        obs['pcd_world'] = np.concatenate(all_points, axis=0)
    else:
        obs['pcd_world'] = np.zeros((0, 3), dtype=np.float32)
    
    # Debug log: point cloud size
    # if _model_state.get('step_count', 0) == 0:
    #     print(f"[DEBUG encode_obs] pcd_world shape: {obs['pcd_world'].shape}")
    
    # Convert to EE frame for policy
    T_w_e = obs['T_w_e']
    obs['pcd_ee'] = transform_pcd(obs['pcd_world'], np.linalg.inv(T_w_e))
    
    return obs

def get_model(usr_args: Dict) -> Any:
    """
    Load and initialize the Instant Policy model.

    Args:
        usr_args: Configuration from deploy_policy.yml + eval.sh overrides

    Returns:
        Initialized model wrapper
    """
    global _model_state
    
    from ip.models.diffusion import GraphDiffusion
    from ip.utils.data_proc import save_sample, sample_to_cond_demo
    import yaml
    
    # Force debug mode for now to ensure Rerun saves
    usr_args['debug_mode'] = True
    
    checkpoint_path = Path(usr_args.get('checkpoint_path', '/home/hantao/playground/RoboTwin/checkpoints'))
    task_name = usr_args.get('task_name', '')
    num_demos = usr_args.get('num_demos', 2)
    num_waypoints = usr_args.get('num_waypoints', 10)
    num_diffusion_iters = usr_args.get('num_diffusion_iters', 8)
    device = usr_args.get('device', 'cuda')
    
    # Model config
    config = {
        'device': device,
        'num_demos': num_demos,
        'traj_horizon': num_waypoints,  # Required by AGI model
        'num_diffusion_iters_test': num_diffusion_iters,
        'num_diffusion_iters_train': 100,
        'num_layers': 2,
        'compile_models': False,
        'batch_size': 1,
        'pre_horizon': 8,
        'local_num_freq': 10,
        'local_nn_dim': 512,
        'hidden_dim': 1024,
        'num_scenes_nodes': 16,
        'pos_in_nodes': True,
        'scene_encoder_path': str(checkpoint_path / 'instant_policy_scene_encoder.pt'),
        'pre_trained_encoder': True,
        'freeze_encoder': True,
        'min_actions': torch.tensor([-0.01] * 3 + [-np.deg2rad(3)] * 3, dtype=torch.float32),
        'max_actions': torch.tensor([0.01] * 3 + [np.deg2rad(3)] * 3, dtype=torch.float32),
        # Training-related configs (not used in inference but required by GraphDiffusion)
        'record': False,
        'save_dir': '/tmp',
        'save_every': 1000,
        'randomise_num_demos': False,
        'use_lr_scheduler': False,
    }
    
    # Load model
    model_path = checkpoint_path / 'instant_policy_model.pt'
    model = GraphDiffusion.load_from_checkpoint(
        str(model_path),
        config=config,
        strict=False,
        map_location=device
    ).to(device)
    
    model.model.reinit_graphs(1, num_demos=max(num_demos, 1))
    model.eval()
    
    # Load demonstrations
    data_root = Path(__file__).parent.parent.parent.parent.parent / 'data'
    demo_files, demo_format = find_demo_files(task_name, str(data_root))
    
    demo_files, demo_format = find_demo_files(task_name, str(data_root))
    
    demos = []
    arm_tag = 'right'
    target_ids = []
    
    for i in range(min(num_demos, len(demo_files))):
        if demo_format == 'pkl':
            demo = load_demo_from_pkl(demo_files[i], num_waypoints=num_waypoints)
        else:  # hdf5
            demo = load_demo_from_hdf5(demo_files[i], num_waypoints=num_waypoints)
            # Get target IDs from first HDF5 demo
            if i == 0:
                import h5py
                with h5py.File(demo_files[i], 'r') as f:
                    target_ids = list(f.attrs.get('target_object_ids', []))
        
        demos.append(demo)
        arm_tag = demo.get('arm_tag', 'right')
    
    # Initialize debugger if enabled
    debugger = None
    debug_mode = usr_args.get('debug_mode', False)
    if debug_mode:
        from .debug_viz import PolicyDebugger
        debug_dir = usr_args.get('debug_dir', 'debug_viz')
        debugger = PolicyDebugger(debug_dir, task_name=task_name)
    
    # Store in global state
    _model_state = {
        'model': model,
        'config': config,
        'demos': demos,
        'arm_tag': arm_tag,
        'target_object_ids': target_ids,
        'num_waypoints': num_waypoints,
        'save_sample': save_sample,
        'sample_to_cond_demo': sample_to_cond_demo,
        'demo_embeddings': None,  # Computed on first step
        'step_count': 0,
        'debugger': debugger,
    }
    
    
    return _model_state


def eval(TASK_ENV, model: Dict, observation: Dict) -> None:
    """
    Main evaluation loop.

    Args:
        TASK_ENV: RoboTwin environment instance
        model: Model state dict from get_model
        observation: Current observation
    """
    obs = encode_obs(observation)
    
    arm_tag = obs['arm_tag']
    T_w_e = obs['T_w_e']
    grip = obs['grip']
    pcd_world = obs['pcd_world']
    pcd_ee_raw = obs['pcd_ee']
    
    config = model['config']
    ip_model = model['model']
    demos = model['demos']
    save_sample_fn = model['save_sample']
    
    # Prepare point cloud in EE frame (already computed in encode_obs)
    pcd_ee = subsample_pcd(pcd_ee_raw, 2048)
    
    # Debug info for cameras (gathered during encode_obs if possible, or reconstruct here)
    debugger = model.get('debugger')
    camera_debug_info = {}
    cam_obs = observation.get('observation', {})
    for cam_name, cam_data in cam_obs.items():
        if cam_name.startswith('cam_'):
             info = {}
             if 'extrinsic_cv' in cam_data: info['extrinsic'] = cam_data['extrinsic_cv']
             if 'intrinsic_cv' in cam_data: info['intrinsic'] = cam_data['intrinsic_cv']
             if info: camera_debug_info[cam_name] = info

    # Build sample structure

    # Build sample structure
    full_sample = {
        'demos': demos,
        'live': {
            'obs': [pcd_ee],
            'grips': [grip],
            'actions_grip': [np.zeros(config['pre_horizon'])],
            'T_w_es': [T_w_e],
            'actions': [np.eye(4).reshape(1, 4, 4).repeat(config['pre_horizon'], axis=0).astype(np.float32)],
        },
    }
    
    # Convert to torch geometric data
    data = save_sample_fn(full_sample, None)
    
    # Compute demo embeddings (cached after first step)
    device = config['device']
    if model['demo_embeddings'] is None:
        with torch.no_grad():
            demo_embds, demo_pos = ip_model.model.get_demo_scene_emb(data.to(device))
        model['demo_embeddings'] = (demo_embds, demo_pos)
    
    demo_embds, demo_pos = model['demo_embeddings']
    
    # Compute live scene embedding
    with torch.no_grad():
        data.live_scene_node_embds, data.live_scene_node_pos = \
            ip_model.model.get_live_scene_emb(data.to(device))
    
    data.demo_scene_node_embds = demo_embds.clone()
    data.demo_scene_node_pos = demo_pos.clone()
    
    # Run inference
    with torch.no_grad():
        with torch.autocast(dtype=torch.float32, device_type=device):
            actions, grips = ip_model.test_step(data.to(device), 0)
        
        actions = actions.squeeze().cpu().numpy()  # (T, 4, 4)
        grips = grips.squeeze().cpu().numpy()  # (T, 1)
    
    model['step_count'] += 1
    
    # Debug info for robot kinematics
    robot_debug_info = {}
    if hasattr(TASK_ENV, 'robot'):
        for link in TASK_ENV.robot.get_links():
             # Get global pose of link
             # Get global pose of link
             pose = link.get_pose()
             robot_debug_info[link.name] = pose_to_matrix(np.concatenate([pose.p, pose.q]))

    # Log to Rerun
    if debugger:
        debugger.log_step(
            step=_model_state['step_count'],
            pcd_world=pcd_world,
            T_w_e=T_w_e,
            actions=actions, 
            grips=grips,
            demo_pcd_ee=None,  # Disabled as per request (demo scene mismatches live scene)
            camera_debug_info=camera_debug_info,
            robot_debug_info=robot_debug_info
        )
    
    # Execute first action
    T_rel = actions[0]  # Relative SE3 transform
    grip_cmd = float(grips[0])  # -1=close, 1=open
    
    # Compute next absolute pose in world frame
    T_next = T_w_e @ T_rel
    
    # Convert to 7D pose [x, y, z, qw, qx, qy, qz]
    next_pos = T_next[:3, 3]
    next_rot = Rot.from_matrix(T_next[:3, :3])
    next_quat_scipy = next_rot.as_quat()  # [qx, qy, qz, qw]
    next_quat = np.array([next_quat_scipy[3], next_quat_scipy[0], 
                          next_quat_scipy[1], next_quat_scipy[2]])  # [qw, qx, qy, qz]
    
    next_pose = np.concatenate([next_pos, next_quat])
    
    # Gripper: BIP uses -1 (close) / 1 (open)
    # RoboTwin: 0 = closed, 1 = open
    grip_val = (grip_cmd + 1) / 2  # -1 -> 0, 1 -> 1
    
    # Build full action for bimanual robot
    # For 'ee' action_type: [left_pose (7) + left_gripper (1) + right_pose (7) + right_gripper (1)]
    if arm_tag == 'left':
        # Active left arm, stationary right arm
        right_pose = observation['endpose']['right_endpose']
        right_grip = observation['endpose']['right_gripper']
        full_action = np.concatenate([next_pose, [grip_val], right_pose, [right_grip]])
    else:
        # Stationary left arm, active right arm
        left_pose = observation['endpose']['left_endpose']
        left_grip = observation['endpose']['left_gripper']
        full_action = np.concatenate([left_pose, [left_grip], next_pose, [grip_val]])
    
    # Execute action with absolute end-effector pose
    TASK_ENV.take_action(full_action, action_type='ee')


def reset_model(model: Dict, seed: int = 0) -> None:
    """Reset model state between episodes."""
    model['demo_embeddings'] = None
    model['step_count'] = 0
    
    # Start new episode recording if debugger is enabled
    debugger = model.get('debugger')
    if debugger is not None:
        debugger.start_episode(seed=seed)

