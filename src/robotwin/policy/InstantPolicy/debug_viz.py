"""
Debug visualization using Rerun for Instant Policy.

Saves point clouds, EE poses, and predicted actions to .rrd files for offline analysis.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional

try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False
    print("[DEBUG] Rerun not available, debug visualization disabled")


class PolicyDebugger:
    """
    Logs policy observations and actions to Rerun for debugging.
    
    Records:
    - Point clouds (live and demo)
    - End-effector poses
    - Predicted actions (relative transforms)
    - Gripper commands
    """
    
    def __init__(self, output_dir: str, task_name: str = "instant_policy"):
        self._output_dir = Path(output_dir)
        # Don't create directory yet - wait until start_episode or explicit setter
        self.task_name = task_name
        self.episode_count = 0
        self.recording_active = False
        self.current_seed = 0

    @property
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, new_dir: Path):
        new_dir = Path(new_dir)
        if self._output_dir != new_dir:
            print(f"[DEBUG RERUN] Changing output dir from {self._output_dir} to {new_dir}")
            self._output_dir = new_dir
            self._output_dir.mkdir(parents=True, exist_ok=True)
            # If we are already recording, we need to save to the new location
            if self.recording_active:
                print(f"[DEBUG RERUN] Restarting recording for new directory...")
                self.start_episode(self.current_seed, increment_episode=False)

    def start_episode(self, seed: int = 0, increment_episode: bool = True):
        """Start a new recording for this episode."""
        if not RERUN_AVAILABLE:
            return
            
        if increment_episode:
            self.episode_count += 1
        
        self.current_seed = seed
        self.step_count = 0
        
        # Save to .rrd file (force absolute path)
        rrd_path = self.output_dir.resolve() / f"{self.task_name}_episode{self.episode_count}_seed{seed}.rrd"
        
        # Ensure parent directory exists before saving
        if not rrd_path.parent.exists():
            print(f"[DEBUG RERUN] Creating directory: {rrd_path.parent}")
            rrd_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize rerun with unique recording ID
        rec_id = f"{self.task_name}_ep{self.episode_count}_{seed}"
        rr.init(rec_id, spawn=False)
        rr.save(str(rrd_path))
        
        self.recording_active = True
        print(f"[DEBUG RERUN] Recording started. ID: {rec_id}")
        print(f"[DEBUG RERUN] Saving to: {rrd_path}")
        if not rrd_path.parent.exists():
            print(f"[DEBUG RERUN] Creating directory: {rrd_path.parent}")
            rrd_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log_step(
        self,
        step: int,
        pcd_world: np.ndarray,
        T_w_e: np.ndarray,
        actions: Optional[np.ndarray] = None,
        grips: Optional[np.ndarray] = None,
        demo_pcd_ee: Optional[np.ndarray] = None,
        camera_debug_info: Optional[Dict] = None,
        robot_debug_info: Optional[Dict] = None,
    ):
        """Log a single step of policy execution."""
        if not RERUN_AVAILABLE or not self.recording_active:
            return
        
        rr.set_time(timeline="step", sequence=step)
        
        # Log live point cloud in world frame
        if pcd_world is not None and len(pcd_world) > 0:
            rr.log("world/pcd_live", rr.Points3D(pcd_world, colors=[0, 255, 0]))
        
        # Log EE pose as coordinate frame
        if T_w_e is not None:
            pos = T_w_e[:3, 3]
            rot_mat = T_w_e[:3, :3]
            rr.log(
                "world/ee_pose",
                rr.Transform3D(
                    translation=pos,
                    mat3x3=rot_mat,
                )
            )
            # Also log position as point for trajectory visualization
            rr.log("world/ee_trajectory", rr.Points3D([pos], colors=[255, 0, 0], radii=[0.01]))
        
        # Log predicted actions (relative transforms)
        if actions is not None and T_w_e is not None:
            # Actions are relative SE3 transforms from current EE
            # Visualize predicted trajectory
            action_positions = []
            T_current = T_w_e.copy()
            
            for i, action in enumerate(actions):
                if action.shape == (4, 4):
                    T_next = T_current @ action
                    action_positions.append(T_next[:3, 3])
                    T_current = T_next
            
            if action_positions:
                action_positions = np.array(action_positions)
                rr.log(
                    "world/predicted_trajectory",
                    rr.Points3D(action_positions, colors=[0, 0, 255], radii=[0.005])
                )
                # Connect with lines
                rr.log(
                    "world/predicted_path",
                    rr.LineStrips3D([action_positions], colors=[0, 0, 255])
                )
        
        # Log gripper state
        if grips is not None:
            grip_val = grips[0] if len(grips) > 0 else 0
            rr.log("gripper/command", rr.Scalars([grip_val]))
        
        # Log world origin for reference
        rr.log("world/origin", rr.Transform3D(translation=[0, 0, 0]))

        # Log camera debug info
        if camera_debug_info:
            for cam_name, info in camera_debug_info.items():
                if 'cam2world_gl' in info: # OpenGL frame
                     c2w = info['cam2world_gl']
                     rr.log(f"world/cameras/{cam_name}_gl", rr.Transform3D(
                         translation=c2w[:3, 3],
                         mat3x3=c2w[:3, :3],
                         axis_length=0.1
                     ))
                if 'extrinsic' in info: # RoboTwin extrinsic (usually W2C)
                    # If W2C, inverse is C2W
                    w2c = info['extrinsic']
                    c2w_cv = np.linalg.inv(np.vstack([w2c, [0,0,0,1]]))
                    rr.log(f"world/cameras/{cam_name}_cv", rr.Transform3D(
                        translation=c2w_cv[:3, 3],
                        mat3x3=c2w_cv[:3, :3],
                        axis_length=0.1
                    ))
                if 'intrinsic' in info:
                    K = info['intrinsic']
                    width = 640
                    height = 480
                    # Estimate resolution from principal point if needed, or use defaults
                    # if K is not None:
                    #     width = int(K[0, 2] * 2)
                    #     height = int(K[1, 2] * 2)
                    
                    # Log pinhole under the _cv frame (since we use extrinsic_cv)
                    if 'extrinsic' in info:
                         rr.log(
                            f"world/cameras/{cam_name}_cv/pinhole",
                            rr.Pinhole(
                                image_from_camera=K,
                                width=width,
                                height=height
                            )
                         )

        # Log robot EEF and gripper links only
        if robot_debug_info:
            for link_name, pose_mat in robot_debug_info.items():
                # Filter to only EEF and gripper-related links
                if any(kw in link_name.lower() for kw in ['eef', 'gripper', 'tcp', 'flange', 'tool']):
                    rr.log(f"world/robot/{link_name}", rr.Transform3D(
                        translation=pose_mat[:3, 3],
                        mat3x3=pose_mat[:3, :3],
                        axis_length=0.05
                    ))

        # Log demo point cloud (in EE frame, transform to world for visualization)
        if demo_pcd_ee is not None and len(demo_pcd_ee) > 0 and T_w_e is not None:
             demo_pcd_world = (T_w_e[:3, :3] @ demo_pcd_ee.T).T + T_w_e[:3, 3]
             rr.log("world/pcd_demo", rr.Points3D(demo_pcd_world, colors=[255, 255, 0]))
        
        self.step_count += 1
    
    def end_episode(self, success: bool = False):
        """End the current recording."""
        if not RERUN_AVAILABLE or not self.recording_active:
            return
        
        rr.log("episode/success", rr.Scalars([1.0 if success else 0.0]))
        self.recording_active = False
        print(f"[DEBUG] Episode {self.episode_count} ended ({'SUCCESS' if success else 'FAIL'})")


# Global debugger instance
_debugger: Optional[PolicyDebugger] = None


def init_debugger(output_dir: str, task_name: str = "instant_policy"):
    """Initialize the global debugger."""
    global _debugger
    _debugger = PolicyDebugger(output_dir, task_name)
    return _debugger


def get_debugger() -> Optional[PolicyDebugger]:
    """Get the global debugger instance."""
    return _debugger
