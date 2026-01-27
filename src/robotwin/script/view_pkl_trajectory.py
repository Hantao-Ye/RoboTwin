import pickle
import numpy as np
try:
    import rerun as rr
except ImportError:
    print("Rerun not installed. converting to mock")
    class MockRR:
        def __getattr__(self, _): return lambda *a, **k: None
        ViewCoordinates = type('VC', (), {'RIGHT_HAND_Z_UP': None})
    rr = MockRR()

import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pkl_path', type=str)
    args = parser.parse_args()

    rr.init("pkl_viewer", spawn=True)

    print(f"Loading {args.pkl_path}...")
    with open(args.pkl_path, 'rb') as f:
        data = pickle.load(f)

    pcds_dict = data.get('pcds', {})
    poses = data.get('poses', [])
    grippers = data.get('grippers', [])
    camera_params = data.get('camera_params', {})
    
    # Log Coordinate Frame
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Log Cameras if available
    for cam_name, params in camera_params.items():
        K = params['K']
        T_w2c = params.get('T_w2c') # 4x4
        width = 640 # Assumption or need to find
        height = 480
        
        if T_w2c is not None:
             # Rerun uses Camera-to-World for Pinhole
             # params['T_c2w'] is available in preprepare_data
             T_c2w = params.get('T_c2w')
             if T_c2w is None:
                 T_c2w = np.linalg.inv(T_w2c)
                 
             rr.log(
                f"world/cameras/{cam_name}",
                rr.Transform3D(
                    translation=T_c2w[:3, 3],
                    mat3x3=T_c2w[:3, :3]
                ),
                static=True
            )
             rr.log(
                f"world/cameras/{cam_name}/pinhole",
                rr.Pinhole(
                    image_from_camera=K,
                    width=width,
                    height=height
                ),
                static=True
             )

    num_frames = len(poses)
    print(f"Visualizing {num_frames} frames...")
    
    for i in range(num_frames):
        rr.set_time(timeline="step", sequence=i)
        
        # Log EE pose
        if i < len(poses):
            T_w_e = poses[i] # 4x4 matrix
            rr.log(
                "world/ee",
                rr.Transform3D(
                    translation=T_w_e[:3, 3],
                    mat3x3=T_w_e[:3, :3]
                )
            )
        
        # Log gripper
        if i < len(grippers):
            rr.log("gripper", rr.Scalars([grippers[i]]))
        
        # Log Point Clouds
        all_points = []
        all_colors = []
        
        for obj_id, pcd_list in pcds_dict.items():
            if pcd_list and i < len(pcd_list) and pcd_list[i] is not None:
                pcd = pcd_list[i]
                pts = pcd[:, :3]
                cols = None
                if pcd.shape[1] >= 6:
                    cols = pcd[:, 3:6]
                
                all_points.append(pts)
                if cols is not None:
                    # Color is likely 0-1 float if preprepare_data divides by 255
                    all_colors.append(cols)
                    
        if all_points:
            merged_pts = np.vstack(all_points)
            merged_cols = np.vstack(all_colors) if all_colors else None
            
            rr.log(
                "world/pcd",
                rr.Points3D(
                    merged_pts,
                    colors=merged_cols,
                    radii=0.005
                )
            )
            
    print("Done logging.")

if __name__ == '__main__':
    main()
