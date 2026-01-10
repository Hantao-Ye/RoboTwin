import argparse
import os
import pickle
import subprocess
import tempfile

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# Config
CAM_NAMES = ["cam_left", "cam_right"]


def pose7d_to_matrix(pose_7d):
    """Transform [x, y, z, qw, qx, qy, qz] to 4x4 Homogenerous matrix."""
    t = pose_7d[:3]
    qw, qx, qy, qz = pose_7d[3], pose_7d[4], pose_7d[5], pose_7d[6]
    r = R.from_quat([qx, qy, qz, qw])
    T = np.eye(4)
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = t
    return T


def generate_masked_point_cloud(
    depth_frame,
    rgb_mask_frame,
    target_id,
    K,
    rgb_frame=None,
    T_cam_to_world=None,
    depth_scale=1.0,
):
    mask_boolean = rgb_mask_frame == target_id
    if np.sum(mask_boolean) == 0:
        return None

    h, w = depth_frame.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u_map, v_map = np.meshgrid(np.arange(w), np.arange(h))
    valid_indices = mask_boolean & (depth_frame > 0)

    z = depth_frame[valid_indices] / depth_scale
    u = u_map[valid_indices]
    v = v_map[valid_indices]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points_cam = np.stack((x, y, z), axis=-1)

    if T_cam_to_world is not None:
        points_cam_homo = np.hstack((points_cam, np.ones((points_cam.shape[0], 1))))
        points_world_homo = (T_cam_to_world @ points_cam_homo.T).T
        points_final = points_world_homo
    else:
        points_final = points_cam

    if rgb_frame is not None:
        colors = rgb_frame[valid_indices].astype(np.float32) / 255.0
        return np.hstack((points_final, colors))
    return points_final


def analyze_mask_colors(mask_frame, frame_idx=0, cam_name=""):
    unique_ids, counts = np.unique(mask_frame, return_counts=True)
    total_pixels = mask_frame.size
    print(f"\n[Analysis] Objects found in {cam_name} Frame {frame_idx}:")
    print(f"{'Actor ID':<10} | {'Count':<10} | {'Area %':<10}")
    print("-" * 40)
    for obj_id, count in zip(unique_ids, counts):
        pct = (count / total_pixels) * 100
        print(f"{obj_id:<10} | {count:<10} | {pct:>6.2f} %")
    print("-" * 40 + "\n")


def read_and_process_hdf5(file_path, TARGET_OBJECT_ID=None):
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None, None, None, None, None, {}

    with h5py.File(file_path, "r") as f:
        # Auto-detect target object ID
        if TARGET_OBJECT_ID is None and "target_object_ids" in f.attrs:
            t_ids = f.attrs["target_object_ids"]
            if len(t_ids) > 0:
                TARGET_OBJECT_ID = int(t_ids[0])
                print(
                    f"Auto-detected TARGET_OBJECT_ID from attributes: {TARGET_OBJECT_ID}"
                )

        # Read images
        image_data = {}
        for cam_name in CAM_NAMES:
            dataset_path = f"observation/{cam_name}/rgb"
            if dataset_path not in f:
                continue
            dataset = f[dataset_path]
            num_frames = dataset.shape[0]
            frame_list = []
            for i in range(num_frames):
                img_bgr = cv2.imdecode(
                    np.frombuffer(dataset[i], np.uint8), cv2.IMREAD_COLOR
                )
                frame_list.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            image_data[cam_name] = frame_list

        # Read arm state
        if "arm_tag" not in f.attrs:
            print("Error: Could not determine arm tag.")
            return None, None, None, None, None, {}

        arm_tag = f.attrs["arm_tag"]
        if isinstance(arm_tag, bytes):
            arm_tag = arm_tag.decode("utf-8")
        print(f"Detected Active Arm: {arm_tag}")

        prefix = "left" if "left" in arm_tag else "right"
        actions = {"endpose": [], "gripper": []}

        endpose_key = f"endpose/{prefix}_endpose"
        if endpose_key in f:
            actions["endpose"] = [pose7d_to_matrix(p) for p in f[endpose_key][:]]

        gripper_key = f"joint_action/{prefix}_gripper"
        if gripper_key in f:
            actions["gripper"] = f[gripper_key][:].tolist()

        # Point clouds
        camera_data_cache = {}
        depth_data_raw = {}
        num_frames = 0

        for cam_name in CAM_NAMES:
            keys = {
                "depth": f"observation/{cam_name}/depth",
                "mask": f"observation/{cam_name}/actor_segmentation",
                "K": f"observation/{cam_name}/intrinsic_cv",
                "extrinsic": f"observation/{cam_name}/extrinsic_cv",
            }
            if all(k in f for k in keys.values()):
                camera_data_cache[cam_name] = {k: f[v][:] for k, v in keys.items()}
                depth_data_raw[cam_name] = camera_data_cache[cam_name]["depth"]
                if num_frames == 0:
                    num_frames = camera_data_cache[cam_name]["depth"].shape[0]

        if TARGET_OBJECT_ID is None:
            print("TARGET_OBJECT_ID is None. Analyzing frames...")
            for cam_name in CAM_NAMES:
                if cam_name in camera_data_cache:
                    analyze_mask_colors(
                        camera_data_cache[cam_name]["mask"][0], 0, cam_name
                    )
            print("Please provide --target_id based on table above.")
            return (
                image_data,
                depth_data_raw,
                actions["endpose"],
                actions["gripper"],
                [],
                {},
            )

        print(
            f"Generating Point Clouds for {num_frames} frames (Object ID {TARGET_OBJECT_ID})..."
        )
        all_merged_pcds = []

        for t in range(num_frames):
            current_frame_pcds = []
            for cam_name in CAM_NAMES:
                if cam_name not in camera_data_cache:
                    continue
                data = camera_data_cache[cam_name]

                # Invert Extrinsic (W2C -> C2W)
                extrinsic_w2c = data["extrinsic"][t]  # 3x4
                T_w2c = np.eye(4)
                T_w2c[:3, :] = extrinsic_w2c
                T_c2w = np.linalg.inv(T_w2c)[:3, :]

                pcd = generate_masked_point_cloud(
                    data["depth"][t],
                    data["mask"][t],
                    TARGET_OBJECT_ID,
                    data["K"][t],
                    rgb_frame=image_data[cam_name][t],
                    T_cam_to_world=T_c2w,
                    depth_scale=1000.0,
                )
                if pcd is not None:
                    current_frame_pcds.append(pcd)

            all_merged_pcds.append(
                np.vstack(current_frame_pcds) if current_frame_pcds else None
            )
            if t % 50 == 0:
                print(f"Frame {t}: Merged {len(current_frame_pcds)} cameras.")

    masked_first_frames = {}
    if TARGET_OBJECT_ID is not None:
        for cam_name in CAM_NAMES:
            if cam_name in image_data and cam_name in camera_data_cache:
                rgb_curr = image_data[cam_name][0]
                mask_curr = camera_data_cache[cam_name]["mask"][0]
                # Apply mask
                mask_boolean = mask_curr == TARGET_OBJECT_ID
                masked_img = rgb_curr.copy()
                masked_img[~mask_boolean] = 0  # Set background to black
                masked_first_frames[cam_name] = masked_img

    return (
        image_data,
        depth_data_raw,
        actions["endpose"],
        actions["gripper"],
        all_merged_pcds,
        masked_first_frames,
    )


def save_video(frame_list, output_path, fps=30, is_depth=False):
    """Save video using system ffmpeg (robust H.264 encoding)."""
    if not frame_list:
        return
    h, w = frame_list[0].shape[:2]

    with tempfile.TemporaryDirectory() as temp_dir:
        frame_pattern = os.path.join(temp_dir, "frame_%04d.png")
        for i, frame in enumerate(frame_list):
            if is_depth:
                valid_mask = frame > 0
                if np.sum(valid_mask) > 0:
                    d_min, d_max = frame[valid_mask].min(), frame[valid_mask].max()
                    denom = d_max - d_min
                    if denom < 1e-5:
                        denom = 1e-5
                    norm = (frame - d_min) / denom * 255
                    save_frame = cv2.applyColorMap(
                        norm.astype(np.uint8), cv2.COLORMAP_JET
                    )
                    save_frame[~valid_mask] = 0
                else:
                    save_frame = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                save_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(frame_pattern % i, save_frame)

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            frame_pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
            print(f"Saved video: {output_path}")
        except Exception as e:
            print(f"FFmpeg failed for {output_path}: {e}")


def save_ply(points, filename, downsample=None):
    if points is None or len(points) == 0:
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    if points.shape[1] >= 6:
        pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6])

    if downsample is not None:
        pcd = pcd.voxel_down_sample(voxel_size=downsample)

    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved Point Cloud: {filename}")


def render_pcd_video(pcd_list, output_path, fps=30):
    valid_frames = [p for p in pcd_list if p is not None]
    if not valid_frames:
        return

    print("Rendering Point Cloud Video...")
    all_pts = np.vstack(valid_frames)
    min_b, max_b = all_pts[:, :3].min(axis=0), all_pts[:, :3].max(axis=0)
    center = (min_b + max_b) / 2
    span = (max_b - min_b).max() / 2

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    save_img_list = []

    for i, points in enumerate(pcd_list):
        ax.clear()
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(center[2] - span, center[2] + span)
        ax.set_title(f"Frame {i:03d}")
        ax.view_init(elev=30, azim=45)

        if points is not None and len(points) > 0:
            limit = 2000
            pts = (
                points
                if len(points) <= limit
                else points[np.random.choice(len(points), limit, replace=False)]
            )
            colors = pts[:, 3:6] if pts.shape[1] >= 6 else "b"
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5, c=colors, depthshade=True)

        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())
        save_img_list.append(
            cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        )  # Matplotlib gives RGBA, want RGB
        if i % 10 == 0:
            print(f"Rendered {i}/{len(pcd_list)}", end="\r")

    plt.close(fig)
    save_video(save_img_list, output_path, fps=fps)


def visualize_trajectory(poses, grippers, output_path):
    positions = np.array([T[:3, 3] for T in poses])
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], label="Path", color="b")
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c="g", label="Start")
    ax1.scatter(
        positions[-1, 0],
        positions[-1, 1],
        positions[-1, 2],
        c="r",
        marker="x",
        label="End",
    )
    ax1.set_title("3D Trajectory")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(grippers, color="orange")
    ax2.set_title("Gripper Width")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved Trajectory Plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Process RoboTwin HDF5 data.")
    parser.add_argument("file_path", help="Path to HDF5 file")
    parser.add_argument(
        "--target_id", type=int, default=None, help="Target Object ID (optional)"
    )
    parser.add_argument("--output_dir", default=None, help="Output directory")
    args = parser.parse_args()

    file_path = os.path.abspath(args.file_path)

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Default: .../data/episode0.hdf5 -> .../vis_result
        output_dir = os.path.join(os.path.dirname(file_path), "../vis_result")

    output_dir = os.path.normpath(output_dir)
    save_pkl_path = os.path.join(os.path.dirname(output_dir), "processed_data.pkl")

    # Run processing
    rgb, depth, poses, grippers, pcds, masked_first_frames = read_and_process_hdf5(file_path, args.target_id)

    if rgb and len(pcds) > 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save Pickle
        with open(save_pkl_path, "wb") as f:
            pickle.dump(
                {
                    "rgb": rgb,
                    "depth": depth,
                    "poses": poses,
                    "grippers": grippers,
                    "pcd_trajectory": pcds,
                },
                f,
            )

        # Visualize
        print(f"Saving results to {output_dir}")
        for cam, frames in rgb.items():
            save_video(frames, os.path.join(output_dir, f"{cam}_rgb.mp4"))
        for cam, frames in depth.items():
            save_video(
                list(frames),
                os.path.join(output_dir, f"{cam}_depth.mp4"),
                is_depth=True,
            )
        
        # Save masked first frames
        for cam, img in masked_first_frames.items():
            save_path = os.path.join(output_dir, f"{cam}_first_masked.png")
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"Saved masked first frame: {save_path}")

        if poses:
            visualize_trajectory(
                poses, grippers, os.path.join(output_dir, "trajectory_gripper.png")
            )

        first_pcd = next((p for p in pcds if p is not None), None)
        if first_pcd is not None:
            valid_pcds = [p for p in pcds if p is not None and len(p) > 0]
            if valid_pcds:
                all_points = np.vstack(valid_pcds)
                save_ply(all_points, os.path.join(output_dir, "combined_object.ply"), downsample=0.002)
                
            render_pcd_video(pcds, os.path.join(output_dir, "point_cloud_motion.mp4"))

            # Preview image
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            pts = first_pcd[
                np.random.choice(
                    len(first_pcd), min(2000, len(first_pcd)), replace=False
                )
            ]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                s=1,
                c=pts[:, 3:6] if pts.shape[1] >= 6 else "b",
            )
            plt.savefig(os.path.join(output_dir, "first_frame_preview.png"))
            plt.close()

    print("Done.")


if __name__ == "__main__":
    main()
