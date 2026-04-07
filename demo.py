import argparse
import os
import warnings

import numpy as np
import rerun as rr  # pip install rerun-sdk==0.21.0
import torch
from huggingface_hub import hf_hub_download

from mvtracker.utils.visualizer_rerun import log_pointclouds_to_rerun, log_tracks_to_rerun


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rerun",
        choices=["save", "spawn", "stream"],
        default="save",
        help=(
            "Whether to save recording to disk, spawn a new Rerun instance, or stream to an existing one. "
            "If 'spawn', make sure a rerun window can be spawned in your environment. "
            "If 'stream', make sure a rerun instance is running at port 9876. "
            "If 'save', the recording will be saved to a `.rrd` file that can be drag-and-dropped into "
            "a running rerun viewer, including the online viewer at https://app.rerun.io/version/0.21.0. "
            "For the online viewer, you want to create low memory-usage recordings with --lightweight."
        ),
    )
    p.add_argument(
        "--lightweight",
        action="store_true",
        help=(
            "Use lightweight rerun logging (less memory usage). This is recommended if you want to "
            "view the recording in the online Rerun viewer at https://app.rerun.io/version/0.21.0."
        ),
    )
    p.add_argument(
        "--random_query_points",
        action="store_true",
        help="Use random query points instead of demo ones.",
    )
    p.add_argument(
        "--cameras",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Which camera view indices to use (0-based). "
            "E.g. --cameras 0 2 to use only cameras 0 and 2. "
            "If not specified, all cameras in the data sample are used."
        ),
    )
    p.add_argument(
        "--repeat-frames",
        type=int,
        default=None,
        help=(
            "Number of frames to select from the start of the sequence. "
            "These frames will be tiled (repeated) to fill a longer sequence. "
            "E.g. --repeat-frames 100 --repeat-count 1000 tiles the first 100 frames 1000 times."
        ),
    )
    p.add_argument(
        "--repeat-count",
        type=int,
        default=1,
        help="How many times to repeat the selected frames (requires --repeat-frames).",
    )
    p.add_argument(
        "--rrd",
        default="mvtracker_demo.rrd",
        help=(
            "Path to save a .rrd file if `--rerun save` is used. "
            "Note that rerun prefers recordings to have a .rrd suffix."
        ),
    )
    args = p.parse_args()
    np.random.seed(72)
    torch.manual_seed(72)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load MVTracker predictor
    mvtracker = torch.hub.load("ethz-vlg/mvtracker", "mvtracker", pretrained=True, device=device)

    # Download demo sample from Hugging Face Hub
    sample_path = hf_hub_download(
        repo_id="ethz-vlg/mvtracker",
        filename="data_sample.npz",
        token=os.getenv("HF_TOKEN"),
        repo_type="model",
    )
    sample = np.load(sample_path)

    rgbs = torch.from_numpy(sample["rgbs"]).float()
    depths = torch.from_numpy(sample["depths"]).float()
    intrs = torch.from_numpy(sample["intrs"]).float()
    extrs = torch.from_numpy(sample["extrs"]).float()
    query_points = torch.from_numpy(sample["query_points"]).float()

    num_views = rgbs.shape[0]

    # Select camera subset if specified
    if args.cameras is not None:
        camera_indices = sorted(set(args.cameras))
        for idx in camera_indices:
            if idx < 0 or idx >= num_views:
                raise ValueError(
                    f"Camera index {idx} is out of range. "
                    f"Available cameras: 0..{num_views - 1} ({num_views} views)."
                )
        print(f"Using cameras {camera_indices} out of {num_views} available views.")
        cam_idx = torch.tensor(camera_indices, dtype=torch.long)
        rgbs = rgbs[cam_idx]
        depths = depths[cam_idx]
        intrs = intrs[cam_idx]
        extrs = extrs[cam_idx]
        num_views = len(camera_indices)
    else:
        camera_indices = list(range(num_views))
        print(f"Using all {num_views} cameras.")

    if args.repeat_frames is not None:
        num_frames_available = rgbs.shape[1]
        n = min(args.repeat_frames, num_frames_available)
        repeat_count = max(1, args.repeat_count)
        print(
            f"Selecting first {n} frames (of {num_frames_available}) and "
            f"repeating {repeat_count}x → {n * repeat_count} total frames."
        )
        # rgbs/depths shape: [V, T, ...], intrs/extrs shape: [V, T, ...]
        rgbs = rgbs[:, :n].repeat(1, repeat_count, *([1] * (rgbs.dim() - 2)))
        depths = depths[:, :n].repeat(1, repeat_count, *([1] * (depths.dim() - 2)))
        intrs = intrs[:, :n].repeat(1, repeat_count, *([1] * (intrs.dim() - 2)))
        extrs = extrs[:, :n].repeat(1, repeat_count, *([1] * (extrs.dim() - 2)))

    # pseudo_confs = torch.ones_like(depths) * 10
    # scale, translation = compute_auto_scene_normalization(depths, pseudo_confs, extrs, intrs)
    # rot = torch.eye(3, dtype=torch.float32)
    # depths, extrs, query_points, _, _ = transform_scene(
    #     transformation_scale=scale,
    #     transformation_rotation=rot,
    #     transformation_translation=translation,
    #     depth=depths,
    #     extrs=extrs,
    #     query_points=query_points,
    #     traj3d_world=None
    # )

    # Optionally, sample random queries in a cylinder of radius 12, height [-1, +10] and replace the demo queries
    if args.random_query_points:
        from mvtracker.models.core.model_utils import init_pointcloud_from_rgbd
        num_queries = 512
        t0 = 0
        xy_radius = 12.0
        z_min, z_max = -1.0, 10.0
        xyz, _ = init_pointcloud_from_rgbd(
            fmaps=rgbs[None],  # [1,V,T,1,H,W], uint8 0–255
            depths=depths[None],  # [1,V,T,1,H,W]
            intrs=intrs[None],  # [1,V,T,3,3]
            extrs=extrs[None],  # [1,V,T,3,4]
            stride=1,
            level=0,
        )
        pts = xyz[t0]  # [V*H*W, 3] at t=0
        assert pts.numel() > 0, "No valid depth points to sample queries from."

        r2 = pts[:, 0] ** 2 + pts[:, 1] ** 2
        mask = (r2 <= xy_radius ** 2) & (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
        pool = pts[mask]
        assert pool.shape[0] > 0, "Cylinder mask removed all points; increase radius or z-range."

        idx = torch.randperm(pool.shape[0])[:num_queries]
        pts = pool[idx]
        ts = torch.full((pts.shape[0], 1), float(t0), device=pts.device)
        query_points = torch.cat([ts, pts], dim=1).float()  # (N,4): (t,x,y,z)
        print(f"Sampled {pts.shape[0]} queries from depth at t={t0} within r<={xy_radius}, z∈[{z_min},{z_max}].")

    # Run prediction
    torch.set_float32_matmul_precision("high")
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device == "cuda", dtype=amp_dtype):
        results = mvtracker(
            rgbs=rgbs[None].to(device) / 255.0,
            depths=depths[None].to(device),
            intrs=intrs[None].to(device),
            extrs=extrs[None].to(device),
            query_points_3d=query_points[None].to(device),
        )
    pred_tracks = results["traj_e"].cpu()  # [T,N,3]
    pred_vis = results["vis_e"].cpu()  # [T,N]

    # Visualize results
    rr.init("3dpt", recording_id="v0.16")
    if args.rerun == "stream":
        rr.connect_tcp()
    elif args.rerun == "spawn":
        rr.spawn()
    log_pointclouds_to_rerun(
        dataset_name="demo",
        datapoint_idx=0,
        rgbs=rgbs[None],
        depths=depths[None],
        intrs=intrs[None],
        extrs=extrs[None],
        depths_conf=None,
        conf_thrs=[5.0],
        log_only_confident_pc=False,
        radii=-2.45,
        fps=12,
        bbox_crop=None,
        sphere_radius_crop=12.0,
        sphere_center_crop=np.array([0, 0, 0]),
        log_rgb_image=False,
        log_depthmap_as_image_v1=False,
        log_depthmap_as_image_v2=False,
        log_camera_frustrum=True,
        log_rgb_pointcloud=True,
    )
    log_tracks_to_rerun(
        dataset_name="demo",
        datapoint_idx=0,
        predictor_name="MVTracker",
        gt_trajectories_3d_worldspace=None,
        gt_visibilities_any_view=None,
        query_points_3d=query_points[None],
        pred_trajectories=pred_tracks,
        pred_visibilities=pred_vis,
        per_track_results=None,
        radii_scale=1.0,
        fps=12,
        sphere_radius_crop=12.0,
        sphere_center_crop=np.array([0, 0, 0]),
        log_per_interval_results=False,
        max_tracks_to_log=100 if args.lightweight else None,
        track_batch_size=50,
        method_id=None,
        color_per_method_id=None,
        memory_lightweight_logging=args.lightweight,
    )
    if args.rerun == "save":
        rr.save(args.rrd)
        print(f"Saved Rerun recording to: {os.path.abspath(args.rrd)}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*DtypeTensor constructors are no longer.*", module="pointops.query")
    warnings.filterwarnings("ignore", message=".*Plan failed with a cudnnException.*", module="torch.nn.modules.conv")
    main()
