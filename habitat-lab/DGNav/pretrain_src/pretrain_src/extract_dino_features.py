#!/usr/bin/env python3
"""
Extract 36-view DINOv2 image features for pretraining datasets.

Each dataset key keeps the same naming as existing CLIP features:
    <scan_id>_<viewpoint_id> -> (36, feat_dim)
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import h5py
import numpy as np
import torch
from tqdm import tqdm


DINO_MODEL_CHOICES = [
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov2_vitl14",
    "dinov2_vitg14",
    "dinov2_vits14_reg",
    "dinov2_vitb14_reg",
    "dinov2_vitl14_reg",
    "dinov2_vitg14_reg",
]

# Local weight names available in this repository (EffoNav copy).
LOCAL_DINO_WEIGHT_NAMES = {
    "dinov2_vits14": "dinov2_vits14_pretrain.pth",
    "dinov2_vits14_reg": "dinov2_vits14_reg4_pretrain.pth",
}


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
        type=str,
        help="JSON config file; command line flags override config values.",
    )
    parser.add_argument(
        "--scanvp_cands_file",
        default="pretrain_src/datasets/R2R/annotations/scanvp_candview_relangles.json",
        type=str,
    )
    parser.add_argument(
        "--connectivity_dir",
        default="pretrain_src/datasets/R2R/connectivity",
        type=str,
    )
    parser.add_argument(
        "--scenes_dir",
        default="data/scene_datasets/mp3d",
        type=str,
        help="Root dir that contains <scan>/<scan>.glb.",
    )
    parser.add_argument(
        "--output_file",
        default="pretrain_src/img_features/DINOv2-ViT-S-14-views-habitat.hdf5",
        type=str,
    )
    parser.add_argument(
        "--dino_repo_path",
        default="vlnce_baselines/models/train/models/EffoNav/dinov2",
        type=str,
    )
    parser.add_argument(
        "--dino_model",
        default="dinov2_vits14",
        choices=DINO_MODEL_CHOICES,
        type=str,
    )
    parser.add_argument(
        "--dino_weights_path",
        default=None,
        type=str,
        help="Local pretrained weights path. If None, infer from model name when possible.",
    )
    parser.add_argument(
        "--use_hub_pretrained",
        action="store_true",
        help="Use torch.hub pretrained download when no local weights are provided.",
    )
    parser.add_argument("--image_size", default=224, type=int)
    parser.add_argument("--hfov", default=90.0, type=float)
    parser.add_argument("--batch_size", default=36, type=int)
    parser.add_argument("--sim_gpu_id", default=0, type=int)
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Feature extractor device: cuda or cpu.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max_viewpoints",
        default=-1,
        type=int,
        help="For debug: process at most N viewpoints. -1 means all.",
    )
    return parser


def parse_with_config(parser):
    args = parser.parse_args()
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {
            arg[2:].split("=")[0] for arg in sys.argv[1:] if arg.startswith("--")
        }
        for key, value in config_args.items():
            if (key not in override_keys) and hasattr(args, key):
                setattr(args, key, value)
    return args


def patch_torch_load_compat():
    # torch<=1.9 may miss `_rebuild_from_type_v2`.
    import torch._tensor as torch_tensor

    if (
        not hasattr(torch_tensor, "_rebuild_from_type_v2")
        and hasattr(torch_tensor, "_rebuild_from_type")
    ):
        torch_tensor._rebuild_from_type_v2 = torch_tensor._rebuild_from_type


def resolve_local_weights_path(dino_repo_path, dino_model, dino_weights_path):
    if dino_weights_path:
        return dino_weights_path
    if dino_model in LOCAL_DINO_WEIGHT_NAMES:
        return os.path.join(
            dino_repo_path, "weights", LOCAL_DINO_WEIGHT_NAMES[dino_model]
        )
    return None


def load_dino_model(args, device):
    repo_path = os.path.abspath(args.dino_repo_path)
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(f"DINO repo path not found: {repo_path}")

    weights_path = resolve_local_weights_path(
        repo_path, args.dino_model, args.dino_weights_path
    )
    use_hub_pretrained = args.use_hub_pretrained and (weights_path is None)

    model = torch.hub.load(
        repo_path,
        args.dino_model,
        source="local",
        pretrained=use_hub_pretrained,
    )

    if not use_hub_pretrained:
        if weights_path is None:
            raise ValueError(
                "No local weights found for the selected model. "
                "Set --dino_weights_path or enable --use_hub_pretrained."
            )
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"DINO weights file not found: {weights_path}")

        patch_torch_load_compat()
        state_dict = torch.load(weights_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if isinstance(state_dict, dict) and all(
            k.startswith("module.") for k in state_dict.keys()
        ):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)

    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad_(False)

    return model, (weights_path if weights_path is not None else "torch.hub pretrained")


def load_scanvp_keys(scanvp_cands_file):
    scanvp_cands = json.load(open(scanvp_cands_file))
    scan_to_vps = defaultdict(set)
    for key in scanvp_cands.keys():
        scan, vp = key.split("_", 1)
        scan_to_vps[scan].add(vp)
    return {scan: sorted(list(vps)) for scan, vps in scan_to_vps.items()}


def load_connectivity_positions(connectivity_dir):
    scans_file = os.path.join(connectivity_dir, "scans.txt")
    scans = [x.strip() for x in open(scans_file).readlines()]

    scan_to_positions = {}
    for scan in scans:
        connectivity_file = os.path.join(connectivity_dir, f"{scan}_connectivity.json")
        data = json.load(open(connectivity_file))
        positions = {}
        for item in data:
            if item["included"]:
                pose = item["pose"]
                positions[item["image_id"]] = np.array(
                    [pose[3], pose[7], pose[11]], dtype=np.float32
                )
        scan_to_positions[scan] = positions
    return scan_to_positions


def mp3d_to_habitat_position(mp_pos):
    # Matterport connectivity pose uses (x, y, z_up).
    # Habitat-Sim uses (x, y_up, z).
    return np.array([mp_pos[0], mp_pos[2], -mp_pos[1]], dtype=np.float32)


def view_index_to_rotation_quat(view_idx):
    from habitat_sim.utils.common import quat_from_angle_axis

    # Match Matterport 36-view indexing:
    # view_idx % 12: heading buckets, view_idx // 12: elevation {-30, 0, +30}.
    heading = -math.radians((view_idx % 12) * 30.0)
    elevation = math.radians((view_idx // 12 - 1) * 30.0)

    q_heading = quat_from_angle_axis(heading, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    q_elevation = quat_from_angle_axis(
        elevation, np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )
    return q_heading * q_elevation


def build_simulator(scene_path, image_size, hfov, sim_gpu_id):
    import habitat_sim

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    if hasattr(sim_cfg, "gpu_device_id"):
        sim_cfg.gpu_device_id = sim_gpu_id

    if hasattr(habitat_sim, "CameraSensorSpec"):
        sensor_spec = habitat_sim.CameraSensorSpec()
    else:
        sensor_spec = habitat_sim.SensorSpec()
    sensor_spec.uuid = "rgb"
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.resolution = [image_size, image_size]
    if hasattr(sensor_spec, "hfov"):
        sensor_spec.hfov = float(hfov)
    elif hasattr(sensor_spec, "parameters"):
        sensor_spec.parameters["hfov"] = str(hfov)
    sensor_spec.position = [0.0, 0.0, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_spec]

    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def render_36_views(sim, mp_pos):
    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = mp3d_to_habitat_position(mp_pos)

    rgb_views = []
    for view_idx in range(36):
        state.rotation = view_index_to_rotation_quat(view_idx)
        try:
            agent.set_state(state, reset_sensors=True)
        except TypeError:
            agent.set_state(state)

        obs = sim.get_sensor_observations()
        rgb = obs["rgb"][..., :3]
        rgb_views.append(rgb)

    return np.stack(rgb_views, axis=0)


def to_dino_input(batch_rgb, device):
    # batch_rgb: uint8 [B, H, W, 3]
    batch = torch.from_numpy(batch_rgb).to(device=device, dtype=torch.float32)
    batch = batch.permute(0, 3, 1, 2).contiguous() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    batch = (batch - mean) / std
    return batch


def encode_views(dino_model, rgb_views, device, batch_size):
    outputs = []
    with torch.no_grad():
        for start in range(0, rgb_views.shape[0], batch_size):
            end = min(start + batch_size, rgb_views.shape[0])
            model_input = to_dino_input(rgb_views[start:end], device)
            model_output = dino_model(model_input)

            if isinstance(model_output, dict):
                if "x_norm_clstoken" in model_output:
                    model_output = model_output["x_norm_clstoken"]
                else:
                    raise ValueError("Unexpected DINO output dict keys.")
            if isinstance(model_output, (list, tuple)):
                raise ValueError("Unexpected DINO output type: list/tuple.")

            outputs.append(model_output.float().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def main(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but --device=cuda is set.")

    dino_model, weights_used = load_dino_model(args, device)
    feature_dim = getattr(dino_model, "embed_dim", None)

    scan_to_vps = load_scanvp_keys(args.scanvp_cands_file)
    scan_to_positions = load_connectivity_positions(args.connectivity_dir)

    output_file = os.path.abspath(args.output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    all_scan_vps = []
    for scan in sorted(scan_to_vps.keys()):
        for vp in scan_to_vps[scan]:
            all_scan_vps.append((scan, vp))
    if args.max_viewpoints > 0:
        all_scan_vps = all_scan_vps[: args.max_viewpoints]

    print("=" * 72)
    print("[DINO Extractor] Start")
    print(f"[DINO Extractor] DINO model: {args.dino_model}")
    print(f"[DINO Extractor] DINO weights: {weights_used}")
    print(f"[DINO Extractor] scenes_dir: {os.path.abspath(args.scenes_dir)}")
    print(f"[DINO Extractor] output_file: {output_file}")
    print(f"[DINO Extractor] total viewpoints to process: {len(all_scan_vps)}")
    print("=" * 72)

    with h5py.File(output_file, "a") as f:
        f.attrs["feature_extractor"] = args.dino_model
        f.attrs["weights"] = str(weights_used)
        f.attrs["image_size"] = int(args.image_size)
        f.attrs["num_views"] = 36
        if feature_dim is not None:
            f.attrs["feature_dim"] = int(feature_dim)

        pbar = tqdm(total=len(all_scan_vps), desc="Extract DINO features")
        processed = 0

        current_scan = None
        sim = None
        for scan, vp in all_scan_vps:
            key = f"{scan}_{vp}"

            if (not args.overwrite) and (key in f):
                pbar.update(1)
                continue

            if scan != current_scan:
                if sim is not None:
                    sim.close()
                    sim = None

                scene_path = os.path.join(args.scenes_dir, scan, f"{scan}.glb")
                if not os.path.isfile(scene_path):
                    raise FileNotFoundError(f"Scene file not found: {scene_path}")
                sim = build_simulator(
                    scene_path=scene_path,
                    image_size=args.image_size,
                    hfov=args.hfov,
                    sim_gpu_id=args.sim_gpu_id,
                )
                current_scan = scan

            if scan not in scan_to_positions:
                raise KeyError(f"Scan not in connectivity set: {scan}")
            if vp not in scan_to_positions[scan]:
                raise KeyError(f"Viewpoint not in connectivity set: {scan}_{vp}")

            rgb_views = render_36_views(sim, scan_to_positions[scan][vp])
            dino_feats = encode_views(
                dino_model=dino_model,
                rgb_views=rgb_views,
                device=device,
                batch_size=args.batch_size,
            )

            if feature_dim is None:
                feature_dim = int(dino_feats.shape[-1])
                f.attrs["feature_dim"] = int(feature_dim)

            if key in f:
                del f[key]
            f.create_dataset(key, data=dino_feats, dtype=np.float32)

            processed += 1
            pbar.update(1)

        if sim is not None:
            sim.close()
        pbar.close()

    print("=" * 72)
    print("[DINO Extractor] Finished")
    print(f"[DINO Extractor] processed viewpoints: {processed}")
    print(f"[DINO Extractor] output_file: {output_file}")
    if feature_dim is not None:
        print(f"[DINO Extractor] feature_dim: {feature_dim}")
    print("=" * 72)


if __name__ == "__main__":
    parser = build_parser()
    args = parse_with_config(parser)
    main(args)
