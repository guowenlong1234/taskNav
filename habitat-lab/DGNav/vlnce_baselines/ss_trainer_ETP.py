import gc
import os
import sys
import random
import warnings
from collections import defaultdict
from typing import Dict, List
import jsonlines

import lmdb
import msgpack_numpy
import numpy as np
import math
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel as DDP

import tqdm
from gym import Space
from habitat import Config, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs, construct_envs_for_rl, is_slurm_batch_job
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.models.graph_utils import GraphMap, MAX_DIST
from vlnce_baselines.utils import reduce_loss

from .utils import get_camera_orientations12
from .utils import (
    length2mask, dir_angle_feature_with_ele,
)
from vlnce_baselines.common.utils import dis_to_con, gather_list_and_concat
from habitat_extensions.measures import NDTW, StepsTaken
from fastdtw import fastdtw

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    # import tensorflow as tf  # noqa: F401

import torch.distributed as distr
import gzip
import json
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from vlnce_baselines.common.ops import pad_tensors_wgrad, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence
from yacs.config import CfgNode as CN


def _ensure_iterator_options(task_config):
    task_config.set_new_allowed(True)
    if "ENVIRONMENT" not in task_config:
        task_config.ENVIRONMENT = CN()
    if "ITERATOR_OPTIONS" not in task_config.ENVIRONMENT:
        task_config.ENVIRONMENT.ITERATOR_OPTIONS = CN()
    if "SHUFFLE" not in task_config.ENVIRONMENT.ITERATOR_OPTIONS:
        task_config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    if (
        "MAX_SCENE_REPEAT_STEPS"
        not in task_config.ENVIRONMENT.ITERATOR_OPTIONS
    ):
        task_config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    task_config.set_new_allowed(False)
    return task_config.ENVIRONMENT.ITERATOR_OPTIONS


def _append_measurement_once(task_config, measurement_name: str) -> None:
    task_config.set_new_allowed(True)
    if "TASK" not in task_config:
        task_config.TASK = CN()
    if "MEASUREMENTS" not in task_config.TASK:
        task_config.TASK.MEASUREMENTS = []

    measurements = list(task_config.TASK.MEASUREMENTS)
    if measurement_name not in measurements:
        measurements.append(measurement_name)
        task_config.TASK.MEASUREMENTS = measurements
    task_config.set_new_allowed(False)


def _get_collision_rate(info: Dict, path_len: int):
    denom = max(int(path_len), 1)
    if not isinstance(info, dict):
        return 0.0, False

    if "collisions" in info:
        collisions = info["collisions"]
        if isinstance(collisions, dict):
            if "count" in collisions:
                return float(collisions["count"]) / denom, True
            if "is_collision" in collisions:
                return float(bool(collisions["is_collision"])) / denom, True
        elif isinstance(collisions, (list, tuple, np.ndarray)):
            return float(np.asarray(collisions).astype(np.float32).sum()) / denom, True
        else:
            try:
                return float(collisions) / denom, True
            except (TypeError, ValueError):
                return 0.0, False

    if "collisions.is_collision" in info:
        col_flag = info["collisions.is_collision"]
        if isinstance(col_flag, (list, tuple, np.ndarray)):
            return float(np.asarray(col_flag).astype(np.float32).sum()) / denom, True
        return float(bool(col_flag)) / denom, True

    return 0.0, False


@baseline_registry.register_trainer(name="SS-ETP")
class RLTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.max_len = int(config.IL.max_traj_len) #  * 0.97 transfered gt path got 0.96 spl
        # Used to accumulate dynamic graph weight statistics (every 200 iterations)
        self.dynamic_graph_weight_history = {}  # {layer_idx: {'w1': [], 'w2': [], 'w3': []}}
        # Dedicated rollout timing log (lightweight, train-only).
        self._perf_timing_fh = None
        self._perf_timing_path = None
        self._train_rollout_counter = 0
        self._warned_missing_collisions = False

    def _init_perf_timing_log(self):
        if self.local_rank != 0:
            return
        perf_dir = os.path.join(self.config.CHECKPOINT_FOLDER, "perf_timing")
        os.makedirs(perf_dir, exist_ok=True)
        exp_name = getattr(self.config, "EXP_NAME", "exp")
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._perf_timing_path = os.path.join(
            perf_dir, f"{exp_name}_train_rollout_timing_rank{self.local_rank}_{ts}.log"
        )
        self._perf_timing_fh = open(
            self._perf_timing_path, "a", encoding="utf-8", buffering=1
        )
        self._train_rollout_counter = 0
        self._perf_timing_fh.write(
            "timestamp,rollout_id,steps,env_instances_avg,total_actions,"
            "waypoint_s,env_call_at_s,navigation_s,env_step_s,"
            "tracked_total_s,rollout_total_s,"
            "waypoint_pct,env_call_at_pct,navigation_pct,env_step_pct,"
            "call_at_requests\n"
        )
        logger.info(f"Perf timing log file: {self._perf_timing_path}")

    def _close_perf_timing_log(self):
        if self._perf_timing_fh is not None:
            self._perf_timing_fh.close()
            self._perf_timing_fh = None

    def _write_perf_timing_log(self, timing_info):
        if self.local_rank != 0 or self._perf_timing_fh is None:
            return
        tracked_total = (
            timing_info["waypoint"]
            + timing_info["env_call_at"]
            + timing_info["navigation"]
            + timing_info["env_step"]
        )
        denom = tracked_total if tracked_total > 1e-8 else 1.0
        line = (
            f"{time.strftime('%Y-%m-%d %H:%M:%S')},"
            f"{timing_info['rollout_id']},"
            f"{timing_info['steps']},"
            f"{timing_info['env_instances_avg']:.3f},"
            f"{timing_info['total_actions']},"
            f"{timing_info['waypoint']:.6f},"
            f"{timing_info['env_call_at']:.6f},"
            f"{timing_info['navigation']:.6f},"
            f"{timing_info['env_step']:.6f},"
            f"{tracked_total:.6f},"
            f"{timing_info['rollout_total']:.6f},"
            f"{timing_info['waypoint'] / denom * 100.0:.2f},"
            f"{timing_info['env_call_at'] / denom * 100.0:.2f},"
            f"{timing_info['navigation'] / denom * 100.0:.2f},"
            f"{timing_info['env_step'] / denom * 100.0:.2f},"
            f"{timing_info['env_call_at_requests']}\n"
        )
        self._perf_timing_fh.write(line)

    def _make_dirs(self):
        if self.config.local_rank == 0:
            os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)
            if self.config.EVAL.SAVE_RESULTS:
                os.makedirs(self.config.RESULTS_DIR, exist_ok=True)

    def save_checkpoint(self, iteration: int):
        checkpoint_dict = {
            "state_dict": self.policy.state_dict(),
            "config": self.config,
            "iteration": iteration,
        }
        # Only save optimizer state if optimizer exists (in training mode)
        if hasattr(self, 'optimizer'):
            checkpoint_dict["optim_state"] = self.optimizer.state_dict()
        torch.save(
            obj=checkpoint_dict,
            f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.iter{iteration}.pth"),
        )

    def _record_dynamic_graph_weights(self):
        """
        Record current dynamic graph weight values (for statistics)
        """
        try:
            model = self.policy.net
            if hasattr(model, 'vln_bert') and hasattr(model.vln_bert, 'global_encoder'):
                global_encoder = model.vln_bert.global_encoder
                if hasattr(global_encoder, 'encoder') and hasattr(global_encoder.encoder, 'x_layers'):
                    x_layers = global_encoder.encoder.x_layers
                    
                    for layer_idx, layer in enumerate(x_layers):
                        # Record dynamic graph weights
                        if hasattr(layer, 'use_dynamic_graph') and layer.use_dynamic_graph:
                            if layer_idx not in self.dynamic_graph_weight_history:
                                self.dynamic_graph_weight_history[layer_idx] = {
                                    'w1': [], 'w2': [], 'w3': []
                                }
                            
                            if hasattr(layer, 'w1'):
                                self.dynamic_graph_weight_history[layer_idx]['w1'].append(float(layer.w1.item()))
                            if hasattr(layer, 'w2'):
                                self.dynamic_graph_weight_history[layer_idx]['w2'].append(float(layer.w2.item()))
                            if hasattr(layer, 'w3'):
                                self.dynamic_graph_weight_history[layer_idx]['w3'].append(float(layer.w3.item()))
        except Exception as e:
            # Fail silently, do not affect training
            pass

    def save_dynamic_graph_weights(self, iteration: int):
        """
        Save dynamic graph weight information (w1, w2, w3) to JSON file
        Save every 200 iterations, including:
        1. Weight values for each of the 200 iterations (complete history)
        2. Statistics (max, min, mean, std)
        """
        try:
            # Get model
            model = self.policy.net
            if hasattr(model, 'vln_bert') and hasattr(model.vln_bert, 'global_encoder'):
                global_encoder = model.vln_bert.global_encoder
                if hasattr(global_encoder, 'encoder') and hasattr(global_encoder.encoder, 'x_layers'):
                    x_layers = global_encoder.encoder.x_layers
                    
                    # Collect weight information for all layers
                    weights_data = {
                        'iteration': iteration,
                        'layers': []
                    }
                    
                    for layer_idx, layer in enumerate(x_layers):
                        layer_data = {
                            'layer_index': layer_idx,
                            'weights': {}
                        }
                        
                        # Process dynamic graph weights
                        if hasattr(layer, 'use_dynamic_graph') and layer.use_dynamic_graph:
                            # Current values
                            current_w1 = float(layer.w1.item()) if hasattr(layer, 'w1') else None
                            current_w2 = float(layer.w2.item()) if hasattr(layer, 'w2') else None
                            current_w3 = float(layer.w3.item()) if hasattr(layer, 'w3') else None
                            
                            # Prepare weight data
                            weights_info = {}
                            
                            if layer_idx in self.dynamic_graph_weight_history:
                                hist = self.dynamic_graph_weight_history[layer_idx]
                                
                                for weight_name in ['w1', 'w2', 'w3']:
                                    if hist[weight_name]:
                                        values = hist[weight_name]  # Keep all historical values
                                        current_val = current_w1 if weight_name == 'w1' else (current_w2 if weight_name == 'w2' else current_w3)
                                        
                                        # Calculate statistics
                                        values_array = np.array(values)
                                        weights_info[weight_name] = {
                                            'history': values,  # Save all historical values
                                            'current': current_val,
                                            'max': float(np.max(values_array)),
                                            'min': float(np.min(values_array)),
                                            'mean': float(np.mean(values_array)),
                                            'std': float(np.std(values_array)),
                                            'count': len(values)
                                        }
                                    else:
                                        # If no historical data, only save current value
                                        current_val = current_w1 if weight_name == 'w1' else (current_w2 if weight_name == 'w2' else current_w3)
                                        if current_val is not None:
                                            weights_info[weight_name] = {
                                                'history': [current_val],  # At least save current value
                                                'current': current_val,
                                                'max': current_val,
                                                'min': current_val,
                                                'mean': current_val,
                                                'std': 0.0,
                                                'count': 1
                                            }
                            else:
                                # If no historical data, only save current value
                                for weight_name, current_val in [('w1', current_w1), ('w2', current_w2), ('w3', current_w3)]:
                                    if current_val is not None:
                                        weights_info[weight_name] = {
                                            'history': [current_val],  # At least save current value
                                            'current': current_val,
                                            'max': current_val,
                                            'min': current_val,
                                            'mean': current_val,
                                            'std': 0.0,
                                            'count': 1
                                        }
                            
                            layer_data['weights'] = weights_info
                        
                        # Only add to results if layer has dynamic graph weights
                        if layer_data['weights']:
                            weights_data['layers'].append(layer_data)
                    
                    # Save to JSON file
                    weights_dir = os.path.join(self.config.CHECKPOINT_FOLDER, 'dynamic_graph_weights')
                    os.makedirs(weights_dir, exist_ok=True)
                    weights_file = os.path.join(weights_dir, f'weights_iter{iteration}.json')
                    
                    with open(weights_file, 'w', encoding='utf-8') as f:
                        json.dump(weights_data, f, indent=2, ensure_ascii=False)
                    
                    # Clear history for next 200 iterations
                    self.dynamic_graph_weight_history.clear()
                    
                    # logger.info(f'Saved dynamic graph weights to {weights_file}')
        except Exception as e:
            logger.warning(f'Failed to save dynamic graph weights: {e}')

    def _set_config(self):
        self.split = self.config.TASK_CONFIG.DATASET.SPLIT
        self.config.defrost()
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.split
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.split
        iterator_options = _ensure_iterator_options(self.config.TASK_CONFIG)
        iterator_options.MAX_SCENE_REPEAT_STEPS = -1
        self.config.SIMULATOR_GPU_IDS = self.config.SIMULATOR_GPU_IDS[self.config.local_rank]
        self.config.use_pbar = not is_slurm_batch_job()
        ''' if choosing image '''
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],            # Back
                'Down': [-math.pi / 2, 0 + shift, 0],       # Down
                'Front':[0, 0 + shift, 0],                  # Front
                'Right':[0, math.pi / 2 + shift, 0],        # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
                'Up':   [math.pi / 2, 0 + shift, 0],        # Up
            }
            sensor_uuids = []
            H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    camera_config.WIDTH = H
                    camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        self.batch_size = self.config.IL.batch_size
        torch.cuda.set_device(self.device)
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
            torch.cuda.set_device(self.device)

    def _init_envs(self):
        # for DDP to load different data
        self.config.defrost()
        self.config.TASK_CONFIG.SEED = self.config.TASK_CONFIG.SEED + self.local_rank
        self.config.freeze()

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False
        )
        env_num = self.envs.num_envs
        dataset_len = sum(self.envs.number_of_episodes)
        logger.info(f'LOCAL RANK: {self.local_rank}, ENV NUM: {env_num}, DATASET LEN: {dataset_len}')
        observation_space = self.envs.observation_spaces[0]
        action_space = self.envs.action_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        return observation_space, action_space

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
        create_optimizer: bool = True,
    ):
        #如果检查点中有三层mlp权重，优先从检查点中加载三层权重
        start_iter = 0
        projector_primary_ckpt = ""
        if load_from_ckpt:
            if getattr(config.IL, "is_requeue", False):
                import glob

                ckpt_list = list(
                    filter(
                        os.path.isfile,
                        glob.glob(os.path.join(config.CHECKPOINT_FOLDER, "*")),
                    )
                )
                if len(ckpt_list) > 0:
                    ckpt_list.sort(key=os.path.getmtime)
                    projector_primary_ckpt = ckpt_list[-1]
            else:
                projector_primary_ckpt = getattr(config.IL, "ckpt_to_load", "")

        config.defrost()
        config.MODEL.projector_ckpt_path = projector_primary_ckpt
        config.freeze()

        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        ''' initialize the waypoint predictor here '''
        from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
        self.waypoint_predictor = BinaryDistPredictor_TRM(device=self.device)
        cwp_fn = 'data/wp_pred/check_cwp_bestdist_hfov63' if self.config.MODEL.task_type == 'rxr' else 'data/wp_pred/check_cwp_bestdist_hfov90'
        self.waypoint_predictor.load_state_dict(torch.load(cwp_fn, map_location = torch.device('cpu'))['predictor']['state_dict'])
        for param in self.waypoint_predictor.parameters():
            param.requires_grad_(False)

        self.policy.to(self.device)
        self.waypoint_predictor.to(self.device)
        self.num_recurrent_layers = self.policy.net.num_recurrent_layers

        if self.config.GPU_NUMBERS > 1:
            print('Using', self.config.GPU_NUMBERS,'GPU!')
            # find_unused_parameters=False fix ddp bug
            self.policy.net = DDP(self.policy.net.to(self.device), device_ids=[self.device],
                output_device=self.device, find_unused_parameters=False, broadcast_buffers=False)
        
        # Setup optimizer, support separate learning rates for dynamic graph and node gating (only created in training mode)
        if create_optimizer:
            use_dynamic_graph = getattr(self.config.MODEL, 'use_dynamic_graph', False)
            use_node_gating = getattr(self.config.MODEL, 'use_node_gating', False)
            
            if use_dynamic_graph or use_node_gating:
                # Separate different parameter groups
                dynamic_graph_params = []
                node_gating_params = []
                other_params = []
                
                for name, param in self.policy.named_parameters():
                    if use_dynamic_graph and ('w1' in name or 'w2' in name or 'w3' in name or 
                                             'semantic_sim_mlp' in name or 'instruction_rel_mlp' in name):
                        dynamic_graph_params.append(param)
                    elif use_node_gating and 'node_gating_mlp' in name:
                        node_gating_params.append(param)
                    else:
                        other_params.append(param)
                
                # Create parameter groups
                param_groups = [
                    {'params': other_params, 'lr': self.config.IL.lr},
                ]
                
                if use_dynamic_graph and dynamic_graph_params:
                    dynamic_graph_lr = getattr(self.config.MODEL, 'dynamic_graph_lr', self.config.IL.lr)
                    param_groups.append({
                        'params': dynamic_graph_params,
                        'lr': dynamic_graph_lr
                    })
                    logger.info(f'Using separate learning rate {dynamic_graph_lr} for dynamic graph parameters')
                
                if use_node_gating and node_gating_params:
                    node_gating_lr = getattr(self.config.MODEL, 'node_gating_lr', self.config.IL.lr)
                    param_groups.append({
                        'params': node_gating_params,
                        'lr': node_gating_lr
                    })
                    logger.info(f'Using separate learning rate {node_gating_lr} for node gating parameters')
                
                self.optimizer = torch.optim.AdamW(param_groups)
            else:
                self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.config.IL.lr)

        if load_from_ckpt:
            if config.IL.is_requeue:
                import glob
                ckpt_list = list(filter(os.path.isfile, glob.glob(config.CHECKPOINT_FOLDER + "/*")) )
                ckpt_list.sort(key=os.path.getmtime)
                ckpt_path = ckpt_list[-1]
            else:
                ckpt_path = config.IL.ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            start_iter = ckpt_dict["iteration"]

            if 'module' in list(ckpt_dict['state_dict'].keys())[0] and self.config.GPU_NUMBERS == 1:
                self.policy.net = torch.nn.DataParallel(self.policy.net.to(self.device),
                    device_ids=[self.device], output_device=self.device)
                self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
                self.policy.net = self.policy.net.module
                self.waypoint_predictor = torch.nn.DataParallel(self.waypoint_predictor.to(self.device),
                    device_ids=[self.device], output_device=self.device)
            else:
                self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
            if config.IL.is_requeue and hasattr(self, 'optimizer'):
                self.optimizer.load_state_dict(ckpt_dict["optim_state"])
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}, iteration: {start_iter}")
			
        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params/1e6:.2f} MB. Trainable: {params_t/1e6:.2f} MB.")
        logger.info("Finished setting up policy.")

        return start_iter

    def _teacher_action(self, batch_angles, batch_distances, candidate_lengths):
        if self.config.MODEL.task_type == 'r2r':
            cand_dists_to_goal = [[] for _ in range(len(batch_angles))]
            oracle_cand_idx = []
            for j in range(len(batch_angles)):
                for k in range(len(batch_angles[j])):
                    angle_k = batch_angles[j][k]
                    forward_k = batch_distances[j][k]
                    dist_k = self.envs.call_at(j, "cand_dist_to_goal", {"angle": angle_k, "forward": forward_k})
                    cand_dists_to_goal[j].append(dist_k)
                curr_dist_to_goal = self.envs.call_at(j, "current_dist_to_goal")
                # if within target range (which def as 3.0)
                if curr_dist_to_goal < 1.5:
                    oracle_cand_idx.append(candidate_lengths[j] - 1)
                else:
                    oracle_cand_idx.append(np.argmin(cand_dists_to_goal[j]))
            return oracle_cand_idx
        elif self.config.MODEL.task_type == 'rxr':
            kargs = []
            current_episodes = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                kargs.append({
                    'ref_path':self.gt_data[str(current_episodes[i].episode_id)]['locations'],
                    'angles':batch_angles[i],
                    'distances':batch_distances[i],
                    'candidate_length':candidate_lengths[i]
                })
            oracle_cand_idx = self.envs.call(["get_cand_idx"]*self.envs.num_envs, kargs)
            return oracle_cand_idx

    def _teacher_action_new(self, batch_gmap_vp_ids, batch_no_vp_left):
        teacher_actions = []
        cur_episodes = self.envs.current_episodes()
        for i, (gmap_vp_ids, gmap, no_vp_left) in enumerate(zip(batch_gmap_vp_ids, self.gmaps, batch_no_vp_left)):
            curr_dis_to_goal = self.envs.call_at(i, "current_dist_to_goal")
            if curr_dis_to_goal < 1.5:
                teacher_actions.append(0)
            else:
                if no_vp_left:
                    teacher_actions.append(-100)
                elif self.config.IL.expert_policy == 'spl':
                    ghost_vp_pos = [(vp, random.choice(pos)) for vp, pos in gmap.ghost_real_pos.items()]
                    ghost_dis_to_goal = [
                        self.envs.call_at(i, "point_dist_to_goal", {"pos": p[1]})
                        for p in ghost_vp_pos
                    ]
                    target_ghost_vp = ghost_vp_pos[np.argmin(ghost_dis_to_goal)][0]
                    teacher_actions.append(gmap_vp_ids.index(target_ghost_vp))
                elif self.config.IL.expert_policy == 'ndtw':
                    ghost_vp_pos = [(vp, random.choice(pos)) for vp, pos in gmap.ghost_real_pos.items()]
                    target_ghost_vp = self.envs.call_at(i, "ghost_dist_to_ref", {
                        "ghost_vp_pos": ghost_vp_pos,
                        "ref_path": self.gt_data[str(cur_episodes[i].episode_id)]['locations'],
                    })
                    teacher_actions.append(gmap_vp_ids.index(target_ghost_vp))
                else:
                    raise NotImplementedError
       
        return torch.tensor(teacher_actions).cuda()
    


    def _vp_feature_variable(self, obs):
            # obs = {
            #     'cand_rgb': cand_rgb,               # [K x 2048]，对应路点的视觉特征向量
            #     'cand_depth': cand_depth,           # [K x 128]，对应路点的深度特征向量
            #     'cand_angle_fts': cand_angle_fts,   # [K x 4]，对应路点的角度特征向量
            #     'cand_img_idxes': cand_img_idxes,   # [K]，对应路点的视觉图片索引
            #     'cand_angles': cand_angles,         # [K]，对应路点的逆时针角度（弧度值）
            #     'cand_distances': cand_distances,   # [K]，对应路点的真实距离（m）

            #     'pano_rgb': pano_rgb,               # B x 12 x 512，全景照片的特征向量
            #     'pano_depth': pano_depth,           # B x 12 x 128，全景照片的维度向量
            #     'pano_angle_fts': pano_angle_fts,   # 12 x 4，全景照片每个角度特征
            #     'pano_img_idxes': pano_img_idxes,   # 12 ，0-11的标号，照片索引数组。
            # }
            # 输出一组相对位置，极坐标表示形式

        batch_rgb_fts, batch_dep_fts, batch_loc_fts = [], [], []
        batch_nav_types, batch_view_lens = [], []

        for i in range(self.envs.num_envs): #对于每个环境循环
            rgb_fts, dep_fts, loc_fts , nav_types = [], [], [], []
            cand_idxes = np.zeros(12, dtype=bool)
            cand_idxes[obs['cand_img_idxes'][i]] = True
            # cand
            rgb_fts.append(obs['cand_rgb'][i])
            dep_fts.append(obs['cand_depth'][i])
            loc_fts.append(obs['cand_angle_fts'][i])
            nav_types += [1] * len(obs['cand_angles'][i])
            # non-cand
            rgb_fts.append(obs['pano_rgb'][i][~cand_idxes]) #对布尔数组取反
            dep_fts.append(obs['pano_depth'][i][~cand_idxes])
            loc_fts.append(obs['pano_angle_fts'][~cand_idxes])

            #nav_types 1 表示 candidate view，0 表示 non-candidate view
            nav_types += [0] * (12-np.sum(cand_idxes))
            
            #合成一个完整的视角张量，前K个是有候选路点的方向，后面的是非候选的
            batch_rgb_fts.append(torch.cat(rgb_fts, dim=0))
            batch_dep_fts.append(torch.cat(dep_fts, dim=0))
            batch_loc_fts.append(torch.cat(loc_fts, dim=0))

            batch_nav_types.append(torch.LongTensor(nav_types))     #把当前环境的 nav_types 从 Python list 变成 LongTensor
            batch_view_lens.append(len(nav_types))                  #记录当前的视角数量
        # collate
        #把一个由不同长度 tensor 组成的 list，padding 到同样长度，再 stack 成一个 batch tensor
        batch_rgb_fts = pad_tensors_wgrad(batch_rgb_fts)
        batch_dep_fts = pad_tensors_wgrad(batch_dep_fts)
        batch_loc_fts = pad_tensors_wgrad(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'rgb_fts': batch_rgb_fts, 'dep_fts': batch_dep_fts, 'loc_fts': batch_loc_fts,
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
        }
        
    def _nav_gmap_variable(self, cur_vp, cur_pos, cur_ori):
        #cur_vp前每个环境所在真实节点的 viewpoint id 列表，cur_pos当前每个环境 agent 的真实三维位置列表，cur_ori当前每个环境 agent 的真实朝向列表
        batch_gmap_vp_ids, batch_gmap_step_ids, batch_gmap_lens = [], [], []
        batch_gmap_img_fts, batch_gmap_pos_fts = [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []

        for i, gmap in enumerate(self.gmaps):
            node_vp_ids = list(gmap.node_pos.keys())
            ghost_vp_ids = list(gmap.ghost_pos.keys())
            if len(ghost_vp_ids) == 0:
                batch_no_vp_left.append(True)
            else:
                batch_no_vp_left.append(False)

            gmap_vp_ids = [None] + node_vp_ids + ghost_vp_ids
            gmap_step_ids = [0] + [gmap.node_stepId[vp] for vp in node_vp_ids] + [0]*len(ghost_vp_ids)
            gmap_visited_masks = [0] + [1] * len(node_vp_ids) + [0] * len(ghost_vp_ids)

            gmap_img_fts = [gmap.get_node_embeds(vp) for vp in node_vp_ids] + \
                           [gmap.get_node_embeds(vp) for vp in ghost_vp_ids]
            gmap_img_fts = torch.stack(
                [torch.zeros_like(gmap_img_fts[0])] + gmap_img_fts, dim=0
            )

            gmap_pos_fts = gmap.get_pos_fts(
                cur_vp[i], cur_pos[i], cur_ori[i], gmap_vp_ids
            )
            gmap_pair_dists = np.zeros((len(gmap_vp_ids), len(gmap_vp_ids)), dtype=np.float32)
            for j in range(1, len(gmap_vp_ids)):
                for k in range(j+1, len(gmap_vp_ids)):
                    vp1 = gmap_vp_ids[j]
                    vp2 = gmap_vp_ids[k]
                    if not vp1.startswith('g') and not vp2.startswith('g'):
                        dist = gmap.shortest_dist[vp1][vp2]
                    elif not vp1.startswith('g') and vp2.startswith('g'):
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = gmap.shortest_dist[vp1][front_vp2] + front_dis2
                    elif vp1.startswith('g') and vp2.startswith('g'):
                        front_dis1, front_vp1 = gmap.front_to_ghost_dist(vp1)
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = front_dis1 + gmap.shortest_dist[front_vp1][front_vp2] + front_dis2
                    else:
                        raise NotImplementedError
                    gmap_pair_dists[j, k] = gmap_pair_dists[k, j] = dist / MAX_DIST
            
            batch_gmap_vp_ids.append(gmap_vp_ids)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_lens.append(len(gmap_vp_ids))
            batch_gmap_img_fts.append(gmap_img_fts)
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
        
        # collate
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        batch_gmap_pos_fts = pad_tensors_wgrad(batch_gmap_pos_fts).cuda()
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        bs = self.envs.num_envs
        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(bs, max_gmap_len, max_gmap_len).float()
        for i in range(bs):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vp_ids': batch_gmap_vp_ids, #图里有哪些点
            'gmap_step_ids': batch_gmap_step_ids,   #这些点什么时候来的
            'gmap_img_fts': batch_gmap_img_fts,     #这些点长什么样
            'gmap_pos_fts': batch_gmap_pos_fts,     #这些点相对我在哪
            'gmap_masks': batch_gmap_masks,         #哪些点有效
            'gmap_visited_masks': batch_gmap_visited_masks,     #哪些点已访问
            'gmap_pair_dists': gmap_pair_dists,     #点和点之间有多远
            'no_vp_left': batch_no_vp_left,         #还有没有可探索 ghost
        }

    def _history_variable(self, obs):
        batch_size = obs['pano_rgb'].shape[0]
        hist_rgb_fts = obs['pano_rgb'][:, 0, ...].cuda()
        hist_pano_rgb_fts = obs['pano_rgb'].cuda()
        hist_pano_ang_fts = obs['pano_angle_fts'].unsqueeze(0).expand(batch_size, -1, -1).cuda()

        return hist_rgb_fts, hist_pano_rgb_fts, hist_pano_ang_fts

    @staticmethod
    def _pause_envs(envs, batch, envs_to_pause):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
            
            for k, v in batch.items():
                batch[k] = v[state_index]

        return envs, batch

    def train(self):
        self._set_config()
        if self.config.MODEL.task_type == 'rxr':
            self.gt_data = {}
            for role in self.config.TASK_CONFIG.DATASET.ROLES:
                with gzip.open(
                    self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                        split=self.split, role=role
                    ), "rt") as f:
                    self.gt_data.update(json.load(f))

        observation_space, action_space = self._init_envs()
        start_iter = self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )

        total_iter = self.config.IL.iters
        log_every  = self.config.IL.log_every
        writer     = TensorboardWriter(self.config.TENSORBOARD_DIR if self.local_rank < 1 else None)

        self.scaler = GradScaler()
        logger.info('Traning Starts... GOOD LUCK!')

        #记录个部分的运行时间，方便做性能分析
        self._init_perf_timing_log()
        try:
            for idx in range(start_iter, total_iter, log_every):
                interval = min(log_every, max(total_iter-idx, 0))
                cur_iter = idx + interval

                sample_ratio = self.config.IL.sample_ratio ** (idx // self.config.IL.decay_interval + 1)
                # sample_ratio = self.config.IL.sample_ratio ** (idx // self.config.IL.decay_interval)
                logs = self._train_interval(interval, self.config.IL.ml_weight, sample_ratio)

                if self.local_rank < 1:
                    loss_str = f'iter {cur_iter}: '
                    for k, v in logs.items():
                        logs[k] = np.mean(v)
                        loss_str += f'{k}: {logs[k]:.3f}, '
                        writer.add_scalar(f'loss/{k}', logs[k], cur_iter)
                    logger.info(loss_str)
                    self.save_checkpoint(cur_iter)
                    
                    # If dynamic graph or node gating is enabled, save weight information every 200 iterations
                    if (getattr(self.config.MODEL, 'use_dynamic_graph', False) or 
                        getattr(self.config.MODEL, 'use_node_gating', False)) and cur_iter % 200 == 0:
                        self.save_dynamic_graph_weights(cur_iter)
        finally:
            self._close_perf_timing_log()
        
    def _train_interval(self, interval, ml_weight, sample_ratio):
        #切换到训练模式
        self.policy.train()

        #如果是多线程的，有包装
        if self.world_size > 1:
            self.policy.net.module.rgb_encoder.eval()
            self.policy.net.module.depth_encoder.eval()
        else:
        #单线程的，没有包装，将深度编码器和rgb编码器冻结，切换为验证模式
            self.policy.net.rgb_encoder.eval()
            self.policy.net.depth_encoder.eval()

        #路点模块切换为验证模式
        self.waypoint_predictor.eval()

        #主进程显示进度条
        if self.local_rank < 1:
            pbar = tqdm.trange(
                interval, leave=False, dynamic_ncols=True, file=sys.stdout
            )
        else:
            pbar = range(interval)

        #前这段训练区间里的标量日志收集器
        self.logs = defaultdict(list)

        #对于每一个循环
        for idx in pbar:

            #清空损失和梯度累积
            self.optimizer.zero_grad()
            self.loss = 0.

            #自动混合精度反向传播
            with autocast():
                self.rollout('train', ml_weight, sample_ratio)
            self.scaler.scale(self.loss).backward() # self.loss.backward()
            self.scaler.step(self.optimizer)        # self.optimizer.step()
            self.scaler.update()

            # If dynamic graph is enabled, record weight values (for statistics)
            if self.local_rank < 1 and getattr(self.config.MODEL, 'use_dynamic_graph', False):
                self._record_dynamic_graph_weights()

            if self.local_rank < 1:
                #主进程显示训练进度
                pbar.set_postfix({'iter': f'{idx+1}/{interval}'})
            
        return deepcopy(self.logs)

    @torch.no_grad()
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ):
        if self.local_rank < 1:
            logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        iterator_options = _ensure_iterator_options(self.config.TASK_CONFIG)
        iterator_options.SHUFFLE = False
        iterator_options.MAX_SCENE_REPEAT_STEPS = -1
        _append_measurement_once(self.config.TASK_CONFIG, "COLLISIONS")
        self.config.IL.ckpt_to_load = checkpoint_path
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],            # Back
                'Down': [-math.pi / 2, 0 + shift, 0],       # Down
                'Front':[0, 0 + shift, 0],                  # Front
                'Right':[0, math.pi / 2 + shift, 0],        # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
                'Up':   [math.pi / 2, 0 + shift, 0],        # Up
            }
            sensor_uuids = []
            H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    camera_config.WIDTH = H
                    camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        if self.config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                self.config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{self.config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname) and not os.path.isfile(self.config.EVAL.CKPT_PATH_DIR):
                print("skipping -- evaluation exists.")
                return
        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=self.traj[::5] if self.config.EVAL.fast_eval else self.traj,
            auto_reset_done=False, # unseen: 11006 
        )
        dataset_length = sum(self.envs.number_of_episodes)
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
            create_optimizer=False,  # No optimizer needed for evaluation
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        if self.config.EVAL.EPISODE_COUNT == -1:
            eps_to_eval = sum(self.envs.number_of_episodes)
        else:
            eps_to_eval = min(self.config.EVAL.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.stat_eps = {}
        # Always initialize loc_noise_history to record loc_noise values for each episode
        self.loc_noise_history = defaultdict(list)
        # Record start time of each episode for calculating episode duration
        self.episode_start_times = {}
        self.pbar = (
            tqdm.tqdm(total=eps_to_eval, dynamic_ncols=True, file=sys.stdout)
            if self.config.use_pbar
            else None
        )

        while len(self.stat_eps) < eps_to_eval:
            self.rollout('eval')
        self.envs.close()

        if self.world_size > 1:
            distr.barrier()
        aggregated_states = {}
        num_episodes = len(self.stat_eps)
        for stat_key in next(iter(self.stat_eps.values())).keys():
            aggregated_states[stat_key] = (
                sum(v[stat_key] for v in self.stat_eps.values()) / num_episodes
            )
        total = torch.tensor(num_episodes).cuda()
        if self.world_size > 1:
            distr.reduce(total,dst=0)
        total = total.item()

        if self.world_size > 1:
            logger.info(f"rank {self.local_rank}'s {num_episodes}-episode results: {aggregated_states}")
            for k,v in aggregated_states.items():
                v = torch.tensor(v*num_episodes).cuda()
                cat_v = gather_list_and_concat(v,self.world_size)
                v = (sum(cat_v)/total).item()
                aggregated_states[k] = v
        
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        
        # Merge loc_noise_history into stat_eps
        for ep_id, metric in self.stat_eps.items():
            if ep_id in self.loc_noise_history:
                metric['loc_noise_history'] = self.loc_noise_history[ep_id]
            else:
                # If no record, set to empty list
                metric['loc_noise_history'] = []
        
        fname = os.path.join(
            self.config.RESULTS_DIR,
            f"stats_ep_ckpt_{checkpoint_index}_{split}_r{self.local_rank}_w{self.world_size}.json",
        )
        with open(fname, "w") as f:
            json.dump(self.stat_eps, f, indent=2)

        if self.local_rank < 1:
            if self.config.EVAL.SAVE_RESULTS:
                fname = os.path.join(
                    self.config.RESULTS_DIR,
                    f"stats_ckpt_{checkpoint_index}_{split}.json",
                )
                with open(fname, "w") as f:
                    json.dump(aggregated_states, f, indent=2)

            # loc_noise_history has been merged into stat_eps, no need to save separately
            logger.info(f"Episodes evaluated: {total}")
            checkpoint_num = checkpoint_index + 1
            for k, v in aggregated_states.items():
                logger.info(f"Average episode {k}: {v:.6f}")
                writer.add_scalar(f"eval_{k}/{split}", v, checkpoint_num)

    @torch.no_grad()
    def inference(self):
        checkpoint_path = self.config.INFERENCE.CKPT_PATH
        logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.IL.ckpt_to_load = checkpoint_path
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.INFERENCE.SPLIT
        self.config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        self.config.TASK_CONFIG.DATASET.LANGUAGES = self.config.INFERENCE.LANGUAGES
        iterator_options = _ensure_iterator_options(self.config.TASK_CONFIG)
        iterator_options.SHUFFLE = False
        iterator_options.MAX_SCENE_REPEAT_STEPS = -1
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION_INFER']
        self.config.TASK_CONFIG.TASK.SENSORS = [s for s in self.config.TASK_CONFIG.TASK.SENSORS if "INSTRUCTION" in s]
        self.config.SIMULATOR_GPU_IDS = [self.config.SIMULATOR_GPU_IDS[self.config.local_rank]]
        # if choosing image
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        self.config.freeze()

        torch.cuda.set_device(self.device)
        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            torch.cuda.set_device(self.device)
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
        self.traj = self.collect_infer_traj()

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=self.traj,
            auto_reset_done=False,
        )

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
            create_optimizer=False,  # No optimizer needed for inference
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        if self.config.INFERENCE.EPISODE_COUNT == -1:
            eps_to_infer = sum(self.envs.number_of_episodes)
        else:
            eps_to_infer = min(self.config.INFERENCE.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.path_eps = defaultdict(list)
        self.inst_ids: Dict[str, int] = {}   # transfer submit format
        self.pbar = tqdm.tqdm(
            total=eps_to_infer, dynamic_ncols=True, file=sys.stdout
        )
        
        # If dynamic or random loc_noise is enabled, initialize recording
        use_dynamic_loc_noise = getattr(self.config.IL, 'use_dynamic_loc_noise', False)
        use_random_loc_noise = getattr(self.config.IL, 'use_random_loc_noise', False)
        # Always initialize loc_noise_history to record loc_noise values for each episode
        self.loc_noise_history = defaultdict(list)

        while len(self.path_eps) < eps_to_infer:
            self.rollout('infer')
        self.envs.close()

        if self.world_size > 1:
            aggregated_path_eps = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_path_eps, self.path_eps)
            tmp_eps_dict = {}
            for x in aggregated_path_eps:
                tmp_eps_dict.update(x)
            self.path_eps = tmp_eps_dict

            aggregated_inst_ids = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_inst_ids, self.inst_ids)
            tmp_inst_dict = {}
            for x in aggregated_inst_ids:
                tmp_inst_dict.update(x)
            self.inst_ids = tmp_inst_dict


        if self.config.MODEL.task_type == "r2r":
            with open(self.config.INFERENCE.PREDICTIONS_FILE, "w") as f:
                json.dump(self.path_eps, f, indent=2)
            logger.info(f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}")
        else:  # use 'rxr' format for rxr-habitat leaderboard
            preds = []
            for k,v in self.path_eps.items():
                # save only positions that changed
                path = [v[0]["position"]]
                for p in v[1:]:
                    if p["position"] != path[-1]: path.append(p["position"])
                preds.append({"instruction_id": self.inst_ids[k], "path": path})
            preds.sort(key=lambda x: x["instruction_id"])
            with jsonlines.open(self.config.INFERENCE.PREDICTIONS_FILE, mode="w") as writer:
                writer.write_all(preds)
            logger.info(f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}")
        
        # loc_noise_history is not merged into path_eps in inference mode (because path_eps has different structure)
        # If needed, can be saved separately, but according to user requirements, mainly focus on eval mode stats_ep file

    def get_pos_ori(self):
        #这个函数是在从所有并行环境里取出当前 agent 的位置和朝向
        pos_ori = self.envs.call(['get_pos_ori']*self.envs.num_envs)
        pos = [x[0] for x in pos_ori]
        ori = [x[1] for x in pos_ori]
        return pos, ori

    def rollout(self, mode, ml_weight=None, sample_ratio=None):
        #真正的训练循环与环境交互

        if mode == 'train':
            feedback = 'sample'
        elif mode == 'eval' or mode == 'infer':
            feedback = 'argmax'
        else:
            raise NotImplementedError

        #重新设置所有环境
        self.envs.resume_all()
        observations = self.envs.reset()

        #设置最长步长
        instr_max_len = self.config.IL.max_text_len # r2r 80, rxr 200

        #设置不同的pad_id
        instr_pad_id = 1 if self.config.MODEL.task_type == 'rxr' else 0

        #把观测中的指令字段进行处理，过长的截断，不足的补足pad，是指令长度与维度统一。
        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                                                  max_length=instr_max_len, pad_id=instr_pad_id)
        
        #abitat Baselines 提供的通用数据处理工具，将原始数据转化成batchtensor，放在指定的GPU上
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        
        #验证模式和推理模式使用
        if mode == 'eval':
            curr_eps = self.envs.current_episodes()
            # Record start time of new episode
            for i, ep in enumerate(curr_eps):
                ep_id = ep.episode_id
                if ep_id not in self.episode_start_times:
                    self.episode_start_times[ep_id] = time.time()
            
            env_to_pause = [i for i, ep in enumerate(curr_eps) 
                            if ep.episode_id in self.stat_eps]    
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
        if mode == 'infer':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes()) 
                            if ep.episode_id in self.path_eps]    
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
            curr_eps = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                if self.config.MODEL.task_type == 'rxr':
                    ep_id = curr_eps[i].episode_id
                    k = curr_eps[i].instruction.instruction_id
                    self.inst_ids[ep_id] = int(k)

        # encode instructions编码指令
        all_txt_ids = batch['instruction']
        all_txt_masks = (all_txt_ids != instr_pad_id)
        all_txt_embeds = self.policy.net(
            mode='language',
            txt_ids=all_txt_ids,
            txt_masks=all_txt_masks,
        )

        loss = 0.
        total_actions = 0.

        #生成一个环境序列，记录哪些环境还没有停止
        not_done_index = list(range(self.envs.num_envs))

        #如果开启了视频或者是训练模式，就获取真实位置信息
        have_real_pos = (mode == 'train' or self.config.VIDEO_OPTION)
        ghost_aug = self.config.IL.ghost_aug if mode == 'train' else 0

        #为每一个环境创建一个拓扑图对象
        self.gmaps = [GraphMap(have_real_pos,   #是否有真实位置
                               self.config.IL.loc_noise,    #位置匹配时候的容忍半径
                               self.config.MODEL.merge_ghost,   #是否把接近的合并
                               ghost_aug) for _ in range(self.envs.num_envs)]   #ghost 位置扰动强度
        
        #初始化每个环境上一时刻所在 viewpoint
        prev_vp = [None] * self.envs.num_envs

        timing_enabled = (
            mode == 'train'
            and self.local_rank == 0
            and self._perf_timing_fh is not None
        )
        if timing_enabled:
            rollout_t0 = time.perf_counter()
            timing_acc = {
                'waypoint': 0.0,
                'env_call_at': 0.0,
                'navigation': 0.0,
                'env_step': 0.0,
            }
            step_counter = 0
            env_instance_sum = 0
            env_call_at_requests = 0

        #对于每一个时间步K
        for stepk in range(self.max_len):
            total_actions += self.envs.num_envs
            #性能分析使用
            if timing_enabled:
                step_counter += 1
                env_instance_sum += self.envs.num_envs

            #只取出还没有停止的环境的对应指令和编码
            txt_masks = all_txt_masks[not_done_index]
            txt_embeds = all_txt_embeds[not_done_index]
            
            # cand waypoint prediction
            '''
            outputs = {
                'cand_rgb': cand_rgb,               # [K x 2048]，对应路点的视觉特征向量
                'cand_depth': cand_depth,           # [K x 128]，对应路点的深度特征向量
                'cand_angle_fts': cand_angle_fts,   # [K x 4]，对应路点的角度特征向量
                'cand_img_idxes': cand_img_idxes,   # [K]，对应路点的视觉图片索引
                'cand_angles': cand_angles,         # [K]，对应路点的逆时针角度（弧度值）
                'cand_distances': cand_distances,   # [K]，对应路点的真实距离（m）

                'pano_rgb': pano_rgb,               # B x 12 x 512，全景照片的特征向量
                'pano_depth': pano_depth,           # B x 12 x 128，全景照片的维度向量
                'pano_angle_fts': pano_angle_fts,   # 12 x 4，全景照片每个角度特征
                'pano_img_idxes': pano_img_idxes,   # 12 ，0-11的标号，照片索引数组。
            }
            输出一组相对位置，极坐标表示形式
            '''
            if timing_enabled:  #性能分析使用
                t_waypoint = time.perf_counter()
            wp_outputs = self.policy.net(
                mode = "waypoint",
                waypoint_predictor = self.waypoint_predictor,
                observations = batch,
                #config.IL.waypoint_aug是否进行采样增强，训练的时候按照概率再nms周围选出一定的点
                in_train = (mode == 'train' and self.config.IL.waypoint_aug),
            )

            # pano encoder
            vp_inputs = self._vp_feature_variable(wp_outputs)
            #将这里面的都pad到相同长度，组织成batch，转换成tensor
            #             return {
            #     'rgb_fts': batch_rgb_fts, 'dep_fts': batch_dep_fts, 'loc_fts': batch_loc_fts,
            #     'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
            # }
            #向字典里新增或者覆盖一个键值对
            vp_inputs.update({
                'mode': 'panorama',
            })
            #进入forward()执行，
            #最终返回的是经过上下文融合之后的全景编码，包括角度、位置、深度、rgb等信息，形状为[B, L, 768]。还有一个mask
            pano_embeds, pano_masks = self.policy.net(**vp_inputs)

            #这一步是在把一整圈全景视角 token，压缩成“当前节点的单个全景摘要表示”。[B, L, H] -> [B, H],将12个视角特征进行融合
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)
            if timing_enabled:  #性能分析使用
                timing_acc['waypoint'] += (time.perf_counter() - t_waypoint)

            # get vp_id, vp_pos of cur_node and cand_ndoe
            cur_pos, cur_ori = self.get_pos_ori()   #批量读取当前 agent 的位置和朝向，并分别整理成两个列表返回
            cur_vp, cand_vp, cand_pos = [], [], []

            for i in range(self.envs.num_envs):
                # cur_vp，当前节点的 id；cand_vp当前时刻所有候选点的 id 列表，cand_pos当前时刻所有候选点的估计位置列表
                cur_vp_i, cand_vp_i, cand_pos_i = self.gmaps[i].identify_node(
                    cur_pos[i], cur_ori[i], wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i]
                )
                cur_vp.append(cur_vp_i)
                cand_vp.append(cand_vp_i)
                cand_pos.append(cand_pos_i)
            
            if mode == 'train' or self.config.VIDEO_OPTION:
                #获取真实的位置和朝向
                if timing_enabled:
                    t_call_at = time.perf_counter()
                cand_real_pos = []
                for i in range(self.envs.num_envs):
                    if timing_enabled:
                        env_call_at_requests += len(wp_outputs['cand_angles'][i])
                    cand_real_pos_i = [
                        self.envs.call_at(i, "get_cand_real_pos", {"angle": ang, "forward": dis})
                        for ang, dis in zip(wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i])
                    ]
                    cand_real_pos.append(cand_real_pos_i)
                if timing_enabled:
                    timing_acc['env_call_at'] += (time.perf_counter() - t_call_at)
            else:
                cand_real_pos = [None] * self.envs.num_envs

            # Calculate loc_noise (priority: dynamic > random > fixed)
            use_dynamic_loc_noise = getattr(self.config.IL, 'use_dynamic_loc_noise', False) #是否启用“动态 loc_noise”
            use_random_loc_noise = getattr(self.config.IL, 'use_random_loc_noise', False)   #是否启用“随机 loc_noise”
            loc_noise_values = [None] * self.envs.num_envs
            
            if use_dynamic_loc_noise:
                # Dynamic loc_noise: calculated based on candidate waypoint angle divergence
                loc_noise_min = getattr(self.config.IL, 'dynamic_loc_noise_min', 0.40)
                loc_noise_max = getattr(self.config.IL, 'dynamic_loc_noise_max', 0.60)
                loc_noise_base = getattr(self.config.IL, 'loc_noise', 0.5)  # Base value, used when insufficient candidate points
                # Read formula coefficients from config
                #alpha：基准值
                # beta：调节强度
                # mapping_type：选哪种映射曲线
                # sigmoid_k：sigmoid 曲线陡峭度
                # exponential_k：指数曲线曲率
                alpha = getattr(self.config.IL, 'dynamic_loc_noise_alpha', 0.65)
                beta = getattr(self.config.IL, 'dynamic_loc_noise_beta', 0.25)
                mapping_type = getattr(self.config.IL, 'dynamic_loc_noise_mapping', 'linear')
                sigmoid_k = getattr(self.config.IL, 'dynamic_loc_noise_sigmoid_k', 12.0)
                exponential_k = getattr(self.config.IL, 'dynamic_loc_noise_exponential_k', 4.0)
                
                def compute_loc_noise_from_std(std_val, mapping='linear'):
                    """
                    “根据当前候选 waypoint 角度分布的离散程度 std_val，动态计算这一步图构建要使用的 loc_noise。”
                    动态阈值建图的核心实现
                    Calculate loc_noise from std value, supports three mapping methods:
                    - linear: loc_noise = alpha - beta * std
                    - sigmoid: use sigmoid function mapping, refer to linear_compare.py
                    - exponential: use exponential function mapping, refer to linear_compare.py
                    
                    All mappings use alpha and beta parameters to determine reference points:
                    - When std=0, loc_noise should be close to loc_noise_max (or alpha)
                    - When std increases, loc_noise should decrease
                    - Use alpha and beta to determine the reference range of std
                    """
                    # Determine reference range of std: when std=std_ref, linear mapping's loc_noise reaches minimum
                    # i.e.: alpha - beta * std_ref = loc_noise_min
                    # Therefore: std_ref = (alpha - loc_noise_min) / beta
                    std_ref = (alpha - loc_noise_min) / beta if beta > 0 else 1.0
                    
                    if mapping == 'linear':
                        # Linear mapping: loc_noise = alpha - beta * std
                        loc_noise = alpha - beta * std_val
                    elif mapping == 'sigmoid':
                        # Sigmoid mapping: refer to implementation in linear_compare.py
                        # Use std_ref as reference point, similar to s_max in linear_compare.py
                        # When std=0, loc_noise=loc_noise_max (similar to y_start=0.5)
                        # When std=std_ref, loc_noise=loc_noise_min (similar to y_end=0.25)
                        if std_val <= 0:
                            return loc_noise_max
                        if std_val >= std_ref:
                            return loc_noise_min
                        
                        # Normalize std to [0, 1] range
                        x_norm = std_val / std_ref  # 0 -> 1
                        # Map to sigmoid's effective interval
                        x_mapped = sigmoid_k * (x_norm - 0.5)  # -k/2 -> k/2
                        
                        sigmoid_val = 1 / (1 + np.exp(-x_mapped))
                        
                        # Calibrate boundaries (because sigmoid(-k/2) != 0, sigmoid(k/2) != 1)
                        s_min = 1 / (1 + np.exp(-sigmoid_k * (-0.5)))
                        s_max_val = 1 / (1 + np.exp(-sigmoid_k * (0.5)))
                        
                        ratio = (sigmoid_val - s_min) / (s_max_val - s_min) if (s_max_val - s_min) > 0 else 0
                        
                        # Map to [loc_noise_min, loc_noise_max] range
                        total_drop = loc_noise_max - loc_noise_min
                        loc_noise = loc_noise_max - total_drop * ratio
                    elif mapping == 'exponential':
                        # Exponential mapping: refer to implementation in linear_compare.py
                        if std_val <= 0:
                            return loc_noise_max
                        if std_val >= std_ref:
                            return loc_noise_min
                        
                        # Normalize std to [0, 1] range
                        x_norm = std_val / std_ref
                        
                        # (e^kx - 1) / (e^k - 1)
                        exp_ratio = (np.exp(exponential_k * x_norm) - 1) / (np.exp(exponential_k) - 1)
                        
                        # Map to [loc_noise_min, loc_noise_max] range
                        total_drop = loc_noise_max - loc_noise_min
                        loc_noise = loc_noise_max - total_drop * exp_ratio
                    else:
                        # Default to linear mapping
                        loc_noise = alpha - beta * std_val
                    
                    # Clip to [min, max] range (although theoretically already in range, for safety)
                    return np.clip(loc_noise, loc_noise_min, loc_noise_max)
                
                for i in range(self.envs.num_envs):
                    cand_angles_i = wp_outputs['cand_angles'][i]
                    if len(cand_angles_i) > 1:
                        # Calculate angle standard deviation (in radians)
                        #计算标准差
                        std = float(np.std(cand_angles_i))
                        # Calculate loc_noise based on mapping type
                        #计算loc_noise
                        dynamic_loc_noise = compute_loc_noise_from_std(std, mapping=mapping_type)
                        loc_noise_values[i] = float(dynamic_loc_noise)
                    else:
                        # If only one or no candidate points, use base value
                        #只有1个或者0个，没有办法计算标准差，就使用基础的
                        loc_noise_values[i] = loc_noise_base
                
                # Record std and loc_noise in eval/infer mode
                if mode in ['eval', 'infer']:
                    curr_eps = self.envs.current_episodes()
                    for i in range(self.envs.num_envs):
                        ep_id = curr_eps[i].episode_id
                        cand_angles_i = wp_outputs['cand_angles'][i]
                        std = float(np.std(cand_angles_i)) if len(cand_angles_i) > 1 else 0.0
                        self.loc_noise_history[ep_id].append({
                            'step': stepk,
                            'std': std,
                            'loc_noise': loc_noise_values[i],
                            'type': 'dynamic',
                            'mapping': mapping_type
                        })
            elif use_random_loc_noise:
                #使用随机loc_noise
                # Random loc_noise: random sampling within specified range
                random_loc_noise_min = getattr(self.config.IL, 'random_loc_noise_min', 0.40)
                random_loc_noise_max = getattr(self.config.IL, 'random_loc_noise_max', 0.60)
                
                for i in range(self.envs.num_envs):
                    # Independent random sampling for each environment
                    random_loc_noise = random.uniform(random_loc_noise_min, random_loc_noise_max)
                    loc_noise_values[i] = float(random_loc_noise)
                
                # Record random loc_noise in eval/infer mode
                if mode in ['eval', 'infer']:
                    curr_eps = self.envs.current_episodes()
                    for i in range(self.envs.num_envs):
                        ep_id = curr_eps[i].episode_id
                        self.loc_noise_history[ep_id].append({
                            'step': stepk,
                            'loc_noise': loc_noise_values[i],
                            'type': 'random'
                        })
            else:
                #使用固定loc_noise
                # If both are disabled, use fixed loc_noise value, also need to record
                fixed_loc_noise = getattr(self.config.IL, 'loc_noise', 0.5)
                if mode in ['eval', 'infer']:
                    curr_eps = self.envs.current_episodes()
                    for i in range(self.envs.num_envs):
                        ep_id = curr_eps[i].episode_id
                        self.loc_noise_history[ep_id].append({
                            'step': stepk,
                            'loc_noise': fixed_loc_noise,
                            'type': 'fixed'
                        })
            # If both are disabled, loc_noise_values remains None, will use fixed loc_noise value in GraphMap

            for i in range(self.envs.num_envs):
                #遍历每一个并行环境，更新各自的图
                cur_embeds = avg_pano_embeds[i]

                #cand_embeds 是候选路点方向对应的局部特征
                cand_embeds = pano_embeds[i][vp_inputs['nav_types'][i]==1]

                # If dynamic or random loc_noise is enabled, pass calculated value; otherwise pass None to use default value
                loc_noise_to_use = loc_noise_values[i] if (use_dynamic_loc_noise or use_random_loc_noise) else None

                #更新了一下拓扑图结构，该合并的合并，该新建的新建。将当前的观测添加到全局的拓扑图中，但是当前的拓扑图中只有几何信息，只有节点之间的距离信息。
                self.gmaps[i].update_graph(prev_vp[i], stepk+1,
                                           cur_vp[i], cur_pos[i], cur_embeds,
                                           cand_vp[i], cand_pos[i], cand_embeds,
                                           cand_real_pos[i], loc_noise=loc_noise_to_use)

            ##cur_vp前每个环境所在真实节点的 viewpoint id 列表，cur_pos当前每个环境 agent 的真实三维位置列表，cur_ori当前每个环境 agent 的真实朝向列表
            #把已经更新好的图，打包成下一步全局导航决策所需的输入表示。
            if timing_enabled:
                t_navigation = time.perf_counter()
            nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori)  
        #             return {
        #     'gmap_vp_ids': batch_gmap_vp_ids, #图里有哪些点
        #     'gmap_step_ids': batch_gmap_step_ids,   #这些点什么时候来的
        #     'gmap_img_fts': batch_gmap_img_fts,     #这些点长什么样
        #     'gmap_pos_fts': batch_gmap_pos_fts,     #这些点相对我在哪
        #     'gmap_masks': batch_gmap_masks,         #哪些点有效
        #     'gmap_visited_masks': batch_gmap_visited_masks,     #哪些点已访问
        #     'gmap_pair_dists': gmap_pair_dists,     #点和点之间有多远
        #     'no_vp_left': batch_no_vp_left,         #还有没有可探索 ghost
        # }
            nav_inputs.update({
                'mode': 'navigation',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
            })
            no_vp_left = nav_inputs.pop('no_vp_left')

            nav_outs = self.policy.net(**nav_inputs)
            if timing_enabled:
                timing_acc['navigation'] += (time.perf_counter() - t_navigation)

        # outs = {
        #     'gmap_embeds': gmap_embeds, #经过全局图导航编码器更新后的图节点表示[B, L, H]
        #     'global_logits': global_logits, # 对图中每个可选节点的打分[B, L]
        # }

            nav_logits = nav_outs['global_logits']
            nav_probs = F.softmax(nav_logits, 1)
            for i, gmap in enumerate(self.gmaps):
                #给节点打一个适合停止的分数，后面进行全局选择
                #把当前节点如果选择 STOP 的概率，存成这个节点的 stop score。
                gmap.node_stop_scores[cur_vp[i]] = nav_probs[i, 0].data.item()

            # random sample demo
            # logits = torch.randn(nav_inputs['gmap_masks'].shape).cuda()
            # logits.masked_fill_(~nav_inputs['gmap_masks'], -float('inf'))
            # logits.masked_fill_(nav_inputs['gmap_visited_masks'], -float('inf'))

            if mode == 'train' or self.config.VIDEO_OPTION:
                #给当前每个环境算“老师应该选哪个图节点”
                teacher_actions = self._teacher_action_new(nav_inputs['gmap_vp_ids'], no_vp_left)
            if mode == 'train':
                #模型预测 nav_logits 和专家动作 teacher_actions 做交叉熵
                loss += F.cross_entropy(nav_logits, teacher_actions, reduction='sum', ignore_index=-100)

            # determine action
            if feedback == 'sample':
                #一部分时候跟模型自己采样，一部分时候跟专家动作
                c = torch.distributions.Categorical(nav_probs)  #把 nav_probs 看成一个离散概率分布
                a_t = c.sample().detach()   #从这个分布里采样一个动作索引，作为模型自己想执行的动作
                #前期更多的按照tf走，后期更多的按照模型自己选择的走。
                a_t = torch.where(torch.rand_like(a_t, dtype=torch.float)<=sample_ratio, teacher_actions, a_t)
            elif feedback == 'argmax':
                a_t = nav_logits.argmax(dim=-1)
            else:
                raise NotImplementedError
            #GPU 上的动作张量 a_t 转成 CPU 上的 numpy 数组
            cpu_a_t = a_t.cpu().numpy()

            # make equiv action
            env_actions = []

            #是否在严格无滑动的执行条件下，使用 tryout 机制来辅助目标节点执行
            use_tryout = (self.config.IL.tryout and not self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING)

            #enumerate遍历一个可迭代元素是，同时拿到下标和元素
            for i, gmap in enumerate(self.gmaps):

                if cpu_a_t[i] == 0 or stepk == self.max_len - 1 or no_vp_left[i]:
                    #如果要停止或者步数耗尽
                    # stop at node with max stop_prob
                    #取出来每一个节点和他的停止分数
                    vp_stop_scores = [(vp, stop_score) for vp, stop_score in gmap.node_stop_scores.items()]
                    #取出停止分数
                    stop_scores = [s[1] for s in vp_stop_scores]
                    #取出来停止分数最大的那个节点
                    stop_vp = vp_stop_scores[np.argmax(stop_scores)][0]
                    #取出停止节点的位置
                    stop_pos = gmap.node_pos[stop_vp]

                    if self.config.IL.back_algo == 'control':   #只有在回退策略设成 control 时，才会真的规划一条路径去控制 agent 走回去
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][stop_vp]]  #取出当前节点到停止目标节点的最短路径
                        back_path = back_path[1:]   #back_path 变成一条路径列表，里面每一项都带：节点 id。节点坐标
                    else:
                        back_path = None

                    vis_info = {
                            'nodes': list(gmap.node_pos.values()),  #取出节点地址数列
                            'ghosts': list(gmap.ghost_aug_pos.values()),    #取出ghost节点数列
                            'predict_ghost': stop_pos,      #取出停止节点的位置
                    }
                    env_actions.append(
                        {
                            'action': {
                                'act': 0,#高层动作类型编号
                                'cur_vp': cur_vp[i],    #当前所在节点 id
                                'stop_vp': stop_vp,     #最终决定要停下来的那个图节点 id
                                'stop_pos': stop_pos,   #该停止节点的真实位置坐标
                                'back_path': back_path, #如果当前不在 stop_vp 上，需要沿图最短路回退过去时，这里给出回退路径
                                'tryout': use_tryout,   #是否启用 tryout 机制辅助执行这个动作
                            },
                            'vis_info': vis_info,
                        }
                    )
                else:#如果没有停止，继续前进执行分支
                    #取出模型决策的目标点
                    ghost_vp = nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]]
                    #取出目标点的真实位置
                    ghost_pos = gmap.ghost_aug_pos[ghost_vp]
                    #如果是ghost节点，找到里他最近的以访问节点id
                    _, front_vp = gmap.front_to_ghost_dist(ghost_vp)
                    front_pos = gmap.node_pos[front_vp]#获取最近节点的位置
                    if self.config.VIDEO_OPTION:#处理视频显示相关内容
                        teacher_action_cpu = teacher_actions[i].cpu().item()
                        if teacher_action_cpu in [0, -100]:
                            teacher_ghost = None
                        else:
                            teacher_ghost = gmap.ghost_aug_pos[nav_inputs['gmap_vp_ids'][i][teacher_action_cpu]]
                        vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            'ghosts': list(gmap.ghost_aug_pos.values()),
                            'predict_ghost': ghost_pos,
                            'teacher_ghost': teacher_ghost,
                        }
                    else:
                        vis_info = None
                    # teleport to front, then forward to ghost
                    if self.config.IL.back_algo == 'control':#如果要回退，给出回退的路径
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][front_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    env_actions.append(
                        {
                            'action': {
                                'act': 4,
                                'cur_vp': cur_vp[i],    #当前所在节点id
                                'front_vp': front_vp, 
                                'front_pos': front_pos,
                                'ghost_vp': ghost_vp, 
                                'ghost_pos': ghost_pos,
                                'back_path': back_path,
                                'tryout': use_tryout,   #是否适用tryout
                            },
                            'vis_info': vis_info,
                        }
                    )
                    prev_vp[i] = front_vp
                    if self.config.MODEL.consume_ghost:
                        gmap.delete_ghost(ghost_vp)

            if timing_enabled:
                t_env_step = time.perf_counter()
            outputs = self.envs.step(env_actions)   #发送给环境，有一个返还观测
            if timing_enabled:
                timing_acc['env_step'] += (time.perf_counter() - t_env_step)
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            if mode == 'eval' and self.pbar is not None:
                # Keep progress feedback alive even before first episode is done.
                self.pbar.set_postfix(
                    {"active_envs": self.envs.num_envs, "rollout_step": stepk + 1},
                    refresh=False,
                )

            # calculate metric
            if mode == 'eval':
                #在评估模式下，负责把每个 episode 的结果统计、保存和收尾处理做好
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float64)
                    pred_path = np.array(info['position']['position'])
                    distances = np.array(info['position']['distance'])
                    metric = {}
                    metric['steps_taken'] = info['steps_taken']
                    metric['distance_to_goal'] = distances[-1]
                    metric['success'] = 1. if distances[-1] <= 3. else 0.
                    metric['oracle_success'] = 1. if (distances <= 3.).any() else 0.
                    metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1],axis=1).sum())
                    metric['collisions'], has_collision_info = _get_collision_rate(
                        info, len(pred_path)
                    )
                    if not has_collision_info and not self._warned_missing_collisions:
                        logger.warning(
                            "Missing collision metrics in env info; defaulting collisions to 0 for this run."
                        )
                        self._warned_missing_collisions = True
                    gt_length = distances[0]
                    metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
                    dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
                    metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
                    metric['sdtw'] = metric['ndtw'] * metric['success']
                    metric['ghost_cnt'] = self.gmaps[i].ghost_cnt
                    # Calculate episode duration (in seconds)
                    if ep_id in self.episode_start_times:
                        episode_duration = time.time() - self.episode_start_times[ep_id]
                        metric['episode_time'] = episode_duration
                        # Clean up start time record for completed episode
                        del self.episode_start_times[ep_id]
                    else:
                        # If start time is not recorded, set to 0 (should not happen theoretically)
                        metric['episode_time'] = 0.0
                    self.stat_eps[ep_id] = metric
                    self.pbar.update()

            # record path
            if mode == 'infer':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    self.path_eps[ep_id] = [
                        {
                            'position': info['position_infer']['position'][0],
                            'heading': info['position_infer']['heading'][0],
                            'stop': False
                        }
                    ]
                    for p, h in zip(info['position_infer']['position'][1:], info['position_infer']['heading'][1:]):
                        if p != self.path_eps[ep_id][-1]['position']:
                            self.path_eps[ep_id].append({
                                'position': p,
                                'heading': h,
                                'stop': False
                            })
                    self.path_eps[ep_id] = self.path_eps[ep_id][:500]
                    self.path_eps[ep_id][-1]['stop'] = True
                    self.pbar.update()

            # pause env
            if sum(dones) > 0:#当前并行环境里，每个环境这一步执行后是否已经结束 episode 的标记列表
                #如果当前这一步之后，至少有一个环境结束了 episode就进入后续结束处理逻辑
                #reversed数列反向
                for i in reversed(list(range(self.envs.num_envs))):
                    if dones[i]:#如果是这个环境停止了，删除相关信息和数组
                        not_done_index.pop(i)  
                        self.envs.pause_at(i)
                        observations.pop(i)
                        # graph stop
                        self.gmaps.pop(i)
                        prev_vp.pop(i)

            if self.envs.num_envs == 0:#所有环境都停止后，循环结束
                break

            # obs for next step
            #处理观测，为下一步循环做准备
            observations = extract_instruction_tokens(observations,self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if timing_enabled:
            self._train_rollout_counter += 1
            rollout_total = time.perf_counter() - rollout_t0
            self._write_perf_timing_log(
                {
                    'rollout_id': self._train_rollout_counter,
                    'steps': step_counter,
                    'env_instances_avg': env_instance_sum / max(step_counter, 1),
                    'total_actions': total_actions,
                    'waypoint': timing_acc['waypoint'],
                    'env_call_at': timing_acc['env_call_at'],
                    'navigation': timing_acc['navigation'],
                    'env_step': timing_acc['env_step'],
                    'rollout_total': rollout_total,
                    'env_call_at_requests': env_call_at_requests,
                }
            )

        if mode == 'train': #如果是训练模式下，统计损失信息。
            loss = ml_weight * loss / total_actions
            self.loss += loss
            self.logs['IL_loss'].append(loss.item())
