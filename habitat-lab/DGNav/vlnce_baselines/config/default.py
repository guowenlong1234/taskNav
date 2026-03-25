from typing import List, Optional, Union

from habitat.config.default import CONFIG_FILE_SEPARATOR
from yacs.config import CfgNode as CN

from habitat_extensions.config.default import (
    get_extended_config as get_task_config,
)

# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.BASE_TASK_CONFIG_PATH = ""
_C.TASK_CONFIG = get_task_config()  # task_config will be stored as a config node
_C.TRAINER_NAME = "dagger"
_C.ENV_NAME = "VLNCEDaggerEnv"
_C.SIMULATOR_GPU_IDS = [0]
_C.TORCH_GPU_ID = 0
_C.TORCH_GPU_IDS = [0]
_C.GPU_NUMBERS = 1
_C.NUM_ENVIRONMENTS = 1
_C.local_rank = 0
_C.VIDEO_OPTION = []  # options: "disk", "tensorboard"
_C.VIDEO_DIR = "videos/debug"
_C.TENSORBOARD_DIR = "data/tensorboard_dirs/debug"
_C.CHECKPOINT_FOLDER = "data/checkpoints/debug"
_C.EVAL_CKPT_PATH_DIR = "data/checkpoints/debug"
_C.RESULTS_DIR = "data/checkpoints/pretrained/evals"
_C.LOG_DIR = "data/logs/running_log"
_C.LOG_FILE = "train.log"

# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
# The split to evaluate on
_C.EVAL.USE_CKPT_CONFIG = True
_C.EVAL.SPLIT = "val_seen"
_C.EVAL.EPISODE_COUNT = -1
_C.EVAL.EPISODE_ID_FILE = ""
_C.EVAL.LANGUAGES = ["en-US", "en-IN"]
_C.EVAL.SAMPLE = False
_C.EVAL.SAVE_RESULTS = True
_C.EVAL.CKPT_PATH_DIR = ""
_C.EVAL.CKPT_PATH_LIST = []
_C.EVAL.fast_eval = False
_C.EVAL.EVAL_NONLEARNING = False
_C.EVAL.ENV_REFILL_POLICY = "legacy_batch"
_C.EVAL.NONLEARNING = CN()
_C.EVAL.NONLEARNING.AGENT = "RandomAgent"

# -----------------------------------------------------------------------------
# INFERENCE CONFIG
# -----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.SPLIT = "test"
_C.INFERENCE.LANGUAGES = ["en-US", "en-IN"]
_C.INFERENCE.SAMPLE = False
_C.INFERENCE.USE_CKPT_CONFIG = True
_C.INFERENCE.CKPT_PATH = "data/checkpoints/CMA_PM_DA_Aug.pth"
_C.INFERENCE.PREDICTIONS_FILE = "predictions.json"
_C.INFERENCE.INFERENCE_NONLEARNING = False
_C.INFERENCE.NONLEARNING = CN()
_C.INFERENCE.NONLEARNING.AGENT = "RandomAgent"
_C.INFERENCE.FORMAT = "rxr"  # either 'rxr' or 'r2r'
_C.INFERENCE.EPISODE_COUNT = -1
# -----------------------------------------------------------------------------
# STANDALONE TEACHER COLLECTOR CONFIG
# -----------------------------------------------------------------------------
_C.COLLECTOR = CN()
_C.COLLECTOR.enable = False
_C.COLLECTOR.output_dir = "/home/gwl/project/dataset/rxr_teacher_collect/"
_C.COLLECTOR.source_splits = ["train", "val_seen", "val_unseen"]

_C.COLLECTOR.image = CN()
_C.COLLECTOR.image.uuid = "collect_rgb"
_C.COLLECTOR.image.width = 320
_C.COLLECTOR.image.height = 240
_C.COLLECTOR.image.hfov = 90

_C.COLLECTOR.geometry = CN()
_C.COLLECTOR.geometry.turn_angle = 15
_C.COLLECTOR.geometry.forward_step_size = 0.25
_C.COLLECTOR.geometry.allow_sliding = False

_C.COLLECTOR.trace = CN()
_C.COLLECTOR.trace.max_primitive_steps = 400
_C.COLLECTOR.trace.min_frames_after_filter = 68

_C.COLLECTOR.filter_static = CN()
_C.COLLECTOR.filter_static.pos_eps = 1e-3
_C.COLLECTOR.filter_static.yaw_eps = 1e-3
_C.COLLECTOR.filter_static.run_k = 3

_C.COLLECTOR.dataset = CN()
_C.COLLECTOR.dataset.roles = ["guide"]
_C.COLLECTOR.dataset.languages = ["*"]

_C.COLLECTOR.teacher = CN()
_C.COLLECTOR.teacher.tryout = True

_C.COLLECTOR.target_counts = CN()
_C.COLLECTOR.target_counts.train = -1
_C.COLLECTOR.target_counts.test = -1

_C.COLLECTOR.prefilter = CN()
_C.COLLECTOR.prefilter.min_estimated_steps = -1

_C.COLLECTOR.runtime = CN()
_C.COLLECTOR.runtime.seed = 0
_C.COLLECTOR.runtime.log_every_rollouts = 50

# -----------------------------------------------------------------------------
# LEGACY COLLECT PLACEHOLDER
# -----------------------------------------------------------------------------
_C.COLLECT = CN()
_C.COLLECT.enable = False
_C.COLLECT.profile = "deprecated"
_C.COLLECT.collect_visual_debug = False
_C.COLLECT.collect_target_supervision = False
_C.COLLECT.seed = 0
_C.COLLECT.episode_source = CN()
_C.COLLECT.episode_source.splits = []
_C.COLLECT.image_sensor = CN()
_C.COLLECT.image_sensor.uuid = "collect_rgb"
_C.COLLECT.image_sensor.width = 320
_C.COLLECT.image_sensor.height = 240
_C.COLLECT.image_sensor.hfov = 90
_C.COLLECT.trace = CN()
_C.COLLECT.trace.max_primitive_steps = 400
_C.COLLECT.trace.max_decision_steps = 64
_C.COLLECT.trace.min_frames_after_filter = 68
_C.COLLECT.filter_static = CN()
_C.COLLECT.filter_static.pos_eps = 1e-3
_C.COLLECT.filter_static.yaw_eps = 1e-3
_C.COLLECT.filter_static.run_k = 3
_C.COLLECT.diagnostic = CN()
_C.COLLECT.diagnostic.enable = False
_C.COLLECT.diagnostic.straight_forward_steps = 8
_C.COLLECT.diagnostic.spin_turn_steps = 8
# -----------------------------------------------------------------------------
# IMITATION LEARNING CONFIG
# -----------------------------------------------------------------------------
_C.IL = CN()
_C.IL.lr = 2.5e-4
_C.IL.batch_size = 5
_C.IL.epochs = 4
_C.IL.iters = 20000
_C.IL.log_every = 200
_C.IL.ml_weight = 1.0
_C.IL.expert_policy = "spl"
_C.IL.sample_ratio = 0.75
_C.IL.decay_interval = 3000
_C.IL.use_iw = True
# inflection coefficient for RxR training set GT trajectories (guide): 1.9
# inflection coefficient for R2R training set GT trajectories: 3.2
_C.IL.inflection_weight_coef = 3.2
# load an already trained model for fine tuning
_C.IL.waypoint_aug = False
_C.IL.load_from_ckpt = False
_C.IL.ckpt_to_load = "data/checkpoints/ckpt.0.pth"
# if True, loads the optimizer state, epoch, and step_id from the ckpt dict.
_C.IL.is_requeue = False
_C.IL.max_traj_len = 20
_C.IL.max_text_len = 80
_C.IL.ghost_aug = 0.0
_C.IL.back_algo = "teleport"
_C.IL.tryout = True
_C.IL.TRAIN_ENV_REFILL_POLICY = "legacy_batch"
_C.IL.TRAIN_STATIC_SCENE_POOLS_ENABLE = False
_C.IL.TRAIN_SLOW_SCENES = []
_C.IL.TRAIN_FAST_POOL_NUM_ENVS = 0
_C.IL.TRAIN_SLOW_POOL_NUM_ENVS = 0
_C.IL.TRAIN_POOL_FAST_ITERS = 10
_C.IL.TRAIN_POOL_SLOW_ITERS = 1
# it True, start training from the saved epoch
# loc_noise configuration (fixed, dynamic, random are parallel, priority: dynamic > random > fixed)
_C.IL.loc_noise = 0.5  # Fixed loc_noise value (used when both dynamic and random are disabled)
# Dynamic loc_noise configuration
_C.IL.use_dynamic_loc_noise = False  # Whether to enable dynamic loc_noise (based on candidate waypoint angle divergence)
_C.IL.dynamic_loc_noise_min = 0.40  # Minimum value for dynamic loc_noise
_C.IL.dynamic_loc_noise_max = 0.60  # Maximum value for dynamic loc_noise
_C.IL.dynamic_loc_noise_alpha = 0.65  # Dynamic loc_noise formula coefficient alpha: loc_noise = clip(alpha - beta * std, min, max)
_C.IL.dynamic_loc_noise_beta = 0.25  # Dynamic loc_noise formula coefficient beta: loc_noise = clip(alpha - beta * std, min, max)
_C.IL.dynamic_loc_noise_mapping = "linear"  # Mapping function type: 'linear', 'sigmoid', 'exponential'
_C.IL.dynamic_loc_noise_sigmoid_k = 12.0  # Sigmoid mapping steepness parameter (only effective when mapping='sigmoid')
_C.IL.dynamic_loc_noise_exponential_k = 4.0  # Exponential mapping curvature parameter (only effective when mapping='exponential')
# Random loc_noise configuration
_C.IL.use_random_loc_noise = False  # Whether to enable random loc_noise (random sampling within specified range)
_C.IL.random_loc_noise_min = 0.40  # Minimum value for random loc_noise
_C.IL.random_loc_noise_max = 0.60  # Maximum value for random loc_noise
# -----------------------------------------------------------------------------
# IL: RXR TRAINER CONFIG
# -----------------------------------------------------------------------------
_C.IL.RECOLLECT_TRAINER = CN()
_C.IL.RECOLLECT_TRAINER.preload_trajectories_file = True
_C.IL.RECOLLECT_TRAINER.trajectories_file = (
    "data/trajectories_dirs/debug/trajectories.json.gz"
)
# if set to a positive int, episodes with longer paths are ignored in training
_C.IL.RECOLLECT_TRAINER.max_traj_len = -1
# if set to a positive int, effective_batch_size must be some multiple of
# IL.batch_size. Gradient accumulation enables an arbitrarily high "effective"
# batch size.
_C.IL.RECOLLECT_TRAINER.effective_batch_size = -1
_C.IL.RECOLLECT_TRAINER.preload_size = 30
_C.IL.RECOLLECT_TRAINER.use_iw = True
_C.IL.RECOLLECT_TRAINER.gt_file = (
    "data/datasets/RxR_VLNCE_v0/{split}/{split}_{role}_gt.json.gz"
)
# -----------------------------------------------------------------------------
# IL: DAGGER CONFIG
# -----------------------------------------------------------------------------
_C.IL.DAGGER = CN()
_C.IL.DAGGER.iterations = 10
_C.IL.DAGGER.update_size = 5000
_C.IL.DAGGER.p = 0.75
_C.IL.DAGGER.expert_policy_sensor = "SHORTEST_PATH_SENSOR"
_C.IL.DAGGER.expert_policy_sensor_uuid = "shortest_path_sensor"
_C.IL.DAGGER.load_space = False
# if True, load saved observation space and action space
_C.IL.DAGGER.lmdb_map_size = 1.0e12
# if True, saves data to disk in fp16 and converts back to fp32 when loading.
_C.IL.DAGGER.lmdb_fp16 = False
# How often to commit the writes to the DB, less commits is
# better, but everything must be in memory until a commit happens/
_C.IL.DAGGER.lmdb_commit_frequency = 500
# If True, load precomputed features directly from lmdb_features_dir.
_C.IL.DAGGER.preload_lmdb_features = False
_C.IL.DAGGER.lmdb_features_dir = (
    "data/trajectories_dirs/debug/trajectories.lmdb"
)
# -----------------------------------------------------------------------------
# RL CONFIG
# -----------------------------------------------------------------------------
_C.RL = CN()
_C.RL.POLICY = CN()
_C.RL.POLICY.OBS_TRANSFORMS = CN()
_C.RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS = [
    "CenterCropperPerSensor",
]
_C.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR = CN()
_C.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = [
    ("rgb", (224, 224)),
    ("depth", (256, 256)),
]
_C.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR = CN()
_C.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = [
    ("rgb", (224, 298)),
    ("depth", (256, 341)),
]
# -----------------------------------------------------------------------------
# ORACLE CONFIG
# -----------------------------------------------------------------------------
_C.ORACLE = CN()
_C.ORACLE.enable = False
_C.ORACLE.mode = "O1"
_C.ORACLE.provider = "SimulatorPeekOracleProvider"
_C.ORACLE.force_have_real_pos = True
_C.ORACLE.query_pipeline = "future_node_avg_pano"
_C.ORACLE.query_pos_strategy = "ghost_real_pos_mean"
_C.ORACLE.query_pos_fallback = "nearest_real_pos"
_C.ORACLE.query_heading_strategy = "face_frontier"
_C.ORACLE.target_ghost_scope = "all"
_C.ORACLE.enable_in_train = False
_C.ORACLE.enable_in_eval = True
_C.ORACLE.apply_mode = "hard"
_C.ORACLE.soft_alpha = 1.0
_C.ORACLE.refresh_policy = "on_change"
_C.ORACLE.multi_heading_pool_size = 4
_C.ORACLE.multi_heading_pool_mode = "mean"
_C.ORACLE.strict_scope = True
_C.ORACLE.shadow_rerun_planner = True
_C.ORACLE.max_scope_ghosts = -1
_C.ORACLE.scope_trace_enable = False
_C.ORACLE.scope_summary_enable = False
_C.ORACLE.scope_trace_dir = "data/logs/oracle_scope_traces/"
_C.ORACLE.scope_summary_dir = "data/logs/oracle_scope_summaries/"
_C.ORACLE.query_only_new_or_changed = True
_C.ORACLE.requery_on_realpos_update = True
_C.ORACLE.requery_min_pos_delta = 0.10
_C.ORACLE.navigability_check = True
_C.ORACLE.navigability_search_radius = 1.0
_C.ORACLE.navigability_num_samples = 16
_C.ORACLE.navigability_y_lock = True
_C.ORACLE.cache_enable = True
_C.ORACLE.cache_radius = 0.35
_C.ORACLE.cache_max_items_per_scene = 8192
_C.ORACLE.batch_query_enable = True
_C.ORACLE.batch_query_adaptive = True
_C.ORACLE.batch_query_micro_size = -1
_C.ORACLE.batch_query_max_micro_size = 32
_C.ORACLE.batch_query_fallback_to_serial = True
_C.ORACLE.max_queries_per_step = -1
_C.ORACLE.max_queries_per_episode = -1
_C.ORACLE.use_amp = False
_C.ORACLE.embed_dtype = "fp32"
_C.ORACLE.persistent_writeback = True
_C.ORACLE.hard_replace = True

_C.ORACLE.trace = CN()
_C.ORACLE.trace.enable = True
_C.ORACLE.trace.dir = "data/logs/oracle_traces/"
_C.ORACLE.trace.format = "jsonl"
_C.ORACLE.trace.log_every_n_steps = 1
_C.ORACLE.trace.counterfactual_trace = False
_C.ORACLE.trace.include_embed_vector = False
_C.ORACLE.trace.include_embed_norm = True
_C.ORACLE.trace.include_positions = True
_C.ORACLE.trace.include_failures = True
_C.ORACLE.trace.buffer_enable = True
_C.ORACLE.trace.buffer_flush_records = 200
_C.ORACLE.trace.flush_on_checkpoint = True
_C.ORACLE.trace.flush_on_run_end = True
# -----------------------------------------------------------------------------
# MODELING CONFIG
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.policy_name = "CMAPolicy"  # or "Seq2SeqPolicy"
_C.MODEL.task_type = "r2r"
_C.MODEL.ablate_depth = False
_C.MODEL.ablate_rgb = False
_C.MODEL.ablate_instruction = False
_C.MODEL.NUM_ANGLES = 12
_C.MODEL.pretrained_path = ""
_C.MODEL.fix_lang_embedding = False
_C.MODEL.fix_pano_embedding = False
_C.MODEL.use_depth_embedding = True
_C.MODEL.use_sprels = True
_C.MODEL.merge_ghost = True
_C.MODEL.consume_ghost = True

_C.MODEL.INSTRUCTION_ENCODER = CN()
_C.MODEL.INSTRUCTION_ENCODER.sensor_uuid = "instruction"
_C.MODEL.INSTRUCTION_ENCODER.vocab_size = 2504
_C.MODEL.INSTRUCTION_ENCODER.use_pretrained_embeddings = True
_C.MODEL.INSTRUCTION_ENCODER.embedding_file = (
    "data/datasets/R2R_VLNCE_v1-2_preprocessed/embeddings.json.gz"
)
_C.MODEL.INSTRUCTION_ENCODER.dataset_vocab = (
    "data/datasets/R2R_VLNCE_v1-2_preprocessed/train/train.json.gz"
)
_C.MODEL.INSTRUCTION_ENCODER.fine_tune_embeddings = False
_C.MODEL.INSTRUCTION_ENCODER.embedding_size = 50
_C.MODEL.INSTRUCTION_ENCODER.hidden_size = 128
_C.MODEL.INSTRUCTION_ENCODER.rnn_type = "LSTM"
_C.MODEL.INSTRUCTION_ENCODER.final_state_only = True
_C.MODEL.INSTRUCTION_ENCODER.bidirectional = False

_C.MODEL.rgb_feature_extractor = "clip"  # options: "clip", "dino"
_C.MODEL.dino = "EffoNav"
_C.MODEL.projector_ckpt_path = ""
_C.MODEL.ORACLE_SOFT_ALPHA = 1.0
_C.MODEL.spatial_output = True
_C.MODEL.RGB_ENCODER = CN()
_C.MODEL.RGB_ENCODER.cnn_type = "TorchVisionResNet50"
_C.MODEL.RGB_ENCODER.output_size = 256

_C.MODEL.VISUAL_DIM = CN()
_C.MODEL.VISUAL_DIM.vis_hidden = 768
_C.MODEL.VISUAL_DIM.directional = 128

_C.MODEL.DEPTH_ENCODER = CN()
_C.MODEL.DEPTH_ENCODER.cnn_type = "VlnResnetDepthEncoder"
_C.MODEL.DEPTH_ENCODER.output_size = 128
# type of resnet to use
_C.MODEL.DEPTH_ENCODER.backbone = "resnet50"
# path to DDPPO resnet weights
_C.MODEL.DEPTH_ENCODER.ddppo_checkpoint = (
    "data/ddppo-models/gibson-2plus-resnet50.pth"
)

_C.MODEL.STATE_ENCODER = CN()
_C.MODEL.STATE_ENCODER.hidden_size = 512
_C.MODEL.STATE_ENCODER.rnn_type = "GRU"

_C.MODEL.SEQ2SEQ = CN()
_C.MODEL.SEQ2SEQ.use_prev_action = False

_C.MODEL.PROGRESS_MONITOR = CN()
_C.MODEL.PROGRESS_MONITOR.use = False
_C.MODEL.PROGRESS_MONITOR.alpha = 1.0  # loss multiplier

_C.MODEL.ORACLE_FT = CN()
_C.MODEL.ORACLE_FT.enable = False
_C.MODEL.ORACLE_FT.hidden_dim = 768
_C.MODEL.ORACLE_FT.num_layers = 3
_C.MODEL.ORACLE_FT.dropout = 0.1
_C.MODEL.ORACLE_FT.activation = "gelu"
_C.MODEL.ORACLE_FT.use_layer_norm = True
_C.MODEL.ORACLE_FT.identity_init = True
_C.MODEL.ORACLE_FT.gain_init = 1.0
_C.MODEL.ORACLE_FT.fusion_alpha = 0.25
_C.MODEL.ORACLE_FT.use_config_soft_alpha = True
_C.MODEL.ORACLE_FT.train_scope = "oracle_only"
_C.MODEL.ORACLE_FT.unfreeze_global_encoder = True
_C.MODEL.ORACLE_FT.unfreeze_input_proj = False
_C.MODEL.ORACLE_FT.oracle_mlp_lr = 5e-5
_C.MODEL.ORACLE_FT.graph_lr = 5e-6
_C.MODEL.ORACLE_FT.input_proj_lr = 1e-5
_C.MODEL.ORACLE_FT.weight_decay = 0.01
_C.MODEL.ORACLE_FT.log_feature_stats = True
_C.MODEL.ORACLE_FT.eval_with_oracle_off = True

# Dynamic graph configuration
_C.MODEL.use_dynamic_graph = False  # Whether to enable dynamic graph method
_C.MODEL.dynamic_graph_lr = 1e-5  # Learning rate for dynamic graph related parameters (if dynamic graph is enabled)

# Geometric Dropout configuration
# Strategy 1: During training, force geometric distance to 0 with a certain probability, forcing the model to rely on semantic and instruction information
_C.MODEL.use_geo_dropout = False  # Whether to enable geometric dropout (only effective when use_dynamic_graph=True)
_C.MODEL.geo_dropout_prob = 0.3  # Probability of geometric dropout (recommended range: 0.2-0.3)

# Geometry-Conditioned Semantic Edge configuration
# Strategy 2: Use geometric distance information as conditional input to semantic similarity MLP, allowing the model to learn "similar appearance and close distance indicate true association"
_C.MODEL.use_geo_conditioned_semantic = False  # Whether to enable geometry-conditioned semantic edge (only effective when use_dynamic_graph=True)

# Ablation study configuration: control whether to use Esem and Einst in dynamic graph
_C.MODEL.use_esem = True  # Whether to enable semantic edge Esem (only effective when use_dynamic_graph=True)
_C.MODEL.use_einst = True  # Whether to enable instruction edge Einst (only effective when use_dynamic_graph=True)

# Node Gating configuration
# Node gating mechanism: gate visual node features after Cross-Attn and before Self-Attn, suppress noisy nodes, highlight instruction-related nodes
_C.MODEL.use_node_gating = False  # Whether to enable node gating mechanism
_C.MODEL.node_gating_lr = 1e-5  # Learning rate for node gating related parameters (if node gating is enabled)


def purge_keys(config: CN, keys: List[str]) -> None:
    for k in keys:
        del config[k]
        config.register_deprecated_key(k)


_ORACLE_LEGACY_ALIASES = {
    "ENABLE": "enable",
    "ENABLED": "enable",
    "MODE": "mode",
    "PROVIDER": "provider",
    "FORCE_HAVE_REAL_POS": "force_have_real_pos",
    "QUERY_PIPELINE": "query_pipeline",
    "QUERY_POS_STRATEGY": "query_pos_strategy",
    "QUERY_POS_FALLBACK": "query_pos_fallback",
    "QUERY_HEADING_STRATEGY": "query_heading_strategy",
    "TARGET_GHOST_SCOPE": "target_ghost_scope",
    "ENABLE_IN_TRAIN": "enable_in_train",
    "ENABLE_IN_EVAL": "enable_in_eval",
    "SCOPE": "target_ghost_scope",
    "APPLY_MODE": "apply_mode",
    "REPLACE_POLICY": "apply_mode",
    "SOFT_ALPHA": "soft_alpha",
    "REFRESH_POLICY": "refresh_policy",
    "MULTI_HEADING_POOL_SIZE": "multi_heading_pool_size",
    "MULTI_HEADING_POOL_MODE": "multi_heading_pool_mode",
    "STRICT_SCOPE": "strict_scope",
    "SHADOW_RERUN_PLANNER": "shadow_rerun_planner",
    "MAX_SCOPE_GHOSTS": "max_scope_ghosts",
    "LOG_SCOPE_TRACE": "scope_trace_enable",
    "LOG_SCOPE_SUMMARY": "scope_summary_enable",
    "SCOPE_TRACE_DIR": "scope_trace_dir",
    "SCOPE_SUMMARY_DIR": "scope_summary_dir",
    "PERSIST": "persistent_writeback",
    "QUERY_ONLY_NEW_OR_CHANGED": "query_only_new_or_changed",
    "REQUERY_ON_REALPOS_UPDATE": "requery_on_realpos_update",
    "REQUERY_MIN_POS_DELTA": "requery_min_pos_delta",
    "NAVIGABILITY_CHECK": "navigability_check",
    "NAVIGABILITY_SEARCH_RADIUS": "navigability_search_radius",
    "NAVIGABILITY_NUM_SAMPLES": "navigability_num_samples",
    "NAVIGABILITY_Y_LOCK": "navigability_y_lock",
    "CACHE_ENABLE": "cache_enable",
    "CACHE_RADIUS": "cache_radius",
    "CACHE_MAX_ITEMS_PER_SCENE": "cache_max_items_per_scene",
    "MAX_QUERIES_PER_STEP": "max_queries_per_step",
    "MAX_QUERIES_PER_EPISODE": "max_queries_per_episode",
    "USE_AMP": "use_amp",
    "EMBED_DTYPE": "embed_dtype",
    "PERSISTENT_WRITEBACK": "persistent_writeback",
    "HARD_REPLACE": "hard_replace",
}

_ORACLE_TRACE_LEGACY_ALIASES = {
    "ENABLE": "enable",
    "DIR": "dir",
    "FORMAT": "format",
    "LOG_EVERY_N_STEPS": "log_every_n_steps",
    "COUNTERFACTUAL_TRACE": "counterfactual_trace",
    "INCLUDE_EMBED_VECTOR": "include_embed_vector",
    "INCLUDE_EMBED_NORM": "include_embed_norm",
    "INCLUDE_POSITIONS": "include_positions",
    "INCLUDE_FAILURES": "include_failures",
}


def _promote_legacy_cfg_keys(config: CN, aliases) -> None:
    for legacy_key, canonical_key in aliases.items():
        if legacy_key in config:
            config[canonical_key] = config[legacy_key]
            del config[legacy_key]


def _normalize_legacy_oracle_opts(opts: List[str]) -> List[str]:
    normalized_opts: List[str] = []
    i = 0
    while i < len(opts):
        key = opts[i]
        if i + 1 >= len(opts):
            normalized_opts.append(key)
            break

        value = opts[i + 1]
        normalized_key = key
        if isinstance(key, str) and key.startswith("ORACLE."):
            suffix = key.split(".", 1)[1]
            mapped_suffix = _ORACLE_LEGACY_ALIASES.get(
                suffix, _ORACLE_LEGACY_ALIASES.get(suffix.upper())
            )
            if mapped_suffix is not None:
                normalized_key = f"ORACLE.{mapped_suffix}"
            elif suffix.startswith("TRACE."):
                trace_suffix = suffix.split(".", 1)[1]
                mapped_trace_suffix = _ORACLE_TRACE_LEGACY_ALIASES.get(
                    trace_suffix,
                    _ORACLE_TRACE_LEGACY_ALIASES.get(trace_suffix.upper()),
                )
                if mapped_trace_suffix is not None:
                    normalized_key = f"ORACLE.trace.{mapped_trace_suffix}"

        normalized_opts.extend([normalized_key, value])
        i += 2

    return normalized_opts


def _normalize_oracle_config(config: CN) -> None:
    if "ORACLE" not in config:
        return

    _promote_legacy_cfg_keys(config.ORACLE, _ORACLE_LEGACY_ALIASES)

    if "TRACE" in config.ORACLE:
        legacy_trace = config.ORACLE["TRACE"]
        for legacy_key, canonical_key in _ORACLE_TRACE_LEGACY_ALIASES.items():
            if legacy_key in legacy_trace:
                config.ORACLE.trace[canonical_key] = legacy_trace[legacy_key]
        del config.ORACLE["TRACE"]

    apply_mode = str(getattr(config.ORACLE, "apply_mode", "hard")).lower()
    if apply_mode == "hard" and not bool(getattr(config.ORACLE, "hard_replace", True)):
        apply_mode = "soft"
    if apply_mode not in {"hard", "soft"}:
        raise ValueError(
            f"Unsupported ORACLE.apply_mode={config.ORACLE.apply_mode!r}"
        )
    config.ORACLE.apply_mode = apply_mode
    config.ORACLE.soft_alpha = float(getattr(config.ORACLE, "soft_alpha", 1.0))
    if not 0.0 <= config.ORACLE.soft_alpha <= 1.0:
        raise ValueError(
            f"ORACLE.soft_alpha must be in [0, 1], got {config.ORACLE.soft_alpha}"
        )
    config.ORACLE.hard_replace = apply_mode == "hard"

    query_pipeline = str(
        getattr(config.ORACLE, "query_pipeline", "future_node_avg_pano")
    ).lower()
    allowed_pipelines = {"future_node_avg_pano"}
    if query_pipeline not in allowed_pipelines:
        raise ValueError(
            f"Unsupported ORACLE.query_pipeline={config.ORACLE.query_pipeline!r}. "
            f"Supported values: {sorted(allowed_pipelines)}"
        )
    config.ORACLE.query_pipeline = query_pipeline

    query_heading_strategy = str(
        getattr(config.ORACLE, "query_heading_strategy", "face_frontier")
    ).lower()
    allowed_heading_strategies = {
        "face_frontier",
        "travel_dir",
        "multi_heading_pool",
    }
    if query_heading_strategy not in allowed_heading_strategies:
        raise ValueError(
            "Unsupported ORACLE.query_heading_strategy="
            f"{config.ORACLE.query_heading_strategy!r}. "
            f"Supported values: {sorted(allowed_heading_strategies)}"
        )
    config.ORACLE.query_heading_strategy = query_heading_strategy
    config.ORACLE.multi_heading_pool_size = int(
        getattr(config.ORACLE, "multi_heading_pool_size", 4)
    )
    if config.ORACLE.multi_heading_pool_size <= 0:
        raise ValueError(
            "ORACLE.multi_heading_pool_size must be > 0, got "
            f"{config.ORACLE.multi_heading_pool_size}"
        )
    config.ORACLE.multi_heading_pool_mode = str(
        getattr(config.ORACLE, "multi_heading_pool_mode", "mean")
    ).lower()

    target_ghost_scope = str(
        getattr(config.ORACLE, "target_ghost_scope", "all")
    ).lower()
    allowed_scopes = {"all", "new_only", "local_frontier", "top1_shadow"}
    if target_ghost_scope not in allowed_scopes:
        raise ValueError(
            "Unsupported ORACLE.target_ghost_scope="
            f"{config.ORACLE.target_ghost_scope!r}. "
            f"Supported values: {sorted(allowed_scopes)}"
        )
    config.ORACLE.target_ghost_scope = target_ghost_scope

    config.ORACLE.enable_in_train = bool(
        getattr(config.ORACLE, "enable_in_train", False)
    )
    config.ORACLE.enable_in_eval = bool(
        getattr(config.ORACLE, "enable_in_eval", True)
    )
    config.ORACLE.strict_scope = bool(
        getattr(config.ORACLE, "strict_scope", True)
    )
    config.ORACLE.shadow_rerun_planner = bool(
        getattr(config.ORACLE, "shadow_rerun_planner", True)
    )
    config.ORACLE.max_scope_ghosts = int(
        getattr(config.ORACLE, "max_scope_ghosts", -1)
    )
    config.ORACLE.scope_trace_enable = bool(
        getattr(config.ORACLE, "scope_trace_enable", False)
    )
    config.ORACLE.scope_summary_enable = bool(
        getattr(config.ORACLE, "scope_summary_enable", False)
    )
    config.ORACLE.scope_trace_dir = str(
        getattr(config.ORACLE, "scope_trace_dir", "data/logs/oracle_scope_traces/")
    )
    config.ORACLE.scope_summary_dir = str(
        getattr(
            config.ORACLE,
            "scope_summary_dir",
            "data/logs/oracle_scope_summaries/",
        )
    )

    default_refresh_policy = "on_change"
    default_bool_pair = (True, True)
    current_bool_pair = (
        bool(getattr(config.ORACLE, "query_only_new_or_changed", True)),
        bool(getattr(config.ORACLE, "requery_on_realpos_update", True)),
    )
    refresh_policy = str(
        getattr(config.ORACLE, "refresh_policy", default_refresh_policy)
    ).lower()
    refresh_policy_pairs = {
        "on_change": (True, True),
        "first_only": (True, False),
        "every_step": (False, True),
        "manual": current_bool_pair,
    }
    if refresh_policy not in refresh_policy_pairs:
        raise ValueError(
            f"Unsupported ORACLE.refresh_policy={config.ORACLE.refresh_policy!r}"
        )

    policy_is_default = refresh_policy == default_refresh_policy
    bools_are_default = current_bool_pair == default_bool_pair
    derived_pair = refresh_policy_pairs[refresh_policy]

    if (not policy_is_default) and bools_are_default:
        final_bool_pair = derived_pair
        final_refresh_policy = refresh_policy
    elif policy_is_default and (not bools_are_default):
        final_bool_pair = current_bool_pair
        final_refresh_policy = {
            (True, True): "on_change",
            (True, False): "first_only",
            (False, True): "every_step",
        }.get(current_bool_pair, "manual")
    elif (
        (not policy_is_default)
        and (not bools_are_default)
        and refresh_policy != "manual"
        and current_bool_pair != derived_pair
    ):
        final_bool_pair = current_bool_pair
        final_refresh_policy = {
            (True, True): "on_change",
            (True, False): "first_only",
            (False, True): "every_step",
        }.get(current_bool_pair, "manual")
    else:
        final_bool_pair = derived_pair
        final_refresh_policy = refresh_policy

    config.ORACLE.query_only_new_or_changed = final_bool_pair[0]
    config.ORACLE.requery_on_realpos_update = final_bool_pair[1]
    config.ORACLE.refresh_policy = final_refresh_policy
    if "MODEL" in config:
        config.MODEL.ORACLE_SOFT_ALPHA = config.ORACLE.soft_alpha


def _normalize_env_refill_policies(config: CN) -> None:
    allowed = {"legacy_batch", "streaming_refill"}

    eval_policy = str(
        getattr(config.EVAL, "ENV_REFILL_POLICY", "legacy_batch")
    ).lower()
    if eval_policy not in allowed:
        raise ValueError(
            f"Invalid EVAL.ENV_REFILL_POLICY={config.EVAL.ENV_REFILL_POLICY!r}. "
            f"Supported values: {sorted(allowed)}"
        )
    config.EVAL.ENV_REFILL_POLICY = eval_policy

    train_policy = str(
        getattr(config.IL, "TRAIN_ENV_REFILL_POLICY", "legacy_batch")
    ).lower()
    if train_policy not in allowed:
        raise ValueError(
            "Invalid IL.TRAIN_ENV_REFILL_POLICY="
            f"{config.IL.TRAIN_ENV_REFILL_POLICY!r}. "
            f"Supported values: {sorted(allowed)}"
        )
    config.IL.TRAIN_ENV_REFILL_POLICY = train_policy


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values. Initialized from the
    habitat_baselines default config. Overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
    """
    config = _C.clone()
    config.set_new_allowed(True)

    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        prev_task_config = ""
        for config_path in config_paths:
            config.merge_from_file(config_path)
            _normalize_oracle_config(config)
            if config.BASE_TASK_CONFIG_PATH != prev_task_config:
                if config.BASE_TASK_CONFIG_PATH:
                    config.TASK_CONFIG = get_task_config(
                        config.BASE_TASK_CONFIG_PATH
                    )
                prev_task_config = config.BASE_TASK_CONFIG_PATH

    if opts:
        normalized_opts = _normalize_legacy_oracle_opts(opts)
        config.CMD_TRAILING_OPTS = normalized_opts
        config.merge_from_list(normalized_opts)

    _normalize_oracle_config(config)
    _normalize_env_refill_policies(config)
    config.set_new_allowed(False)
    config.freeze()
    return config
