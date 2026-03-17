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
_C.BASE_TASK_CONFIG_PATH = "habitat_extensions/config/vlnce_task.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
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
_C.LOG_FILE = "train.log"

# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
# The split to evaluate on
_C.EVAL.USE_CKPT_CONFIG = True
_C.EVAL.SPLIT = "val_seen"
_C.EVAL.EPISODE_COUNT = -1
_C.EVAL.LANGUAGES = ["en-US", "en-IN"]
_C.EVAL.SAMPLE = False
_C.EVAL.SAVE_RESULTS = True
_C.EVAL.CKPT_PATH_DIR = ""
_C.EVAL.fast_eval = False
_C.EVAL.EVAL_NONLEARNING = False
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
_C.ORACLE.ENABLE = False
_C.ORACLE.MODE = "O1"
_C.ORACLE.PROVIDER = "SimulatorPeekOracleProvider"
_C.ORACLE.FORCE_HAVE_REAL_POS = True
_C.ORACLE.QUERY_PIPELINE = "future_node_avg_pano"
_C.ORACLE.QUERY_POS_STRATEGY = "ghost_real_pos_mean"
_C.ORACLE.QUERY_POS_FALLBACK = "nearest_real_pos"
_C.ORACLE.QUERY_HEADING_STRATEGY = "face_frontier"
_C.ORACLE.TARGET_GHOST_SCOPE = "all"
_C.ORACLE.QUERY_ONLY_NEW_OR_CHANGED = True
_C.ORACLE.REQUERY_ON_REALPOS_UPDATE = True
_C.ORACLE.REQUERY_MIN_POS_DELTA = 0.10
_C.ORACLE.NAVIGABILITY_CHECK = True
_C.ORACLE.NAVIGABILITY_SEARCH_RADIUS = 1.0
_C.ORACLE.NAVIGABILITY_NUM_SAMPLES = 16
_C.ORACLE.NAVIGABILITY_Y_LOCK = True
_C.ORACLE.CACHE_ENABLE = True
_C.ORACLE.CACHE_RADIUS = 0.25
_C.ORACLE.CACHE_MAX_ITEMS_PER_SCENE = 4096
_C.ORACLE.MAX_QUERIES_PER_STEP = -1
_C.ORACLE.MAX_QUERIES_PER_EPISODE = -1
_C.ORACLE.USE_AMP = False
_C.ORACLE.EMBED_DTYPE = "fp32"
_C.ORACLE.PERSISTENT_WRITEBACK = True
_C.ORACLE.HARD_REPLACE = True

_C.ORACLE.TRACE = CN()
_C.ORACLE.TRACE.ENABLE = True
_C.ORACLE.TRACE.DIR = "data/logs/oracle_traces/"
_C.ORACLE.TRACE.FORMAT = "jsonl"
_C.ORACLE.TRACE.LOG_EVERY_N_STEPS = 1
_C.ORACLE.TRACE.COUNTERFACTUAL_TRACE = False
_C.ORACLE.TRACE.INCLUDE_EMBED_VECTOR = False
_C.ORACLE.TRACE.INCLUDE_EMBED_NORM = True
_C.ORACLE.TRACE.INCLUDE_POSITIONS = True
_C.ORACLE.TRACE.INCLUDE_FAILURES = True
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
    "enable": "ENABLE",
    "mode": "MODE",
    "provider": "PROVIDER",
    "force_have_real_pos": "FORCE_HAVE_REAL_POS",
    "query_pipeline": "QUERY_PIPELINE",
    "query_pos_strategy": "QUERY_POS_STRATEGY",
    "query_pos_fallback": "QUERY_POS_FALLBACK",
    "query_heading_strategy": "QUERY_HEADING_STRATEGY",
    "target_ghost_scope": "TARGET_GHOST_SCOPE",
    "query_only_new_or_changed": "QUERY_ONLY_NEW_OR_CHANGED",
    "requery_on_realpos_update": "REQUERY_ON_REALPOS_UPDATE",
    "requery_min_pos_delta": "REQUERY_MIN_POS_DELTA",
    "navigability_check": "NAVIGABILITY_CHECK",
    "navigability_search_radius": "NAVIGABILITY_SEARCH_RADIUS",
    "navigability_num_samples": "NAVIGABILITY_NUM_SAMPLES",
    "navigability_y_lock": "NAVIGABILITY_Y_LOCK",
    "cache_enable": "CACHE_ENABLE",
    "cache_radius": "CACHE_RADIUS",
    "cache_max_items_per_scene": "CACHE_MAX_ITEMS_PER_SCENE",
    "max_queries_per_step": "MAX_QUERIES_PER_STEP",
    "max_queries_per_episode": "MAX_QUERIES_PER_EPISODE",
    "use_amp": "USE_AMP",
    "embed_dtype": "EMBED_DTYPE",
    "persistent_writeback": "PERSISTENT_WRITEBACK",
    "hard_replace": "HARD_REPLACE",
}

_ORACLE_TRACE_LEGACY_ALIASES = {
    "enable": "ENABLE",
    "dir": "DIR",
    "format": "FORMAT",
    "log_every_n_steps": "LOG_EVERY_N_STEPS",
    "counterfactual_trace": "COUNTERFACTUAL_TRACE",
    "include_embed_vector": "INCLUDE_EMBED_VECTOR",
    "include_embed_norm": "INCLUDE_EMBED_NORM",
    "include_positions": "INCLUDE_POSITIONS",
    "include_failures": "INCLUDE_FAILURES",
}


def _promote_legacy_cfg_keys(config: CN, aliases) -> None:
    for legacy_key, canonical_key in aliases.items():
        if legacy_key in config:
            config[canonical_key] = config[legacy_key]
            del config[legacy_key]


def _normalize_oracle_config(config: CN) -> None:
    if "ORACLE" not in config:
        return

    _promote_legacy_cfg_keys(config.ORACLE, _ORACLE_LEGACY_ALIASES)

    if "trace" in config.ORACLE:
        legacy_trace = config.ORACLE["trace"]
        for legacy_key, canonical_key in _ORACLE_TRACE_LEGACY_ALIASES.items():
            if legacy_key in legacy_trace:
                config.ORACLE.TRACE[canonical_key] = legacy_trace[legacy_key]
        del config.ORACLE["trace"]

    if "TRACE" in config.ORACLE:
        _promote_legacy_cfg_keys(
            config.ORACLE.TRACE, _ORACLE_TRACE_LEGACY_ALIASES
        )


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
            if config.BASE_TASK_CONFIG_PATH != prev_task_config:
                config.TASK_CONFIG = get_task_config(
                    config.BASE_TASK_CONFIG_PATH
                )
                prev_task_config = config.BASE_TASK_CONFIG_PATH

    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    _normalize_oracle_config(config)
    config.set_new_allowed(False)
    config.freeze()
    return config
