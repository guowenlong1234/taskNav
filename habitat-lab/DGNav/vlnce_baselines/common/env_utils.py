import os
import random
import sys
import importlib
from contextlib import contextmanager
from typing import List, Optional, Tuple, Type, Union

import habitat
from habitat import logger
from habitat import Env, RLEnv, VectorEnv, make_dataset
from yacs.config import CfgNode as CN

try:
    from habitat import Config
except ImportError:
    from omegaconf import DictConfig as Config

random.seed(0)

SLURM_JOBID = os.environ.get("SLURM_JOB_ID", None)


def _ensure_content_scenes(config: Config):
    try:
        scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    except AttributeError:
        config.defrost()
        config.TASK_CONFIG.DATASET.CONTENT_SCENES = ["*"]
        config.freeze()
        scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES

    if isinstance(scenes, str):
        scenes = [scenes]
    elif isinstance(scenes, tuple):
        scenes = list(scenes)
    return scenes


def _to_plain_config(value):
    if isinstance(value, CN):
        return {k: _to_plain_config(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {k: _to_plain_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_config(v) for v in value]
    return value


def _sync_task_config_for_habitat_core(config: Config):
    config.defrost()
    task_config = config.TASK_CONFIG

    if not hasattr(task_config, "SEED"):
        task_config.SEED = 0
    task_config.seed = task_config.SEED

    if not hasattr(task_config, "ENVIRONMENT"):
        task_config.ENVIRONMENT = CN()
    env_cfg = task_config.ENVIRONMENT
    if not hasattr(env_cfg, "MAX_EPISODE_STEPS"):
        env_cfg.MAX_EPISODE_STEPS = 1000
    if not hasattr(env_cfg, "MAX_EPISODE_SECONDS"):
        env_cfg.MAX_EPISODE_SECONDS = 10000000
    if not hasattr(env_cfg, "ITERATOR_OPTIONS"):
        env_cfg.ITERATOR_OPTIONS = CN()
    task_config.environment = env_cfg
    task_config.environment.max_episode_steps = env_cfg.MAX_EPISODE_STEPS
    task_config.environment.max_episode_seconds = env_cfg.MAX_EPISODE_SECONDS
    task_config.environment.iterator_options = env_cfg.ITERATOR_OPTIONS

    if not hasattr(task_config, "SIMULATOR"):
        task_config.SIMULATOR = CN()
    sim_cfg = task_config.SIMULATOR
    task_config.simulator = sim_cfg
    if hasattr(sim_cfg, "TYPE"):
        task_config.simulator.type = sim_cfg.TYPE
    if hasattr(sim_cfg, "HABITAT_SIM_V0"):
        task_config.simulator.habitat_sim_v0 = sim_cfg.HABITAT_SIM_V0
    if hasattr(sim_cfg, "SCENE"):
        task_config.simulator.scene = sim_cfg.SCENE
    if not hasattr(sim_cfg, "additional_object_paths"):
        sim_cfg.additional_object_paths = []
    if not hasattr(sim_cfg, "default_agent_id"):
        sim_cfg.default_agent_id = 0
    if not hasattr(sim_cfg, "agents_order"):
        sim_cfg.agents_order = ["agent_0"]
    if not hasattr(sim_cfg, "agents"):
        sim_cfg.agents = CN()
    if hasattr(sim_cfg, "AGENT_0"):
        agent_cfg = sim_cfg.AGENT_0
        if not hasattr(agent_cfg, "sim_sensors"):
            agent_cfg.sim_sensors = CN()
        if hasattr(agent_cfg, "SENSORS"):
            for sensor_name in list(agent_cfg.SENSORS):
                if hasattr(sim_cfg, sensor_name):
                    sensor_cfg = getattr(sim_cfg, sensor_name)
                    if not hasattr(sensor_cfg, "TYPE"):
                        upper_name = sensor_name.upper()
                        if "DEPTH" in upper_name:
                            sensor_cfg.TYPE = "HabitatSimDepthSensor"
                        elif "SEMANTIC" in upper_name:
                            sensor_cfg.TYPE = "HabitatSimSemanticSensor"
                        else:
                            sensor_cfg.TYPE = "HabitatSimRGBSensor"
                    if not hasattr(sensor_cfg, "UUID"):
                        sensor_uuid = sensor_name.lower()
                        if sensor_uuid.endswith("_sensor"):
                            sensor_uuid = sensor_uuid[: -len("_sensor")]
                        sensor_cfg.UUID = sensor_uuid
                    if "DEPTH" in sensor_name.upper():
                        if not hasattr(sensor_cfg, "MIN_DEPTH"):
                            sensor_cfg.MIN_DEPTH = 0.0
                        if not hasattr(sensor_cfg, "MAX_DEPTH"):
                            sensor_cfg.MAX_DEPTH = 10.0
                        if not hasattr(sensor_cfg, "NORMALIZE_DEPTH"):
                            sensor_cfg.NORMALIZE_DEPTH = True
                    for key in (
                        "TYPE",
                        "UUID",
                        "HEIGHT",
                        "WIDTH",
                        "HFOV",
                        "MIN_DEPTH",
                        "MAX_DEPTH",
                        "NORMALIZE_DEPTH",
                        "POSITION",
                        "ORIENTATION",
                    ):
                        if hasattr(sensor_cfg, key):
                            setattr(sensor_cfg, key.lower(), getattr(sensor_cfg, key))
                    agent_cfg.sim_sensors[sensor_name.lower()] = sensor_cfg
        if not hasattr(agent_cfg, "is_set_start_state"):
            agent_cfg.is_set_start_state = False
        if not hasattr(agent_cfg, "start_position"):
            agent_cfg.start_position = [0.0, 0.0, 0.0]
        if not hasattr(agent_cfg, "start_rotation"):
            agent_cfg.start_rotation = [0.0, 0.0, 0.0, 1.0]
        sim_cfg.agents.agent_0 = agent_cfg
        if len(sim_cfg.agents_order) == 0:
            sim_cfg.agents_order = ["agent_0"]

    if not hasattr(task_config, "TASK"):
        task_config.TASK = CN()
    task_cfg = task_config.TASK
    task_config.task = task_cfg
    if hasattr(task_cfg, "TYPE"):
        task_config.task.type = task_cfg.TYPE
    if not hasattr(task_cfg, "physics_target_sps"):
        task_cfg.physics_target_sps = 60.0
    if not hasattr(task_cfg, "actions"):
        task_cfg.actions = CN()
        if not hasattr(task_cfg, "ACTIONS"):
            task_cfg.ACTIONS = CN()
        default_action_types = {
            "STOP": "StopAction",
            "MOVE_FORWARD": "MoveForwardAction",
            "TURN_LEFT": "TurnLeftAction",
            "TURN_RIGHT": "TurnRightAction",
            "HIGHTOLOW": "MoveHighToLowAction",
            "HIGHTOLOWEVAL": "MoveHighToLowActionEval",
            "HIGHTOLOWINFERENCE": "MoveHighToLowActionInference",
        }
        action_names = (
            list(task_cfg.POSSIBLE_ACTIONS)
            if hasattr(task_cfg, "POSSIBLE_ACTIONS")
            else []
        )
        for action_name in action_names:
            if hasattr(task_cfg.ACTIONS, action_name):
                action_cfg = getattr(task_cfg.ACTIONS, action_name)
            else:
                action_cfg = CN()
                if action_name in default_action_types:
                    action_cfg.TYPE = default_action_types[action_name]
            if hasattr(action_cfg, "TYPE"):
                action_cfg.type = action_cfg.TYPE
            task_cfg.actions[action_name] = _to_plain_config(action_cfg)
    if not hasattr(task_cfg, "lab_sensors"):
        task_cfg.lab_sensors = CN()
        sensor_names = (
            list(task_cfg.SENSORS) if hasattr(task_cfg, "SENSORS") else []
        )
        default_sensor_types = {
            "INSTRUCTION_SENSOR": "InstructionSensor",
            "RXR_INSTRUCTION_SENSOR": "RxRInstructionSensor",
            "SHORTEST_PATH_SENSOR": "ShortestPathSensor",
            "VLN_ORACLE_PROGRESS_SENSOR": "VLNOracleProgressSensor",
        }
        for sensor_name in sensor_names:
            if hasattr(task_cfg, sensor_name):
                sensor_cfg = getattr(task_cfg, sensor_name)
            elif sensor_name in default_sensor_types:
                sensor_cfg = CN()
                sensor_cfg.TYPE = default_sensor_types[sensor_name]
                setattr(task_cfg, sensor_name, sensor_cfg)
            else:
                continue
            if hasattr(sensor_cfg, "TYPE"):
                sensor_cfg.type = sensor_cfg.TYPE
            task_cfg.lab_sensors[sensor_name] = _to_plain_config(sensor_cfg)
    if not hasattr(task_cfg, "measurements"):
        task_cfg.measurements = CN()
        measurement_names = (
            list(task_cfg.MEASUREMENTS)
            if hasattr(task_cfg, "MEASUREMENTS")
            else []
        )
        default_measurement_types = {
            "DISTANCE_TO_GOAL": "DistanceToGoal",
            "SUCCESS": "Success",
            "SPL": "SPL",
            "NDTW": "NDTW",
            "SDTW": "SDTW",
            "PATH_LENGTH": "PathLength",
            "ORACLE_SUCCESS": "OracleSuccess",
            "STEPS_TAKEN": "StepsTaken",
            "POSITION": "Position",
            "POSITION_INFER": "PositionInfer",
            "TOP_DOWN_MAP_VLNCE": "TopDownMapVLNCE",
        }
        for measurement_name in measurement_names:
            if hasattr(task_cfg, measurement_name):
                measurement_cfg = getattr(task_cfg, measurement_name)
            elif measurement_name in default_measurement_types:
                measurement_cfg = CN()
                measurement_cfg.TYPE = default_measurement_types[
                    measurement_name
                ]
                setattr(task_cfg, measurement_name, measurement_cfg)
            else:
                continue
            if hasattr(measurement_cfg, "TYPE"):
                measurement_cfg.type = measurement_cfg.TYPE
            task_cfg.measurements[measurement_name] = _to_plain_config(
                measurement_cfg
            )

    if not hasattr(task_config, "DATASET"):
        task_config.DATASET = CN()
    dataset_cfg = task_config.DATASET
    task_config.dataset = dataset_cfg
    if hasattr(dataset_cfg, "TYPE"):
        task_config.dataset.type = dataset_cfg.TYPE

    config.freeze()


def _patch_habitat_read_write_for_yacs():
    try:
        import habitat.config as habitat_config_pkg
        import habitat.core.env as habitat_core_env
        import habitat.tasks.nav.nav as habitat_nav_task
        from omegaconf import Container
        habitat_read_write_mod = importlib.import_module(
            "habitat.config.read_write"
        )

        if getattr(habitat_read_write_mod, "_dgnav_read_write_compat", False):
            return

        _orig_read_write = habitat_read_write_mod.read_write

        @contextmanager
        def _compat_read_write(config):
            if isinstance(config, Container):
                with _orig_read_write(config):
                    yield config
            else:
                was_frozen = (
                    config.is_frozen()
                    if hasattr(config, "is_frozen")
                    else False
                )
                if hasattr(config, "defrost"):
                    config.defrost()
                try:
                    yield config
                finally:
                    if was_frozen and hasattr(config, "freeze"):
                        config.freeze()

        habitat_read_write_mod.read_write = _compat_read_write
        habitat_read_write_mod._dgnav_read_write_compat = True
        habitat_config_pkg.read_write = _compat_read_write
        habitat_core_env.read_write = _compat_read_write
        habitat_nav_task.read_write = _compat_read_write
    except Exception:
        pass


def is_slurm_job() -> bool:
    return SLURM_JOBID is not None


def is_slurm_batch_job() -> bool:
    r"""Heuristic to determine if a slurm job is a batch job or not. Batch jobs
    will have a job name that is not a shell unless the user specifically set the job
    name to that of a shell. Interactive jobs have a shell name as their job name.
    """
    return is_slurm_job() and os.environ.get("SLURM_JOB_NAME", None) not in (
        None,
        "bash",
        "zsh",
        "fish",
        "tcsh",
        "sh",
    )


def make_env_fn(config: Config, env_class: Type[Union[Env, RLEnv]]):
    _patch_habitat_read_write_for_yacs()
    _sync_task_config_for_habitat_core(config)
    try:
        dataset = make_dataset(
            config.TASK_CONFIG.DATASET.TYPE,
            config=config.TASK_CONFIG.DATASET,
        )
    except TypeError:
        dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)

    env = env_class(config=config, dataset=dataset)
    if hasattr(env, "seed"):
        env.seed(config.TASK_CONFIG.SEED)
    return env


def _normalize_scene_name(scene: str) -> str:
    return os.path.splitext(os.path.basename(str(scene)))[0]


def get_dataset_scenes_to_load(config: Config) -> List[str]:
    _sync_task_config_for_habitat_core(config)
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    scenes = _ensure_content_scenes(config)
    if "*" in scenes:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)
    return [_normalize_scene_name(scene) for scene in scenes]


def split_static_scene_pools(
    all_scenes: List[str],
    slow_scenes: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    slow_targets = []
    seen_targets = set()
    for scene in slow_scenes:
        name = _normalize_scene_name(scene)
        if name not in seen_targets:
            slow_targets.append(name)
            seen_targets.add(name)

    available = []
    seen_available = set()
    for scene in all_scenes:
        name = _normalize_scene_name(scene)
        if name not in seen_available:
            available.append(name)
            seen_available.add(name)

    available_set = set(available)
    selected_slow = [scene for scene in slow_targets if scene in available_set]
    missing = [scene for scene in slow_targets if scene not in available_set]
    selected_slow_set = set(selected_slow)
    selected_fast = [scene for scene in available if scene not in selected_slow_set]
    return selected_fast, selected_slow, missing


def construct_envs(
    config: Config,
    env_class: Type[Union[Env, RLEnv]],
    workers_ignore_signals: bool = False,
    auto_reset_done: bool = True,
    episodes_allowed: Optional[List[str]] = None,
    content_scenes_override: Optional[List[str]] = None,
) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.
    :param config: configs that contain num_environments as well as information
    :param necessary to create individual environments.
    :param env_class: class type of the envs to be created.
    :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor
    :param auto_reset_done: Whether or not to automatically reset the env on done
    :return: VectorEnv object created according to specification.
    """

    _sync_task_config_for_habitat_core(config)

    num_envs_per_gpu = config.NUM_ENVIRONMENTS
    if isinstance(config.SIMULATOR_GPU_IDS, list):
        gpus = config.SIMULATOR_GPU_IDS
    else:
        gpus = [config.SIMULATOR_GPU_IDS]
    num_gpus = len(gpus)
    num_envs = num_gpus * num_envs_per_gpu

    if episodes_allowed is not None:
        config.defrost()
        config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = episodes_allowed
        config.freeze()

    configs = []
    env_classes = [env_class for _ in range(num_envs)]
    if content_scenes_override is not None:
        scenes = [_normalize_scene_name(scene) for scene in content_scenes_override]
    else:
        dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
        scenes = _ensure_content_scenes(config)
        if "*" in scenes:
            scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)
        scenes = [_normalize_scene_name(scene) for scene in scenes]
    logger.info(f"SPLTI: {config.TASK_CONFIG.DATASET.SPLIT}, NUMBER OF SCENES: {len(scenes)}")

    if num_envs > 1:
        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multi-process logic relies on being able"
                " to split scenes uniquely between processes"
            )

        if len(scenes) < num_envs and len(scenes) != 1:
            raise RuntimeError(
                "reduce the number of GPUs or envs as there"
                " aren't enough number of scenes"
            )

        random.shuffle(scenes)

    if len(scenes) == 1:
        scene_splits = [[scenes[0]] for _ in range(num_envs)]
    else:
        scene_splits = [[] for _ in range(num_envs)]
        for idx, scene in enumerate(scenes):
            scene_splits[idx % len(scene_splits)].append(scene)

        assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_gpus):
        for j in range(num_envs_per_gpu):
            proc_config = config.clone()
            proc_config.defrost()
            proc_id = (i * num_envs_per_gpu) + j

            task_config = proc_config.TASK_CONFIG
            task_config.SEED += proc_id
            if len(scenes) > 0:
                task_config.DATASET.CONTENT_SCENES = scene_splits[proc_id]

            task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpus[i]

            task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS

            proc_config.freeze()
            configs.append(proc_config) 

    is_debug = True if sys.gettrace() else False
    if os.environ.get("DGNAV_FORCE_THREADED_ENV", "0") == "1":
        is_debug = True
    env_entry = habitat.ThreadedVectorEnv if is_debug else habitat.VectorEnv
    envs = env_entry(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(zip(configs, env_classes)), 
        auto_reset_done=auto_reset_done,
        workers_ignore_signals=workers_ignore_signals,
    )
    return envs


def construct_envs_auto_reset_false(
    config: Config, env_class: Type[Union[Env, RLEnv]]
) -> VectorEnv:
    return construct_envs(config, env_class, auto_reset_done=False)

def construct_envs_for_rl(
    config: Config,
    env_class: Type[Union[Env, RLEnv]],
    workers_ignore_signals: bool = False,
    auto_reset_done: bool = True,
    episodes_allowed: Optional[List[str]] = None,
) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.
    :param config: configs that contain num_environments as well as information
    :param necessary to create individual environments.
    :param env_class: class type of the envs to be created.
    :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor
    :param auto_reset_done: Whether or not to automatically reset the env on done
    :return: VectorEnv object created according to specification.
    """

    _sync_task_config_for_habitat_core(config)

    num_envs_per_gpu = config.NUM_ENVIRONMENTS
    if isinstance(config.SIMULATOR_GPU_IDS, list):
        gpus = config.SIMULATOR_GPU_IDS
    else:
        gpus = [config.SIMULATOR_GPU_IDS]
    num_gpus = len(gpus)
    num_envs = num_gpus * num_envs_per_gpu

    if episodes_allowed is not None:
        config.defrost()
        config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = episodes_allowed
        config.freeze()

    configs = []
    env_classes = [env_class for _ in range(num_envs)]
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    scenes = _ensure_content_scenes(config)
    if "*" in scenes:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)

    if num_envs > 1:
        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multi-process logic relies on being able"
                " to split scenes uniquely between processes"
            )

        if len(scenes) < num_envs and len(scenes) != 1:
            raise RuntimeError(
                "reduce the number of GPUs or envs as there"
                " aren't enough number of scenes"
            )
        random.shuffle(scenes)

    if len(scenes) == 1:
        scene_splits = [[scenes[0]] for _ in range(num_envs)]
    else:
        scene_splits = [[] for _ in range(num_envs)]
        for idx, scene in enumerate(scenes):
            scene_splits[idx % len(scene_splits)].append(scene)

        assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_gpus):
        for j in range(num_envs_per_gpu):
            proc_config = config.clone()
            proc_config.defrost()
            proc_id = (i * num_envs_per_gpu) + j

            task_config = proc_config.TASK_CONFIG
            task_config.SEED += proc_id
            if len(scenes) > 0:
                task_config.DATASET.CONTENT_SCENES = scene_splits[proc_id]

            task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpus[i]

            task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS

            proc_config.freeze()
            configs.append(proc_config)

    is_debug = True if sys.gettrace() else False
    if os.environ.get("DGNAV_FORCE_THREADED_ENV", "0") == "1":
        is_debug = True
    env_entry = habitat.ThreadedVectorEnv if is_debug else habitat.VectorEnv
    envs = env_entry(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(zip(configs, env_classes)),
        auto_reset_done=auto_reset_done,
        workers_ignore_signals=workers_ignore_signals,
    )
    return envs
