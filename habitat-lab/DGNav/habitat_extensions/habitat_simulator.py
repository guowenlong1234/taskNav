#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Union,
    cast,
)

import numpy as np
from gym import spaces
from gym.spaces.box import Box
from numpy import ndarray

if TYPE_CHECKING:
    from torch import Tensor

import habitat_sim

from habitat_sim.simulator import MutableMapping, MutableMapping_T
from habitat.sims.habitat_simulator.habitat_simulator import (
    HabitatSim,
    overwrite_config,
)
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    DepthSensor,
    Observations,
    RGBSensor,
    SemanticSensor,
    Sensor,
    SensorSuite,
    ShortestPathPoint,
    Simulator,
    VisualObservation,
)
from habitat.core.spaces import Space

try:
    from habitat.core.simulator import Config
except ImportError:
    from omegaconf import DictConfig as Config

# inherit habitat-lab/habitat/sims/habitat_simulator/habitat_simulator.py
@registry.register_simulator(name="Sim-v1")
class Simulator(HabitatSim):
    r"""Simulator wrapper over habitat-sim

    habitat-sim repo: https://github.com/facebookresearch/habitat-sim

    Args:
        config: configuration for initializing the simulator.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def _get_agent_config(self):
        if hasattr(self.habitat_config, "AGENT_0"):
            return self.habitat_config.AGENT_0
        if (
            hasattr(self.habitat_config, "agents_order")
            and hasattr(self.habitat_config, "agents")
        ):
            agent_name = self.habitat_config.agents_order[
                self.habitat_config.default_agent_id
            ]
            return self.habitat_config.agents[agent_name]
        raise AttributeError("No agent config found")

    def create_sim_config(
        self, _sensor_suite: SensorSuite
    ) -> habitat_sim.Configuration:
        """Compat create_sim_config for both old/new habitat-sim SensorSpec APIs."""
        sim_config = habitat_sim.SimulatorConfiguration()
        if not hasattr(sim_config, "scene_id"):
            raise RuntimeError(
                "Incompatible version of Habitat-Sim detected, please upgrade habitat_sim"
            )

        sim_backend_cfg = (
            self.habitat_config.HABITAT_SIM_V0
            if hasattr(self.habitat_config, "HABITAT_SIM_V0")
            else self.habitat_config.habitat_sim_v0
        )
        overwrite_config(
            config_from=sim_backend_cfg,
            config_to=sim_config,
            ignore_keys={"gpu_gpu"},
        )
        scene_id = (
            self.habitat_config.SCENE
            if hasattr(self.habitat_config, "SCENE")
            else self.habitat_config.scene
        )
        sim_config.scene_id = scene_id

        agent_config = habitat_sim.AgentConfiguration()
        overwrite_config(
            config_from=self._get_agent_config(),
            config_to=agent_config,
            ignore_keys={
                "is_set_start_state",
                "sensors",
                "sim_sensors",
                "start_position",
                "start_rotation",
            },
        )

        sensor_specifications = []
        for sensor in _sensor_suite.sensors.values():
            # habitat-sim>=0.2 visual sensors use CameraSensorSpec instead of SensorSpec.
            if hasattr(habitat_sim, "CameraSensorSpec"):
                sim_sensor_cfg = habitat_sim.CameraSensorSpec()
            else:
                sim_sensor_cfg = habitat_sim.SensorSpec()

            overwrite_config(
                config_from=sensor.config,
                config_to=sim_sensor_cfg,
                ignore_keys={
                    "height",
                    "hfov",
                    "max_depth",
                    "min_depth",
                    "normalize_depth",
                    "type",
                    "width",
                },
            )
            sim_sensor_cfg.uuid = sensor.uuid
            sim_sensor_cfg.resolution = list(sensor.observation_space.shape[:2])

            if hasattr(sim_sensor_cfg, "parameters"):
                sim_sensor_cfg.parameters["hfov"] = str(sensor.config.HFOV)
            elif hasattr(sim_sensor_cfg, "hfov"):
                sim_sensor_cfg.hfov = float(sensor.config.HFOV)

            sensor_sim_type = getattr(sensor, "sim_sensor_type", None)
            if sensor_sim_type is not None:
                sim_sensor_cfg.sensor_type = sensor_sim_type
            if hasattr(sim_sensor_cfg, "gpu2gpu_transfer"):
                gpu_gpu = (
                    sim_backend_cfg.GPU_GPU
                    if hasattr(sim_backend_cfg, "GPU_GPU")
                    else getattr(sim_backend_cfg, "gpu_gpu", False)
                )
                sim_sensor_cfg.gpu2gpu_transfer = (
                    gpu_gpu
                )
            sensor_specifications.append(sim_sensor_cfg)

        agent_config.sensor_specifications = sensor_specifications
        forward_step_size = (
            self.habitat_config.FORWARD_STEP_SIZE
            if hasattr(self.habitat_config, "FORWARD_STEP_SIZE")
            else self.habitat_config.forward_step_size
        )
        turn_angle = (
            self.habitat_config.TURN_ANGLE
            if hasattr(self.habitat_config, "TURN_ANGLE")
            else self.habitat_config.turn_angle
        )
        agent_config.action_space = {
            0: habitat_sim.ActionSpec("stop"),
            1: habitat_sim.ActionSpec(
                "move_forward",
                habitat_sim.ActuationSpec(amount=forward_step_size),
            ),
            2: habitat_sim.ActionSpec(
                "turn_left",
                habitat_sim.ActuationSpec(amount=turn_angle),
            ),
            3: habitat_sim.ActionSpec(
                "turn_right",
                habitat_sim.ActuationSpec(amount=turn_angle),
            ),
        }

        return habitat_sim.Configuration(sim_config, [agent_config])

    def step_without_obs(self,
        action: Union[str, int, MutableMapping_T[int, Union[str, int]]],
        dt: float = 1.0 / 60.0,):
        self._num_total_frames += 1
        if isinstance(action, MutableMapping):
            return_single = False
        else:
            action = cast(Dict[int, Union[str, int]], {self._default_agent_id: action})
            return_single = True
        collided_dict: Dict[int, bool] = {}
        for agent_id, agent_act in action.items():
            agent = self.get_agent(agent_id)
            collided_dict[agent_id] = agent.act(agent_act)
            self.__last_state[agent_id] = agent.get_state()

        # # step physics by dt
        # step_start_Time = time.time()
        # super().step_world(dt)
        # self._previous_step_time = time.time() - step_start_Time

        multi_observations = {}
        for agent_id in action.keys():
            agent_observation = {}
            agent_observation["collided"] = collided_dict[agent_id]
            multi_observations[agent_id] = agent_observation

        if return_single:
            sim_obs = multi_observations[self._default_agent_id]
        else:
            sim_obs = multi_observations

        self._prev_sim_obs = sim_obs
