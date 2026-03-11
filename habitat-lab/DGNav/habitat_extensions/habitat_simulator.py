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
    HabitatSimVizSensors,
    overwrite_config,
)
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    Config,
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

    def create_sim_config(
        self, _sensor_suite: SensorSuite
    ) -> habitat_sim.Configuration:
        """Compat create_sim_config for both old/new habitat-sim SensorSpec APIs."""
        sim_config = habitat_sim.SimulatorConfiguration()
        if not hasattr(sim_config, "scene_id"):
            raise RuntimeError(
                "Incompatible version of Habitat-Sim detected, please upgrade habitat_sim"
            )

        overwrite_config(
            config_from=self.habitat_config.HABITAT_SIM_V0,
            config_to=sim_config,
            ignore_keys={"gpu_gpu"},
        )
        sim_config.scene_id = self.habitat_config.SCENE

        agent_config = habitat_sim.AgentConfiguration()
        overwrite_config(
            config_from=self._get_agent_config(),
            config_to=agent_config,
            ignore_keys={
                "is_set_start_state",
                "sensors",
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

            sensor = cast(HabitatSimVizSensors, sensor)
            sim_sensor_cfg.sensor_type = sensor.sim_sensor_type
            if hasattr(sim_sensor_cfg, "gpu2gpu_transfer"):
                sim_sensor_cfg.gpu2gpu_transfer = (
                    self.habitat_config.HABITAT_SIM_V0.GPU_GPU
                )
            sensor_specifications.append(sim_sensor_cfg)

        agent_config.sensor_specifications = sensor_specifications
        agent_config.action_space = registry.get_action_space_configuration(
            self.habitat_config.ACTION_SPACE_CONFIG
        )(self.habitat_config).get()

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
