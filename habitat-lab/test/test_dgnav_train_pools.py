#!/usr/bin/env python3

import logging
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace


REPO_HABITAT_DIR = Path(__file__).resolve().parents[1]
DGNAV_DIR = REPO_HABITAT_DIR / "DGNav"
HABITAT_LAB_DIR = REPO_HABITAT_DIR / "habitat-lab"
HABITAT_BASELINES_DIR = REPO_HABITAT_DIR / "habitat-baselines"

for path in (DGNAV_DIR, HABITAT_BASELINES_DIR, HABITAT_LAB_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

try:
    import importlib
    from contextlib import contextmanager

    import habitat
    import habitat.config as habitat_config
    import habitat.config.default as habitat_config_default
    import habitat.core.env as habitat_core_env
    import habitat.core.utils as habitat_core_utils
    import habitat.tasks.nav.nav as habitat_nav_task
    import habitat.utils.visualizations.utils as habitat_viz_utils
    import habitat_baselines.config.default as hb_default
    from omegaconf import Container
    from yacs.config import CfgNode

    habitat_read_write_mod = importlib.import_module("habitat.config.read_write")

    if not hasattr(habitat, "Config"):
        habitat.Config = CfgNode
    if not hasattr(habitat_config, "Config"):
        habitat_config.Config = CfgNode
    if not hasattr(habitat_config_default, "Config"):
        habitat_config_default.Config = CfgNode
    if not hasattr(habitat_core_utils, "try_cv2_import"):
        def _try_cv2_import():
            import cv2
            return cv2

        habitat_core_utils.try_cv2_import = _try_cv2_import
    if (
        not hasattr(habitat_viz_utils, "append_text_to_image")
        and hasattr(habitat_viz_utils, "append_text_underneath_image")
    ):
        habitat_viz_utils.append_text_to_image = (
            habitat_viz_utils.append_text_underneath_image
        )
    if not hasattr(habitat_read_write_mod, "_dgnav_read_write_compat"):
        orig_read_write = habitat_read_write_mod.read_write

        @contextmanager
        def _compat_read_write(config):
            if isinstance(config, Container):
                with orig_read_write(config):
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
        habitat_config.read_write = _compat_read_write
        habitat_core_env.read_write = _compat_read_write
        habitat_nav_task.read_write = _compat_read_write
    if not hasattr(hb_default, "_C"):
        hb_default._C = CfgNode()
except ImportError:
    pass

try:
    from vlnce_baselines.ss_trainer_ETP import RLTrainer
except ImportError as exc:  # pragma: no cover - depends on local env
    raise unittest.SkipTest(f"DGNav trainer dependencies unavailable: {exc}")


class _FakeEnvPool:
    def __init__(self, num_envs: int, resumed_num_envs: int = None):
        self.num_envs = int(num_envs)
        self._resumed_num_envs = (
            self.num_envs
            if resumed_num_envs is None
            else int(resumed_num_envs)
        )
        self.resume_all_calls = 0

    def resume_all(self):
        self.resume_all_calls += 1
        self.num_envs = self._resumed_num_envs


def _make_trainer(
    *,
    static_pool_active: bool,
    envs=None,
    fast_envs=None,
    slow_envs=None,
    train_iteration_counter: int = 0,
    fast_iters: int = 10,
    slow_iters: int = 1,
):
    trainer = RLTrainer.__new__(RLTrainer)
    trainer._train_static_scene_pool_active = bool(static_pool_active)
    trainer.envs = envs
    trainer.fast_envs = fast_envs
    trainer.slow_envs = slow_envs
    trainer._train_iteration_counter = int(train_iteration_counter)
    trainer.local_rank = 0
    trainer.config = SimpleNamespace(
        IL=SimpleNamespace(
            TRAIN_POOL_FAST_ITERS=int(fast_iters),
            TRAIN_POOL_SLOW_ITERS=int(slow_iters),
        )
    )
    return trainer


class TrainPoolActivationTests(unittest.TestCase):
    def test_single_pool_activation_resumes_before_use(self):
        envs = _FakeEnvPool(num_envs=0, resumed_num_envs=3)
        trainer = _make_trainer(static_pool_active=False, envs=envs)

        preferred_pool_name, preferred_env = trainer._select_train_pool()
        actual_pool_name, actual_env = trainer._activate_train_pool(
            preferred_pool_name, preferred_env
        )

        self.assertEqual(preferred_pool_name, "fast")
        self.assertIs(preferred_env, envs)
        self.assertEqual(actual_pool_name, "fast")
        self.assertIs(actual_env, envs)
        self.assertEqual(envs.resume_all_calls, 1)
        self.assertEqual(envs.num_envs, 3)

    def test_select_train_pool_ignores_zero_num_envs_when_scheduling(self):
        fast_env = _FakeEnvPool(num_envs=0, resumed_num_envs=4)
        slow_env = _FakeEnvPool(num_envs=0, resumed_num_envs=2)
        trainer = _make_trainer(
            static_pool_active=True,
            envs=fast_env,
            fast_envs=fast_env,
            slow_envs=slow_env,
            train_iteration_counter=10,
        )

        pool_name, pool_env = trainer._select_train_pool()

        self.assertEqual(pool_name, "slow")
        self.assertIs(pool_env, slow_env)

    def test_activate_train_pool_uses_preferred_pool_after_resume(self):
        fast_env = _FakeEnvPool(num_envs=0, resumed_num_envs=4)
        slow_env = _FakeEnvPool(num_envs=0, resumed_num_envs=2)
        trainer = _make_trainer(
            static_pool_active=True,
            envs=fast_env,
            fast_envs=fast_env,
            slow_envs=slow_env,
        )

        actual_pool_name, actual_env = trainer._activate_train_pool("fast", fast_env)

        self.assertEqual(actual_pool_name, "fast")
        self.assertIs(actual_env, fast_env)
        self.assertEqual(fast_env.resume_all_calls, 1)
        self.assertEqual(slow_env.resume_all_calls, 0)
        self.assertEqual(fast_env.num_envs, 4)

    def test_activate_train_pool_falls_back_to_alternate_pool(self):
        fast_env = _FakeEnvPool(num_envs=0, resumed_num_envs=4)
        slow_env = _FakeEnvPool(num_envs=0, resumed_num_envs=0)
        trainer = _make_trainer(
            static_pool_active=True,
            envs=fast_env,
            fast_envs=fast_env,
            slow_envs=slow_env,
            train_iteration_counter=11,
        )

        with patch("vlnce_baselines.ss_trainer_ETP.logger.warning") as warn_mock:
            actual_pool_name, actual_env = trainer._activate_train_pool(
                "slow", slow_env
            )

        self.assertEqual(actual_pool_name, "fast")
        self.assertIs(actual_env, fast_env)
        self.assertEqual(slow_env.resume_all_calls, 1)
        self.assertEqual(fast_env.resume_all_calls, 1)
        warn_mock.assert_called_once()
        self.assertIn(
            "fallback preferred_pool=%s actual_pool=%s "
            "train_iteration_counter=%d preferred_num_envs=%s actual_num_envs=%s",
            warn_mock.call_args[0][0],
        )

    def test_activate_train_pool_raises_detailed_error_when_all_pools_empty(self):
        fast_env = _FakeEnvPool(num_envs=0, resumed_num_envs=0)
        slow_env = _FakeEnvPool(num_envs=0, resumed_num_envs=0)
        trainer = _make_trainer(
            static_pool_active=True,
            envs=fast_env,
            fast_envs=fast_env,
            slow_envs=slow_env,
            train_iteration_counter=11,
        )

        with self.assertRaises(RuntimeError) as exc_info:
            trainer._activate_train_pool("slow", slow_env)

        message = str(exc_info.exception)
        self.assertIn("preferred_pool=slow", message)
        self.assertIn("train_iteration_counter=11", message)
        self.assertIn("fast_num_envs=0", message)
        self.assertIn("slow_num_envs=0", message)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
