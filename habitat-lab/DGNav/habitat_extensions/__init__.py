def _patch_habitat_action_name_compat() -> None:
    """Allow legacy uppercase HabitatSimActions lookups on Habitat-Lab 3.3."""
    try:
        from habitat.sims.habitat_simulator.actions import HabitatSimActions

        action_cls = HabitatSimActions.__class__
        if getattr(action_cls, "_dgnav_case_compat", False):
            return

        orig_getattr = action_cls.__getattr__
        orig_getitem = action_cls.__getitem__

        def _compat_getattr(self, name):
            try:
                return orig_getattr(self, name)
            except KeyError:
                if isinstance(name, str):
                    lowered = name.lower()
                    if lowered != name:
                        return orig_getattr(self, lowered)
                raise

        def _compat_getitem(self, name):
            try:
                return orig_getitem(self, name)
            except KeyError:
                if isinstance(name, str):
                    lowered = name.lower()
                    if lowered != name:
                        return orig_getitem(self, lowered)
                raise

        action_cls.__getattr__ = _compat_getattr
        action_cls.__getitem__ = _compat_getitem
        action_cls._dgnav_case_compat = True
    except Exception:
        # Best effort: keep imports working even if action API changes.
        pass


_patch_habitat_action_name_compat()

from habitat_extensions import measures, obs_transformers, sensors, nav
from habitat_extensions.config.default import get_extended_config
from habitat_extensions.task import VLNCEDatasetV1
from habitat_extensions.habitat_simulator import Simulator
