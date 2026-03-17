#!/usr/bin/env python3

from habitat.core.registry import registry


def get_env_class(env_name: str):
    env_cls = registry.get_env(env_name)
    if env_cls is None:
        raise RuntimeError(f"Environment {env_name} is not registered")
    return env_cls
