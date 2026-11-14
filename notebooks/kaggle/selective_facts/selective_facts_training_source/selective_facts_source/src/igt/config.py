"""Config file per system (extension #1, id `cfg`; docs/extensions.md). A config writes down one system's command:
which runner, which arguments. It changes no result: run_config.py builds exactly the command line you would type.

    # configs/systems/stage2_proposed.yaml
    system: proposed
    runner: eval/runner_stage2.py
    needs: gpu                      # cpu | gpu | docker (informational)
    args:
      preset: proposed
      backend: hf
      agent-4bit: true              # true -> --agent-4bit ; false / null -> left out
      seeds: [0]                    # list -> --seeds 0
      stop_after_no_progress: 3     # underscores become dashes: --stop-after-no-progress 3
    positional: []                  # positional arguments, before the flags

    from igt.config import load_system_config, to_argv
    cfg = load_system_config("configs/systems/stage2_proposed.yaml")
    argv = to_argv(cfg, overrides=["--limit", "5"])     # explicit flags come last, so they win (argparse: last wins)
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SYSTEM_CONFIGS = ROOT / "configs" / "systems"
KEYS = {"system", "runner", "args", "positional", "needs", "description", "extension", "notes"}


class ConfigError(ValueError):
    pass


def load_system_config(path):
    path = Path(path)
    if not path.exists() and (SYSTEM_CONFIGS / path).exists():
        path = SYSTEM_CONFIGS / path
    cfg = yaml.safe_load(path.read_text()) or {}
    unknown = set(cfg) - KEYS
    if unknown:
        raise ConfigError(f"{path}: unknown keys {sorted(unknown)}")
    for k in ("system", "runner"):
        if not cfg.get(k):
            raise ConfigError(f"{path}: '{k}' is required")
    if not (ROOT / cfg["runner"]).exists():
        raise ConfigError(f"{path}: runner {cfg['runner']} not found")
    if not isinstance(cfg.get("args") or {}, dict):
        raise ConfigError(f"{path}: 'args' must be a mapping")
    cfg.setdefault("args", {})
    cfg.setdefault("positional", [])
    cfg["_path"] = str(path)
    return cfg


def flag(key):
    return "--" + str(key).replace("_", "-")


def to_argv(cfg, overrides=()):
    """Runner arguments (without the interpreter and runner path)."""
    argv = [str(p) for p in cfg.get("positional") or []]
    for k, v in (cfg.get("args") or {}).items():
        if v is None or v is False:
            continue
        if v is True:
            argv.append(flag(k))
        elif isinstance(v, (list, tuple)):
            argv += [flag(k), *map(str, v)]
        else:
            argv += [flag(k), str(v)]
    return argv + list(overrides)


def command(cfg, python="python", overrides=()):
    return [python, cfg["runner"], *to_argv(cfg, overrides)]


def all_configs():
    return {p.stem: load_system_config(p) for p in sorted(SYSTEM_CONFIGS.glob("*.yaml"))}
