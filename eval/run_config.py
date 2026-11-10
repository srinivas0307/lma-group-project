"""Print or run the command written in a system config (extension #1; configs/systems/*.yaml).

    .venv/bin/python eval/run_config.py --list
    .venv/bin/python eval/run_config.py configs/systems/stage1_baseline2.yaml            # print the command
    .venv/bin/python eval/run_config.py stage1_baseline2.yaml --run                      # run it
    .venv/bin/python eval/run_config.py stage2_proposed.yaml -- --limit 5 --backend pipeline_check   # extra flags win

Arguments after "--" are appended to the command, so they override the config (argparse keeps the last value).
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from igt.config import all_configs, command, load_system_config  # noqa: E402


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--python", default=sys.executable if Path(sys.executable).exists() else "python")
    args = ap.parse_args(argv)
    if args.list or not args.config:
        for name, c in all_configs().items():
            print(f"{name:<40} {c['system']:<36} {c.get('needs', '-'):<7} {c['runner']}")
        return 0
    cfg = load_system_config(args.config)
    cmd = command(cfg, args.python, extra)
    print(" ".join(shlex.quote(c) for c in cmd))
    if args.run:
        return subprocess.run(cmd, cwd=ROOT).returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
