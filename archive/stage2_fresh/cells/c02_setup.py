# ===== Setup: packages, imports, input discovery =====
import os, sys, re, json, math, time, glob, shutil, zipfile, hashlib, difflib, textwrap, traceback, subprocess, pickle, ast
from pathlib import Path
import numpy as np
import pandas as pd

ON_KAGGLE = Path("/kaggle/input").exists()

def _ensure_packages():
    """Install the GPU-side packages only when they are missing (Kaggle images differ between versions)."""
    need = []
    for mod, pip_name in [("peft", "peft"), ("bitsandbytes", "bitsandbytes"), ("accelerate", "accelerate"), ("yaml", "pyyaml")]:
        try:
            __import__(mod)
        except Exception:
            need.append(pip_name)
    if need:
        print("pip install:", need)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=True)

if RUN_MODELS:
    _ensure_packages()
import yaml


def find_input_root(marker_relpath, search_roots):
    """Return the first directory that contains marker_relpath (searched recursively, zips are extracted)."""
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        if (root / marker_relpath).exists():
            return root
        for hit in sorted(root.rglob(Path(marker_relpath).name)):
            cand = hit
            for _ in Path(marker_relpath).parts:
                cand = cand.parent
            if (cand / marker_relpath).exists():
                return cand
        for z in sorted(root.rglob("*.zip")):
            dest = WORK_DIR / "_unzipped" / z.stem
            if not dest.exists():
                try:
                    with zipfile.ZipFile(z) as zf:
                        if not any(n.endswith(Path(marker_relpath).name) for n in zf.namelist()):
                            continue
                        zf.extractall(dest)
                except zipfile.BadZipFile:
                    continue
            found = find_input_root(marker_relpath, [dest])
            if found:
                return found
    return None


def find_adapter(protocol, search_roots):
    """Find an adapter folder whose igt_prompt_version.txt equals the expected protocol string."""
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for ver in sorted(root.rglob("igt_prompt_version.txt")):
            if ver.read_text().strip() == protocol and (ver.parent / "adapter_config.json").exists():
                return ver.parent
        for z in sorted(root.rglob("*.zip")):
            dest = WORK_DIR / "_unzipped" / z.stem
            if not dest.exists():
                try:
                    with zipfile.ZipFile(z) as zf:
                        if not any(n.endswith("adapter/igt_prompt_version.txt") for n in zf.namelist()):
                            continue
                        zf.extractall(dest, members=[n for n in zf.namelist() if n.startswith("adapter/")])
                except zipfile.BadZipFile:
                    continue
            hit = find_adapter(protocol, [dest])
            if hit:
                return hit
    return None


WORK_DIR = Path(WORK_DIR)
WORK_DIR.mkdir(parents=True, exist_ok=True)
SEARCH_ROOTS = [p for p in INPUT_SEARCH_ROOTS if Path(p).exists()]

DATA_ROOT = Path(DATA_ROOT) if DATA_ROOT else find_input_root("benchmark/tasks/INDEX.md", SEARCH_ROOTS)
assert DATA_ROOT is not None, "Benchmark data not found: attach the stage2_fresh_data dataset (benchmark/tasks + data/*.csv)."
print("DATA_ROOT     :", DATA_ROOT)

if RUN_MODELS:
    PARSER_ADAPTER = Path(PARSER_ADAPTER) if PARSER_ADAPTER else find_adapter(PARSER_PROTOCOL, SEARCH_ROOTS)
    assert PARSER_ADAPTER is not None, f"No adapter with protocol {PARSER_PROTOCOL!r} found under {SEARCH_ROOTS}."
    print("PARSER_ADAPTER:", PARSER_ADAPTER)
    cfg = json.loads((PARSER_ADAPTER / "adapter_config.json").read_text())
    assert cfg["base_model_name_or_path"] == PARSER_BASE, (cfg["base_model_name_or_path"], PARSER_BASE)
