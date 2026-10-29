"""Exit criterion of Phase 3: 100% of red-team samples blocked (static) or contained (resource abuse)."""
from pathlib import Path

import pandas as pd
import pytest
import yaml

from igt.sandbox import check, run, safe_run

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = yaml.safe_load((ROOT / "eval" / "redteam" / "samples.yaml").read_text())["samples"]
DF = pd.DataFrame({"a": range(5)})
LIMITS = dict(timeout_s=10, cpu_s=5, mem_gb=1)


def test_redteam_set_is_well_formed():
    ids = [s["id"] for s in SAMPLES]
    assert len(ids) == len(set(ids)) and len(ids) >= 15
    assert all(s["expect"] in {"blocked", "contained"} for s in SAMPLES)


@pytest.mark.parametrize("s", [s for s in SAMPLES if s["expect"] == "blocked"], ids=lambda s: s["id"])
def test_blocked_by_guard(s):
    r = check(s["code"])
    assert not r.ok, f"{s['id']} was not blocked"


@pytest.mark.slow
@pytest.mark.parametrize("s", [s for s in SAMPLES if s["expect"] == "contained"], ids=lambda s: s["id"])
def test_contained_by_sandbox(s):
    assert check(s["code"]).ok, "resource samples should pass the static check; the sandbox must stop them"
    r = safe_run(s["code"], DF, **LIMITS)
    assert r.status in s["contained_as"], (s["id"], r.status, r.error)
    assert r.duration_s is None or r.duration_s < 30


@pytest.mark.slow
@pytest.mark.parametrize("s", [s for s in SAMPLES if s.get("bypass_check")], ids=lambda s: s["id"])
def test_second_layer_holds_without_guard(s):
    r = run(s["code"], DF, **LIMITS)
    assert r.status in s["contained_as"], (s["id"], r.status, r.error)
    assert r.df_out is None
    if s["category"] == "network" and not r.network_isolated:
        assert "network access is disabled" in r.error, r.error      # refused by the stub, not by accident
