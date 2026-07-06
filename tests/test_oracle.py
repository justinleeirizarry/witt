"""Claim B, gated by an independent execution oracle.

Runs the mutation harness (examples/oracle_eval.py) against BFCL's real
stateful classes and asserts the invariants that must hold — not the
findings. The findings (value/ordering blindness, dependency-induced
false positives) are printed by the example script; here we lock in:

  * spec-derived rules never block a valid ground-truth sequence
    (the "0 false positives by construction" claim, on an oracle the
    tool has no access to), and
  * they catch structural (missing-argument) errors with high recall.

Skips unless the gorilla repo (BFCL execution backend + mpmath) is
importable and BFCL_DATA_DIR / GORILLA_ROOT are set.
"""

import importlib.util
import os

import pytest

GORILLA_ROOT = os.environ.get(
    "GORILLA_ROOT", "gorilla/berkeley-function-call-leaderboard")
DATA_DIR = os.environ.get("BFCL_DATA_DIR", os.path.join(GORILLA_ROOT, "bfcl_eval/data"))


def _load_harness():
    here = os.path.dirname(__file__)
    path = os.path.normpath(os.path.join(here, "..", "examples", "oracle_eval.py"))
    if not os.path.exists(path):
        pytest.skip("oracle_eval.py not found")
    spec = importlib.util.spec_from_file_location("oracle_eval", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as ex:  # pragma: no cover
        pytest.skip(f"harness import failed: {ex}")
    return mod


@pytest.fixture(scope="module")
def results():
    if not os.path.isdir(os.path.join(DATA_DIR, "multi_turn_func_doc")):
        pytest.skip("BFCL data not present")
    oe = _load_harness()
    try:
        oe.load(GORILLA_ROOT, DATA_DIR)  # probe: backend + mpmath importable?
    except Exception as ex:
        pytest.skip(f"BFCL execution backend unavailable: {ex}")
    return oe.evaluate(GORILLA_ROOT, DATA_DIR, limit=50)


def _recall(cm):
    tp, fn = cm["TP"], cm["FN"]
    return tp / (tp + fn) if (tp + fn) else float("nan")


def test_spec_rules_have_zero_false_positives_on_ground_truth(results):
    # Every required-param present in a valid sequence → never blocked.
    # This is the guarantee, and it's exact, so we assert it exactly.
    assert results["fp_gold"]["spec-only"] == 0


def test_structural_recall_far_exceeds_semantic_recall(results):
    # The competence boundary: presence-checking catches missing arguments
    # but is blind to wrong-but-valid values. Absolute recall wobbles with
    # the stochastic simulators; this qualitative gap does not.
    spec = results["cm"]["spec-only"]
    assert _recall(spec["drop_arg"]) >= _recall(spec["change_value"]) + 0.3
    assert _recall(spec["drop_arg"]) >= 0.6


def test_oracle_actually_labelled_broken_mutations(results):
    # Sanity: the oracle is doing its job (mutations do break tasks).
    dropped = results["cm"]["spec-only"]["drop_arg"]
    assert dropped["TP"] + dropped["FN"] > 0
