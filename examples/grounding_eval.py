"""
Measure the Grounding feature against the same independent oracle as
oracle_eval.py: rule-blind mutations on BFCL multi-turn ground truth,
judged by actually executing the sequences.

Result (seed 0): value-mutation recall 0% -> ~58% (identifier mode),
overall recall 42.6% -> ~60%. Known FP class: semantic translations
(city -> airport code, prose date -> ISO date), ~13% of valid runs in
strict mode — which is why the default mode is "warn".

Run:
  BFCL_DATA_DIR=... GORILLA_ROOT=... python examples/grounding_eval.py
"""

import copy
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from oracle_eval import (  # noqa: E402
    CLASS_MODULES, MUTATORS, load, build_param_order, to_params, execute,
    mut_drop_arg, mut_change_value, mut_reorder, mut_swap_tool,
)
from witt import Grounding, Supervisor, generate_rules  # noqa: E402


def user_text(rec):
    out = []
    for turn in rec.get("question", []):
        for msg in (turn if isinstance(turn, list) else [turn]):
            if isinstance(msg, dict):
                out.append(str(msg.get("content", "")))
    return " ".join(out)


def gate_sequence(seq, engine, grounding, order, involved, cfg, registry):
    """Gate with grounding in strict mode, feeding real tool results back
    into the corpus via record_success(result=...)."""
    gate = Supervisor(engine, grounding=grounding)
    owner = {}
    for cls in involved:
        inst = registry[cls]()
        if hasattr(inst, "_load_scenario"):
            inst._load_scenario(copy.deepcopy((cfg or {}).get(cls, {})))
        for m in dir(inst):
            if not m.startswith("_") and callable(getattr(inst, m)):
                owner.setdefault(m, inst)
    for idx, call in enumerate(seq):
        try:
            name, params = to_params(call, order)
        except Exception:
            return idx
        if not gate.check(name, params=params).allowed:
            return idx
        result = None
        inst = owner.get(name)
        if inst is not None:
            try:
                result = getattr(inst, name)(**params)
            except Exception:
                pass
        gate.record_success(name, result=result)
    return None


def main():
    gorilla_root = os.environ.get(
        "GORILLA_ROOT", "gorilla/berkeley-function-call-leaderboard")
    data_dir = os.environ.get(
        "BFCL_DATA_DIR", os.path.join(gorilla_root, "bfcl_eval/data"))
    registry, specs, base, ans = load(gorilla_root, data_dir)
    order = build_param_order(specs)
    rng = random.Random(0)
    cases = [cid for cid, r in base.items()
             if set(r["involved_classes"]) <= set(CLASS_MODULES)]
    shuffled = list(cases)
    rng.shuffle(shuffled)
    test = shuffled[len(shuffled) // 2:]

    cm = {m: {"TP": 0, "FN": 0, "FP": 0, "TN": 0} for m in MUTATORS}
    fp_gold = checked = unstable = 0

    for cid in test:
        rec = base[cid]
        involved = rec["involved_classes"]
        cfg = rec.get("initial_config")
        flat = [c for turn in ans[cid]["ground_truth"] for c in turn]
        if not flat:
            continue
        gt = execute(flat, involved, cfg, registry, order)
        if gt != execute(flat, involved, cfg, registry, order):
            unstable += 1
            continue
        checked += 1
        tools = [t for cls in involved for t in specs.get(cls, {}).values()]

        def fresh_gate():
            g = Grounding(user_text=user_text(rec), tool_specs=tools,
                          config=cfg, mode="strict")
            return generate_rules(tools, auto_detect_destructive=False), g

        eng, g = fresh_gate()
        if gate_sequence(flat, eng, g, order, involved, cfg,
                         registry) is not None:
            fp_gold += 1

        for m in MUTATORS:
            if m == "swap_tool":
                res = mut_swap_tool(flat, rng, order, specs, involved)
            else:
                res = {"drop_arg": mut_drop_arg,
                       "change_value": mut_change_value,
                       "reorder": mut_reorder}[m](flat, rng, order)
            if res is None:
                continue
            mseq, _ = res
            broke = execute(mseq, involved, cfg, registry, order) != gt
            eng, g = fresh_gate()
            flagged = gate_sequence(mseq, eng, g, order, involved, cfg,
                                    registry) is not None
            key = ("TP" if broke and flagged else "FN" if broke else
                   "FP" if flagged else "TN")
            cm[m][key] += 1

    print(f"### spec + grounding(strict)   "
          f"FP on ground truth: {fp_gold}/{checked} "
          f"(skipped {unstable} nondeterministic)")
    tot = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
    print(f"  {'mutation':<14}{'broke':>7}{'caught':>8}{'recall':>9}{'FP':>5}")
    for m in MUTATORS:
        c = cm[m]
        for k in tot:
            tot[k] += c[k]
        broke = c["TP"] + c["FN"]
        rr = c["TP"] / broke if broke else float("nan")
        print(f"  {m:<14}{broke:>7}{c['TP']:>8}{rr:>9.2f}{c['FP']:>5}")
    broke = tot["TP"] + tot["FN"]
    print(f"  overall: {tot['TP']}/{broke} = {tot['TP']/broke:.1%}   "
          f"FP on harmless: {tot['FP']}/{tot['FP'] + tot['TN']}")


if __name__ == "__main__":
    main()
