"""
Does *scoped* grounding beat flat grounding?

Flat grounding asks "does this value appear anywhere the agent could have
gotten it?" — one corpus, every source pooled. So a value produced by tool
A grounds an argument to tool B, and a fabricated value survives whenever
it happens to occur somewhere in the transcript.

Scoped grounding asks the narrower question: "did it come from the source
that supplies *this* argument?" Scopes are mined from the train split with
infer_scopes_from_logs — the provenance of each (tool, param) across runs
known to be correct — and enforced on the test split.

Both configurations are gated against the *same* mutations on the same
split, so the comparison is paired: every difference is the scopes.

Same oracle as oracle_eval.py: rule-blind mutations on BFCL multi-turn
ground truth, judged by executing the sequences. One mutator is added —
reuse_value (below) — because the existing four cannot express the failure
mode scopes exist to catch.

Result (seed 0). On the four original mutators scoping changes essentially
nothing, which is the expected finding: change_value writes "DECOY_x" and
n+7, values that occur in *no* source, so flat grounding already catches
them and a narrower question cannot help. The gain is on reuse_value, where
the value is real and only its provenance is wrong:

  scopes            reuse_value recall    FP on valid runs
  none (flat)         0.24 (19/79)           18/86
  mined, support 0.0  0.33 (26/79)           19/86     <- the default
  mined, support 0.25 0.37 (29/79)           23/86
  mined, support 0.5  0.42 (33/79)           30/86
  mined, support 0.9  0.49 (39/79)           31/86

Flat grounding catches a quarter of cross-source confusion. Mined scopes
at the permissive default lift that by ~37% relative for one additional
false positive; tightening the threshold keeps buying recall at a steepening
FP cost — the same correlation-not-causation trade as mined dependencies.
Hand-declared scopes carry no such trade (see examples/scoped_grounding.py);
only the mining is a guess.

Run:
  BFCL_DATA_DIR=... GORILLA_ROOT=... python examples/scoped_grounding_eval.py
  SCOPE_MIN_SUPPORT=0.5 python examples/scoped_grounding_eval.py
"""

import copy
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from oracle_eval import (  # noqa: E402
    CLASS_MODULES, MUTATORS, load, build_param_order, to_params, execute,
    render, mut_drop_arg, mut_change_value, mut_reorder, mut_swap_tool,
)
from grounding_eval import user_text  # noqa: E402
from witt import (  # noqa: E402
    Grounding, Supervisor, generate_rules, infer_scopes_from_logs,
)

MIN_SUPPORT = float(os.environ.get("SCOPE_MIN_SUPPORT", "0.0"))


def mut_reuse_value(seq, rng, order):
    """Replace an argument with a real value used elsewhere in the same
    task — right shape, wrong object.

    The existing change_value mutator writes "DECOY_x" and n+7, values that
    by construction occur nowhere, so any corpus check catches them. This
    one models the fabrication agents actually commit: refunding the order
    you looked at last, messaging the contact from the previous thread. The
    value is real and present in the environment; only its provenance is
    wrong.

    Rule-blind like the others: it picks from the values in this task, with
    no knowledge of sources, scopes, or what the gate will check.
    """
    calls = []
    for i, c in enumerate(seq):
        try:
            name, params = to_params(c, order)
        except Exception:
            continue
        calls.append((i, name, params))

    def usable(v):
        return not isinstance(v, bool) and isinstance(v, (str, int, float))

    by_param, by_type = defaultdict(list), defaultdict(list)
    for _, _, params in calls:
        for k, v in params.items():
            if usable(v):
                by_param[k].append(v)
                by_type[type(v)].append(v)

    slots = [(i, name, k, v) for i, name, params in calls
             for k, v in params.items() if usable(v)]
    rng.shuffle(slots)
    for i, name, k, v in slots:
        # Prefer a different object under the same parameter name; fall
        # back to any other value of the same type in this task.
        alts = ([x for x in by_param[k] if x != v]
                or [x for x in by_type[type(v)] if x != v])
        if not alts:
            continue
        _, _, params = next(c for c in calls if c[0] == i)
        params = dict(params)
        params[k] = rng.choice(alts)
        s = list(seq)
        s[i] = render(name, params)
        return s, i
    return None


ALL_MUTATORS = list(MUTATORS) + ["reuse_value"]


def mutate(m, flat, rng, order, specs, involved):
    if m == "swap_tool":
        return mut_swap_tool(flat, rng, order, specs, involved)
    if m == "reuse_value":
        return mut_reuse_value(flat, rng, order)
    return {"drop_arg": mut_drop_arg, "change_value": mut_change_value,
            "reorder": mut_reorder}[m](flat, rng, order)


def simulators(involved, cfg, registry):
    """Fresh BFCL simulator instances, keyed by the method they own."""
    owner = {}
    for cls in involved:
        inst = registry[cls]()
        if hasattr(inst, "_load_scenario"):
            inst._load_scenario(copy.deepcopy((cfg or {}).get(cls, {})))
        for m in dir(inst):
            if not m.startswith("_") and callable(getattr(inst, m)):
                owner.setdefault(m, inst)
    return owner


def run(seq, engine, grounding, order, involved, cfg, registry,
        stop_on_block=True):
    """Execute a call sequence through the gate, feeding real tool results
    back into the corpus. Returns (index of the first blocked call or None,
    the gate's log)."""
    gate = Supervisor(engine, grounding=grounding)
    owner = simulators(involved, cfg, registry)
    blocked = None
    for idx, call in enumerate(seq):
        try:
            name, params = to_params(call, order)
        except Exception:
            return (idx if blocked is None else blocked), gate.log
        if not gate.check(name, params=params).allowed:
            if stop_on_block:
                return idx, gate.log
            blocked = idx if blocked is None else blocked
        result = None
        inst = owner.get(name)
        if inst is not None:
            try:
                result = getattr(inst, name)(**params)
            except Exception:
                pass
        gate.record_success(name, result=result)
    return blocked, gate.log


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
    train, test = shuffled[:len(shuffled) // 2], shuffled[len(shuffled) // 2:]

    def tools_for(rec):
        return [t for cls in rec["involved_classes"]
                for t in specs.get(cls, {}).values()]

    def fresh(rec, scopes):
        tools = tools_for(rec)
        g = Grounding(user_text=user_text(rec), tool_specs=tools,
                      config=rec.get("initial_config"), mode="strict",
                      scopes=scopes)
        return generate_rules(tools, auto_detect_destructive=False), g

    # ── Mine scopes from valid runs on the train split ───────────────
    logs = []
    for cid in train:
        rec = base[cid]
        flat = [c for turn in ans[cid]["ground_truth"] for c in turn]
        if not flat:
            continue
        eng, g = fresh(rec, None)
        # Don't stop on a block: we want the whole valid trace's provenance.
        _, log = run(flat, eng, g, order, rec["involved_classes"],
                     rec.get("initial_config"), registry, stop_on_block=False)
        logs.append(log)
    scopes = infer_scopes_from_logs(logs, min_support=MIN_SUPPORT)
    n_scoped = sum(len(p) for p in scopes.values())
    print(f"mined {n_scoped} param scopes over {len(scopes)} tools "
          f"from {len(logs)} valid train runs (min_support={MIN_SUPPORT})\n")

    # ── Evaluate both configurations on the same test mutations ──────
    CONFIGS = {"flat": None, "scoped": scopes}
    cm = {c: {m: {"TP": 0, "FN": 0, "FP": 0, "TN": 0} for m in ALL_MUTATORS}
          for c in CONFIGS}
    fp_gold = {c: 0 for c in CONFIGS}
    checked = unstable = 0

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

        for name, sc in CONFIGS.items():
            eng, g = fresh(rec, sc)
            if run(flat, eng, g, order, involved, cfg, registry)[0] is not None:
                fp_gold[name] += 1

        for m in ALL_MUTATORS:
            res = mutate(m, flat, rng, order, specs, involved)
            if res is None:
                continue
            mseq, _ = res
            broke = execute(mseq, involved, cfg, registry, order) != gt
            for name, sc in CONFIGS.items():
                eng, g = fresh(rec, sc)
                flagged = run(mseq, eng, g, order, involved, cfg,
                              registry)[0] is not None
                key = ("TP" if broke and flagged else "FN" if broke else
                       "FP" if flagged else "TN")
                cm[name][m][key] += 1

    for name in CONFIGS:
        print(f"### spec + grounding(strict, {name})   "
              f"FP on ground truth: {fp_gold[name]}/{checked} "
              f"(skipped {unstable} nondeterministic)")
        tot = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
        print(f"  {'mutation':<14}{'broke':>7}{'caught':>8}"
              f"{'recall':>9}{'FP':>5}")
        for m in ALL_MUTATORS:
            c = cm[name][m]
            for k in tot:
                tot[k] += c[k]
            broke = c["TP"] + c["FN"]
            rr = c["TP"] / broke if broke else float("nan")
            print(f"  {m:<14}{broke:>7}{c['TP']:>8}{rr:>9.2f}{c['FP']:>5}")
        broke = tot["TP"] + tot["FN"]
        print(f"  overall: {tot['TP']}/{broke} = {tot['TP']/broke:.1%}   "
              f"FP on harmless: {tot['FP']}/{tot['FP'] + tot['TN']}\n")


if __name__ == "__main__":
    main()
