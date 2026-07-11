"""
Prototype: can mined ARGUMENT BINDINGS close the change_value gap?

Same protocol as oracle_eval.py (independent execution oracle, rule-blind
mutations, train/test split) with two additions:

  * infer_bindings_from_traces — mines (tool.param must equal an earlier
    dep.dep_param value) relations from the train half only. Requires
    value diversity (>=2 distinct values seen) so constants can't bind
    by coincidence, and picks the strongest dep per (tool, param).

  * a 'spec+deps+bindings' config enforced through witt's existing
    Supervisor binding mechanism (no engine changes needed).
"""

import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from oracle_eval import (  # noqa: E402
    CLASS_MODULES, MUTATORS, load, build_param_order, to_params,
    execute, mut_drop_arg, mut_change_value, mut_reorder, mut_swap_tool,
)
from witt import Supervisor, generate_rules, infer_dependencies_from_traces  # noqa: E402


def infer_bindings_from_traces(param_traces, min_support=0.95, min_occurrences=3):
    """Mine argument-bound dependencies from (tool, params) traces.

    A candidate binding (tool.param -> dep.dep_param) is kept when, in at
    least `min_support` of tool.param's occurrences, some EARLIER dep call
    in the same trace had dep_param equal to the value. Constants are
    excluded by requiring >=2 distinct observed values.
    """
    occ = defaultdict(int)
    match = defaultdict(lambda: defaultdict(int))
    values = defaultdict(set)

    for trace in param_traces:
        seen = []
        for tool, params in trace:
            for p, v in params.items():
                if v is None:
                    continue
                occ[(tool, p)] += 1
                values[(tool, p)].add(repr(v))
                found = set()
                for dtool, dparams in seen:
                    if dtool == tool:
                        continue
                    for dp, dv in dparams.items():
                        if dv == v or str(dv) == str(v):
                            found.add((dtool, dp))
                for f in found:
                    match[(tool, p)][f] += 1
            seen.append((tool, params))

    bindings = defaultdict(list)
    for (tool, p), deps in match.items():
        if occ[(tool, p)] < min_occurrences or len(values[(tool, p)]) < 2:
            continue
        best, best_count = None, 0
        for (dtool, dp), cnt in deps.items():
            if cnt / occ[(tool, p)] >= min_support and cnt > best_count:
                best, best_count = (dtool, dp), cnt
        if best:
            bindings[tool].append({"tool": best[0], "param": p,
                                   "dep_param": best[1]})
    return dict(bindings)


def gate_sequence(seq, engine, order):
    gate = Supervisor(engine)
    for idx, call in enumerate(seq):
        try:
            name, params = to_params(call, order)
        except Exception:
            return idx
        if not gate.check(name, params=params).allowed:
            return idx
        gate.record_success(name)
    return None


def evaluate(gorilla_root, data_dir, seed=0):
    registry, specs, base, ans = load(gorilla_root, data_dir)
    order = build_param_order(specs)
    rng = random.Random(seed)

    cases = [cid for cid, r in base.items()
             if set(r["involved_classes"]) <= set(CLASS_MODULES)]
    shuffled = list(cases)
    rng.shuffle(shuffled)
    split = len(shuffled) // 2
    train, test = shuffled[:split], shuffled[split:]

    def parsed_trace(cid):
        return [to_params(c, order)
                for turn in ans[cid]["ground_truth"] for c in turn]

    train_traces = [parsed_trace(c) for c in train]
    mined_deps = infer_dependencies_from_traces(
        [[t for t, _ in tr] for tr in train_traces], min_support=0.9)
    strict_deps = infer_dependencies_from_traces(
        [[t for t, _ in tr] for tr in train_traces], min_support=1.0)
    mined_binds = infer_bindings_from_traces(train_traces)

    def build(involved, deps=None, binds=None):
        tools = [t for cls in involved for t in specs.get(cls, {}).values()]
        names = {t["name"] for t in tools}
        d = None
        if deps:
            d = {k: [x for x in v if x in names]
                 for k, v in deps.items() if k in names}
            d = {k: v for k, v in d.items() if v}
        b = None
        if binds:
            b = {k: [e for e in v if e["tool"] in names]
                 for k, v in binds.items() if k in names}
            b = {k: v for k, v in b.items() if v}
        return generate_rules(tools, dependencies=d, bindings=b,
                              auto_detect_destructive=False)

    configs = {
        "spec-only":            lambda inv: build(inv),
        "spec+deps(0.9)":       lambda inv: build(inv, deps=mined_deps),
        "spec+deps(1.0)":       lambda inv: build(inv, deps=strict_deps),
        "spec+bindings":        lambda inv: build(inv, binds=mined_binds),
        "spec+deps1.0+binds":   lambda inv: build(inv, deps=strict_deps,
                                                  binds=mined_binds),
    }
    cm = {n: {m: {"TP": 0, "FN": 0, "FP": 0, "TN": 0} for m in MUTATORS}
          for n in configs}
    fp_gold = {n: 0 for n in configs}
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

        engines = {n: f(involved) for n, f in configs.items()}
        for n, eng in engines.items():
            if gate_sequence(flat, eng, order) is not None:
                fp_gold[n] += 1

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
            for n, eng in engines.items():
                flagged = gate_sequence(mseq, eng, order) is not None
                key = ("TP" if broke and flagged else "FN" if broke else
                       "FP" if flagged else "TN")
                cm[n][m][key] += 1

    print(f"Test cases: {checked} (skipped {unstable} nondeterministic); "
          f"train: {len(train)}")
    print(f"Mined: {sum(len(v) for v in mined_deps.values())} deps@0.9, "
          f"{sum(len(v) for v in strict_deps.values())} deps@1.0, "
          f"{sum(len(v) for v in mined_binds.values())} bindings\n")
    for n in configs:
        tot = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
        print(f"### {n}   (FP on ground truth: {fp_gold[n]}/{checked})")
        print(f"  {'mutation':<14}{'broke':>7}{'caught':>8}{'recall':>9}{'FP':>5}")
        for m in MUTATORS:
            c = cm[n][m]
            for k in tot:
                tot[k] += c[k]
            broke = c["TP"] + c["FN"]
            rr = c["TP"] / broke if broke else float("nan")
            print(f"  {m:<14}{broke:>7}{c['TP']:>8}{rr:>9.2f}{c['FP']:>5}")
        broke = tot["TP"] + tot["FN"]
        print(f"  overall: {tot['TP']}/{broke} = {tot['TP']/broke:.1%}   "
              f"FP on harmless: {tot['FP']}/{tot['FP'] + tot['TN']}\n")


if __name__ == "__main__":
    evaluate(
        os.environ.get("GORILLA_ROOT",
                       "gorilla/berkeley-function-call-leaderboard"),
        os.environ.get("BFCL_DATA_DIR",
                       "gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"),
    )


def grounding_eval(gorilla_root, data_dir, seed=0):
    """Config: every string param must be grounded in the user request or
    a prior tool return (substring match, case-insensitive). No mining."""
    registry, specs, base, ans = load(gorilla_root, data_dir)
    order = build_param_order(specs)
    rng = random.Random(seed)
    cases = [cid for cid, r in base.items()
             if set(r["involved_classes"]) <= set(CLASS_MODULES)]
    shuffled = list(cases); rng.shuffle(shuffled)
    test = shuffled[len(shuffled) // 2:]

    def user_text(rec):
        out = []
        for turn in rec.get("question", []):
            for msg in (turn if isinstance(turn, list) else [turn]):
                if isinstance(msg, dict):
                    out.append(str(msg.get("content", "")))
        return " ".join(out)

    def gate_grounded(seq, engine, corpus0, involved, cfg, registry):
        gate = Supervisor(engine)
        corpus = corpus0.lower()
        insts, owner = {}, {}
        for cls in involved:
            inst = registry[cls]()
            if hasattr(inst, "_load_scenario"):
                import copy as _c
                inst._load_scenario(_c.deepcopy((cfg or {}).get(cls, {})))
            insts[cls] = inst
            for m in dir(inst):
                if not m.startswith("_") and callable(getattr(inst, m)):
                    owner.setdefault(m, inst)
        for idx, call in enumerate(seq):
            try:
                name, params = to_params(call, order)
            except Exception:
                return idx
            for v in params.values():
                if isinstance(v, str) and len(v) >= 3 and v.lower() not in corpus:
                    return idx
            if not gate.check(name, params=params).allowed:
                return idx
            gate.record_success(name)
            inst = owner.get(name)
            if inst is not None:
                try:
                    corpus += " " + repr(getattr(inst, name)(**params)).lower()
                except Exception:
                    pass
        return None

    cm = {m: {"TP": 0, "FN": 0, "FP": 0, "TN": 0} for m in MUTATORS}
    fp_gold = checked = unstable = 0
    for cid in test:
        rec = base[cid]; involved = rec["involved_classes"]
        cfg = rec.get("initial_config")
        flat = [c for turn in ans[cid]["ground_truth"] for c in turn]
        if not flat:
            continue
        gt = execute(flat, involved, cfg, registry, order)
        if gt != execute(flat, involved, cfg, registry, order):
            unstable += 1; continue
        checked += 1
        tools = [t for cls in involved for t in specs.get(cls, {}).values()]
        eng = generate_rules(tools, auto_detect_destructive=False)
        corpus0 = user_text(rec)
        if gate_grounded(flat, eng, corpus0, involved, cfg, registry) is not None:
            fp_gold += 1
        for m in MUTATORS:
            if m == "swap_tool":
                res = mut_swap_tool(flat, rng, order, specs, involved)
            else:
                res = {"drop_arg": mut_drop_arg, "change_value": mut_change_value,
                       "reorder": mut_reorder}[m](flat, rng, order)
            if res is None:
                continue
            mseq, _ = res
            broke = execute(mseq, involved, cfg, registry, order) != gt
            flagged = gate_grounded(mseq, eng, corpus0, involved, cfg,
                                    registry) is not None
            key = ("TP" if broke and flagged else "FN" if broke else
                   "FP" if flagged else "TN")
            cm[m][key] += 1

    print(f"### spec+grounding   (FP on ground truth: {fp_gold}/{checked})")
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
