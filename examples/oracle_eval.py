"""
Claim B, definitively: does witt catch the errors that actually break
a task — judged by an oracle it has no access to?

The current README benchmark injects errors by falsifying the exact premise
a rule checks (circular) and never measures what the tool *misses*. This
harness fixes both flaws:

  * Independent oracle. BFCL multi-turn ships executable stateful classes
    (a real file system, trading bot, etc.). We execute each call sequence
    and label it by its actual effect — final state + return values —
    compared to the ground-truth sequence. witt never sees this.

  * Rule-blind adversary. Mutations (drop an argument, change a value,
    reorder calls, swap a tool) are applied to valid ground-truth
    sequences by a generator that knows nothing about witt's rules.

  * The false-negative cell. We report the full confusion matrix, so the
    errors witt MISSES are counted — per mutation category.

The engine is built purely mechanically from the tool specs
(generate_rules) and frozen. No hand-tuning against the mutations.

Run:
  BFCL_DATA_DIR=gorilla/berkeley-function-call-leaderboard/bfcl_eval/data \
  GORILLA_ROOT=gorilla/berkeley-function-call-leaderboard \
  python examples/oracle_eval.py
"""

import ast
import copy
import importlib
import json
import os
import random
import sys

from witt import Supervisor, generate_rules, infer_dependencies_from_traces

CLASS_MODULES = {
    "GorillaFileSystem": "gorilla_file_system",
    "MathAPI": "math_api",
    "MessageAPI": "message_api",
    "TwitterAPI": "posting_api",
    "TicketAPI": "ticket_api",
    "TradingBot": "trading_bot",
    "TravelAPI": "travel_booking",
    "VehicleControlAPI": "vehicle_control",
}
BACKEND = "bfcl_eval.eval_checker.multi_turn_eval.func_source_code"


def load(gorilla_root, data_dir):
    sys.path.insert(0, gorilla_root)
    registry, specs = {}, {}
    for cls, mod in CLASS_MODULES.items():
        registry[cls] = getattr(importlib.import_module(f"{BACKEND}.{mod}"), cls)
        specs[cls] = {}
        path = os.path.join(data_dir, "multi_turn_func_doc", f"{mod}.json")
        if os.path.exists(path):
            for line in open(path):
                t = json.loads(line)
                specs[cls][t["name"]] = t
    base = {json.loads(l)["id"]: json.loads(l)
            for l in open(os.path.join(data_dir, "BFCL_v4_multi_turn_base.json"))}
    ans = {json.loads(l)["id"]: json.loads(l) for l in
           open(os.path.join(data_dir, "possible_answer/BFCL_v4_multi_turn_base.json"))}
    return registry, specs, base, ans


def build_param_order(specs):
    """tool name -> ordered list of parameter names (for binding positional
    args). BFCL calls mix positional and keyword arguments."""
    order = {}
    for cls in specs.values():
        for name, t in cls.items():
            order[name] = list((t.get("parameters", {}) or {})
                               .get("properties", {}).keys())
    return order


def to_params(call, order):
    """Parse a call and bind BOTH positional and keyword args to parameter
    names, so a positionally-passed required arg is not seen as missing."""
    node = ast.parse(call.strip(), mode="eval").body
    name = node.func.id
    params = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
    names = order.get(name, [])
    for i, a in enumerate(node.args):
        params[names[i] if i < len(names) else f"_pos{i}"] = ast.literal_eval(a)
    return name, params


def render(name, params):
    return f"{name}(" + ", ".join(f"{k}={v!r}" for k, v in params.items()) + ")"


def execute(seq, involved, cfg, registry, order):
    """Run a sequence against fresh instances; return (returns, state sig)."""
    insts, owner = {}, {}
    for cls in involved:
        inst = registry[cls]()
        if hasattr(inst, "_load_scenario"):
            inst._load_scenario(copy.deepcopy((cfg or {}).get(cls, {})))
        insts[cls] = inst
        for m in dir(inst):
            if not m.startswith("_") and callable(getattr(inst, m)):
                owner.setdefault(m, inst)
    returns = []
    for call in seq:
        try:
            name, params = to_params(call, order)
        except Exception as ex:
            returns.append(f"PARSE_ERR:{type(ex).__name__}")
            continue
        inst = owner.get(name)
        if inst is None:
            returns.append(f"NO_METHOD:{name}")
            continue
        try:
            returns.append(repr(getattr(inst, name)(**params)))
        except Exception as ex:
            returns.append(f"ERR:{type(ex).__name__}")
    state = "||".join(f"{c}={vars(i)!r}" for c, i in sorted(insts.items()))
    return returns, state


def gate_sequence(seq, engine, order):
    """Return the index witt blocks at, or None if it allows all."""
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


# ── Mutations — applied blind to the rules ───────────────────────────
def mut_drop_arg(seq, rng, order):
    idxs = [i for i, c in enumerate(seq) if to_params(c, order)[1]]
    if not idxs:
        return None
    i = rng.choice(idxs)
    name, params = to_params(seq[i], order)
    params.pop(rng.choice(list(params)))
    s = list(seq); s[i] = render(name, params)
    return s, i


def mut_change_value(seq, rng, order):
    idxs = [i for i, c in enumerate(seq) if to_params(c, order)[1]]
    if not idxs:
        return None
    i = rng.choice(idxs)
    name, params = to_params(seq[i], order)
    k = rng.choice(list(params))
    v = params[k]
    if isinstance(v, bool):
        params[k] = not v
    elif isinstance(v, (int, float)):
        params[k] = v + 7
    elif isinstance(v, str):
        params[k] = "DECOY_" + v
    else:
        return None
    s = list(seq); s[i] = render(name, params)
    return s, i


def mut_reorder(seq, rng, order):
    if len(seq) < 2:
        return None
    i = rng.randint(0, len(seq) - 2)
    s = list(seq); s[i], s[i + 1] = s[i + 1], s[i]
    return s, i


def mut_swap_tool(seq, rng, order, specs, involved):
    pool = [n for cls in involved for n in specs.get(cls, {})]
    idxs = list(range(len(seq)))
    rng.shuffle(idxs)
    for i in idxs:
        name, params = to_params(seq[i], order)
        alts = [n for n in pool if n != name]
        if alts:
            s = list(seq); s[i] = render(rng.choice(alts), params)
            return s, i
    return None


MUTATORS = ["drop_arg", "change_value", "reorder", "swap_tool"]


def evaluate(gorilla_root, data_dir, seed=0, limit=None):
    """Run the full oracle evaluation and return the raw results, so both
    the CLI report and the test suite can consume them."""
    registry, specs, base, ans = load(gorilla_root, data_dir)
    order = build_param_order(specs)
    rng = random.Random(seed)

    supported = set(CLASS_MODULES)
    cases = [cid for cid, r in base.items()
             if set(r["involved_classes"]) <= supported]

    # Train/test split — dependencies are mined ONLY from the train half,
    # so the ordering rules can never have seen the test mutations.
    shuffled = list(cases)
    rng.shuffle(shuffled)
    split = len(shuffled) // 2
    train, test = shuffled[:split], shuffled[split:]
    if limit:
        test = test[:limit]

    def names_of(cid):
        return [to_params(c, order)[0]
                for turn in ans[cid]["ground_truth"] for c in turn]

    mined_deps = infer_dependencies_from_traces(
        [names_of(c) for c in train], min_support=0.9)

    def build_engine(involved, with_deps):
        tools = [t for cls in involved for t in specs.get(cls, {}).values()]
        tool_names = {t["name"] for t in tools}
        deps = None
        if with_deps:
            deps = {k: [d for d in v if d in tool_names]
                    for k, v in mined_deps.items() if k in tool_names}
            deps = {k: v for k, v in deps.items() if v}
        return generate_rules(tools, dependencies=deps, auto_detect_destructive=False)

    configs = {"spec-only": False, "spec+mined-deps": True}
    cm = {name: {m: {"TP": 0, "FN": 0, "FP": 0, "TN": 0} for m in MUTATORS}
          for name in configs}
    fp_gold = {name: 0 for name in configs}
    checked = unstable = 0

    for cid in test:
        rec = base[cid]
        involved = rec["involved_classes"]
        cfg = rec.get("initial_config")
        flat = [c for turn in ans[cid]["ground_truth"] for c in turn]
        if not flat:
            continue

        # Oracle baseline. Skip cases whose state isn't stable across
        # identical runs (random simulators) — can't be judged.
        gt = execute(flat, involved, cfg, registry, order)
        if gt != execute(flat, involved, cfg, registry, order):
            unstable += 1
            continue
        checked += 1

        engines = {name: build_engine(involved, wd) for name, wd in configs.items()}
        for name, eng in engines.items():
            if gate_sequence(flat, eng, order) is not None:
                fp_gold[name] += 1

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
            for name, eng in engines.items():
                flagged = gate_sequence(mseq, eng, order) is not None
                key = ("TP" if broke and flagged else "FN" if broke else
                       "FP" if flagged else "TN")
                cm[name][m][key] += 1

    return {"checked": checked, "unstable": unstable, "train": len(train),
            "mined_deps": mined_deps, "fp_gold": fp_gold, "cm": cm,
            "configs": list(configs)}


def main():
    gorilla_root = os.environ.get("GORILLA_ROOT",
        "gorilla/berkeley-function-call-leaderboard")
    data_dir = os.environ.get("BFCL_DATA_DIR",
        os.path.join(gorilla_root, "bfcl_eval/data"))
    r = evaluate(gorilla_root, data_dir)
    checked, unstable, cm, fp_gold = r["checked"], r["unstable"], r["cm"], r["fp_gold"]

    # ── Report ──
    print(f"Test cases evaluated: {checked}  (skipped {unstable} nondeterministic)")
    print(f"Dependencies mined from {r['train']} held-out train cases: "
          f"{sum(len(v) for v in r['mined_deps'].values())} ordering constraints\n")

    for name in r["configs"]:
        print(f"### {name}")
        print(f"  false positives on ground truth: {fp_gold[name]}/{checked}")
        tot = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
        print(f"  {'mutation':<14}{'broke':>7}{'caught':>8}{'missed':>8}"
              f"{'recall':>9}{'FP':>5}")
        for m in MUTATORS:
            c = cm[name][m]
            for k in tot:
                tot[k] += c[k]
            broke = c["TP"] + c["FN"]
            r = c["TP"] / broke if broke else float("nan")
            print(f"  {m:<14}{broke:>7}{c['TP']:>8}{c['FN']:>8}{r:>9.2f}{c['FP']:>5}")
        broke = tot["TP"] + tot["FN"]
        print(f"  overall recall on task-breaking mutations: "
              f"{tot['TP']}/{broke} = {tot['TP']/broke:.1%}   "
              f"FP on harmless: {tot['FP']}/{tot['FP']+tot['TN']}\n")

    print("Read per-category recall as the competence boundary: structural")
    print("errors (drop_arg) are caught near-perfectly and at zero false")
    print("positives; value errors are invisible to presence-checking; and")
    print("ordering recall is exactly as good as the dependency rules you give it.")


if __name__ == "__main__":
    main()
