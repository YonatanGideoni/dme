"""
Probability of improvement of DME over IID RS (BoN) and SCS vs program budget N
Run from experiments/math_bounds: uv run python -m dme_math.prob_improvement
"""
from __future__ import annotations

import argparse
import json
import math
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammaln

from dme_math.paths import PLOTS_DIR, RUNS_DIR
from dme_math.problems.problem_utils import ProblemLoader

MODEL_SLUG = "openai-gpt-oss-20b"
# defaults: the paper's runs, see the README
DME_PATTERN = "dme/results_{problem}_dme_beta1e+06_lencorroff_nchains*_bs*_{model}.json"
BON_PATTERN = "results_{problem}_*_{model}.json"
SCS_PATTERN = "scs/trial*/results_{problem}_scs_numalgs*_ngens*_k*_{model}.json"

PROBLEM_LABELS = {
    "pack_circ26": "Circle packing",
    "maxmin_dist_ratio": "Max–min dist. ratio",
    "kissing_11d": "Kissing number in 11D",
    "heilbronn_triangles": "Heilbronn triangles",
    "uncert_ineq": "Uncertainty ineq.",
    "first_autocorr": "First autocorr. ineq.",
    "sec_autocorr": "Second autocorr. ineq.",
    "sumdiff_sets": "Sums/differences of sets",
    "min_overlap": "Erdős' min. overlap",
}
BON_COLOR = "#8c564b"
SCS_COLOR = "#9467bd"
LOAD_WORKERS = 6  # each JSON parse can take a few GB


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def reward_of(record: dict, conf_key: str, lower_is_better: bool) -> float:
    """Higher is always better; failures / missing / non-finite metrics are -inf."""
    if not record["success"]:
        return -np.inf
    value = record["metrics"].get(conf_key)
    if value is None or not np.isfinite(value):
        return -np.inf
    return -value if lower_is_better else value


def source_files(runs_dir: Path, problem: str, args) -> dict[str, list[Path]]:
    fmt = dict(problem=problem, model=args.model_slug)
    dme = sorted(runs_dir.glob(args.dme_pattern.format(**fmt)))
    # BoN files are results_<problem>_<num samples>_<model>.json; the regex keeps e.g. other problems or DME/SCS files
    # whose names happen to match the glob out
    bon_re = re.compile(re.escape(f"results_{problem}_") + r"\d+" + re.escape(f"_{args.model_slug}.json"))
    bon = sorted(p for p in runs_dir.glob(args.bon_pattern.format(**fmt)) if bon_re.fullmatch(p.name))
    scs = sorted(runs_dir.glob(args.scs_pattern.format(**fmt)),
                 key=lambda p: (int(m.group(1)) if (m := re.search(r"trial(\d+)", str(p))) else 0, str(p)))
    return dict(dme=dme, bon=bon, scs=scs)


def dme_file_layout(path: Path) -> tuple[int, int]:
    """(n_chains, batch size per chain) of a DME run file, from its name."""
    m = re.search(r"_nchains(\d+)_bs(\d+)_", path.name)
    if m is None:
        raise ValueError(f"can't read n_chains/bs from DME run file name {path.name}")
    return int(m.group(1)), int(m.group(2))


def split_dme_chains(rewards: np.ndarray, n_chains: int, bs: int) -> np.ndarray:
    """(n_chains, n_steps * bs): within each step of n_chains*bs records, chain c owns records [c*bs, (c+1)*bs).
    A trailing incomplete step (e.g. an interrupted run) is dropped."""
    n_steps = rewards.size // (n_chains * bs)
    rewards = rewards[: n_steps * n_chains * bs]
    return rewards.reshape(n_steps, n_chains, bs).transpose(1, 0, 2).reshape(n_chains, -1)


def _file_rewards(args) -> np.ndarray:
    path, problem = args
    _, _, config = ProblemLoader.load_problem(problem)
    key, lib = config["conf_metric_name"], config.get("lower_is_better", False)
    records = json.load(open(path))["results"]
    assert all(r["algorithm_id"] == i for i, r in enumerate(records)), f"{path}: records not in generation order"
    return np.array([reward_of(r, key, lib) for r in records], dtype=float)


def _fingerprint(files: dict[str, list[Path]]) -> str:
    return json.dumps({k: [(str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in v] for k, v in files.items()})


def _equal_length(arrays: list[np.ndarray], what: str, problem: str) -> np.ndarray:
    """Stack 1D arrays, truncating to the shortest (with a warning if lengths differ)."""
    length = min(a.size for a in arrays)
    if any(a.size != length for a in arrays):
        print(f"  warning: {problem}: {what} have different lengths {sorted({a.size for a in arrays})}, "
              f"truncating all to {length}", flush=True)
    return np.stack([a[:length] for a in arrays])


def load_rewards(problems: list[str], runs_dir: Path, cache_dir: Path, args) -> dict[str, dict]:
    """{problem: dict(dme=(C chains, L), bon=(n,), scs=(n trials, T))}, cached per problem."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out, todo = {}, {}
    for p in problems:
        files = source_files(runs_dir, p, args)
        missing = [k for k, v in files.items() if not v]
        if missing:
            raise FileNotFoundError(f"{p}: no {', '.join(missing)} run files found under {runs_dir}")
        fp = _fingerprint(files)
        path = cache_dir / f"rewards_{p}.npz"
        if path.exists():
            with np.load(path) as d:
                if str(d["fingerprint"]) == fp:
                    out[p] = dict(dme=d["dme"], bon=d["bon"], scs=d["scs"])
                    continue
        todo[p] = (files, fp, path)

    tasks = [(f, p, kind) for p, (files, _, _) in todo.items() for kind, fs in files.items() for f in fs]
    arrays = []
    if tasks:
        print(f"Parsing {len(tasks)} JSON files...", flush=True)
        with ProcessPoolExecutor(max_workers=LOAD_WORKERS) as ex:
            arrays = list(ex.map(_file_rewards, [(str(f), p) for f, p, _ in tasks]))
    for p, (files, fp, path) in todo.items():
        got = {k: [(f, a) for (f, pp, kind), a in zip(tasks, arrays) if pp == p and kind == k] for k in files}
        chains = [c for f, a in got["dme"] for c in split_dme_chains(a, *dme_file_layout(f))]
        dme = _equal_length(chains, "DME chains", p)
        bon = np.concatenate([a for _, a in got["bon"]])
        scs = _equal_length([a for _, a in got["scs"]], "SCS trials", p)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, dme=dme, bon=bon, scs=scs, fingerprint=np.array(fp))
        tmp.replace(path)
        out[p] = dict(dme=dme, bon=bon, scs=scs)
    return {p: out[p] for p in problems}


def available_problems(runs_dir: Path, args) -> list[str]:
    """Problems with DME, BoN and SCS runs."""
    return [p for p in sorted(n.split("/")[-1] for n in ProblemLoader.list_problems())
            if all(source_files(runs_dir, p, args).values())]


def max_budget(data: dict, dme_chains_per_run: int) -> int:
    """Largest budget all three methods' runs support."""
    n_chains, chain_len = data["dme"].shape
    if n_chains < dme_chains_per_run:
        raise ValueError(f"{n_chains} DME chains available, fewer than --dme-chains-per-run={dme_chains_per_run}")
    return min(dme_chains_per_run * chain_len, data["bon"].size, data["scs"].size)


# ---------------------------------------------------------------------------
# Probability of improvement
# ---------------------------------------------------------------------------

def log_comb(n, k) -> np.ndarray:
    n, k = np.asarray(n, dtype=float), np.asarray(k, dtype=float)
    with np.errstate(invalid="ignore"):
        out = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    return np.where((k >= 0) & (k <= n), out, -np.inf)


def dme_atoms(chain_best: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Distinct values w of the max over a random k-of-C chain subset, and their probabilities."""
    w = np.unique(chain_best)
    n = chain_best.size
    le = np.exp(log_comb((chain_best[None, :] <= w[:, None]).sum(1), k) - log_comb(n, k))
    lt = np.exp(log_comb((chain_best[None, :] < w[:, None]).sum(1), k) - log_comb(n, k))
    return w, le - lt


def below_masks(vals: np.ndarray, w: float) -> tuple[np.ndarray, np.ndarray]:
    """Per value: DME value w wins outright (A), or wins or ties (B); ties via np.isclose(x, w)."""
    close = np.isclose(vals, w)
    return (vals < w) & ~close, (vals < w) | close


def credit_from_all_below(p_a: float, p_b: float) -> float:
    # p_a / p_b: P(baseline run lies entirely in A / B); A and B are half-lines, so this is its max.
    return p_a + 0.5 * (p_b - p_a)


def theta_bon(w, q, bon: np.ndarray, N: int) -> float:
    n = bon.size
    total = 0.0
    for wi, qi in zip(w, q):
        a, b = (m.sum() for m in below_masks(bon, wi))
        p_a, p_b = (np.exp(log_comb(c, N) - log_comb(n, N)) for c in (a, b))
        total += qi * credit_from_all_below(p_a, p_b)
    return total


def p_scs_run_in(T_in: np.ndarray, P_in: np.ndarray, t: int, r: int) -> float:
    """P(t full + (if r>0) 1 partial trial, drawn w/o replacement, all land in the set)."""
    n = T_in.size
    a = int(T_in.sum())
    if r == 0:
        return float(np.exp(log_comb(a, t) - log_comb(n, t)))
    b = int((P_in & ~T_in).sum())  # partial trial in the set but its full run isn't
    num = np.logaddexp(np.log(a) + log_comb(a - 1, t) if a > 0 else -np.inf,
                       np.log(b) + log_comb(a, t) if b > 0 else -np.inf)
    return float(np.exp(num - np.log(n) - log_comb(n - 1, t)))


def theta_scs(w, q, scs: np.ndarray, N: int) -> float:
    t, r = divmod(N, scs.shape[1])
    T = scs.max(axis=1)
    P = scs[:, :r].max(axis=1) if r > 0 else np.full(scs.shape[0], -np.inf)
    total = 0.0
    for wi, qi in zip(w, q):
        (Ta, Tb), (Pa, Pb) = below_masks(T, wi), below_masks(P, wi)
        total += qi * credit_from_all_below(p_scs_run_in(Ta, Pa, t, r), p_scs_run_in(Tb, Pb, t, r))
    return total


def compute(rewards: dict[str, dict], budgets: np.ndarray, k: int) -> dict[str, dict]:
    per_problem = {}
    for p, data in rewards.items():
        bon, scs = np.empty(budgets.size), np.empty(budgets.size)
        for i, N in enumerate(budgets):
            w, q = dme_atoms(data["dme"][:, : N // k].max(axis=1), k)
            bon[i], scs[i] = theta_bon(w, q, data["bon"], N), theta_scs(w, q, data["scs"], N)
        per_problem[p] = dict(bon=bon, scs=scs)
    return per_problem


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_aggregate(budgets, agg: dict, out_path: Path) -> None:
    max_n = budgets[-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    l_scs, = ax.plot(budgets, agg["scs"], linewidth=4, label="Prob(DME>SCS)", color=SCS_COLOR)
    l_bon, = ax.plot(budgets, agg["bon"], linewidth=4, label="Prob(DME>BoN)", color=BON_COLOR)
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=1.5, zorder=0)
    ax.set_xlim(0, max_n)
    ax.set_ylim(0.35, 1)
    ax.grid(alpha=0.3)
    ax.legend(reverse=True, fontsize=18, loc="lower right")
    ax.set_xlabel("# Generated Programs", fontsize=18)
    ax.set_ylabel("Prob(DME>Method)", fontsize=18)
    ax.tick_params(labelsize=15)

    # Final-value labels to the right of each curve, nudged apart if close.
    entries = sorted([(l_scs, agg["scs"][-1]), (l_bon, agg["bon"][-1])], key=lambda e: -e[1])
    ys = []
    for _, val in entries:
        ys.append(val if not ys else min(val, ys[-1] - 0.05))
    for (line, val), y in zip(entries, ys):
        ax.text(max_n * 1.015, y + 0.005, f"{val * 100:.0f}%", va="center", ha="left",
                fontsize=16, color=line.get_color(), fontweight="bold", clip_on=False)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=400)
    plt.close()


def plot_per_problem(budgets, per_problem: dict, out_path: Path) -> None:
    n_cols = min(3, len(per_problem))
    n_rows = math.ceil(len(per_problem) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16 * n_cols / 3, 14 * n_rows / 3), sharex=True, sharey=True,
                             constrained_layout=True, squeeze=False)
    axes = axes.ravel()
    for ax, (p, d) in zip(axes, per_problem.items()):
        l_scs, = ax.plot(budgets, d["scs"], linewidth=4.5, label="Prob(DME>SCS)", color=SCS_COLOR)
        l_bon, = ax.plot(budgets, d["bon"], linewidth=4.5, label="Prob(DME>BoN)", color=BON_COLOR)
        ax.axhline(0.5, linestyle="--", color="gray", linewidth=1.2, zorder=0)
        ax.set_title(PROBLEM_LABELS.get(p, p), fontsize=14, pad=8)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=14)
        ax.label_outer()
    for ax in axes[len(per_problem):]:
        ax.set_visible(False)
    handles = [l_bon, l_scs]
    fig.legend(handles=handles, labels=[h.get_label() for h in handles],
               loc="upper center", ncol=2, fontsize=18, bbox_to_anchor=(0.5, 1.06), frameon=True)
    fig.supxlabel("# Generated Programs", fontsize=18)
    fig.supylabel("Prob(DME>Method)", fontsize=18)
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=400)
    plt.close()


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    p.add_argument("--output-dir", type=Path, default=PLOTS_DIR)
    p.add_argument("--problems", nargs="+", default=None,
                   help="Defaults to every problem with DME, BoN and SCS runs in --runs-dir")
    p.add_argument("--dme-chains-per-run", type=int, default=4, help="Chains per DME run (the budget is split evenly)")
    p.add_argument("--budget-step", type=int, default=20)
    p.add_argument("--max-budget", type=int, default=None, help="Defaults to the largest budget the runs support")
    p.add_argument("--model-slug", default=MODEL_SLUG)
    p.add_argument("--dme-pattern", default=DME_PATTERN, help="Glob for DME run files, relative to --runs-dir")
    p.add_argument("--bon-pattern", default=BON_PATTERN, help="Glob for BoN run files, relative to --runs-dir")
    p.add_argument("--scs-pattern", default=SCS_PATTERN, help="Glob for SCS run files (one per trial)")
    return p.parse_args(argv)


def select_budgets(rewards: dict[str, dict], k: int, step: int, max_n: int | None) -> np.ndarray:
    """Budgets step, 2*step, ... up to max_n (by default the largest the runs support), each giving every one of the k
    DME chains at least one program."""
    supported = min(max_budget(d, k) for d in rewards.values())
    max_n = supported if max_n is None else max_n
    if max_n > supported:
        raise ValueError(f"--max-budget {max_n} exceeds the largest budget the runs support, {supported}")
    budgets = np.arange(step, max_n + 1, step)
    budgets = budgets[budgets >= k]
    if budgets.size == 0:
        raise ValueError(f"no budget between {step} and {max_n} gives each of the {k} DME chains a program")
    return budgets


def print_summary(rewards: dict[str, dict], per_problem: dict, agg: dict, max_n: int) -> None:
    for p, d in rewards.items():
        print(f"  {p:22s} DME {d['dme'].shape[0]} chains x {d['dme'].shape[1]}, BoN {d['bon'].size}, "
              f"SCS {d['scs'].shape[0]} trials x {d['scs'].shape[1]}")
    for p, res in per_problem.items():
        print(f"  {p:22s} N={max_n}  vs BoN {res['bon'][-1]:.3f}  vs SCS {res['scs'][-1]:.3f}")
    print(f"  {'aggregate':22s} N={max_n}  vs BoN {agg['bon'][-1]:.3f}  vs SCS {agg['scs'][-1]:.3f}", flush=True)


def main(argv=None) -> dict:
    args = parse_args(argv)
    problems = args.problems or available_problems(args.runs_dir, args)
    if not problems:
        raise FileNotFoundError(f"no problem has DME, BoN and SCS runs under {args.runs_dir}")
    rewards = load_rewards(problems, args.runs_dir, args.output_dir / "cache" / "simple", args)
    budgets = select_budgets(rewards, args.dme_chains_per_run, args.budget_step, args.max_budget)

    per_problem = compute(rewards, budgets, args.dme_chains_per_run)
    agg = {m: np.mean([res[m] for res in per_problem.values()], axis=0) for m in ("bon", "scs")}
    print_summary(rewards, per_problem, agg, budgets[-1])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_aggregate(budgets, agg, args.output_dir / "prob_improvement_agg.pdf")
    plot_per_problem(budgets, per_problem, args.output_dir / "prob_improvement_per_problem.pdf")
    print(f"Wrote plots to {args.output_dir}/", flush=True)
    return dict(budgets=budgets, per_problem=per_problem, agg=agg)


if __name__ == "__main__":
    main()
