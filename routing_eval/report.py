"""Human-readable frontier report, CSV export, and a dependency-free plot."""
from __future__ import annotations

import csv
from typing import Dict, List

from .frontier import FrontierResult
from .modelselect import LocalViability, ModelCategoryRanking


def to_csv(fr: FrontierResult, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau", "remote_tokens", "accuracy", "escalation_rate"])
        for p in fr.points:
            w.writerow([p.tau, p.remote_tokens, round(p.accuracy, 6),
                        round(p.escalation_rate, 6)])


def ascii_plot(fr: FrontierResult, width: int = 56, height: int = 13) -> str:
    xs = [p.remote_tokens for p in fr.points]
    ys = [p.accuracy for p in fr.points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if ymax == ymin:
        ymax = ymin + 1e-9
    grid = [[" "] * width for _ in range(height)]

    def cell(x, y):
        col = int((x - xmin) / (xmax - xmin) * (width - 1)) if xmax > xmin else 0
        row = int((ymax - y) / (ymax - ymin) * (height - 1))
        return row, col

    for x, y in zip(xs, ys):
        r, c = cell(x, y)
        grid[r][c] = "*"
    if fr.operating_point:
        r, c = cell(fr.operating_point.remote_tokens, fr.operating_point.accuracy)
        grid[r][c] = "O"
    body = "\n".join("".join(row) for row in grid)
    return (f"acc {ymax:.3f} |\n{body}\n"
            f"acc {ymin:.3f} +{'-' * width}\n"
            f"       tokens {xmin} .. {xmax}   (* frontier, O operating point)")


def summarize(fr: FrontierResult) -> str:
    lines = [f"-- signal: {fr.signal} " + "-" * 34]
    lines.append(f"all-local : acc {fr.all_local_accuracy:.3f}  tokens 0")
    lines.append(f"all-remote: acc {fr.all_remote_accuracy:.3f}  tokens {fr.all_remote_tokens}")
    lines.append(f"ceiling   : acc {fr.union_ceiling:.3f}  (max any router can reach)")
    if fr.accuracy_floor is not None:
        lines.append(f"floor     : acc {fr.accuracy_floor:.3f}")
        if not fr.feasible:
            lines.append("  INFEASIBLE on this set: no threshold meets the floor "
                         "(capability-limited -> improve the model, not the gate).")
        else:
            op = fr.operating_point
            lines.append(f"  operating : tokens {op.remote_tokens}  acc {op.accuracy:.3f}  "
                         f"escalate {op.escalation_rate:.0%}  "
                         f"margin +{op.accuracy - fr.accuracy_floor:.3f}")
            tag = "exact" if fr.oracle_exact else "approx"
            eff = "n/a" if fr.gate_efficiency is None else f"{fr.gate_efficiency:.2f}"
            lines.append(f"  oracle    : tokens {fr.oracle_tokens} ({tag})   "
                         f"gate efficiency {eff}  (1.00 = perfect-knowledge router)")
    return "\n".join(lines)


def format_local_viability(viability: Dict[str, LocalViability]) -> str:
    lines = ["-- local-tier viability probe " + "-" * 26]
    lines.append(f"{'category':<18}{'n':>3}{'acc':>7}{'avg_s':>9}{'max_s':>9}  verdict")
    for cat in sorted(viability):
        v = viability[cat]
        avg = f"{v.avg_latency_s:.2f}" if v.avg_latency_s is not None else "n/a"
        mx = f"{v.max_latency_s:.2f}" if v.max_latency_s is not None else "n/a"
        verdict = "LOCAL-VIABLE" if v.local_viable else "MUST-ESCALATE"
        lines.append(f"{cat:<18}{v.n:>3}{v.accuracy:>7.2f}{avg:>9}{mx:>9}  "
                     f"{verdict} ({v.reason})")
    return "\n".join(lines)


def format_bakeoff_ranking(ranking: Dict[str, List[ModelCategoryRanking]]) -> str:
    lines = ["-- Fireworks model bake-off (floor-clearing first, ranked by tokens) " + "-" * 4]
    for cat in sorted(ranking):
        lines.append(f"\ncategory: {cat}")
        lines.append(f"  {'model':<42}{'acc':>7}{'tokens':>10}  clears_floor")
        for r in ranking[cat]:
            lines.append(f"  {r.model:<42}{r.accuracy:>7.2f}{r.total_tokens:>10}  "
                         f"{'yes' if r.clears_floor else 'no'}")
    return "\n".join(lines)
