"""Forecast GTA VI (PS5) trophy earn rates on Exophase a week after launch.

Entry for make.fun's gta6-trophy-rarity-forecast-01. Reads one case file
(/input/<id>.in) and writes /output/<id>.out with five quantiles (5, 25, 50,
75, 95%) of each trophy's earn rate at the case's target. Python 3.13,
standard library only, no network, deterministic.

Model
-----
An Exophase earn rate is a ratio: earners / players-tracked. Over launch
week both parts move, in opposite directions for the rate:

  * players-tracked N(t) grows fast as new players join (and Exophase
    refreshes its counts in jumps, which dilutes every rate at once);
  * earners E_i(t) for trophy i only grow, as the tracked players play
    through the game.

Fitting the rate directly mixes the two forces and chases refresh jumps.
Instead this program fits, per trophy, log-earners against log-time and
fits log players-tracked the same way, then divides:

    log E_i(t) = a_i + b_i * x(t),   log N(t) = c + d * x(t),
    x(t) = ln(1 + hours since 2026-11-19T00:00:00Z),  b_i, d >= 0.

The fit uses the case's last 24 hours of reads (the recent regime is what
extrapolates), falling back to the full history when there are too few
reads. Each trophy's slope is shrunk toward its tier's median slope, so a
trophy with a noisy or flat series borrows strength from its peers.

The median forecast is rate* = 100 * E*(target) / N*(target). Story
trophies must nest (everyone who finished Chapter 2 also finished
Chapter 1, and the platinum needs everything), so the medians along the
chain Chapter 1 -> 2 -> 3 -> 4 -> final mission -> 100% -> platinum are
made non-increasing with a pool-adjacent-violators pass in log space.

Spread: the quantiles widen with the fit's residual scatter and with the
extrapolation distance in log-time, so a far target honestly gets a wide
band. The pinball loss punishes a too-narrow band much harder than a
too-wide one, so the spread errs wide.

With a flat history (the practice cases, whose reads are all one unchanged
page) every slope is zero and the forecast is the last rate with a
random-walk-style spread -- the house baseline, as it should be: nothing
in a flat history can foresee a refresh.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]

LAUNCH = datetime(2026, 11, 19, tzinfo=timezone.utc)

# Fit window: the last day of reads carries the regime that extrapolates.
WINDOW_HOURS = 24.0
# Prior strength (in reads) when shrinking a trophy's slope to its tier's.
SHRINK_READS = 6.0
# Spread: residual scatter, plus this per log-hour of extrapolation, plus floor.
SPREAD_PER_LOG_HOUR = 0.45
SPREAD_FLOOR = 0.40
# Hard caps on fitted log-time slopes (log units per log-hour).
MAX_SLOPE = 3.0

# Story chain: medians must be non-increasing along it (id order).
STORY_CHAIN = [3, 4, 5, 6, 7, 14, 1]


def parse_time(text):
    """ISO-8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def log_time(moment):
    """x(t) = ln(1 + hours since launch), floored at 0 before launch."""
    hours = (moment - LAUNCH).total_seconds() / 3600.0
    return math.log1p(max(0.0, hours))


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, as JavaScript Math.round does it."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def wls(xs, ys, ws, lo=None, hi=None):
    """Weighted least squares for y = a + b*x. Returns (a, b, resid_std).

    With fewer than two distinct x values the slope is 0 and the intercept
    is the weighted mean; resid_std is then the weighted spread of y.
    When lo/hi bound the slope, the slope is clamped and the intercept is
    refit at that slope, so the line still passes through the data's level
    (clamping the slope alone would swing the intercept wildly).
    """
    sw = sum(ws)
    if sw <= 0:
        return 0.0, 0.0, 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        var = sum(w * (y - my) ** 2 for w, y in zip(ws, ys)) / sw
        return my, 0.0, math.sqrt(max(0.0, var))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    if lo is not None:
        b = max(lo, b)
    if hi is not None:
        b = min(hi, b)
    # Refit the intercept at the (possibly clamped) slope.
    a = sum(w * (y - b * x) for w, x, y in zip(ws, xs, ys)) / sw
    n = len(xs)
    if n > 2:
        var = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
        resid = math.sqrt(max(0.0, var))
    else:
        resid = 0.0
    return a, b, resid


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def pava_decreasing(values):
    """Pool adjacent violators: least-squares projection onto non-increasing.

    Returns a list of the same length whose entries are non-increasing.
    """
    blocks = []  # each [count, sum]
    for value in values:
        blocks.append([1, value])
        while (len(blocks) >= 2
               and blocks[-2][1] / blocks[-2][0]
               < blocks[-1][1] / blocks[-1][0] - 1e-12):
            merged = [blocks[-2][0] + blocks[-1][0],
                      blocks[-2][1] + blocks[-1][1]]
            blocks[-2:] = [merged]
    out = []
    for count, total in blocks:
        out.extend([total / count] * count)
    return out


def fit_series(points, slope_bounds=(0.0, MAX_SLOPE)):
    """Fit log y against log-time on the recent window.

    points: list of (x, y) with y > 0, oldest first. Returns (a, b, resid,
    n_used). The window is the last 24h of x; if it holds fewer than 3
    points, the whole history is used instead.
    """
    if not points:
        return 0.0, 0.0, 0.0, 0
    x_last = points[-1][0]
    hours_last = math.expm1(x_last)
    cutoff = math.log1p(max(0.0, hours_last - WINDOW_HOURS))
    window = [p for p in points if p[0] >= cutoff]
    used = window if len(window) >= 3 else points
    # Recency weights: halve every 12 log-hours of age (gentle; the window
    # already restricts the fit to recent reads).
    xs = [p[0] for p in used]
    ys = [p[1] for p in used]
    ws = [0.5 ** ((x_last - x) / 12.0) for x in xs]
    lo, hi = slope_bounds if slope_bounds else (None, None)
    a, b, resid = wls(xs, ys, ws, lo=lo, hi=hi)
    return a, b, resid, len(used)


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    target = parse_time(case["target"])
    x_target = log_time(target)
    x_last = log_time(parse_time(history[-1]["at"]))
    dx = max(0.0, x_target - x_last)

    # --- players-tracked series: log N against log-time. ---
    n_points = []
    last_n = None
    for read in history:
        n = read.get("players_tracked")
        if n is None:
            n = last_n
        if n is None or n <= 0:
            continue
        last_n = n
        n_points.append((log_time(parse_time(read["at"])), math.log(n)))
    if n_points:
        a_n, b_n, _, _ = fit_series(n_points)
        n_target = math.exp(a_n + b_n * x_target)
    else:
        n_target = 1.0

    # --- per-trophy earners series and fits. ---
    last_rates = history[-1]["rates"]
    global_last = [r for r in last_rates if r is not None]
    fallback_rate = median(global_last) if global_last else 1.0

    per_trophy = []
    for i, trophy in enumerate(trophies):
        e_points = []
        last_n_i = None
        for read in history:
            rate = read["rates"][i]
            n = read.get("players_tracked")
            if n is None:
                n = last_n_i
            if n is None or n <= 0:
                continue
            last_n_i = n
            if rate is None:
                continue
            earners = rate / 100.0 * n
            # Floor at a quarter of a player so log(0) never appears; the
            # forecast below inverts this same transform, so a flat series
            # forecasts exactly its last rate.
            e_points.append((log_time(parse_time(read["at"])),
                             math.log(max(earners, 0.25))))
        last_rate = last_rates[i]
        if last_rate is None:
            last_rate = fallback_rate
        if not e_points:
            per_trophy.append({
                "id": trophy["id"], "tier": trophy["tier"],
                "last_rate": last_rate, "slope": 0.0, "median": last_rate,
                "resid": 0.0, "n_used": 0,
            })
            continue
        a_e, b_e, resid_e, n_used = fit_series(e_points)
        # Rate-space residuals for the spread: fit logit(rate) on the same
        # window so refresh jumps show up as honest uncertainty.
        r_points = []
        for read in history:
            rate = read["rates"][i]
            if rate is None:
                continue
            r_points.append((log_time(parse_time(read["at"])), logit(rate)))
        _, _, resid_r, _ = fit_series(r_points, slope_bounds=None)
        e_target = math.exp(a_e + b_e * x_target)
        rate_star = 100.0 * e_target / n_target
        per_trophy.append({
            "id": trophy["id"], "tier": trophy["tier"],
            "last_rate": last_rate, "slope": b_e, "median": rate_star,
            "resid": max(resid_e, resid_r), "n_used": n_used,
        })

    # --- shrink slopes toward the tier median (borrow strength). ---
    by_tier = {}
    for t in per_trophy:
        by_tier.setdefault(t["tier"], []).append(t["slope"])
    tier_median = {tier: median(sl) for tier, sl in by_tier.items()}
    global_median_slope = median([t["slope"] for t in per_trophy])
    for t in per_trophy:
        pool = tier_median.get(t["tier"], global_median_slope)
        k = min(t["n_used"], 24)
        shrunk = (k * t["slope"] + SHRINK_READS * pool) / (k + SHRINK_READS)
        if shrunk != t["slope"]:
            # Recompute the median with the shrunk slope: the intercept was
            # fitted at the last read's x, so reuse it via the last rate.
            # log E*(target) = log E(last) + slope * dx, in earners space;
            # equivalently scale the rate forecast by the slope change.
            factor = math.exp((shrunk - t["slope"]) * dx)
            t["median"] = t["median"] * factor
            t["slope"] = shrunk

    # --- story chain: medians non-increasing along it. ---
    by_id = {t["id"]: t for t in per_trophy}
    chain = [by_id[i] for i in STORY_CHAIN if i in by_id]
    if len(chain) >= 2:
        adjusted = pava_decreasing([math.log(max(0.01, t["median"]))
                                    for t in chain])
        for t, lg in zip(chain, adjusted):
            t["median"] = math.exp(lg)

    # --- quantiles. ---
    rows = []
    for t in per_trophy:
        med = clamp_rate(t["median"])
        spread = math.sqrt(t["resid"] ** 2
                           + (SPREAD_PER_LOG_HOUR * dx) ** 2
                           + SPREAD_FLOOR ** 2)
        centre = logit(med)
        qs = []
        prev = None
        for z in Z:
            q = round4(clamp_rate(inv_logit(centre + z * spread)))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": t["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
