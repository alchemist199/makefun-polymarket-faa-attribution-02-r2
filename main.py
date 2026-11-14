# SPDX-License-Identifier: MIT
# polymarket-faa-attribution-02 entry, round 2 (Python 3.13, standard library only).
#
# Forecast: for each market of the case, for each of the 240 minutes after the
# checkpoint, five quantiles (5%, 25%, 50%, 75%, 95%) of the market's Yes price.
#
# Model (original work, calibrated on the practice cases' history, not on their
# answers): a jump-aware random walk around a drifting median.
#
#   q_tau(h) = clamp(med(h) + S_tau * sigma * sqrt(h), 0, 1)
#
# where sigma is the market's own realised 1-minute volatility over the last
# 120 minutes (population standard deviation of the 1-minute price changes),
# floored at 0.0006 and capped at 0.01 price units per square-root minute,
# times 0.95; and med(h) is the median at horizon h.
#
# Why this shape (S = [-0.91, -0.40, -0.15, +0.05, +0.98]):
# Polymarket Yes prices are step functions: most minutes they do not move at
# all, then they jump. So the conditional distribution of the h-step move is
# far more peaked than a Gaussian (the 25%/75% quantiles sit only ~0.4 / ~0.05
# sigma from the median) with fat tails (the 5%/95% quantiles sit ~0.91x /
# ~0.98x sigma*sqrt(h) out). The shape was tuned by coordinate descent on the
# practice cases (0.2655 with the round-1 shape, 0.2484 with this one).
# The median sits 0.15 sigma below the centre: across the practice cases the
# price's conditional median runs below its starting point (favourites drift
# down, small markets drift up only slightly), so a small downward median
# shift beats a centred median on pinball loss.
#
# Median drift (single-winner renormalisation, gated):
# In a single-winner event the markets are mutually exclusive and their Yes
# prices should sum to 1. When the sum deviates from 1 the prices tend to
# revert toward the renormalised price x / sum. So, when the case holds many
# markets (>= 10, the signature of a single-winner event), the median drifts
# linearly from the last price x toward x / sum over the horizon:
#   med(h) = x + (h / horizon) * (x / sum - x).
# The final's FAA family is four INDEPENDENT markets (not mutually exclusive),
# so the gate switches the drift off there (fewer than 10 markets) and the
# median is just the last price plus the shape's own -0.15 sigma shift, which
# on its own still beats the round-1 model on the practice (0.2567 vs 0.2655).
# Gating on the market count keeps the renormalisation from ever touching an
# independent-markets case.
#
# Why a 120-minute volatility window: volatility is strongly non-stationary
# (markets are calm for hours, then jump). A short window tracks the current
# regime; longer windows (6h/24h/72h) drag in stale jumps from the market's
# early chaotic hours and overstate the spread. The window is short, but the
# floor (0.0006) keeps a market that was flat for two hours from getting a
# zero-width forecast, and the cap (0.01) keeps one huge recent jump from
# blowing the spread up. The model needs only the last 120 minutes, so it
# still works when a case holds much less than 72 hours of history (the final
# FAA markets are only ~16 hours old at the first checkpoint).
#
# Deterministic: pure arithmetic over the case file, no randomness, no clock,
# no network. Two runs on the same case write byte-identical output.
#
# Usage (local): MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
# The grader runs: python3 /bot/main.py, reading /input/<id>.in, writing
# /output/<id>.out.

import json
import math
import os
from os import path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"

# The quantile levels the contest scores, in the order each row lists them.
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standardised h-step quantile shape S_tau (see header): how far each quantile
# sits from the median, in units of sigma * sqrt(h).
SHAPE = [-0.91, -0.40, -0.15, 0.05, 0.98]

# Volatility window: the last 120 one-minute price changes.
VOL_WINDOW = 120
# Floor / cap on the raw realised sigma, in price units per square-root
# minute (applied before the shrinkage multiplier below).
SIGMA_FLOOR = 0.0006
SIGMA_CAP = 0.01
# Shrinkage on the realised volatility (it is jump-dominated, so slightly
# overstated for the quiet minutes that dominate the score).
SIGMA_MULT = 0.95
# Fallback sigma when a market has too little history to estimate one.
SIGMA_DEFAULT = 0.0025

# Single-winner renormalisation: only when a case holds at least this many
# markets (the practice's single-winner event holds 19-24; the final's
# independent FAA family holds 4, so the drift stays off there).
SINGLE_WINNER_MIN_MARKETS = 10


def price6(x):
    """x clamped to [0, 1] and rounded to 6 decimals (0, never -0)."""
    v = round(min(1.0, max(0.0, x)) * 1e6) / 1e6
    return v + 0.0


def realised_sigma(prices):
    """Population std of the last VOL_WINDOW 1-minute price changes,
    floored/capped, then shrunk (see header)."""
    n = len(prices)
    if n < 6:
        return SIGMA_DEFAULT
    start = max(0, n - 1 - VOL_WINDOW)
    changes = [prices[i + 1] - prices[i] for i in range(start, n - 1)]
    if len(changes) < 5:
        return SIGMA_DEFAULT
    mean = math.fsum(changes) / len(changes)
    var = math.fsum((c - mean) * (c - mean) for c in changes) / len(changes)
    sigma = math.sqrt(var)
    if sigma < SIGMA_FLOOR:
        sigma = SIGMA_FLOOR
    elif sigma > SIGMA_CAP:
        sigma = SIGMA_CAP
    return sigma * SIGMA_MULT


def forecast_market(prices, horizon, renorm_target):
    """240 rows of five quantiles for one market.

    renorm_target is the price the median drifts toward over the horizon
    (the renormalised price x / sum for a single-winner case, else x itself,
    i.e. no drift)."""
    if not prices:
        # No history at all: stay flat at 0.5 rather than failing the case.
        return [[0.5, 0.5, 0.5, 0.5, 0.5] for _ in range(horizon)]
    x = prices[-1]
    sigma = realised_sigma(prices)
    rows = []
    for h in range(1, horizon + 1):
        med = x + (h / horizon) * (renorm_target - x)
        spread = sigma * math.sqrt(h)
        row = [price6(med + SHAPE[j] * spread) for j in range(5)]
        # Rounding can only tie values, but enforce non-decreasing anyway.
        for j in range(1, 5):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def renorm_targets(markets):
    """Per-market median-drift target for a single-winner case.

    Returns a dict id -> target price (x / sum), or None when the case is
    not a single-winner event (too few markets) or the sum is already ~1
    (no drift to apply)."""
    if len(markets) < SINGLE_WINNER_MIN_MARKETS:
        return None
    total = math.fsum(m["prices"][-1] for m in markets if m["prices"])
    if total <= 0.0 or abs(total - 1.0) < 1e-9:
        return None
    return {m["id"]: (m["prices"][-1] / total if m["prices"] else 0.5)
            for m in markets}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for fname in inputs:
        case_id = fname[: -len(".in")]
        with open(path.join(INPUT_DIR, fname), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        horizon = int(case["horizon_minutes"])
        markets = case["markets"]
        # Single-winner gate: renormalise only when the case holds many
        # markets (mutually exclusive outcomes); independent markets keep
        # their own last price as the drift target.
        targets = renorm_targets(markets)
        forecasts = []
        for market in markets:
            prices = market["prices"]
            if targets is not None:
                target = targets[market["id"]]
            else:
                target = prices[-1] if prices else 0.5
            rows = forecast_market(prices, horizon, target)
            forecasts.append({"id": market["id"], "q": rows})
        out_path = path.join(OUTPUT_DIR, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"forecasts": forecasts}, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
