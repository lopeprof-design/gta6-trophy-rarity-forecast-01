# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/lopeprof-design/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/lopeprof-design/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for Grand Theft Auto VI (PS5) trophy earn rates on Exophase, a week after launch. It reads one case (`/input/<id>.in`: every Exophase read before the checkpoint) and writes five quantiles (5%, 25%, 50%, 75%, 95%) of each trophy's earn rate at the case's target to `/output/<id>.out`.

## How it works

An Exophase earn rate is a ratio — earners over players tracked — and over launch week both halves move against the rate: players-tracked grows fast as new players join (and Exophase refreshes its counts in jumps, diluting every rate at once), while earners only grow as players play through the game. Fitting the rate directly mixes the two forces and chases refresh jumps, so `main.py` fits them separately, both against log-time since launch:

- `log E_i(t) = a_i + b_i * x(t)` per trophy (earners, from rate × players-tracked), slope shrunk toward its tier's median;
- `log N(t) = c + d * x(t)` for players-tracked;

then forecasts `rate* = 100 * E*(target) / N*(target)`. Fits use the last 24 hours of reads. Story trophies must nest (Chapter 1 → 2 → 3 → 4 → final mission → 100% → platinum), so medians along that chain are made non-increasing. Quantiles widen with the fit's residual scatter and with the extrapolation distance, so a far target honestly gets a wide band. On a synthetic final built from the real launch-night and day-one Exophase snapshots, it scores about a third of the house baseline's pinball loss.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The grader runs `python3 /bot/main.py` (Python 3.13, standard library only, no network) with the case in `/input` and writes the forecast to `/output`. No dependencies to install.

---

<sub>Entered on [make.fun](https://make.fun/?ref=tghy00yw&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
