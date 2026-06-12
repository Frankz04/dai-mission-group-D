"""Task 3 — chronological feature engineering for match prediction."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from pathlib import Path

from helpers.loading import FEATURES_DATA_PATH, load_original_data, load_reshaped_data
from helpers.reshape import reshape_matches

LONG_MATCH_MINUTES = 180
FORM_WINDOW = 5

MATCH_KEY = ["tourney_id", "match_num"]
CHRONO_COLS = ["tourney_date", "match_num", "tourney_id"]
PLAYER_FEATURE_COLS = [
    "recent_form",
    "rest_days",
    "long_prev_match",
    "prev_minutes",
    "surface_win_rate",
    "surface_matches_prior",
]


def _add_player_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Player-level features using only information from prior matches."""
    out = df.sort_values(["player_id", *CHRONO_COLS]).copy()
    grouped = out.groupby("player_id", sort=False)

    out["prev_minutes"] = grouped["minutes"].shift(1)
    out["long_prev_match"] = out["prev_minutes"].ge(LONG_MATCH_MINUTES).astype("boolean")

    prev_date = grouped["tourney_date"].shift(1)
    out["rest_days"] = (
        pd.to_datetime(out["tourney_date"], format="%Y%m%d")
        - pd.to_datetime(prev_date, format="%Y%m%d")
    ).dt.days

    prior_wins = grouped["win"].shift(1)
    out["recent_form"] = prior_wins.groupby(out["player_id"]).transform(
        lambda wins: wins.rolling(FORM_WINDOW, min_periods=FORM_WINDOW).mean()
    )

    surface_grouped = out.groupby(["player_id", "surface"], sort=False)
    prior_surface_wins = surface_grouped["win"].shift(1)
    out["surface_matches_prior"] = surface_grouped.cumcount()
    out["surface_win_rate"] = prior_surface_wins.groupby(
        [out["player_id"], out["surface"]]
    ).transform(lambda wins: wins.expanding().mean())

    return out


def _add_head_to_head_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pair-level win history before each match (chronological, no leakage)."""
    out = df.sort_values(CHRONO_COLS).copy()
    pair_stats: dict[tuple[int, int], dict[str, int]] = {}

    h2h_wins: list[float] = []
    h2h_matches: list[int] = []
    h2h_win_rates: list[float] = []

    for row in out.itertuples(index=False):
        player_id = int(row.player_id)
        opp_id = int(row.opp_id)
        key = (min(player_id, opp_id), max(player_id, opp_id))
        stats = pair_stats.get(key, {"min_wins": 0, "total": 0})

        if player_id == key[0]:
            player_wins = stats["min_wins"]
        else:
            player_wins = stats["total"] - stats["min_wins"]

        h2h_wins.append(player_wins)
        h2h_matches.append(stats["total"])
        h2h_win_rates.append(
            player_wins / stats["total"] if stats["total"] > 0 else np.nan
        )

        winner_id = player_id if row.win == 1 else opp_id
        if winner_id == key[0]:
            stats["min_wins"] += 1
        stats["total"] += 1
        pair_stats[key] = stats

    out["h2h_wins"] = h2h_wins
    out["h2h_matches"] = h2h_matches
    out["h2h_win_rate"] = h2h_win_rates
    return out


def _both_perspectives(df: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match so each player's history includes every appearance."""
    return reshape_matches(df, randomize=False)


def _merge_player_and_opponent_features(
    base: pd.DataFrame, featured_both: pd.DataFrame
) -> pd.DataFrame:
    """Attach pre-match features for player and opponent on each match row."""
    lookup = featured_both[[*MATCH_KEY, "tourney_date", "player_id", *PLAYER_FEATURE_COLS]]
    out = base.merge(
        lookup,
        on=[*MATCH_KEY, "tourney_date", "player_id"],
        how="left",
    )
    opp_lookup = lookup.rename(
        columns={"player_id": "opp_id", **{col: f"opp_{col}" for col in PLAYER_FEATURE_COLS}}
    )
    return out.merge(
        opp_lookup,
        on=[*MATCH_KEY, "tourney_date", "opp_id"],
        how="left",
    )


def build_features(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return reshaped match rows with engineered pre-match features."""
    base = load_reshaped_data() if df is None else df.copy()
    featured_both = _add_player_history_features(_both_perspectives(load_original_data()))
    out = _merge_player_and_opponent_features(base, featured_both)
    out = _add_head_to_head_features(out)

    out["form_diff"] = out["recent_form"] - out["opp_recent_form"]
    out["surface_win_rate_diff"] = out["surface_win_rate"] - out["opp_surface_win_rate"]

    return out.sort_values(CHRONO_COLS).reset_index(drop=True)


def engineer_and_save(
    output_path: Path | str | None = None,
    *,
    input_path: Path | str | None = None,
) -> Path:
    """Load reshaped data, engineer features, and write to disk."""
    output_path = Path(output_path) if output_path is not None else FEATURES_DATA_PATH
    df = load_reshaped_data(input_path)
    featured = build_features(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(output_path, index=False)
    print(f"Wrote {len(featured):,} rows with {featured.shape[1]} columns to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Engineer chronological match features and save to CSV."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=FEATURES_DATA_PATH,
        help=f"Output path (default: {FEATURES_DATA_PATH})",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help="Input reshaped CSV (default: data/atp_matches_reshaped.csv)",
    )
    args = parser.parse_args()
    engineer_and_save(args.output, input_path=args.input)


if __name__ == "__main__":
    main()
