"""Task 2 EDA: win rates, ranking baseline, fatigue and form prevalence."""

from __future__ import annotations

import pandas as pd

from helpers.loading import load_reshaped_data

LONG_MATCH_MINUTES = 180
FORM_WINDOW = 5
ROUND_ORDER = {
    "RR": 0,
    "R128": 1,
    "R64": 2,
    "R32": 3,
    "R16": 4,
    "QF": 5,
    "SF": 6,
    "F": 7,
    "ER": 8,
    "BR": 9,
}


def ordered_rounds(series: pd.Series) -> list[str]:
    rounds = series.dropna().unique().tolist()
    return sorted(rounds, key=lambda r: (ROUND_ORDER.get(r, 99), r))


def add_player_history(df: pd.DataFrame) -> pd.DataFrame:
    """Chronological per-player history for fatigue and form indicators."""
    out = df.sort_values(["player_id", "tourney_date", "match_num"]).copy()
    grouped = out.groupby("player_id", sort=False)

    out["prev_minutes"] = grouped["minutes"].shift(1)
    out["long_prev_match"] = out["prev_minutes"].ge(LONG_MATCH_MINUTES)

    prev_date = grouped["tourney_date"].shift(1)
    out["rest_days"] = pd.to_datetime(out["tourney_date"], format="%Y%m%d") - pd.to_datetime(
        prev_date, format="%Y%m%d"
    )
    out["rest_days"] = out["rest_days"].dt.days

    prior_wins = grouped["win"].shift(1)
    out["recent_form"] = (
        prior_wins.groupby(out["player_id"])
        .transform(lambda wins: wins.rolling(FORM_WINDOW, min_periods=1).mean())
    )
    out["prior_matches"] = grouped.cumcount()
    return out


def ranked_matches(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.dropna(subset=["player_rank", "opp_rank"]).copy()
    ranked = ranked[ranked["player_rank"] != ranked["opp_rank"]]
    ranked["player_favored"] = ranked["player_rank"] < ranked["opp_rank"]
    ranked["favorite_wins"] = ranked["player_favored"] == ranked["win"].astype(bool)
    ranked["rank_gap"] = (ranked["player_rank"] - ranked["opp_rank"]).abs()
    return ranked


def print_section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(title)
    print("=" * 72)


def print_table(table: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    print(table.to_string())


def run_eda() -> None:
    df = load_reshaped_data()
    df["year"] = df["tourney_date"] // 10000
    ranked = ranked_matches(df)
    history = add_player_history(df)

    print_section("Dataset overview")
    print(f"Rows: {len(df):,}  |  Years: {int(df['year'].min())}-{int(df['year'].max())}")
    print(f"Matches with both ranks: {len(ranked):,} ({len(ranked) / len(df):.1%} of rows)")

    print_section("1. Favorite win rate by surface")
    by_surface = (
        ranked.groupby("surface", dropna=False)["favorite_wins"]
        .agg(matches="count", favorite_win_rate="mean")
        .sort_values("favorite_win_rate", ascending=False)
    )
    by_surface["favorite_win_rate"] = (by_surface["favorite_win_rate"] * 100).round(1)
    print_table(by_surface, "Higher-ranked player win rate (%) by surface")

    print_section("2. Favorite win rate by round")
    by_round = (
        ranked.groupby("round", dropna=False)["favorite_wins"]
        .agg(matches="count", favorite_win_rate="mean")
    )
    by_round = by_round.reindex(ordered_rounds(by_round.index))
    by_round["favorite_win_rate"] = (by_round["favorite_win_rate"] * 100).round(1)
    print_table(by_round, "Higher-ranked player win rate (%) by round")

    print_section("3. Ranking-gap baseline")
    overall_baseline = ranked["favorite_wins"].mean()
    print(
        f"Higher-ranked player wins: {overall_baseline:.1%} "
        f"({ranked['favorite_wins'].sum():,} / {len(ranked):,} ranked matches)"
    )

    gap_bins = [0, 5, 10, 20, 50, 100, 10_000]
    gap_labels = ["1-5", "6-10", "11-20", "21-50", "51-100", "100+"]
    ranked["gap_bucket"] = pd.cut(
        ranked["rank_gap"],
        bins=gap_bins,
        labels=gap_labels,
        right=True,
    )
    by_gap = (
        ranked.groupby("gap_bucket", observed=True)["favorite_wins"]
        .agg(matches="count", favorite_win_rate="mean")
    )
    by_gap["favorite_win_rate"] = (by_gap["favorite_win_rate"] * 100).round(1)
    print_table(by_gap, "Baseline accuracy by absolute rank gap")

    print_section("4. Fatigue prevalence")
    has_prev_match = history["prev_minutes"].notna()
    long_prev = history["long_prev_match"].fillna(False)

    print(f"Rows with a prior match in dataset: {has_prev_match.mean():.1%}")
    print(
        f"Rows with long previous match (>={LONG_MATCH_MINUTES} min): "
        f"{long_prev.mean():.1%} of all rows, "
        f"{long_prev[has_prev_match].mean():.1%} of rows with prior match"
    )

    fatigue_outcomes = history.loc[has_prev_match].groupby("long_prev_match")["win"].agg(
        matches="count", win_rate="mean"
    )
    fatigue_outcomes["win_rate"] = (fatigue_outcomes["win_rate"] * 100).round(1)
    print_table(
        fatigue_outcomes,
        "Player win rate after normal vs. long previous match (descriptive only)",
    )

    rest = history.loc[history["rest_days"].notna(), "rest_days"]
    rest_summary = rest.describe(percentiles=[0.25, 0.5, 0.75, 0.9]).round(1)
    print("\nRest days since previous match:")
    print(rest_summary.to_string())

    short_rest = (rest <= 1).mean()
    print(f"\nShare with <=1 rest day since previous match: {short_rest:.1%}")

    print_section("5. Form prevalence")
    enough_history = history["prior_matches"] >= FORM_WINDOW
    print(
        f"Rows with at least {FORM_WINDOW} prior matches for rolling form: "
        f"{enough_history.mean():.1%}"
    )

    form_summary = history.loc[history["recent_form"].notna(), "recent_form"].describe(
        percentiles=[0.25, 0.5, 0.75]
    )
    print(f"\nRecent form (win rate over prior {FORM_WINDOW} matches):")
    print((form_summary * 100).round(1).to_string())

    hot_form = history["recent_form"].ge(0.8)
    cold_form = history["recent_form"].le(0.2)
    print(f"\nShare with recent form >= 80%: {hot_form.mean():.1%}")
    print(f"Share with recent form <= 20%: {cold_form.mean():.1%}")

    form_outcomes = history.loc[history["recent_form"].notna()].copy()
    form_outcomes["form_bucket"] = pd.cut(
        form_outcomes["recent_form"],
        bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.01],
        labels=["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"],
    )
    by_form = (
        form_outcomes.groupby("form_bucket", observed=True)["win"]
        .agg(matches="count", win_rate="mean")
    )
    by_form["win_rate"] = (by_form["win_rate"] * 100).round(1)
    print_table(by_form, "Player win rate by recent-form bucket (descriptive only)")


if __name__ == "__main__":
    run_eda()
