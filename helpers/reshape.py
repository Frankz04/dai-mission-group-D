# Reshape winner/loser match rows into symmetric player-vs-opponent records.
import argparse

import numpy as np
import pandas as pd
from pathlib import Path

from loading import RESHAPED_DATA_PATH, load_original_data

DEFAULT_OUTPUT = RESHAPED_DATA_PATH


def _rename_columns(df: pd.DataFrame, player_prefix: str, opp_prefix: str) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        if col.startswith("winner_"):
            rename[col] = f"{player_prefix}{col[len('winner_'):]}"
        elif col.startswith("loser_"):
            rename[col] = f"{opp_prefix}{col[len('loser_'):]}"
        elif col.startswith("w_"):
            rename[col] = f"{player_prefix}{col[len('w_'):]}"
        elif col.startswith("l_"):
            rename[col] = f"{opp_prefix}{col[len('l_'):]}"
    return df.rename(columns=rename)


def _winner_perspective(df: pd.DataFrame) -> pd.DataFrame:
    out = _rename_columns(df, "player_", "opp_")
    out["win"] = 1
    return out


def _loser_perspective(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        if col.startswith("loser_"):
            rename[col] = f"player_{col[len('loser_'):]}"
        elif col.startswith("winner_"):
            rename[col] = f"opp_{col[len('winner_'):]}"
        elif col.startswith("l_"):
            rename[col] = f"player_{col[len('l_'):]}"
        elif col.startswith("w_"):
            rename[col] = f"opp_{col[len('w_'):]}"
    out = df.rename(columns=rename)
    out["win"] = 0
    return out


def reshape_matches(
    df: pd.DataFrame,
    *,
    randomize: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Reshape match-level data to one player-vs-opponent row per match.

    With ``randomize=True`` (default), each match keeps a single row and the
    player/opponent assignment is flipped at random so models cannot trivially
    read the original winner column. With ``randomize=False``, both perspectives
    are kept (two rows per match).
    """
    if not randomize:
        return pd.concat(
            [_winner_perspective(df), _loser_perspective(df)],
            ignore_index=True,
        )

    rng = np.random.default_rng(seed)
    flip = rng.random(len(df)) < 0.5
    winner_view = _winner_perspective(df.loc[~flip])
    loser_view = _loser_perspective(df.loc[flip])
    return pd.concat([winner_view, loser_view], ignore_index=True)


def reshape_and_save(
    output_path: Path | str | None = None,
    *,
    randomize: bool = True,
    seed: int = 42,
) -> Path:
    """Load raw matches, reshape once, and write the result to disk."""
    output_path = Path(output_path) if output_path is not None else DEFAULT_OUTPUT
    df = load_original_data()
    long_df = reshape_matches(df, randomize=randomize, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(output_path, index=False)
    print(f"Wrote {len(long_df):,} rows to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Reshape ATP matches to player-vs-opponent format.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--both-perspectives",
        action="store_true",
        help="Keep both player perspectives per match (two rows per match).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for label assignment.")
    args = parser.parse_args()
    reshape_and_save(
        args.output,
        randomize=not args.both_perspectives,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
