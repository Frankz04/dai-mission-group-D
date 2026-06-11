# Data loading & first inspection (proposal stage)
# Clone https://github.com/JeffSackmann/tennis_atp and place atp_matches_*.csv in ./data/
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESHAPED_DATA_PATH = DATA_DIR / "atp_matches_reshaped.csv"


def load_original_data() -> pd.DataFrame:
    files = sorted(
        f for f in DATA_DIR.glob("atp_matches_*.csv")
        if f.name != RESHAPED_DATA_PATH.name
    )
    print("Loading data from", len(files), "files")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df = df[df.tourney_date // 10000 >= 1991]  # match stats available from 1991
    return df


def load_reshaped_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load the pre-reshaped player-vs-opponent dataset."""
    path = Path(path) if path is not None else RESHAPED_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Reshaped data not found at {path}. "
            "Run `python -m helpers.reshape` first."
        )
    print("Loading reshaped data from", path)
    return pd.read_csv(path)
