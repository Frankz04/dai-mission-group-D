# Data loading & first inspection (proposal stage)
# Clone https://github.com/JeffSackmann/tennis_atp and place atp_matches_*.csv in ./data/
import pandas as pd, numpy as np, glob


def load_data():
    files = sorted(glob.glob("../data/atp_matches_*.csv"))
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df = df[df.tourney_date // 10000 >= 1991]            # match stats available from 1991
    return df
