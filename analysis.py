from helpers.loading import load_data

df = load_data()

print("rows, cols:", df.shape)
print("first columns:", df.columns.tolist()[:25])

# Basic data overview / distributions
df["year"] = df["tourney_date"] // 10000
print("year range:", int(df["year"].min()), "-", int(df["year"].max()))

missing = df.isna().mean().sort_values(ascending=False).head(10)
print("\nTop 10 missing-value ratios:")
print(missing)

def show_counts(col, n=10):
    if col in df.columns:
        print(f"\n{col} value counts (top {n}):")
        print(df[col].value_counts(dropna=False).head(n))

show_counts("surface")
show_counts("round")
show_counts("tourney_level")
show_counts("best_of")

numeric_cols = df.select_dtypes(include=[np.number]).columns
key_numeric = [c for c in [
    "minutes",
    "winner_rank", "loser_rank",
    "winner_age", "loser_age",
    "winner_ht", "loser_ht"
] if c in numeric_cols]

print("\nNumeric summary (key columns):")
print(df[key_numeric].describe().T if key_numeric else df[numeric_cols].describe().T)

print("\nYear distribution (matches per year):")
print(df["year"].value_counts().sort_index())

# Reshape winner/loser -> symmetric player/opponent rows (prevents leakage):
# w = df.rename(columns=lambda c: c.replace("winner_","player_").replace("loser_","opp_")); w["win"] = 1
# l = df.rename(columns=lambda c: c.replace("loser_","player_").replace("winner_","opp_")); l["win"] = 0
# long_df = pd.concat([w, l], ignore_index=True)
#
# Fatigue treatment (causal): did the player's PREVIOUS match run long? (look one match back)
# long_df = long_df.sort_values(["player_id", "tourney_date", "round"])
# long_df["prev_minutes"]    = long_df.groupby("player_id")["minutes"].shift(1)
# long_df["long_prev_match"] = (long_df["prev_minutes"] >= 180).astype(int)   # >= 3 hours
# print(long_df["win"].mean(), long_df["long_prev_match"].mean())
