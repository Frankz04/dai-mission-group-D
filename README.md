# DAI-Tennis-Prediction
Analysing historical tennis data in order to predict future winners and outcomes

# Fatigue, Form, and the Efficiency of Tennis Rankings
## Predicting ATP Match Outcomes (1991–2024)

We ask whether a match's winner can be predicted from pre-match information, and whether a model can beat the "higher-ranked player wins" benchmark. Because the ATP ranking is the sport's consensus signal of player quality, beating it is an information-efficiency test, directly analogous to market-efficiency tests in finance.

**Research question**: Using ATP tour-level matches since 1991, we ask: how accurately can a match's winner be predicted from pre-match information, and can a model beat the "higher-ranked player wins" benchmark? Beating that benchmark would mean the ranking leaves exploitable information on the table; failing to beat it would mean the ranking is already a near-efficient summary of player quality.

To answer this well, two supporting analyses feed the prediction task:

**1**. A causal check of whether fatigue (a long previous-round match) genuinely lowers next-match win probability — i.e. whether it is a real driver worth encoding as a feature rather than a spurious correlate; and

**2**. an unsupervised decomposition of players into playing-style archetypes, used both as predictive features and to characterise how different player types win.

**Why it matters:** The ranking acts as the market's consensus signal of player quality, so asking whether a model can out-predict it is an information-efficiency test, directly analogous to market-efficiency tests in finance. The fatigue sub-analysis additionally speaks to tournament scheduling and athlete welfare — a mechanism-design question for the sport's multi-billion-euro economy.
