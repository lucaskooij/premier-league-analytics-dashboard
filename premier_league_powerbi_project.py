import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances


# -----------------------------
# 1. LOAD DATA
# -----------------------------

df = pd.read_csv("epl_final.csv")

df["MatchDate"] = pd.to_datetime(df["MatchDate"])

print(df.head())
print(df.info())


# -----------------------------
# 2. CREATE ORIGINAL PROJECT FEATURES
# -----------------------------

# Original DS3000-style match vector:
# x = [HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR]

feature_cols = [
    "HomeShots",
    "AwayShots",
    "HomeShotsOnTarget",
    "AwayShotsOnTarget",
    "HomeFouls",
    "AwayFouls",
    "HomeCorners",
    "AwayCorners",
    "HomeYellowCards",
    "AwayYellowCards",
    "HomeRedCards",
    "AwayRedCards"
]

# Convert result to numeric version:
# Home win = 1, Draw = 0, Away win = -1

result_map = {
    "H": 1,
    "D": 0,
    "A": -1
}

df["ResultNumeric"] = df["FullTimeResult"].map(result_map)


# -----------------------------
# 3. HALFTIME + SECOND HALF FEATURES
# -----------------------------

df["HT_GoalDiff"] = df["HalfTimeHomeGoals"] - df["HalfTimeAwayGoals"]
df["FT_GoalDiff"] = df["FullTimeHomeGoals"] - df["FullTimeAwayGoals"]

df["SecondHalfHomeGoals"] = df["FullTimeHomeGoals"] - df["HalfTimeHomeGoals"]
df["SecondHalfAwayGoals"] = df["FullTimeAwayGoals"] - df["HalfTimeAwayGoals"]
df["SecondHalfGoalDiff"] = df["SecondHalfHomeGoals"] - df["SecondHalfAwayGoals"]

df["HomeShotDiff"] = df["HomeShots"] - df["AwayShots"]
df["HomeSOTDiff"] = df["HomeShotsOnTarget"] - df["AwayShotsOnTarget"]
df["HomeCornerDiff"] = df["HomeCorners"] - df["AwayCorners"]
df["HomeFoulDiff"] = df["HomeFouls"] - df["AwayFouls"]
df["HomeYellowDiff"] = df["HomeYellowCards"] - df["AwayYellowCards"]
df["HomeRedDiff"] = df["HomeRedCards"] - df["AwayRedCards"]

# Halftime situation
df["HomeLeadingAtHalf"] = df["HT_GoalDiff"] > 0
df["AwayLeadingAtHalf"] = df["HT_GoalDiff"] < 0
df["DrawAtHalf"] = df["HT_GoalDiff"] == 0

# Comeback logic
df["HomeComebackWin"] = (df["HT_GoalDiff"] < 0) & (df["FullTimeResult"] == "H")
df["AwayComebackWin"] = (df["HT_GoalDiff"] > 0) & (df["FullTimeResult"] == "A")
df["AnyComebackWin"] = df["HomeComebackWin"] | df["AwayComebackWin"]

# Blown lead logic
df["HomeBlewLead"] = (df["HT_GoalDiff"] > 0) & (df["FullTimeResult"] != "H")
df["AwayBlewLead"] = (df["HT_GoalDiff"] < 0) & (df["FullTimeResult"] != "A")
df["AnyBlownLead"] = df["HomeBlewLead"] | df["AwayBlewLead"]

# Did the match result change from halftime to fulltime?
df["ResultChangedAfterHalf"] = df["HalfTimeResult"] != df["FullTimeResult"]


# -----------------------------
# 4. MODEL 1: ORIGINAL STATS ONLY
# -----------------------------

X_original = df[feature_cols]
y = df["FullTimeResult"]

X_train, X_test, y_train, y_test = train_test_split(
    X_original,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

scaler_original = StandardScaler()
X_train_scaled = scaler_original.fit_transform(X_train)
X_test_scaled = scaler_original.transform(X_test)

log_model_original = LogisticRegression(max_iter=1000)
log_model_original.fit(X_train_scaled, y_train)

y_pred_original = log_model_original.predict(X_test_scaled)

print("\nMODEL 1: ORIGINAL MATCH STATS ONLY")
print("Accuracy:", accuracy_score(y_test, y_pred_original))
print(classification_report(y_test, y_pred_original))


# -----------------------------
# 5. MODEL 2: STATS + HALFTIME FEATURES
# -----------------------------

halftime_features = [
    "HT_GoalDiff",
    "HalfTimeHomeGoals",
    "HalfTimeAwayGoals",
    "SecondHalfHomeGoals",
    "SecondHalfAwayGoals",
    "HomeShotDiff",
    "HomeSOTDiff",
    "HomeCornerDiff",
    "HomeFoulDiff",
    "HomeYellowDiff",
    "HomeRedDiff"
]

all_model_features = feature_cols + halftime_features

X_full = df[all_model_features]

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_full,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

scaler_full = StandardScaler()
X_train2_scaled = scaler_full.fit_transform(X_train2)
X_test2_scaled = scaler_full.transform(X_test2)

log_model_full = LogisticRegression(max_iter=1000)
log_model_full.fit(X_train2_scaled, y_train2)

y_pred_full = log_model_full.predict(X_test2_scaled)

print("\nMODEL 2: MATCH STATS + HALFTIME FEATURES")
print("Accuracy:", accuracy_score(y_test2, y_pred_full))
print(classification_report(y_test2, y_pred_full))


# -----------------------------
# 6. RANDOM FOREST MODEL
# -----------------------------

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=8
)

rf_model.fit(X_train2, y_train2)
rf_pred = rf_model.predict(X_test2)

print("\nMODEL 3: RANDOM FOREST")
print("Accuracy:", accuracy_score(y_test2, rf_pred))
print(classification_report(y_test2, rf_pred))


# -----------------------------
# 7. FEATURE IMPORTANCE FOR POWER BI
# -----------------------------

importance_df = pd.DataFrame({
    "Feature": all_model_features,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFEATURE IMPORTANCE")
print(importance_df)


# -----------------------------
# 8. CONFUSION MATRIX FOR POWER BI
# -----------------------------

cm = confusion_matrix(y_test2, rf_pred, labels=["H", "D", "A"])

confusion_df = pd.DataFrame(
    cm,
    index=["Actual_HomeWin", "Actual_Draw", "Actual_AwayWin"],
    columns=["Predicted_HomeWin", "Predicted_Draw", "Predicted_AwayWin"]
)

print("\nCONFUSION MATRIX")
print(confusion_df)


# -----------------------------
# 9. ORIGINAL DS3000 LINEAR ALGEBRA MODEL
# -----------------------------

# This is the closest version to your original project:
# A w = b
# A = match stats matrix
# w = weights
# b = result vector

A = df[feature_cols].values
b = df["ResultNumeric"].values

linear_model = LinearRegression()
linear_model.fit(A, b)

df["LinearPredictionScore"] = linear_model.predict(A)

def classify_linear_score(score):
    if score > 0.25:
        return "H"
    elif score < -0.25:
        return "A"
    else:
        return "D"

df["LinearPredictedResult"] = df["LinearPredictionScore"].apply(classify_linear_score)

linear_accuracy = accuracy_score(df["FullTimeResult"], df["LinearPredictedResult"])

print("\nDS3000 LINEAR ALGEBRA MODEL")
print("Accuracy:", linear_accuracy)

weights_df = pd.DataFrame({
    "Feature": feature_cols,
    "Weight": linear_model.coef_
}).sort_values(by="Weight", ascending=False)

print("\nLINEAR MODEL WEIGHTS")
print(weights_df)


# -----------------------------
# 10. SIMILAR MATCH FINDER USING NORMS
# -----------------------------

scaled_vectors = StandardScaler().fit_transform(df[feature_cols])

distance_matrix = euclidean_distances(scaled_vectors)

def find_similar_matches(match_index, top_n=5):
    distances = distance_matrix[match_index]

    similar_indices = np.argsort(distances)[1:top_n + 1]

    similar_matches = df.iloc[similar_indices][[
        "Season",
        "MatchDate",
        "HomeTeam",
        "AwayTeam",
        "FullTimeHomeGoals",
        "FullTimeAwayGoals",
        "FullTimeResult",
        "HalfTimeHomeGoals",
        "HalfTimeAwayGoals",
        "HalfTimeResult"
    ]].copy()

    similar_matches["Distance"] = distances[similar_indices]

    return similar_matches

example_similar_matches = find_similar_matches(0, top_n=5)

print("\nSIMILAR MATCHES TO FIRST MATCH")
print(example_similar_matches)


# -----------------------------
# 11. TEAM-LEVEL POWER BI TABLE
# -----------------------------

home_team_stats = df.groupby(["Season", "HomeTeam"]).agg(
    HomeMatches=("HomeTeam", "count"),
    HomeWins=("FullTimeResult", lambda x: (x == "H").sum()),
    HomeDraws=("FullTimeResult", lambda x: (x == "D").sum()),
    HomeLosses=("FullTimeResult", lambda x: (x == "A").sum()),
    HomeGoals=("FullTimeHomeGoals", "sum"),
    HomeGoalsAllowed=("FullTimeAwayGoals", "sum"),
    HomeComebackWins=("HomeComebackWin", "sum"),
    HomeBlownLeads=("HomeBlewLead", "sum")
).reset_index().rename(columns={"HomeTeam": "Team"})

away_team_stats = df.groupby(["Season", "AwayTeam"]).agg(
    AwayMatches=("AwayTeam", "count"),
    AwayWins=("FullTimeResult", lambda x: (x == "A").sum()),
    AwayDraws=("FullTimeResult", lambda x: (x == "D").sum()),
    AwayLosses=("FullTimeResult", lambda x: (x == "H").sum()),
    AwayGoals=("FullTimeAwayGoals", "sum"),
    AwayGoalsAllowed=("FullTimeHomeGoals", "sum"),
    AwayComebackWins=("AwayComebackWin", "sum"),
    AwayBlownLeads=("AwayBlewLead", "sum")
).reset_index().rename(columns={"AwayTeam": "Team"})

team_stats = pd.merge(
    home_team_stats,
    away_team_stats,
    on=["Season", "Team"],
    how="outer"
).fillna(0)

team_stats["TotalMatches"] = team_stats["HomeMatches"] + team_stats["AwayMatches"]
team_stats["TotalWins"] = team_stats["HomeWins"] + team_stats["AwayWins"]
team_stats["TotalDraws"] = team_stats["HomeDraws"] + team_stats["AwayDraws"]
team_stats["TotalLosses"] = team_stats["HomeLosses"] + team_stats["AwayLosses"]

team_stats["TotalGoals"] = team_stats["HomeGoals"] + team_stats["AwayGoals"]
team_stats["TotalGoalsAllowed"] = team_stats["HomeGoalsAllowed"] + team_stats["AwayGoalsAllowed"]
team_stats["GoalDifference"] = team_stats["TotalGoals"] - team_stats["TotalGoalsAllowed"]

team_stats["TotalComebackWins"] = team_stats["HomeComebackWins"] + team_stats["AwayComebackWins"]
team_stats["TotalBlownLeads"] = team_stats["HomeBlownLeads"] + team_stats["AwayBlownLeads"]

team_stats["WinRate"] = team_stats["TotalWins"] / team_stats["TotalMatches"]
team_stats["ComebackRate"] = team_stats["TotalComebackWins"] / team_stats["TotalMatches"]
team_stats["BlownLeadRate"] = team_stats["TotalBlownLeads"] / team_stats["TotalMatches"]


# -----------------------------
# 12. PREDICTIONS TABLE FOR POWER BI
# -----------------------------

predictions_df = X_test2.copy()
predictions_df["ActualResult"] = y_test2.values
predictions_df["PredictedResult"] = rf_pred
predictions_df["CorrectPrediction"] = predictions_df["ActualResult"] == predictions_df["PredictedResult"]


# -----------------------------
# 13. EXPORT FILES FOR POWER BI
# -----------------------------

df.to_csv("powerbi_match_level_data.csv", index=False)
team_stats.to_csv("powerbi_team_stats.csv", index=False)
importance_df.to_csv("powerbi_feature_importance.csv", index=False)
confusion_df.to_csv("powerbi_confusion_matrix.csv")
predictions_df.to_csv("powerbi_predictions.csv", index=False)
weights_df.to_csv("powerbi_linear_weights.csv", index=False)
example_similar_matches.to_csv("powerbi_similar_matches_example.csv", index=False)

print("\nExport complete.")
print("Use these files in Power BI:")
print("1. powerbi_match_level_data.csv")
print("2. powerbi_team_stats.csv")
print("3. powerbi_feature_importance.csv")
print("4. powerbi_confusion_matrix.csv")
print("5. powerbi_predictions.csv")
print("6. powerbi_linear_weights.csv")
print("7. powerbi_similar_matches_example.csv")