"""
TASK 1: Credit Scoring Model
Dataset: Kaggle "Give Me Some Credit" -> put cs-training.csv next to this file.
If the file is missing, a synthetic dataset with the same columns is generated
so the script still runs end-to-end.

Run as a script or paste each '# %%' block into a Jupyter cell.
"""

# %% 1. IMPORTS
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # remove this line in Jupyter
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve,
                             precision_recall_curve, classification_report)

RANDOM_STATE = 42
OUT_DIR = "outputs_task1"
os.makedirs(OUT_DIR, exist_ok=True)
TARGET = "SeriousDlqin2yrs"   # 1 = defaulted (bad), 0 = good


# %% 2. LOAD DATA
def make_synthetic(n=20000, seed=RANDOM_STATE):
    """Fake data with the same schema, only used if the real CSV is absent."""
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 90, n)
    income = rng.lognormal(8.6, 0.6, n)
    income[rng.random(n) < 0.2] = np.nan
    util = np.clip(rng.beta(1, 3, n) * rng.choice([1, 1, 1, 3], n), 0, 5)
    debt_ratio = np.clip(rng.lognormal(-1, 1, n), 0, 5000)
    late30 = rng.poisson(0.4, n)
    late60 = rng.poisson(0.2, n)
    late90 = rng.poisson(0.15, n)
    logit = (-4.2 + 1.2 * util + 0.5 * late30 + 0.6 * late60 + 0.9 * late90
             - 0.02 * (age - 45) + 0.0003 * np.minimum(debt_ratio, 50))
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        TARGET: y,
        "RevolvingUtilizationOfUnsecuredLines": util,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": late30,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": income,
        "NumberOfOpenCreditLinesAndLoans": rng.poisson(8, n),
        "NumberOfTimes90DaysLate": late90,
        "NumberRealEstateLoansOrLines": rng.poisson(1, n),
        "NumberOfTime60-89DaysPastDueNotWorse": late60,
        "NumberOfDependents": rng.poisson(0.8, n).astype(float),
    })


if os.path.exists("cs-training.csv"):
    df = pd.read_csv("cs-training.csv", index_col=0)
    print("Loaded real dataset")
else:
    df = make_synthetic()
    print("cs-training.csv not found -> using SYNTHETIC data (numbers are not meaningful)")

print(df.shape)
print(df.head())


# %% 3. EDA
print("\nClass balance:\n", df[TARGET].value_counts(normalize=True).round(4))
print("\nMissing values:\n", df.isna().sum()[df.isna().sum() > 0])

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
df[TARGET].value_counts().plot.bar(ax=ax[0], title="Class distribution")
df["age"].plot.hist(bins=30, ax=ax[1], title="Age")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda.png", dpi=120)
plt.close()


# %% 4. CLEANING (row-wise rules only; imputing happens inside the Pipeline)
df = df[df["age"] >= 18].copy()

# In this dataset, 96 / 98 in the late-payment columns are data-entry codes
late_cols = ["NumberOfTime30-59DaysPastDueNotWorse",
             "NumberOfTimes90DaysLate",
             "NumberOfTime60-89DaysPastDueNotWorse"]
for c in late_cols:
    df.loc[df[c] >= 90, c] = np.nan

# Cap extreme outliers at the 99th percentile
for c in ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome"]:
    df[c] = df[c].clip(upper=df[c].quantile(0.99))


# %% 5. FEATURE ENGINEERING
def add_features(d):
    d = d.copy()
    late30, late60, late90 = late_cols
    d["TotalLatePayments"] = d[late30].fillna(0) + d[late60].fillna(0) + d[late90].fillna(0)
    d["HasLatePayment"] = (d["TotalLatePayments"] > 0).astype(int)
    d["SevereLate"] = (d[late90].fillna(0) > 0).astype(int)
    d["IncomeMissing"] = d["MonthlyIncome"].isna().astype(int)
    d["LogIncome"] = np.log1p(d["MonthlyIncome"])
    d["IncomePerDependent"] = d["MonthlyIncome"] / (d["NumberOfDependents"].fillna(0) + 1)
    d["DebtPerCreditLine"] = d["DebtRatio"] / (d["NumberOfOpenCreditLinesAndLoans"] + 1)
    d["HighUtilization"] = (d["RevolvingUtilizationOfUnsecuredLines"] > 0.8).astype(int)
    d["AgeGroup"] = pd.cut(d["age"], [0, 30, 45, 60, 200], labels=False)
    return d


df = add_features(df)
X = df.drop(columns=[TARGET])
y = df[TARGET]


# %% 6. SPLIT (stratified, so both sets keep the same default rate)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print("\nTrain:", X_train.shape, "Test:", X_test.shape)


# %% 7. MODELS (imputer + scaler live inside the pipeline -> no leakage)
def make_pipe(clf):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),   # needed for LR, harmless for trees
        ("clf", clf),
    ])


models = {
    "Logistic Regression": make_pipe(LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
    "Decision Tree": make_pipe(DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=50, class_weight="balanced",
        random_state=RANDOM_STATE)),
    "Random Forest": make_pipe(RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=20,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)),
}


# %% 8. TRAIN + EVALUATE
def evaluate(name, model, X_te, y_te, threshold=0.5):
    probs = model.predict_proba(X_te)[:, 1]
    preds = (probs >= threshold).astype(int)
    return {
        "Model": name,
        "Precision": precision_score(y_te, preds, zero_division=0),
        "Recall": recall_score(y_te, preds),
        "F1": f1_score(y_te, preds),
        "ROC-AUC": roc_auc_score(y_te, probs),
    }, probs, preds


results, all_probs = [], {}
for name, model in models.items():
    model.fit(X_train, y_train)
    res, probs, preds = evaluate(name, model, X_test, y_test)
    results.append(res)
    all_probs[name] = probs
    print(f"\n=== {name} ===")
    print(classification_report(y_test, preds, target_names=["Good", "Default"]))

results_df = pd.DataFrame(results).set_index("Model").round(4)
print("\nMODEL COMPARISON\n", results_df)
results_df.to_csv(f"{OUT_DIR}/model_comparison.csv")


# %% 9. PLOTS: ROC curves + confusion matrices
plt.figure(figsize=(6, 5))
for name, probs in all_probs.items():
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, probs):.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves"); plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curves.png", dpi=120)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, model) in zip(axes, models.items()):
    ConfusionMatrixDisplay(confusion_matrix(y_test, model.predict(X_test)),
                           display_labels=["Good", "Default"]).plot(ax=ax, colorbar=False)
    ax.set_title(name)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrices.png", dpi=120)
plt.close()


# %% 10. TUNE THE BEST MODEL (Random Forest) with cross-validation
best_name = results_df["ROC-AUC"].idxmax()
print(f"\nBest by ROC-AUC: {best_name}")

param_grid = {
    "clf__n_estimators": [200, 300],
    "clf__max_depth": [8, 12],
    "clf__min_samples_leaf": [10, 30],
}
grid = GridSearchCV(
    make_pipe(RandomForestClassifier(class_weight="balanced", n_jobs=-1,
                                     random_state=RANDOM_STATE)),
    param_grid, scoring="roc_auc",
    cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE), n_jobs=-1)
grid.fit(X_train, y_train)
tuned = grid.best_estimator_
print("Best params:", grid.best_params_)
tuned_res, tuned_probs, _ = evaluate("Random Forest (tuned)", tuned, X_test, y_test)
print({k: round(v, 4) if isinstance(v, float) else v for k, v in tuned_res.items()})


# %% 11. THRESHOLD TUNING (business decision: catch more defaulters)
prec, rec, thr = precision_recall_curve(y_test, tuned_probs)
f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
best_thr = thr[np.argmax(f1s)]
print(f"\nThreshold maximizing F1: {best_thr:.3f}")
res_thr, _, _ = evaluate("Tuned RF @ best threshold", tuned, X_test, y_test, best_thr)
print({k: round(v, 4) if isinstance(v, float) else v for k, v in res_thr.items()})

plt.figure(figsize=(6, 4))
plt.plot(thr, prec[:-1], label="Precision")
plt.plot(thr, rec[:-1], label="Recall")
plt.axvline(best_thr, color="gray", ls="--", label="Best F1 threshold")
plt.xlabel("Threshold"); plt.legend(); plt.title("Precision / Recall vs Threshold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/threshold_tradeoff.png", dpi=120)
plt.close()


# %% 12. EXPLAINABILITY
rf = tuned.named_steps["clf"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
imp.tail(10).plot.barh(figsize=(7, 5), title="Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=120)
plt.close()
print("\nTop features:\n", imp.sort_values(ascending=False).head(10).round(4))

lr = models["Logistic Regression"].named_steps["clf"]
coef = pd.Series(lr.coef_[0], index=X.columns).sort_values()
print("\nLogReg coefficients (+ raises default risk, - lowers it):\n", coef.round(3))

print(f"\nDone. Plots and tables saved in ./{OUT_DIR}/")
