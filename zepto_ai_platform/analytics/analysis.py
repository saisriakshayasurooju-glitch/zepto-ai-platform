from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import skew
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, mean_absolute_error,
    mean_squared_error, r2_score
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
CSV = ROOT / "titanic.csv"
RANDOM_STATE = 42


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close()


def main():
    # Required: load exactly once and save immediately.
    df = sns.load_dataset("titanic")
    df.to_csv(CSV, index=False)

    print("Shape:", df.shape)
    print("\nINFO:")
    df.info()
    print("\nDESCRIBE:")
    print(df.describe(include="all").transpose())
    print("\nMissing %:")
    print((df.isna().mean() * 100).sort_values(ascending=False))

    # Missing-value strategy.
    missing = df.isna().mean() * 100
    low = [c for c in missing.index if 0 < missing[c] < 5]
    medium = [c for c in missing.index if 5 <= missing[c] <= 30]
    high = [c for c in missing.index if missing[c] > 30]
    print("\nMissing strategy")
    print("Drop rows (<5%):", low)
    print("Impute (5-30%):", medium)
    print("High missing (>30%) explicit decision:", high)

    # For EDA we use a working copy. High-missing deck/cabin is not used
    # for modeling, so it is excluded rather than artificially imputed.
    work = df.copy()
    if "age" in work:
        work["age"] = work["age"].fillna(work["age"].median())
    if "embarked" in work:
        work["embarked"] = work["embarked"].fillna(work["embarked"].mode()[0])

    # Histograms and boxplots.
    for col in ["age", "fare"]:
        plt.figure(figsize=(7, 4))
        sns.histplot(work[col], kde=True)
        plt.title(f"Histogram of {col}")
        savefig(f"hist_{col}.png")

        plt.figure(figsize=(7, 4))
        sns.boxplot(x=work[col])
        plt.title(f"Boxplot of {col}")
        savefig(f"box_{col}.png")

    # IQR outlier counts.
    for col in ["age", "fare"]:
        q1, q3 = work[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = ((work[col] < lo) | (work[col] > hi)).sum()
        print(f"{col} IQR outliers: {count}")

    # Fare mean/median/mode/skewness and interpretation.
    fare_mean = work["fare"].mean()
    fare_median = work["fare"].median()
    fare_mode = work["fare"].mode().iloc[0]
    fare_skew = work["fare"].skew()
    print("\nFare mean:", fare_mean)
    print("Fare median:", fare_median)
    print("Fare mode:", fare_mode)
    print("Fare skewness:", fare_skew)

    if fare_mean > fare_median:
        print("Interpretation: mean > median, consistent with positive/right skew.")
    elif fare_mean < fare_median:
        print("Interpretation: mean < median, consistent with negative/left skew.")
    else:
        print("Interpretation: mean and median are approximately equal.")

    # Survival rates.
    print("\nSurvival by sex:")
    print(work.groupby("sex")["survived"].mean())
    print("\nSurvival by pclass:")
    print(work.groupby("pclass")["survived"].mean())
    print("\nSurvival by sex + pclass:")
    print(work.groupby(["sex", "pclass"])["survived"].mean())

    # Exact required six-column correlation matrix.
    corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr = work[corr_cols].corr()
    print("\nRequired correlation matrix:")
    print(corr)

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Titanic Correlation Matrix")
    savefig("correlation_heatmap.png")

    pairs = []
    for i in range(len(corr_cols)):
        for j in range(i + 1, len(corr_cols)):
            pairs.append((abs(corr.iloc[i, j]), corr_cols[i], corr_cols[j], corr.iloc[i, j]))
    pairs.sort(reverse=True)
    print("\nTop two absolute off-diagonal correlations:")
    print(pairs[:2])

    # At least four multivariate charts.
    plt.figure(figsize=(7, 5))
    sns.barplot(data=work, x="pclass", y="survived", hue="sex")
    plt.title("Survival Rate by Class and Sex")
    savefig("multi_01_survival_class_sex.png")

    plt.figure(figsize=(7, 5))
    sns.boxplot(data=work, x="pclass", y="fare", hue="survived")
    plt.title("Fare by Class and Survival")
    savefig("multi_02_fare_class_survival.png")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=work, x="age", y="fare", hue="survived", style="sex")
    plt.title("Age vs Fare by Survival and Sex")
    savefig("multi_03_age_fare_survival.png")

    plt.figure(figsize=(8, 5))
    sns.pointplot(data=work, x="pclass", y="survived", hue="embarked")
    plt.title("Survival by Class and Embarkation Port")
    savefig("multi_04_class_embarked_survival.png")

    # Exploratory z-score standardization of age and fare.
    standardized = work[["age", "fare"]].copy()
    before = standardized.describe().loc[["mean", "std"]]
    standardized = (standardized - standardized.mean()) / standardized.std(ddof=0)
    after = standardized.describe().loc[["mean", "std"]]
    print("\nBefore standardization:\n", before)
    print("\nAfter standardization:\n", after)

    # ---------------- Classification ----------------
    # Drop high-missing cabin and non-predictive identifiers/name/ticket.
    features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    X = df[features].copy()
    y = df["survived"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    numeric = ["pclass", "age", "sibsp", "parch", "fare"]
    categorical = ["sex", "embarked"]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, oob_score=True
        )
    }

    def evaluate(name, estimator, Xtr=X_train, ytr=y_train):
        pipe = Pipeline([("preprocess", clone(preprocessor)), ("model", estimator)])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

        result = {
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, prob) if prob is not None else np.nan
        }
        print("\n", name, result)
        print("Confusion matrix:\n", confusion_matrix(y_test, pred))
        return pipe, result

    fitted = {}
    results = []
    for name, model in models.items():
        pipe, result = evaluate(name, model)
        fitted[name] = pipe
        results.append(result)

    # Decision-tree visualization.
    dt = fitted["Decision Tree"]
    feature_names = dt.named_steps["preprocess"].get_feature_names_out()
    plt.figure(figsize=(20, 10))
    plot_tree(
        dt.named_steps["model"],
        feature_names=feature_names,
        class_names=["not survived", "survived"],
        filled=False,
        max_depth=3,
        fontsize=7
    )
    plt.title("Decision Tree (first three levels shown)")
    savefig("decision_tree.png")

    # Class imbalance experiment:
    # 1) baseline
    # 2) class_weight balanced
    # 3) simple deterministic oversampling on TRAINING data only.
    baseline_pipe, baseline_result = evaluate(
        "Random Forest - baseline",
        RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    )
    balanced_pipe, balanced_result = evaluate(
        "Random Forest - class_weight=balanced",
        RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE
        )
    )

    train = X_train.copy()
    train["__target__"] = y_train.values
    majority = train[train["__target__"] == train["__target__"].value_counts().idxmax()]
    minority = train[train["__target__"] == train["__target__"].value_counts().idxmin()]
    minority_up = minority.sample(
        n=len(majority), replace=True, random_state=RANDOM_STATE
    )
    balanced_train = pd.concat([majority, minority_up]).sample(
        frac=1, random_state=RANDOM_STATE
    )
    X_over = balanced_train.drop(columns="__target__")
    y_over = balanced_train["__target__"]

    over_pipe, over_result = evaluate(
        "Random Forest - training oversampling",
        RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        X_over,
        y_over
    )
    print("\nImbalance comparison:")
    print(pd.DataFrame([baseline_result, balanced_result, over_result]))

    # GridSearchCV RF with pipeline so preprocessing is fitted inside CV folds.
    rf_pipe = Pipeline([
        ("preprocess", clone(preprocessor)),
        ("model", RandomForestClassifier(
            random_state=RANDOM_STATE, oob_score=True
        ))
    ])

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
        "model__max_features": ["sqrt", "log2"]
    }

    grid = GridSearchCV(
        rf_pipe, param_grid=param_grid, cv=5, scoring="roc_auc", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_rf = grid.best_estimator_
    print("\nBest RF params:", grid.best_params_)
    print("Best RF CV ROC-AUC:", grid.best_score_)
    print("Best RF OOB score:", best_rf.named_steps["model"].oob_score_)

    # Persist complete fitted preprocessing + estimator pipeline.
    model_path = ROOT / "best_pipeline.joblib"
    joblib.dump(best_rf, model_path)
    reloaded = joblib.load(model_path)
    print("Reloaded pipeline test predictions:", reloaded.predict(X_test.head(5)))

    # ---------------- Regression: predict fare ----------------
    reg_features = ["pclass", "sex", "age", "sibsp", "parch", "embarked", "survived"]
    reg_df = df[reg_features + ["fare"]].copy()
    Xr = reg_df[reg_features]
    yr = reg_df["fare"]

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, yr, test_size=0.20, random_state=RANDOM_STATE
    )

    reg_pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), ["pclass", "age", "sibsp", "parch", "survived"]),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), ["sex", "embarked"])
    ])

    reg_pipe = Pipeline([
        ("preprocess", reg_pre),
        ("model", LinearRegression())
    ])
    reg_pipe.fit(Xr_train, yr_train)
    pred = reg_pipe.predict(Xr_test)

    mae = mean_absolute_error(yr_test, pred)
    rmse = np.sqrt(mean_squared_error(yr_test, pred))
    r2 = r2_score(yr_test, pred)
    n = len(yr_test)
    p = len(reg_pipe.named_steps["preprocess"].get_feature_names_out())
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan

    print("\nRegression metrics:")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)
    print("Adjusted R2:", adj_r2)

    residuals = yr_test.to_numpy() - pred
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=pred, y=residuals)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted fare")
    plt.ylabel("Residual")
    plt.title("Fare Regression Residual Plot")
    savefig("fare_residuals.png")

    print("\nRegression interpretation:")
    print("A funnel-shaped residual spread would indicate heteroscedasticity; "
          "randomly scattered residuals around zero are more consistent with constant variance.")

    print("\nAnalysis complete. Figures:", FIG)
    print("Saved model:", model_path)


if __name__ == "__main__":
    main()
