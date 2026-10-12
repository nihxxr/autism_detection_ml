"""Train and evaluate a classifier on the UCI Autism Screening Adult dataset.

Usage:
    python scripts/train_model.py                 # downloads UCI dataset 426
    python scripts/train_model.py data/autism.csv # or use a local CSV/ARFF

Writes models/aq10_model.joblib and models/metrics.json.

What this shows: in the UCI data the label (Class/ASD) is assigned from the
AQ-10 total itself (score above 6 means "YES"). Any model trained on the ten
item scores can reach near-perfect accuracy by learning that sum, so the
headline accuracy says nothing about real diagnostic power. The script prints
the rule baseline next to the models so the comparison is visible.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parent.parent
FEATURES = [f"A{i}_Score" for i in range(1, 11)]


def load_data(path: str | None) -> pd.DataFrame:
    if path is None:
        from ucimlrepo import fetch_ucirepo

        ds = fetch_ucirepo(id=426)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    elif path.endswith(".arff"):
        from scipy.io import arff

        raw, _ = arff.loadarff(path)
        df = pd.DataFrame(raw)
        for col in df.select_dtypes([object]):
            df[col] = df[col].str.decode("utf-8")
    else:
        df = pd.read_csv(path)
    return df


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    rename = {}
    for col in df.columns:
        low = col.lower().replace(" ", "_")
        for i in range(1, 11):
            if low in (f"a{i}_score", f"a{i}"):
                rename[col] = f"A{i}_Score"
        if low in ("class/asd", "class_asd", "class"):
            rename[col] = "target"
    df = df.rename(columns=rename)
    missing = [c for c in FEATURES + ["target"] if c not in df.columns]
    if missing:
        raise SystemExit(f"Dataset is missing columns: {missing}")

    df = df[FEATURES + ["target"]].replace("?", np.nan).dropna()
    X = df[FEATURES].astype(int)
    y = df["target"].astype(str).str.strip().str.upper().map({"YES": 1, "NO": 0, "1": 1, "0": 0})
    keep = y.notna()
    return X[keep], y[keep].astype(int)


def evaluate(name: str, y: pd.Series, pred: np.ndarray, proba: np.ndarray | None) -> dict:
    out = {
        "model": name,
        "accuracy": round(accuracy_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
    }
    if proba is not None:
        out["roc_auc"] = round(roc_auc_score(y, proba), 4)
    return out


def main() -> None:
    X, y = prepare(load_data(sys.argv[1] if len(sys.argv) > 1 else None))
    print(f"{len(X)} rows, {int(y.sum())} positive ({y.mean():.1%})")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    total = X.sum(axis=1)
    results.append(evaluate("Rule: AQ-10 total > 6", y, (total > 6).astype(int).values, total.values / 10))

    candidates = {
        "Logistic regression": LogisticRegression(max_iter=1000),
        "Random forest": RandomForestClassifier(n_estimators=300, random_state=42),
    }
    for name, est in candidates.items():
        proba = cross_val_predict(est, X, y, cv=cv, method="predict_proba")[:, 1]
        results.append(evaluate(name, y, (proba >= 0.5).astype(int), proba))

    for r in results:
        print(f"  {r['model']:<26} acc {r['accuracy']:.3f}  f1 {r['f1']:.3f}  auc {r.get('roc_auc', float('nan')):.3f}")

    final = LogisticRegression(max_iter=1000).fit(X, y)
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(final, models_dir / "aq10_model.joblib")
    (models_dir / "metrics.json").write_text(
        json.dumps({"rows": len(X), "positive_rate": round(float(y.mean()), 4), "cv_5fold": results}, indent=2)
    )
    print(f"Saved {models_dir / 'aq10_model.joblib'}")


if __name__ == "__main__":
    main()
