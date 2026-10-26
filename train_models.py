"""
Train the five SpamShield classifiers on the UCI Spambase dataset.

Run once:  python train_models.py
Outputs  :  model/*.joblib, model/metrics.json, model/meta.json,
            test_data.csv, data/spambase_full.csv
"""
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, roc_curve)

from spamshield_features import FEATURE_NAMES, TARGET, log1p_transform

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)
(ROOT / "data").mkdir(exist_ok=True)
SEED = 42


def load_data() -> pd.DataFrame:
    raw = ROOT / "data" / "spambase.tsv.gz"
    df = pd.read_csv(raw, sep="\t")
    X = df.drop(columns=["target"])
    X.columns = FEATURE_NAMES
    out = X.copy()
    out[TARGET] = df["target"].astype(int)
    return out


def make_models():
    """(pipeline, hyper-parameter grid) for every classifier."""
    def prep(scale=True):
        steps = [("log", FunctionTransformer(log1p_transform))]
        if scale:
            steps.append(("scale", StandardScaler()))
        return steps

    return {
        "Logistic Regression": (
            Pipeline(prep() + [("clf", LogisticRegression(max_iter=2000, random_state=SEED))]),
            {"clf__C": [0.01, 0.1, 1, 10, 100]},
        ),
        "Decision Tree": (
            # trees are scale-invariant -> raw features, no transform needed
            Pipeline([("clf", DecisionTreeClassifier(random_state=SEED))]),
            {"clf__max_depth": [4, 6, 8, 10, 14, None],
             "clf__min_samples_leaf": [1, 3, 5, 10]},
        ),
        "KNN": (
            Pipeline(prep() + [("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [3, 5, 7, 11, 15, 21],
             "clf__weights": ["uniform", "distance"]},
        ),
        "Naive Bayes": (
            Pipeline(prep(scale=False) + [("clf", GaussianNB())]),
            {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]},
        ),
        "Support Vector Machine": (
            Pipeline(prep() + [("clf", SVC(kernel="rbf", probability=False, random_state=SEED))]),
            {"clf__C": [0.5, 1, 5, 10, 50], "clf__gamma": ["scale", 0.005, 0.02]},
        ),
    }


def evaluate(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    df = load_data()
    print("Dataset:", df.shape, "| spam share: %.3f" % df[TARGET].mean())
    df.to_csv(ROOT / "data" / "spambase_full.csv", index=False)

    X, y = df[FEATURE_NAMES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    # Hold-out test file shipped with the repo (and used by the Streamlit app)
    test_df = X_te.copy()
    test_df[TARGET] = y_te.values
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, extras = {}, {}

    for name, (pipe, grid) in make_models().items():
        t0 = time.time()
        gs = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1)
        gs.fit(X_tr, y_tr)
        best = gs.best_estimator_

        # SVM: refit best params with probability estimates (Platt scaling)
        if name == "Support Vector Machine":
            best.set_params(clf__probability=True)
            best.fit(X_tr, y_tr)

        proba = best.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        m = evaluate(y_te, pred, proba)
        results[name] = m

        fpr, tpr, _ = roc_curve(y_te, proba)
        step = max(1, len(fpr) // 120)
        extras[name] = {
            "best_params": {k.replace("clf__", ""): (None if v is None else v)
                            for k, v in gs.best_params_.items()},
            "cv_f1_mean": float(gs.best_score_),
            "cv_f1_std": float(gs.cv_results_["std_test_score"][gs.best_index_]),
            "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
            "roc": {"fpr": fpr[::step].tolist(), "tpr": tpr[::step].tolist()},
            "train_seconds": round(time.time() - t0, 1),
        }
        joblib.dump(best, MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib")
        print(f"{name:24s}", {k: round(v, 4) for k, v in m.items()},
              extras[name]["best_params"], f"{extras[name]['train_seconds']}s")

    # Global feature importance on the overall winner (permutation, test set)
    winner = max(results, key=lambda k: results[k]["MCC"])
    best = joblib.load(MODEL_DIR / f"{winner.lower().replace(' ', '_')}.joblib")
    pi = permutation_importance(best, X_te, y_te, scoring="roc_auc",
                                n_repeats=5, random_state=SEED, n_jobs=-1)
    imp = (pd.Series(pi.importances_mean, index=FEATURE_NAMES)
           .sort_values(ascending=False).head(15))

    # class-wise feature means (used for the "why was it flagged" explainer)
    class_means = X_tr.groupby(y_tr.values).mean()
    meta = {
        "winner": winner,
        "dataset": {"rows": int(len(df)), "features": len(FEATURE_NAMES),
                    "spam": int(y.sum()), "ham": int((1 - y).sum()),
                    "train_rows": int(len(X_tr)), "test_rows": int(len(X_te))},
        "permutation_importance": imp.round(5).to_dict(),
        "ham_mean": class_means.loc[0].to_dict(),
        "spam_mean": class_means.loc[1].to_dict(),
        "feature_std": X_tr.std().to_dict(),
    }
    json.dump({"metrics": results, "extras": extras},
              open(MODEL_DIR / "metrics.json", "w"), indent=2)
    json.dump(meta, open(MODEL_DIR / "meta.json", "w"), indent=2)
    print("\nWinner (by MCC):", winner)
    print(imp.round(4).head(8))


if __name__ == "__main__":
    main()
