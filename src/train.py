import argparse, hashlib, json, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from src.data import load


def get_models(seed):
    return {
        "Naive Bayes": (MultinomialNB(), None),
        "KNN": (KNeighborsClassifier(), {"n_neighbors": [5, 7, 9]}),
        "SVM": (SVC(random_state=seed), {"C": [1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale"]}),
        "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=seed), {"C": [0.1, 1, 10]}),
        "Decision Tree": (DecisionTreeClassifier(random_state=seed), {"max_depth": [5, 10, 20], "criterion": ["gini", "entropy"]}),
        "Random Forest": (RandomForestClassifier(random_state=seed, n_jobs=-1), {"n_estimators": [50, 100, 200], "max_depth": [None, 20]}),
        "AdaBoost": (AdaBoostClassifier(random_state=seed), {"n_estimators": [50, 100]}),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=seed), {"n_estimators": [100], "max_depth": [3]}),
    }


def run(data_path, out_dir="results", seed=42, test_size=0.2, max_features=5000):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load(data_path)
    X = CountVectorizer(stop_words="english", max_features=max_features).fit_transform(df["text"])
    y = df["label"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    rows, cms = [], {}
    for name, (m, grid) in get_models(seed).items():
        t0 = time.time()
        best = {}
        if grid:
            g = GridSearchCV(m, grid, cv=3, scoring="f1", n_jobs=-1).fit(Xtr, ytr)
            m, best = g.best_estimator_, g.best_params_
        else:
            m.fit(Xtr, ytr)
        p = m.predict(Xte)
        rows.append({"Model": name, "Accuracy": accuracy_score(yte, p), "Precision": precision_score(yte, p),
                     "Recall": recall_score(yte, p), "F1": f1_score(yte, p),
                     "Time(s)": round(time.time() - t0, 2), "Best Params": json.dumps(best)})
        cms[name] = confusion_matrix(yte, p)

    res = pd.DataFrame(rows).sort_values(["F1", "Model"], ascending=[False, True]).reset_index(drop=True)
    res.to_csv(out / "results.csv", index=False)

    ax = res.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]].plot.bar(figsize=(11, 5), rot=30)
    ax.set_ylim(0, 1.05)
    ax.set_title("Spam Detection: Model Comparison")
    plt.tight_layout()
    plt.savefig(out / "comparison.png", dpi=150)
    plt.close()

    fig, axs = plt.subplots(2, 4, figsize=(14, 6))
    for a, (n, c) in zip(axs.ravel(), cms.items()):
        a.imshow(c, cmap="Blues")
        a.set_title(n, fontsize=9)
        a.set_xticks([0, 1]); a.set_yticks([0, 1])
        a.set_xticklabels(["ham", "spam"]); a.set_yticklabels(["ham", "spam"])
        for i in range(2):
            for j in range(2):
                a.text(j, i, c[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrices.png", dpi=150)
    plt.close()

    meta = {
        "data_file": str(data_path), "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "n_samples": int(len(df)), "n_spam": int(y.sum()), "n_ham": int((y == 0).sum()),
        "seed": seed, "test_size": test_size, "max_features": max_features,
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
    }
    (out / "run_info.json").write_text(json.dumps(meta, indent=2))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/emails.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print(run(a.data, a.out, a.seed).drop(columns="Best Params").round(4).to_string(index=False))
