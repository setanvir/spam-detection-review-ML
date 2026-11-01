import re
from pathlib import Path
import pandas as pd


def _clean(t):
    t = t.lower()
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def load(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        text_col = max(df.select_dtypes(exclude="number").columns, key=lambda c: df[c].astype(str).str.len().mean())
        label_col = [c for c in df.columns if c != text_col and df[c].nunique() <= 2][0]
        df = df[[label_col, text_col]]
    else:
        df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"], encoding="latin-1")
        df = df[["label", "text"]]
    df.columns = ["label", "text"]
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    if df["label"].dtype == object:
        df["label"] = df["label"].str.strip().str.lower().map({"spam": 1, "ham": 0})
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str).map(_clean)
    return df
