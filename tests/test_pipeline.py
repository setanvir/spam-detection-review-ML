import random
import pandas as pd
from src.train import run


def _fake(path):
    random.seed(1)
    sp = ["win money now free prize click", "free offer limited cash claim", "buy cheap pills discount"]
    hm = ["meeting tomorrow project report", "please review the attached schedule", "lunch with team on friday"]
    rows = []
    for i in range(300):
        s = random.random() < 0.3
        rows.append((random.choice(sp if s else hm) + " note %d" % i, int(s)))
    pd.DataFrame(rows, columns=["text", "spam"]).to_csv(path, index=False)


def test_run_and_reproducible(tmp_path):
    f = tmp_path / "emails.csv"
    _fake(f)
    a = run(f, tmp_path / "a", seed=42)
    b = run(f, tmp_path / "b", seed=42)
    assert len(a) == 8
    cols = ["Model", "Accuracy", "Precision", "Recall", "F1"]
    pd.testing.assert_frame_equal(a[cols], b[cols])
    for n in ["results.csv", "comparison.png", "confusion_matrices.png", "run_info.json"]:
        assert (tmp_path / "a" / n).exists()
