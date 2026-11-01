import argparse, io, shutil, urllib.request, zipfile
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def kaggle():
    import kagglehub
    src = Path(kagglehub.dataset_download("karthickveerakumar/spam-filter"))
    f = next(src.rglob("emails.csv"))
    shutil.copy(f, DATA / "emails.csv")
    return DATA / "emails.csv"


def uci():
    raw = urllib.request.urlopen(UCI_URL, timeout=60).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        z.extract("SMSSpamCollection", DATA)
    return DATA / "SMSSpamCollection"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["kaggle", "uci"], default="kaggle")
    a = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    print("saved:", kaggle() if a.source == "kaggle" else uci())
