"""Apply the trained classifier to the analysis text (chapter 67 of Hongloumeng).

Chapter 67 is left out of training because its authorship is disputed in the
scholarly tradition. The script reports the predicted label and P(Gao E) for each
passage, plus an aggregate view.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "output"


def main() -> None:
    bundle = joblib.load(OUT / "authorship_model.joblib")
    vectorizer, model = bundle["vectorizer"], bundle["model"]
    labels = bundle["labels"]

    with (DATA / "analysis_ch67.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    x = vectorizer.transform([r["text"] for r in rows])
    pred = model.predict(x)
    proba = model.predict_proba(x)[:, 1]

    with (OUT / "analysis_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "pred_label", "pred_author", "p_gao", "p_cao", "text"])
        for r, p, pr in zip(rows, pred, proba):
            writer.writerow([r["item_id"], int(p), labels[int(p)],
                             f"{pr:.4f}", f"{1 - pr:.4f}", r["text"]])

    n = len(rows)
    n_gao = int((pred == 1).sum())
    summary = {
        "n_passages": n,
        "predicted_gao": n_gao,
        "predicted_cao": n - n_gao,
        "mean_p_gao": float(proba.mean()),
        "median_p_gao": float(sorted(proba)[n // 2]),
        "min_p_gao": float(proba.min()),
        "max_p_gao": float(proba.max()),
    }
    (OUT / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    ranked = sorted(zip(rows, proba), key=lambda t: t[1])
    print("\nmost Cao-like passage:")
    print(f"p_gao={ranked[0][1]:.3f} {ranked[0][0]['item_id']} {ranked[0][0]['text'][:80]}")
    print("\nmost Gao-like passage:")
    print(f"p_gao={ranked[-1][1]:.3f} {ranked[-1][0]['item_id']} {ranked[-1][0]['text'][:80]}")


if __name__ == "__main__":
    main()
