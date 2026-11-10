"""Train and evaluate a logistic-regression authorship classifier.

Features: TF-IDF over jieba word unigrams and bigrams, fitted on the training set only.
Evaluation: accuracy, confusion matrix, precision, recall and F1 on the held-out test set.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, classification_report)

from textutil import tokenize

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "output"
SEED = 42
LABEL_NAMES = {0: "曹雪芹", 1: "高鶚"}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def set_chinese_font() -> None:
    for name in ("SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "Arial Unicode MS"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = read_csv(DATA / "train.csv")
    test = read_csv(DATA / "test.csv")

    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                                 ngram_range=(1, 2), min_df=3, max_features=6000,
                                 sublinear_tf=True)
    x_train = vectorizer.fit_transform([r["text"] for r in train])
    x_test = vectorizer.transform([r["text"] for r in test])
    y_train = [int(r["label"]) for r in train]
    y_test = [int(r["label"]) for r in test]

    model = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear", random_state=SEED)
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    proba = model.predict_proba(x_test)[:, 1]

    metrics = {
        "train_size": len(train),
        "test_size": len(test),
        "features": int(x_train.shape[1]),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_gao": float(precision_score(y_test, pred, pos_label=1, zero_division=0)),
        "recall_gao": float(recall_score(y_test, pred, pos_label=1, zero_division=0)),
        "f1_gao": float(f1_score(y_test, pred, pos_label=1, zero_division=0)),
        "precision_cao": float(precision_score(y_test, pred, pos_label=0, zero_division=0)),
        "recall_cao": float(recall_score(y_test, pred, pos_label=0, zero_division=0)),
        "f1_cao": float(f1_score(y_test, pred, pos_label=0, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, pred, labels=[0, 1], target_names=["曹雪芹", "高鶚"], digits=4),
        "seed": SEED,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    with (OUT / "test_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "chapter", "true_label", "true_author",
                         "pred_label", "pred_author", "p_gao", "correct"])
        for r, y, p, pr in zip(test, y_test, pred, proba):
            writer.writerow([r["item_id"], r["chapter"], y, LABEL_NAMES[y],
                             int(p), LABEL_NAMES[int(p)], f"{pr:.4f}", int(y == p)])

    # most informative words for each side
    names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]
    order = coef.argsort()
    with (OUT / "top_features.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "feature", "weight", "leans_to"])
        for rank, idx in enumerate(list(order[:30]) + list(order[-30:][::-1]), start=1):
            writer.writerow([rank, names[idx], f"{coef[idx]:.4f}",
                             "高鶚" if coef[idx] > 0 else "曹雪芹"])

    # confusion matrix figure
    set_chinese_font()
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["曹雪芹", "高鶚"])
    ax.set_yticks([0, 1], ["曹雪芹", "高鶚"])
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    joblib.dump({"vectorizer": vectorizer, "model": model,
                 "labels": LABEL_NAMES}, OUT / "authorship_model.joblib")

    print(json.dumps({k: metrics[k] for k in
                      ("train_size", "test_size", "features", "accuracy",
                       "precision_gao", "recall_gao", "f1_gao")},
                     ensure_ascii=False, indent=2))
    print(metrics["classification_report"])
    print("confusion matrix:", metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
