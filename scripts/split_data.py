"""Split passages into training (80%) and test (20%) sets.

The split is done at chapter level so that passages from the same chapter never
appear in both sets. Within each label, chapters are shuffled with a fixed seed
and 80% go to training.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 42
TRAIN_PER_CLASS = 400
TEST_PER_CLASS = 100


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = read_csv(DATA / "passages.csv")
    by_label: dict[int, set[int]] = defaultdict(set)
    for r in rows:
        by_label[int(r["label"])].add(int(r["chapter"]))

    rng = random.Random(SEED)
    info = {"seed": SEED, "split": "chapter-level, stratified by label",
            "train_per_class": TRAIN_PER_CLASS, "test_per_class": TEST_PER_CLASS}
    train_chapters: set[int] = set()
    test_chapters: set[int] = set()
    for label, chapters in sorted(by_label.items()):
        chapters = sorted(chapters)
        rng.shuffle(chapters)
        n_train = round(len(chapters) * 0.8)
        train_chapters.update(chapters[:n_train])
        test_chapters.update(chapters[n_train:])
        info[f"label_{label}"] = {"train_chapters": sorted(chapters[:n_train]),
                                  "test_chapters": sorted(chapters[n_train:])}

    fields = ["item_id", "chapter", "author", "label", "text"]
    train, test = [], []
    for label in sorted(by_label):
        pool_train = [r for r in rows if int(r["label"]) == label
                      and int(r["chapter"]) in train_chapters]
        pool_test = [r for r in rows if int(r["label"]) == label
                     and int(r["chapter"]) in test_chapters]
        rng_train = random.Random(SEED + 100 + label)
        rng_test = random.Random(SEED + 200 + label)
        train.extend(rng_train.sample(pool_train, min(TRAIN_PER_CLASS, len(pool_train))))
        test.extend(rng_test.sample(pool_test, min(TEST_PER_CLASS, len(pool_test))))

    train.sort(key=lambda r: r["item_id"])
    test.sort(key=lambda r: r["item_id"])
    write_csv(DATA / "train.csv", train, fields)
    write_csv(DATA / "test.csv", test, fields)

    info["n_train"] = len(train)
    info["n_test"] = len(test)
    info["n_train_by_label"] = {str(lbl): sum(1 for r in train if int(r["label"]) == lbl)
                                for lbl in sorted(by_label)}
    info["n_test_by_label"] = {str(lbl): sum(1 for r in test if int(r["label"]) == lbl)
                               for lbl in sorted(by_label)}
    (DATA / "split_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    print(f"train={len(train)} test={len(test)}")
    print("train by label:", info["n_train_by_label"])
    print("test by label:", info["n_test_by_label"])


if __name__ == "__main__":
    main()
