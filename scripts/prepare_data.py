"""Prepare labelled passages from Hongloumeng (120 chapters).

Label rule: chapters 1-80 -> Cao Xueqin (label 0); chapters 81-120 -> Gao E (label 1).
Unit: a passage of 120-300 Chinese characters, cut only at sentence-final punctuation.
Chapter 67 is excluded from the training pool and kept as the analysis text.
"""

from __future__ import annotations

import csv
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "红楼梦.txt"
RAW_URL = "https://raw.githubusercontent.com/mcjkurz/qhchina-data/main/corpora/misc/红楼梦.txt"

CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def cn_to_int(s: str) -> int:
    """Convert a Chinese numeral such as 一百二十 or 一零零 to an integer."""
    if "十" not in s and "百" not in s:
        # digit-by-digit notation, e.g. 一零零 = 100, 第一二零回 = 120
        return int("".join(str(CN_DIGITS[ch]) for ch in s))
    total, section, number = 0, 0, 0
    for ch in s:
        if ch in CN_DIGITS:
            number = CN_DIGITS[ch]
        elif ch == "十":
            section += (number or 1) * 10
            number = 0
        elif ch == "百":
            section += (number or 1) * 100
            number = 0
        else:
            raise ValueError(f"unexpected numeral character: {ch}")
    return total + section + number


def load_text() -> str:
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RAW_URL, RAW)
    return RAW.read_text(encoding="utf-8")


HEADING = re.compile(r"^第([一二三四五六七八九十百零〇]+)回[\s　]+(.+)$")


def split_chapters(text: str) -> dict[int, str]:
    chapters: dict[int, list[str]] = {}
    current: int | None = None
    for line in text.splitlines():
        stripped = line.strip()
        m = HEADING.match(stripped)
        if m:
            num = cn_to_int(m.group(1))
            # accept only the next chapter in sequence, so repeated titles inside
            # the body cannot be mistaken for a new chapter
            if 1 <= num <= 120 and (current is None and num == 1 or num == (current or 0) + 1):
                current = num
                chapters.setdefault(current, [])
                continue
        if current is not None:
            chapters[current].append(line)
    return {k: "\n".join(v) for k, v in sorted(chapters.items())}


SENT_END = re.compile(r"(?<=[。！？．!?])")


def make_passages(chapter_text: str) -> list[str]:
    """Group sentences into passages of roughly 150-320 characters."""
    units: list[str] = []
    for line in chapter_text.splitlines():
        line = line.strip()
        if not line or HEADING.match(line):
            continue
        units.extend(u.strip() for u in SENT_END.split(line) if u.strip())

    passages: list[str] = []
    buf = ""
    for unit in units:
        if len(buf) + len(unit) <= 320:
            buf += unit
            if len(buf) >= 150:
                passages.append(buf)
                buf = ""
        else:
            if buf:
                passages.append(buf)
                buf = ""
            while len(unit) > 320:
                passages.append(unit[:300])
                unit = unit[300:]
            buf = unit
    if buf:
        if passages and len(buf) < 100:
            passages[-1] += buf
        else:
            passages.append(buf)

    return [p for p in passages if 100 <= len(p) <= 460]


def main() -> None:
    text = load_text()
    chapters = split_chapters(text)
    if len(chapters) != 120:
        raise SystemExit(f"expected 120 chapters, found {len(chapters)}")

    rows, analysis_rows = [], []
    item_no = 0
    for num, body in chapters.items():
        author, label = ("曹雪芹", 0) if num <= 80 else ("高鶚", 1)
        for p in make_passages(body):
            if num == 67:
                analysis_rows.append({"item_id": f"ch67-{len(analysis_rows) + 1:03d}",
                                      "chapter": 67, "author": author, "label": label,
                                      "text": p})
            else:
                item_no += 1
                rows.append({"item_id": f"hlm-{item_no:05d}", "chapter": num,
                             "author": author, "label": label, "text": p})

    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "passages.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "chapter", "author", "label", "text"])
        writer.writeheader()
        writer.writerows(rows)
    with (data_dir / "analysis_ch67.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "chapter", "author", "label", "text"])
        writer.writeheader()
        writer.writerows(analysis_rows)

    print(f"chapters: {len(chapters)}; passages: {len(rows)}; analysis passages: {len(analysis_rows)}")
    for label, name in ((0, "曹雪芹"), (1, "高鶚")):
        n = sum(1 for r in rows if r["label"] == label)
        print(f"  {name}: {n} passages")


if __name__ == "__main__":
    main()
