"""Turn report.md into a printable HTML file (A4, Chinese typography)."""

from __future__ import annotations

import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def tidy(text: str) -> str:
    text = re.sub(r"([\u4e00-\u9fff])\s+([A-Za-z0-9])", r"\1\2", text)
    text = re.sub(r"([A-Za-z0-9])\s+([\u4e00-\u9fff])", r"\1\2", text)
    return text


def build() -> None:
    md = (ROOT / "report.md").read_text(encoding="utf-8")
    blocks = [b.strip() for b in md.split("\n\n") if b.strip()]

    parts: list[str] = []
    figure_done = False
    for block in blocks:
        if block.startswith("# "):
            parts.append(f"<h1>{html.escape(tidy(block[2:].strip()))}</h1>")
        elif block.startswith("## "):
            parts.append(f"<h2>{html.escape(tidy(block[3:].strip()))}</h2>")
        elif block.startswith("### "):
            parts.append(f"<h3>{html.escape(tidy(block[4:].strip()))}</h3>")
        elif block.startswith("*") and block.endswith("*"):
            parts.append(f'<p class="note">{html.escape(tidy(block.strip("*")))}</p>')
        else:
            para = "".join(block.split("\n"))
            parts.append(f"<p>{html.escape(tidy(para))}</p>")
            if not figure_done and ("二十四個錯誤" in para or "二十四个错误" in para):
                parts.append('<img src="output/confusion_matrix.png" alt="confusion matrix">')
                figure_done = True

    body = "\n".join(parts)
    page = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>以邏輯迴歸操作「作者邊界」：《紅樓夢》前八十回與後四十回</title>
<style>
  @page {{ size: A4; margin: 17mm 17mm 15mm 17mm; }}
  html, body {{ margin: 0; padding: 0; }}
  body {{ font-family: "SimSun", "Songti SC", serif; font-size: 10.0pt;
         line-height: 1.55; color: #1a1a1a; text-align: justify; }}
  h1 {{ font-family: "SimHei", sans-serif; font-size: 17pt; font-weight: normal;
        text-align: center; margin: 0 0 5px 0; letter-spacing: 1px; }}
  h2 {{ font-family: "SimHei", sans-serif; font-size: 11.5pt; font-weight: normal;
        margin: 10px 0 3px 0; color: #1f2d3d; }}
  h3 {{ font-family: "SimHei", sans-serif; font-size: 10.5pt; font-weight: normal;
        margin: 8px 0 2px 0; color: #33475b; }}
  p {{ margin: 0 0 4px 0; text-indent: 2em; }}
  img {{ display: block; width: 4.8cm; margin: 6px auto 3px auto; }}
  .note {{ font-size: 9pt; color: #555; text-indent: 0; border-top: 1px solid #ddd;
           padding-top: 8px; margin-top: 16px; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""
    (ROOT / "report.html").write_text(page, encoding="utf-8")
    print("report.html written")


if __name__ == "__main__":
    build()
