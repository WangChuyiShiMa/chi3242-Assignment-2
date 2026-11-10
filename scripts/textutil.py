"""Shared tokenizer so that the fitted vectorizer can be pickled and reloaded.

jieba tends to fuse a character name with the following speech verb:
“贾政道”“黛玉笑” become one token, so the model can mistake a speaker
label for a style feature.  We split a known name from a speech verb before
tokenizing.  This follows the Assignment 1 feedback about “郭靖道”.
"""

import re

import jieba

PUNCT = re.compile(r"^[\W_]+$")

NAMES = """宝玉 黛玉 宝钗 凤姐 贾母 贾政 王夫人 贾琏 贾珍 贾蓉 贾赦 贾敏
林如海 薛姨妈 薛蟠 香菱 袭人 晴雯 平儿 鸳鸯 紫鹃 雪雁 麝月 秋纹 碧痕
小红 龄官 芳官 藕官 蕊官 葵官 豆官 艾官 茄官 文官 宝官 玉官 秦可卿
秦钟 尤氏 尤二姐 尤三姐 贾迎春 贾探春 贾惜春 贾元春 史湘云 妙玉 李纨
贾兰 贾环 赵姨娘 周瑞家的 刘姥姥 板儿 巧姐 倪二 柳湘莲 蒋玉菡 冯紫英
薛蝌 薛宝琴 邢岫烟 李纹 李绮 甄士隐 贾雨村 空空道人 茫茫大士 渺渺真人
警幻仙子 王熙凤 贾宝玉 林黛玉 薛宝钗 史太君 贾代儒 贾代化 贾代善 贾敬
贾蔷 贾芸 贾芹 贾菌""".split()

_SPEECH = "道笑说问答哭骂叫嚷叹念想"
_NAME_RE = re.compile(
    "(" + "|".join(sorted((re.escape(n) for n in NAMES), key=len, reverse=True)) + ")"
    "([" + _SPEECH + "])"
)


def tokenize(text: str) -> list[str]:
    text = _NAME_RE.sub(r"\1 \2", text)
    return [t for t in jieba.cut(text) if t.strip() and not PUNCT.match(t)]
