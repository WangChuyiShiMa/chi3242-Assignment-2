# 以逻辑回归操作「作者边界」：《红楼梦》前八十回与后四十回

CHI 3242 作业2：训练自己的文本分类器

## 研究问题

《红楼梦》前八十回与后四十回的作者归属，是红学中最持久的争论之一。本项目把它转化为一个二元分类问题：

> 仅凭段落的用词习惯，能否区分曹雪芹（第一至八十回）与高鶚（第八十一至一百二十回）？

分类器训练完成后，用于分析作者存疑的**第六十七回**（该回不参与训练与测试）。

## 标签定义

| 标签 | 作者 | 回目 |
|---|---|---|
| 0 | 曹雪芹 | 第 1–80 回 |
| 1 | 高鶚 | 第 81–120 回 |

- **文本单位**：100–460 字的连续段落，只在句末标点（。！？．）处切断。
- **标注规则**：段落继承所属回目的作者标签。这是"回"级别的粗标签，默认一回只有一位作者。
- **留出文本**：第六十七回全部 55 段不进入 `passages.csv`，单独存为 `data/analysis_ch67.csv`。
- **分词清洗**：jieba 会把“贾政道”“黛玉笑”黏成一个词。旧版 4960 段中共有 1274 处、47 种此类连写；其中“贾政道”在旧特征表中权重 0.62。`scripts/textutil.py` 先把已知人名与“道、笑、说、问、答、哭、骂、叫、嚷、叹、念、想”切开，再分词；同样处理后残留为 0。

## 数据来源与预处理

- 原文：课程语料仓库 [`mcjkurz/qhchina-data`](https://github.com/mcjkurz/qhchina-data)，`corpora/misc/红楼梦.txt`（UTF-8，120 回）。
- 解析：按"第X回"标题切分回目；标题行不计入正文；重复回目标题跳过；第一回之前的目录、题记不进入语料。
- 切分：按句末标点累积成段，长度限定 100–460 字；共得 4960 段（曹雪芹 3350 段、高鶚 1610 段）。诗词、对联与判词仍留在段落中，这是有意保留的妥协。
- 平衡采样：训练集每类 400 段、测试集每类 100 段（随机种子 42）。
- 防泄漏：**按回切分**——同一回的段落只出现在训练集或测试集之一。每类随机取 80% 回目作训练、20% 作测试。

## 方法与特征

- 分词：`jieba`，并在分词前切开“人名+说话动词”的连写。
- 特征：词一元 + 二元的 TF-IDF（`min_df=3`，`max_features=6000`，`sublinear_tf=True`），仅在训练集上拟合，共 5485 维。
- 模型：`LogisticRegression`（`liblinear`，`C=1`，`random_state=42`）。
- 评估：测试集准确率、混淆矩阵、精确率、召回率、F1。

## 主要结果

- 测试集准确率 **0.88**（200 段中 176 段正确）
- 混淆矩阵：`[[86, 14], [10, 90]]`
- 高鶚：精确率 0.865、召回率 0.900、F1 0.882
- 曹雪芹：精确率 0.896、召回率 0.860、F1 0.878
- 第六十七回：55 段中 44 段判为高鶚、11 段判为曹雪芹，平均 P(高鶚) = 0.571（中位 0.574，最低 0.376，最高 0.802）；阈值 0.6 时 21 段判为高鶚。
- 与清洗前相比：准确率由 0.89 降至 0.88。4 段改变预测，3 段由对变错、1 段由错变对，合计少了 2 段；这部分差异来自去掉“贾政道”等说话标签带来的虚高，而不是模型退步。

## 仓库结构

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── passages.csv          # 全部标注段落（4960 段）
│   ├── train.csv             # 训练集（800 段）
│   ├── test.csv              # 测试集（200 段）
│   ├── analysis_ch67.csv     # 分析文本：第六十七回（55 段）
│   └── split_info.json       # 切分记录（回目、种子、数量）
├── scripts/
│   ├── prepare_data.py       # 下载原文、切回、切段、标注
│   ├── split_data.py         # 按回分层切分 80/20，平衡采样
│   ├── train_evaluate.py     # 分词、TF-IDF、训练、评估、特征权重
│   ├── apply_model.py        # 应用到第六十七回
│   ├── build_report_pdf.py   # report.md → report.html（A4 排版）
│   └── textutil.py           # 共用分词函数
├── output/
│   ├── metrics.json          # 评估指标
│   ├── test_predictions.csv  # 测试集逐段预测
│   ├── confusion_matrix.png  # 混淆矩阵
│   ├── top_features.csv      # 权重最大/最小的 60 个特征
│   ├── analysis_predictions.csv  # 第 67 回逐段预测
│   └── analysis_summary.json     # 第 67 回汇总
├── analysis.html             # 可选：互动式预测检视页
├── report.md
└── report.pdf
```

## 运行方法

```bash
python -m pip install -r requirements.txt

python scripts/prepare_data.py    # 自动下载原文并生成段落（约 1 分钟）
python scripts/split_data.py      # 生成 train.csv / test.csv
python scripts/train_evaluate.py  # 训练与评估（约 1 分钟）
python scripts/apply_model.py     # 第 67 回预测
```

报告 PDF 由 `report.md` 排版生成：先运行 `python scripts/build_report_pdf.py` 写出 `report.html`（临时文件，已由 `.gitignore` 排除），再用浏览器（Chrome）以 A4 打印为 `report.pdf`；正文使用宋体、标题黑体，图随文插入。

原文较大，`data/raw/红楼梦.txt` 已加入 `.gitignore`；`prepare_data.py` 会在缺失时自动下载。

## 可选的互动分析页

`analysis.html` 是课程 optional 的互动分析页。数据已内联，直接双击用浏览器打开即可；如果在 GitHub 仓库根目录开启 Pages，也可以访问 `/analysis.html`。

页面包含测试集与第六十七回切换、按回目/误判/概率筛选、搜索、混淆矩阵点选，以及单段原文和概率标尺。

## AI 使用声明

数据处理、模型代码在 AI 工具（豆包）协助下完成；研究设计、报告文字定稿与结果核对由作者负责。分类器仅在训练集上拟合，测试集与第 67 回均未参与训练。
