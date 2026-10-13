# ReviewGuard — final-paper repository

以作者于 2026-10-13 指定的最终 PDF 为基准整理。原工作目录保持原样。

- **最终论文**：`paper/ReviewGuard.pdf`，原文件 SHA-256 为 `594a4e008b00b634ac6851fdb34f9a82b82a28dfd6a7390ff66f6571137fef62`。
- **可编辑论文**：`paper/reviewguard.tex`。它根据最接近的 `6.27_pricai/pricai_final.tex` 与最终 PDF 对齐；原始 TeX 快照及恢复差异见 `docs/`。这是恢复的源码，不能声称是提交时的原始 TeX。
- **核心实现与实验**：`scripts/`。保留原文件名、实验参数和共用实现，代码中的 CARD/CARD-T 对应论文 ReviewGuard 基础模式/模拟验证样本校准模式。
- **已有实验输出**：`reports/`、`results/`。按最终论文筛选，保留原数值，未重新运行或覆盖。
- **文件来源**：`docs/source_manifest.csv` 记录复制前的路径、哈希与大小；`docs/paper_experiment_map.md` 说明图表与实验入口的关系。

## 环境与实验

```powershell
py -3.11 -m pip install -r requirements.txt
py -3.11 scripts/reproduce_final.py --check-inputs
py -3.11 scripts/reproduce_final.py
```

三个论文输入 CSV 已纳入仓库，路径与原代码一致，见 `docs/data.md`。E5 缓存仅本地保留，通过 `.gitignore` 排除。默认运行不包含 E5；完整表示实验用 `--with-e5`，需要现有缓存或下载模型。权重向量 oracle 搜索有额外耗时。依赖文件尚未锁定版本，不保证跨环境逐位数值一致。

统一入口使用保存于原 `tmp/train_only_protocol_refresh.py` 的最终刷新逻辑，实验输出写入 `reproduced/reports/` 与 `reproduced/results/`，不会覆盖仓库中保留的论文证据。不要把早期根目录 `run_full_experiments.ps1` 当作最终论文入口。

`scripts/` 中的绘图辅助文件按原样保留，用于追溯；部分使用历史输出目录，统一入口只调用已重定向输出的绘图步骤。论文包中已保留最终三张图，不要求运行绘图脚本才能编译论文。

## 编译论文

在 `paper/` 中运行：

```powershell
pdflatex -jobname=reviewguard_preview reviewguard.tex
bibtex reviewguard_preview
pdflatex -jobname=reviewguard_preview reviewguard.tex
pdflatex -jobname=reviewguard_preview reviewguard.tex
```

生成的 `paper/reviewguard_preview.pdf` 是本地重编译预览，不覆盖作者提供的 `paper/ReviewGuard.pdf`。字体、引擎版本和浮动体位置可能导致排版差异。

## Git 范围

纳入论文及引用图、相关代码、三个论文输入 CSV、论文对应的结果表和来源说明。未纳入早期分类流水线、Yelp/Jakarta 探索、未采用策略、重复 ZIP、其他草稿或缓存。嵌入缓存可以通过统一入口重新生成，无需 Git LFS。
