# 实验输入

为了保持原代码路径，三个输入保留原位置：

| 输入 | 用途 | Git |
| --- | --- | --- |
| `uber_reviews_without_reviewid.csv` | Uber 原始流，论文有效数据 12,000 条 | 纳入 |
| `reviews.csv` | Google Play Apps 预处理输入 | 纳入 |
| `data/processed/google_play_app_reviews_standardized.csv` | Google Apps 标准化输入，12,495 条 | 纳入 |

标准化入口：`py -3.11 scripts/32_prepare_second_dataset.py`。为复核当前论文，优先使用保留输入，避免预处理环境变化导致差异。输入的来源路径、大小和 SHA-256 见 `source_manifest.csv`。数据来源引用保留于 `paper/references.bib` 的 `kaggleUberReviews` 和 `kaggleGooglePlayApps` 条目。

两个 `embedding_card_*_canonical_20pct.npy` 文件是最终 E5 刷新入口使用的完整流嵌入缓存，已复制到本地 `data/processed/`，不加入 Git。`--with-e5` 缺少缓存时会通过原编码器重新生成，模型默认为 `intfloat/multilingual-e5-small`；下载需要网络访问。

仅纳入本论文使用的上述数据，不纳入 Yelp、Jakarta、无关训练/测试 CSV 或可再生嵌入缓存。此整理不改变原数据授权；来源引用见论文参考文献。
