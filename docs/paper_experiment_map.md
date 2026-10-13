# 最终论文与实验对应

权威稿为作者提供的 `paper/ReviewGuard.pdf`（16 页）。表格数值核对基于现有结果，未重跑全部实验。

| 论文内容 | 结果证据 | 入口 |
| --- | --- | --- |
| 表 1：数据与分析流 | `reports/train_only_protocol_metadata.csv` | `train_only_protocol_refresh.py` |
| 表 2：攻击比例 | `reports/attack_sweep_results.csv`、`trusted_calibrated_card_eval.csv` | `07_attack_sweep.py`、`31_trusted_calibrated_card_eval.py` |
| 五种子稳定性 | `reports/seed_stability.csv` | `20_seed_stability.py` |
| 表 3：噪声标签基线 | `reports/strong_noisy_label_baselines.csv` | `42_strong_noisy_label_baselines.py` |
| 图 3：跨流及表示比较 | `reports/representation_comparison_20pct.csv` | 最终刷新入口中的 `run_e5_corrected`，以及 `33`、`34`、`35` 脚本 |
| 表 4：融合消融 | `reports/formula_multi_ratio_eval.csv`、`table_formula_multi_ratio_mean_tvd.csv` | `40_reliability_interaction_diagnostics.py` |
| 特征消融与相关性 | `reports/cohort_feature_ablation.csv`、`table_reliability_correlation.csv` | `19_cohort_feature_ablation.py`、`40_reliability_interaction_diagnostics.py` |
| 攻击类型分析 | `reports/ablation_results.csv` | 最终刷新入口中的 `run_attack_types`（不是早期独立攻击脚本） |
| 部分自适应攻击 | `results/partially_adaptive_mixed_20pct_*.csv` | `47_partially_adaptive_mixed_eval.py` |
| 表 5：自然突发诊断 | `results/natural_burst_false_positive_summary.csv` | `46_natural_burst_false_positive.py` |
| Oracle 诊断 | `reports/cohort_weight_simplex_search*.csv` | `45_cohort_weight_simplex_search.py` |
| 表 6：规模实验 | `reports/scale_stress_eval.csv` | `39_scale_stress_eval.py`，由最终刷新入口调用 |
| 评论案例 | `reports/review_level_target_examples.csv` | 最终刷新入口中的 `run_review_examples` |
| 图 1、图 2 | `paper/intro_problem_baseline_example.pdf`、`reviewguard_workflow_flow_only.pdf` | 保留最终图文件及 `51`、`52` 绘图辅助源码 |

参数敏感性 CSV 作为辅助证据保留，最终 PDF 没有参数敏感性图。Oracle 搜索使用评价参考，不作为可部署方法。模拟验证样本校准使用额外信息，不能与基础模式混同。

注意：原 `39_scale_stress_eval.py` 的 `SOURCES` 合并了 Uber 与 Google Apps；最终 PDF 表 6 的标题写的是 Uber review pools。这一来源描述差异保留在此，整理期间未改变实验代码或论文数值，需要作者在后续修改时判断。
