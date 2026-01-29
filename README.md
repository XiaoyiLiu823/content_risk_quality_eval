# Content Risk Model Quality Evaluation (误杀/漏放 + 分层定位 + 阈值版本对比)

本项目用一个可复跑的脚本小工程，演示如何对“内容风险模型”做质量评估：
- 模型：TF-IDF + LogisticRegression
- 数据：sklearn 自带 20newsgroups（选两类做二分类：1=风险，0=安全）
- 版本对比：threshold=0.50(v1) vs 0.60(v2)
- 评估：TP/FP/FN/TN、precision/recall/f1、误杀率、漏放率
- 分层：按文本长度分桶（0-50, 51-120, 121-300, 301+）

## 误杀/漏放定义
- 风险=1，安全=0
- 误杀（False Positive, FP）：真实为安全(0)，预测为风险(1)
- 漏放（False Negative, FN）：真实为风险(1)，预测为安全(0)

派生指标：
- precision = TP / (TP + FP)
- recall    = TP / (TP + FN)
- f1        = 2 * precision * recall / (precision + recall)
- 误杀率 (Kill Rate) = FP / (TP + FP)  （即 1 - precision）
- 漏放率 (Miss Rate) = FN / (TP + FN)  （即 1 - recall）

## 输出文件（生成到 outputs/）
1) daily_quality_metrics.csv
   - 每行一个版本的总体指标（以及每个 segment 的汇总列）

2) top_fp_fn_breakdown.csv
   - 每个版本分别统计 FP 与 FN 在 segment 维度的贡献（count、share、rank）

3) action_list.csv
   - 每个版本分别抽取最多 50 条 FP + 50 条 FN 样本，用于人工复核/标注回流
   - 字段：id、version、error_type、score、segment、text_snippet(截断200字符)

## 如何运行
```bash
pip install -r requirements.txt
python run.py
