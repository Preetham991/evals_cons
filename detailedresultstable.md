

# also create a atble for all the values generated and explaination '

**Table 1 – Quantitative \& Calibration Results (dummy 1 000-email run, seed 42)**


| \# | Metric | Value | Ideal / Range | What the number means | Why you care |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 1 | **Accuracy** | **0.845** | 0 – 1 (higher = better)[^8] | 845 of 1 000 top-class predictions were correct. | Quick headline KPI; hides which classes fail. |
| 2 | **Macro-F1** | **0.842** | 0 – 1 (higher)[^8] | Average of per-class F1s; each label contributes equally. | Shows that minority classes (Important 15%) perform on par with majority; guards against accuracy inflation. |
| 3 | **Matthews Corr. Coef.** | **0.805** | −1 … 1 (higher)[^5] | Pearson correlation of one-hot truth vs prediction; robust to class skew. | Confirms genuine discrimination; if this were low while Accuracy was high you’d suspect imbalance tricks. |
| 4 | **Log-loss** (nats) | **0.640** | 0 (perfect) → ∞[^5] | Mean −log p(true-class) ⇒ average true-class probability is e$^{-0.64}$=0.53. | Penalises over/under-confidence exponentially; drives cross-entropy training. |
| 5 | **Brier Score** | **0.325** | 0 (perfect) → 2[^9] | Mean-squared error of full prob. vector. Reliability slice ≈0.06. | Combines calibration + sharpness in a single bounded number. |
| 6 | **Ranked-Probability Score** | **0.335** | 0 (perfect) → 1 (worst) | Cum-squared error on an ordinal label axis; here mirrors Brier because labels unordered. | If classes had real order (e.g. urgency), RPS would punish distance of mis-rank. |
| 7 | **Sharpness** (entropy) | **1.044 nats** | 0 (certain) … ln 5 = 1.609 (uniform) | Predictive distributions are midway between peaky and flat. | Measures *decisiveness*; must be interpreted together with calibration. |

**Calibration / reliability metrics**


| \# | Metric | Value | Ideal / Range | What the number means | Operational signal |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 8 | **ECE** (Expected Cal. Error) | **0.256** | 0 → 1 (lower = better)[^10] | Avg. | confidence − accuracy |
| 9 | **MCE** (Max Cal. Error) | **0.318** | 0 → 1 (lower) | Worst single-bin gap (32 pp) in 0.8–0.9 bin. | Risk guard-rail; many orgs demand MCE < 0.15 before auto-action. |
| 10 | **Reliability slope β₁** | **0.927** | 1 | Curve accuracy = β₀+β₁·confidence rises too slowly. | β₁ < 1 ⇒ under-confident at high scores; fix by dividing logits by T < 1. |
| 11 | **Reliability intercept β₀** | **0.298** | 0 | Even “0% confident” outputs still win 30% of the time. | Large β₀ common with under-confident nets; disappears after scaling. |
| 12 | **Spiegelhalter Z** | **+17.23** | 0 ( | Z | < 1.96 = good )[^10] |
| 13 | **Over-conf. Error (OCE)** | **0.0005** | 0 → 1 | Portion of ECE from conf > acc. | Negligible; model rarely brags. |
| 14 | **Under-conf. Error (UCE)** | **0.256** | 0 → 1 | Portion of ECE from conf < acc. | **All** reliability defect is shyness; treat with warming, not cooling. |

**How to read the table**

- Rows 1–7 tell you *how often* the classifier is right and how “sharp” its distributions are.
- Rows 8–14 tell you *how trustworthy* the probability numbers are.
Despite high accuracy (Row 1), Rows 8–14 show every confidence score is too small by ~25 pp.
A single temperature-scaling pass on a held-out set will push:
- ECE → < 0.05, MCE → < 0.10, β₁ → 1, β₀ → 0
- Log-loss drops (probabilities closer to 1 on correct cases)
- Sharpness falls slightly (entropy ↓) while Accuracy, F1, MCC remain unchanged.
<span style="display:none">[^1][^2][^3][^4][^6][^7]</span>

<div style="text-align: center">⁂</div>

[^1]: https://neptune.ai/blog/performance-metrics-in-machine-learning-complete-guide

[^2]: https://www.geeksforgeeks.org/machine-learning/metrics-for-machine-learning-model/

[^3]: https://pubs.rsna.org/doi/pdf/10.1148/ryai.2021200126

[^4]: https://iariw.org/wp-content/uploads/2021/07/Metrics_Paper.pdf

[^5]: https://scikit-learn.org/stable/modules/model_evaluation.html

[^6]: https://docs.ultralytics.com/guides/yolo-performance-metrics/

[^7]: https://www.kaggle.com/code/ahmedabdulhamid/7-evaluation-metrics-ml-course

[^8]: https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall

[^9]: https://spotintelligence.com/2024/04/07/classification-metrics/

[^10]: https://matthewrkaye.com/posts/2023-03-20-metrics-calibration/calibration.html

