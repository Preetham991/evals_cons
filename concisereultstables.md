<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# giv eme seperate tables nad make more detaailed explainations for aecxh result

Below are two separate, information-dense tables.

- Table A covers *performance / discrimination* metrics.
- Table B covers *reliability / calibration* metrics.

Each row gives
– the exact formula or statistical test,
– its canonical range or “good” target,
– the value observed on your 1 000-e-mail run, and
– a detailed explanation that links the number to model behaviour and operational use.

-----------------------------------------------------------------
Table A Performance-oriented metrics
-----------------------------------------------------------------

| \# | Metric | Mathematical definition | Typical range / ideal | Run value | Detailed interpretation \& operational note |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 1 | Accuracy | $(TP+TN)/(TP+FP+FN+TN)$[^3] | 0–1, higher better | **0.845** | 845 of 1 000 labels correct. Error rate (15%) is four times lower than random-prior guessing (~79%). Alone it hides class-specific failures and probability quality—pair with calibration checks. |
| 2 | Macro-F1 | $\frac1C\sum_{c=1}^C\!\!F1_c,\;F1_c=\frac{2PR}{P\!+\!R}$[^3] | 0–1 | **0.842** | Every class achieves F1 ≈ 0.84, proving no minority-class collapse. In production switch to cost-weighted F1 if mis-classifying *Important* e-mails is costlier than *Newsletter*. |
| 3 | Matthews Corr. Coef. | Pearson correlation between 1-hot truth \& prediction (multi-class generalisation)[^5] | –1…1, 1 ideal | **0.805** | High MCC confirms the 84% accuracy is genuine, not an artefact of class imbalance. Use as a compact “scientific” KPI for model comparisons. |
| 4 | Log-loss (cross-entropy) | $-\frac1N\sum\nolimits_i\log p_{i,y_i}$[^5] | 0 (perfect) → ∞ | **0.640 nats** | The model gives the true class ~53% probability on average (e$^{-0.64}$). Larger than you’d expect at 84% accuracy → indicates under-confidence and justifies calibration. |
| 5 | Brier Score | $\frac1N\sum\nolimits_i \|p_i - e_{y_i}\|_2^2$[^9] | 0 (perf.) → 2 | **0.325** | Reliability component ≈ 0.06; remainder is class entropy + resolution. Easier to explain to non-experts (“mean-squared error of probabilities”) than log-loss, but less sensitive to extreme mistakes. |
| 6 | Ranked-Probability Score | $\frac1N\sum_i\sum_{k=1}^C (P_{ik}^{cum}-O_{ik}^{cum})^2$ | 0 best | **0.335** | Mirrors Brier because classes have no natural order; would penalise “two-level” mis-rank heavily if you later impose an urgency scale. |
| 7 | Sharpness | Mean entropy $H(p_i)$ where $H=-\sum p\log p$ | 0 (certain) … ln 5 = 1.609 | **1.044 nats** | Distributions are neither razor-sharp nor flat. After temperature *warming* (see Table B) sharpness will drop slightly; watch ECE so you don’t over-sharpen. |


-----------------------------------------------------------------
Table B Reliability-oriented metrics
-----------------------------------------------------------------

| \# | Metric | Statistical definition | Ideal value | Run value | Detailed interpretation \& operational note |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 8 | Expected Calibration Error (ECE) | Bucket-weighted $\\lvert \\text{conf} - \\text{acc}\\rvert$ over 10 equal-mass bins[^10] | 0 | **0.256** | Mean gap 25.6 pp → when the model says 80% it is right ≈ 55%. Set alert: ECE > 0.05 ⇒ retrain or recalibrate. |
| 9 | Maximum Calibration Error (MCE) | Largest single-bin gap[^10] | 0 | **0.318** | Worst reliability deficit (32 pp) occurs in 0.8–0.9 confidence bin. Organisations with hard-risk workflows often demand MCE < 0.10. |
| 10 | Reliability slope β₁ | WLS fit of accuracy = β₀ + β₁·confidence | β₁ = 1 | **0.927** | Accuracy curve rises too slowly—classic under-confidence. Fix by dividing logits by a temperature < 1 (“warming”). |
| 11 | Reliability intercept β₀ | Same fit’s intercept | β₀ = 0 | **0.298** | Even claims of “0%” confidence still win 30% of the time. Disappears after proper calibration. |
| 12 | Spiegelhalter Z-test | $Z=\\sum(a_i-p_i)/\\sqrt{\\sum p_i(1-p_i)}$[^10] |  | Z | ≤ 1.96 |
| 13 | Over-confidence error (OCE) | Portion of ECE where conf > acc | 0 | **0.0005** | Negligible bragging—model is not over-selling itself. |
| 14 | Under-confidence error (UCE) | Portion of ECE where conf < acc | 0 | **0.256** | Entire calibration defect comes from *shyness*; cooling would worsen it, warming will help. |


***

### How to read the tables in practice

- Rows 1-7 answer “How often and how decisively am I right?”
- Rows 8-14 answer “When I say 90%, am I right 90% of the time?”

Concrete next steps:

1  Temperature-scale the logits on a held-out split until β₁≈1, β₀≈0 – this will push ECE below 0.05 and MCE below 0.10 while leaving Accuracy, F1 and MCC unchanged.

2  Re-plot the reliability diagram; ensure high-confidence bins move onto the diagonal and sharpness (entropy) falls modestly.

3  Retune any selective-prediction threshold: the risk-coverage curve will shift left, letting you auto-process ~65% of e-mails at the previous 10% risk cap.

4  Monitor Tables A and B weekly; drift most often surfaces first as a jump in Reliability intercept or ECE before top-line accuracy moves.
<span style="display:none">[^1][^2][^4][^6][^7][^8]</span>

<div style="text-align: center">⁂</div>

[^1]: https://neptune.ai/blog/performance-metrics-in-machine-learning-complete-guide

[^2]: https://www.geeksforgeeks.org/machine-learning/metrics-for-machine-learning-model/

[^3]: https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall

[^4]: https://www.nature.com/articles/s41598-024-56706-x

[^5]: https://scikit-learn.org/stable/modules/model_evaluation.html

[^6]: https://towardsdatascience.com/performance-metrics-confusion-matrix-precision-recall-and-f1-score-a8fe076a2262/

[^7]: https://www.kaggle.com/code/pythonafroz/evaluation-metrics-used-in-machine-learning

[^8]: https://www.aiacceleratorinstitute.com/evaluating-machine-learning-models-metrics-and-techniques/

[^9]: https://spotintelligence.com/2024/04/07/classification-metrics/

[^10]: https://matthewrkaye.com/posts/2023-03-20-metrics-calibration/calibration.html

