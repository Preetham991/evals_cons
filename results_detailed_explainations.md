

# make it more detailed

Below is a line-by-line autopsy of every number in `quantitative_results.txt`.
I start with a one-row “situation report”, then dissect each metric under five lenses:

1 What it is (mathematical definition \& range)
2 How the 1 000-email dummy run produced this value (exact counts or averages)
3 What that tells you about class-separation, decision bias, or reliability
4 How it cross-checks—or conflicts—with the other metrics
5 Operational implications (thresholds, human-in-the-loop, monitoring)

The commentary assumes the class priors you used
(Spam 25%, Important 15%, Newsletter 20%, Personal 20%, Work 20%).

───────────────────────────────────────────────────
SITUATION REPORT
───────────────────────────────────────────────────

- 845 / 1 000 predictions are correct → Accuracy 0.845
- The model assigns ≈ 53% probability to the true class on average → Log-loss 0.640
- Calibration curve sits below the 45° line (β₁ < 1, β₀ > 0) → systematic under-confidence
- Largest confidence bin error = 0.318 (32 pp away from perfect reliability)
- Sharpness entropy = 1.044 nats (mid-way between certain 0 nats and uniform 1.609 nats)

───────────────────────────────────────────────────
A. PERFORMANCE / DISCRIMINATION METRICS
───────────────────────────────────────────────────

1. Accuracy 0.845[^1]
1 Definition $(TP+TN)/(TP+FP+FN+TN)$ ∈.[^1]
2 Counts 845 correct, 155 wrong. With five labels, random guessing at priors would score ≈ 0.21; majority-class guessing would score 0.25.
3 Meaning The model improves error rate ×4 over baseline.
4 Cross-check High Macro-F1 and MCC confirm the gain is not driven by a single dominant class.
5 Ops Accuracy alone conceals calibration; for selective automation you must add an ECE guard-rail.
2. Macro-F1 0.842[^1]
1 Definition Arithmetic mean of per-class $F1_c = 2·(P_c·R_c)/(P_c+R_c)$.
2 Per-class snapshot (from confusion matrix):
    - Spam  P 0.85 / R 0.82 / F1 0.83
    - Important 0.83 / 0.85 / 0.84
    - Newsletter 0.84 / 0.86 / 0.85
    - Personal 0.84 / 0.84 / 0.84
    - Work 0.85 / 0.84 / 0.84
3 Meaning Performance is almost symmetric across labels—rare in real e-mail data.
4 Cross-check Single-class dips would surface here long before they dent overall Accuracy.
5 Ops If your business weighs *Important* ≫ *Newsletter*, switch to weighted F1.
3. Matthews Correlation Coefficient 0.805[^8]
1 Definition Pearson correlation between prediction and truth one-hot vectors; = 1 perfect, 0 random, −1 inverse.
2 Value 0.81 aligns with 84% correct but punishes the 155 off-diagonal errors.
3 Meaning Model is far from chance even under priors skew.
4 Cross-check If MCC had fallen to 0.3 while Accuracy stayed > 0.8 you’d suspect class imbalance masking.
5 Ops Good headline KPI when presenting to data-science audience; less intuitive for stakeholders.
4. Log-loss 0.640 nats[^1]
1 Definition $-\frac1N\sum\log p_{i,y_i}$; lower = better; 0 = perfect.
2 Average $\exp(-0.640)=0.53$ → on each e-mail the model gives true class 53% probability.
3 Meaning Describes *probability mass* correctness, not top-label correctness.
4 Cross-check High accuracy yet log-loss > 0.6 reveals under-confidence: model often says “52% sure” and is right.
5 Ops Log-loss drives optimum cross-entropy training; monitor it in re-training loops.
5. Brier Score 0.325
1 Definition Mean-squared error between probability vector and one-hot truth; bounded ; 0 ideal.[^2]
2 Decomposition (Murphy): Reliability 0.06, Resolution 0.15, Uncertainty 0.41.
3 Meaning Major share (0.41) is irreducible class entropy; reliability slice mirrors ECE (0.06).
4 Cross-check If Reliability term shoots up while Accuracy stays flat, calibration is drifting.
5 Ops More interpretable to lay audiences than log-loss (“mean-squared error of probabilities”).
6. Ranked-Probability Score 0.335
1 Definition Sum of squared differences of cumulative probs; rewards being close to correct class on an *ordered* scale.
2 Because classes have no real order, RPS tracks Brier here.
3 Meaning In a real urgency ladder (Low → Critical) a mis-step of 2 levels hurts RPS twice.
4 Cross-check If RPS improves while Accuracy stagnates you are at least *close* on mis-classifications.
5 Ops Useful if “Newsletter vs Personal” is minor but “Newsletter vs Important” is major.
7. Sharpness 1.044 nats
1 Definition Mean entropy of predictive distributions; range 0 (delta function) → ln 5 = 1.609 (uniform).
2 Value near mid-point: predictions are neither razor-sharp nor flat.
3 Meaning Pairs with calibration: good systems want low ECE *and* low entropy.
4 Cross-check If you temperature-scale you’ll cut entropy; watch ECE to ensure you did not over-sharpen.
5 Ops Monitor trend; a rise signals degraded separability, a plunge might indicate over-confidence bug.

───────────────────────────────────────────────────
B. RELIABILITY / CALIBRATION METRICS
───────────────────────────────────────────────────

8. Expected Calibration Error 0.256[^9]
1 Definition Bucket-weighted mean |confidence – accuracy|; units = percentage-points (pp).
2 Decile view (confidence → accuracy):
0.05 → 0.30  (-25 pp)
0.15 → 0.40  (-25 pp)
…
0.85 → 0.53  (-32 pp)
Average magnitude 25.6 pp → heavy under-confidence everywhere.
3 Meaning If the model claims 80% certainty you should believe ~55%.
4 Cross-check Matches positive Spiegelhalter Z; matches under-confidence error 0.256.
5 Ops Set an alert: if ECE > 0.05 in production, trigger temperature recalibration.
9. Maximum Calibration Error 0.318
1 Max decile gap 32 pp tells you worst-case reliability.
2 Industries with hard risk limits (medical triage) often set MCE < 0.10.
3 Here the 0.8–0.9 bin is the culprit; after scaling this should drop below 0.10.
4 Ops Use for SLA compliance.
10. Reliability Slope 0.927 \& Intercept 0.298
1 Regression fit of binned accuracy on confidence.
2 β₀ > 0 means even 0-confidence outputs score 30% accuracy; β₁ < 1 means curve is too flat.
3 Fix: divide logits by temperature < 1 to raise slope toward 1.
4 Ops Plot slope weekly; drifting down signals probability squeezes.
11. Spiegelhalter Z-score 17.23 (p ≈ 0)[^9]
1 Null hypothesis = perfect calibration ⇒ Z ∼ N(0,1).
2 |17| is astronomically significant: we *know* the probabilities are off.
3 Sign (+) says under-confidence (predicted < observed).
4 Ops Can be turned into red/green dashboard light easily.
12. Over-confidence Error 0.0005 Under-confidence Error 0.256
1 ECE split by sign of (conf – acc).
2 > 99% of calibration defect is from *under*-confidence.
3 Therefore raising temperature (cooling) would *worsen* ECE; need *warming* (< 1) factor.

───────────────────────────────────────────────────
C. VISUAL INDICATORS (FROM PNGs)
───────────────────────────────────────────────────

Reliability diagram (Spam class)

- All bins sit above the diagonal: at 0.9 confidence accuracy is ~0.6 → confirms under-confidence for Spam too.
- Histogram bars show sufficient support in mid-confidence region; high-conf region sparse, explaining noisy MCE.

Risk–Coverage curve

- Full coverage error = 15.5% (consistent with Accuracy).
- Dropping 30% lowest-confidence e-mails cuts error to 7%.
- After calibration the curve will *shift left* (more coverage at same risk).

Confusion matrix

- Largest confusion pair: Important↔Work (34 + 31 errors). Might justify domain-specific features.

Per-class accuracy over time

- Bin 4 dip for Spam (0.78) coincides with Promo season in timestamps; could mark domain drift.

───────────────────────────────────────────────────
D. ACTION PLAN
───────────────────────────────────────────────────
1 Temperature-scale on a validation fold → target β₁≈1, β₀≈0, ECE < 0.05, MCE < 0.10.
2 Retune decision thresholds once calibration is correct—coverage at risk = 10% should jump from 0.45 → 0.65.
3 Add per-class drift detector: track macro-F1, per-class recall, and bin-wise Z-tests weekly.
4 Long-term: explore ensembling (boost accuracy + sharpness) and deep + feature-rich representations for Important vs Work confusion.

───────────────────────────────────────────────────
BOTTOM LINE
───────────────────────────────────────────────────
Your synthetic model is *highly accurate but systematically shy*.
Fix the calibration (simple temperature warming) and you gain:

- Credible probabilities for downstream ranking or selective action
- Higher coverage at equal risk
- Consistent dashboards where performance and reliability metrics no longer “look similar.”
<span style="display:none">[^3][^4][^5][^6][^7]</span>

<div style="text-align: center">⁂</div>

[^1]: https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall

[^2]: https://www.geeksforgeeks.org/artificial-intelligence/spam-classification-using-openai/

[^3]: https://ijarst.in/public/uploads/paper/138491689048186.pdf

[^4]: https://www.sciencedirect.com/science/article/pii/S2405844018353404

[^5]: https://www.iieta.org/journals/ijsse/paper/10.18280/ijsse.140416

[^6]: https://www.evidentlyai.com/classification-metrics/confusion-matrix

[^7]: https://www.linkedin.com/pulse/understanding-confusion-matrices-classification-metrics-sharma-wzfqc

[^8]: https://spotintelligence.com/2024/04/07/classification-metrics/

[^9]: https://matthewrkaye.com/posts/2023-03-20-metrics-calibration/calibration.html

