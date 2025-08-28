#!/usr/bin/env python
# dummy_email_confidence.py
# ---------------------------------------------------------------
# Synthetic 1 000-row / 5-class e-mail classification experiment
# Generates full quantitative & visual confidence analysis
# ---------------------------------------------------------------

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             log_loss, precision_recall_curve,
                             confusion_matrix)
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------
# 1  SYNTHETIC DATASET
# ---------------------------------------------------------------
np.random.seed(42)
N, C = 1_000, 5
CLASSES = ["Spam", "Important", "Newsletter", "Personal", "Work"]
PRIORS  = [0.25, 0.15, 0.20, 0.20, 0.20]

y_true = np.random.choice(C, N, p=PRIORS)

def noisy_dirichlet(label, noise=0.7):
    alpha           = np.ones(C)*noise
    alpha[label]   *= 5                    # bias toward ground-truth class
    return np.random.dirichlet(alpha)

probs = np.vstack([noisy_dirichlet(l) for l in y_true])
y_pred = probs.argmax(axis=1)
conf   = probs.max(axis=1)

# ---------------------------------------------------------------
# 2  QUANTITATIVE & CALIBRATION METRICS
# ---------------------------------------------------------------
one_hot = np.eye(C)[y_true]
metrics = {
    "Accuracy"        : accuracy_score(y_true, y_pred),
    "Macro-F1"        : f1_score(y_true, y_pred, average="macro"),
    "MCC"             : matthews_corrcoef(y_true, y_pred),
    "Log-loss"        : log_loss(y_true, probs),
    "Brier Score"     : np.mean(np.sum((probs-one_hot)**2, axis=1)),
}

def ece_mce(y, p, bins=15):
    edges = np.linspace(0,1,bins+1)
    acc   = (p.argmax(1)==y)
    conf  = p.max(1)
    ece = mce = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf>lo)&(conf<=hi)
        if m.any():
            gap  = abs(acc[m].mean()-conf[m].mean())
            ece += m.mean()*gap
            mce  = max(mce, gap)
    return ece, mce

metrics["ECE"], metrics["MCE"] = ece_mce(y_true, probs)

# reliability line
edges = np.linspace(0,1,16)
x=y=w=[]
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (conf>lo)&(conf<=hi)
    if m.any():
        x.append(conf[m].mean()); y.append((y_pred[m]==y_true[m]).mean()); w.append(m.sum())
slope, intercept = LinearRegression().fit(
    np.array(x).reshape(-1,1), y, sample_weight=w).coef_[0], \
    LinearRegression().fit(np.array(x).reshape(-1,1), y, sample_weight=w).intercept_
metrics["Reliability slope"]     = slope
metrics["Reliability intercept"] = intercept

# Spiegelhalter Z
z_num = np.sum((y_pred==y_true)-conf)
z_den = np.sqrt(np.sum(conf*(1-conf)))
metrics["Spiegelhalter Z"] = z_num/z_den if z_den else float("nan")

# over/under–confidence
over = under = 0
for lo, hi in zip(edges[:-1], edges[1:]):
    m=(conf>lo)&(conf<=hi)
    if m.any():
        gap=conf[m].mean()-(y_pred[m]==y_true[m]).mean()
        (over if gap>0 else under).__add__(m.mean()*abs(gap))
metrics["Over-confidence error"]  = over
metrics["Under-confidence error"] = under

# Ranked-Probability Score
cum_p = np.cumsum(probs,1)
cum_o = np.zeros_like(cum_p)
for i in range(N): cum_o[i, y_true[i]:] = 1
metrics["RPS"] = np.mean(np.sum((cum_p-cum_o)**2,1))

# Sharpness (mean entropy)
metrics["Sharpness"] = (-probs*np.log(np.clip(probs,1e-12,1))).sum(1).mean()

# Write text file
out_dir = Path(__file__).resolve().parent
(out_dir/"quantitative_results.txt").write_text(
    "\n".join(f"{k}: {v}" for k,v in metrics.items())
)

# ---------------------------------------------------------------
# 3  VISUAL DIAGNOSTICS
# ---------------------------------------------------------------
# 3-a  Precision–Recall per class
plt.figure(figsize=(8,6))
for i,name in enumerate(CLASSES):
    prec, rec, _ = precision_recall_curve((y_true==i).astype(int), probs[:,i])
    plt.plot(rec, prec, label=name)
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision–Recall Curve per Class")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(out_dir/"pr_curve_per_class.png"); plt.close()

# 3-b  Confusion-matrix heat-map
cm = confusion_matrix(y_true, y_pred, labels=range(C))
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=CLASSES, yticklabels=CLASSES, cmap="Blues")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title("Confusion Matrix"); plt.tight_layout()
plt.savefig(out_dir/"confusion_matrix.png"); plt.close()

# 3-c  Calibration error per confidence decile
edges = np.linspace(0,1,11)
calib = [abs(conf[((conf>lo)&(conf<=hi))].mean() -
             (y_pred[((conf>lo)&(conf<=hi))]==y_true[((conf>lo)&(conf<=hi))]).mean())
         if ((conf>lo)&(conf<=hi)).any() else 0
         for lo,hi in zip(edges[:-1],edges[1:])]
plt.figure(figsize=(8,4))
plt.bar(range(10), calib)
plt.xticks(range(10), [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(10)], rotation=45)
plt.ylabel("Calibration error")
plt.title("Calibration Error per Confidence Decile")
plt.tight_layout(); plt.savefig(out_dir/"calibration_heatmap.png"); plt.close()

# 3-d  Per-class accuracy over synthetic time bins
time  = np.linspace(0,100,N) + np.random.randn(N)*5
bins  = pd.qcut(time, 10, labels=False, duplicates="drop")
acc_t = np.full((10,C), np.nan)
for b in range(10):
    idx = np.where(bins==b)[0]
    for c in range(C):
        cls = idx[y_true[idx]==c]
        acc_t[b,c] = (y_pred[cls]==c).mean() if cls.size else np.nan
plt.figure(figsize=(10,6))
for c in range(C): plt.plot(acc_t[:,c], label=CLASSES[c])
plt.xlabel("Time bins"); plt.ylabel("Accuracy")
plt.title("Per-Class Accuracy Over Time")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(out_dir/"per_class_accuracy_time.png"); plt.close()

print("✓ quantitative_results.txt and four PNGs saved next to script")
