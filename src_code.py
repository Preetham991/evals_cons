"""
email_calibration_full.py
Synthetic 5-class e-mail data  ➜  TF-IDF + Logistic-Regression
➜  quantitative metrics  ➜  seven saved plots  ➜  zipped archive.

Run:
    pip install scikit-learn matplotlib seaborn numpy
    python email_calibration_full.py
"""

import random, zipfile, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from datetime import datetime, timedelta
from collections import Counter
from pathlib   import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model            import LogisticRegression
from sklearn.model_selection         import train_test_split
from sklearn.preprocessing           import label_binarize
from sklearn.metrics import (
    accuracy_score, log_loss, roc_curve, auc,
    precision_recall_curve, average_precision_score, classification_report
)
from sklearn.calibration import calibration_curve

# -------------------------------------------------------------------------
# 1  Synthetic e-mail dataset
# -------------------------------------------------------------------------
BASE  = datetime(2024, 1, 1)
CATS  = ["Spam", "Work", "Personal", "Newsletter", "Important"]
N_PER = [60, 55, 45, 40, 45]                     # samples per class
TEMPL = {
    "Spam":       ["Win big now!", "Cheap pills today!", "Click here—prize!"],
    "Work":       ["Team meeting 2 PM.", "Budget report due Friday."],
    "Personal":   ["Dinner this weekend?", "Happy birthday!"],
    "Newsletter": ["AI weekly roundup.", "Recipe of the month."],
    "Important":  ["URGENT: Server down!", "Security breach detected!"]
}
def hour(cat):
    if cat == "Work":   # business hours
        return random.choices(range(9,18),
                              weights=[5,10,15,15,15,15,10,10,5])[0]
    if cat == "Spam":   # any time
        return random.randint(0,23)
    # evening-tilted for the rest
    return random.choices(range(24),
                          weights=[2]*8 + [6]*10 + [8]*6)[0]
def synth(cat):
    d  = random.randint(0,180)
    ts = BASE + timedelta(days=d, hours=hour(cat))
    txt= random.choice(TEMPL[cat])
    return f"{txt}\nSent: {ts:%Y-%m-%d %H:%M}", cat

texts, labels = [], []
for c, n in zip(CATS, N_PER):
    texts += [synth(c)[0] for _ in range(n)]
    labels += [c]*n

lab2int = {c:i for i,c in enumerate(CATS)}
y = np.array([lab2int[l] for l in labels])

# -------------------------------------------------------------------------
# 2  Model
# -------------------------------------------------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(
    texts, y, test_size=0.30, stratify=y, random_state=42)

vec  = TfidfVectorizer(stop_words="english", max_features=2_000)
X_tr = vec.fit_transform(X_tr)
X_te = vec.transform(X_te)

clf   = LogisticRegression(max_iter=1_000).fit(X_tr, y_tr)
probs = clf.predict_proba(X_te)              # (n_samples, 5)
preds = clf.predict(X_te)
conf  = probs.max(1)
correct = (preds == y_te).astype(float)

# -------------------------------------------------------------------------
# 3  Quantitative metrics
# -------------------------------------------------------------------------
def brier(y_true, p):
    oh = np.zeros_like(p); oh[np.arange(len(p)), y_true] = 1
    return np.mean((p - oh).sum(1)**2)
def ece(n=15):
    err = 0; bins = np.linspace(0,1,n+1)
    for lo,hi in zip(bins[:-1], bins[1:]):
        m = (conf>lo)&(conf<=hi)
        if m.any():
            err += m.mean()*abs(conf[m].mean() - correct[m].mean())
    return err
def slope_int(n=15):
    xs=ys=w=[]; bins=np.linspace(0,1,n+1)
    xs,ys,w=[],[],[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        m=(conf>lo)&(conf<=hi)
        if m.sum(): xs.append(conf[m].mean()); ys.append(correct[m].mean()); w.append(m.sum())
    xs,ys,w=np.array(xs),np.array(ys),np.array(w)
    if len(xs)<2: return float("nan"), float("nan")
    wm = lambda z:(w*z).sum()/w.sum()
    slope = (w*(xs-wm(xs))*(ys-wm(ys))).sum() / (w*(xs-wm(xs))**2).sum()
    intercept = wm(ys) - slope*wm(xs)
    return slope, intercept

acc   = accuracy_score(y_te, preds)
lloss = log_loss(y_te, probs)
brier_v = brier(y_te, probs)
ece_v   = ece()
slope_v, int_v = slope_int()

# -------------------------------------------------------------------------
# 4  Plots saved next to the script
# -------------------------------------------------------------------------
DIR   = Path(__file__).resolve().parent if "__file__" in globals() else Path(".")
plt.rcParams["figure.figsize"] = (6,5)
sns.set_style("whitegrid")
plots = []

def save(fig_name):
    plt.tight_layout()
    plt.savefig(DIR / fig_name, dpi=150)
    plt.close()
    plots.append(fig_name)

# A Reliability + histogram
bins=np.linspace(0,1,11)
mids=accs=sizes=[]; mids,accs,sizes=[],[],[]
for lo,hi in zip(bins[:-1],bins[1:]):
    m=(conf>lo)&(conf<=hi)
    if m.sum():
        mids.append(conf[m].mean()); accs.append(correct[m].mean()); sizes.append(m.sum())
plt.subplot(211)
plt.plot([0,1],[0,1],'k--')
plt.scatter(mids,accs,s=[s*1.5 for s in sizes],edgecolor='k')
plt.xlabel("Mean confidence"); plt.ylabel("Fraction positives"); plt.title("Reliability diagram")
plt.subplot(212)
plt.hist(conf,bins=20,color='skyblue',edgecolor='k')
plt.xlabel("Confidence"); plt.ylabel("Count"); plt.title("Confidence histogram")
save("reliability_and_hist.png")

# B Risk–coverage
thr=np.linspace(0,1,21); cov=risk=[],[]
cov,risk=[],[]
for t in thr:
    m=conf>=t
    if m.any(): cov.append(m.mean()); risk.append(1-correct[m].mean())
plt.plot(cov,risk,'-o'); plt.gca().invert_xaxis()
plt.xlabel("Coverage"); plt.ylabel("Error rate"); plt.title("Risk–coverage curve")
save("risk_coverage.png")

# C Per-class calibration
fig,ax=plt.subplots(2,3,figsize=(15,8)); ax=ax.ravel()
for i,c in enumerate(CATS):
    fpos, mp = calibration_curve((y_te==i).astype(int), probs[:,i],
                                 n_bins=10, strategy="uniform")
    ax[i].plot([0,1],[0,1],'k--',lw=0.8)
    ax[i].plot(mp,fpos,'o-',label=c)
    ax[i].set(xlabel="Mean predicted", ylabel="Fraction positives",
              title=f"Calibration – {c}")
    ax[i].legend(fontsize=8)
fig.suptitle("Per-class calibration curves")
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(DIR/"per_class_calibration.png", dpi=150); plt.close()
plots.append("per_class_calibration.png")

# D Confusion matrix
cm = np.zeros((len(CATS), len(CATS)), int)
for t,p in zip(y_te,preds): cm[t,p]+=1
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',
            xticklabels=CATS, yticklabels=CATS)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion matrix")
save("confusion_matrix.png")

# E ROC curves (micro, macro, per-class)
y_bin = label_binarize(y_te, classes=list(range(len(CATS))))
fpr,tpr,aucv={}, {}, {}
for i in range(len(CATS)):
    fpr[i],tpr[i],_=roc_curve(y_bin[:,i],probs[:,i]); aucv[i]=auc(fpr[i],tpr[i])
fpr["micro"],tpr["micro"],_ = roc_curve(y_bin.ravel(),probs.ravel())
aucv["micro"]=auc(fpr["micro"],tpr["micro"])
all_fpr=np.unique(np.concatenate([fpr[i] for i in range(len(CATS))]))
mean_tpr=np.vstack([np.interp(all_fpr,fpr[i],tpr[i]) for i in range(len(CATS))]).mean(0)
aucv["macro"]=auc(all_fpr,mean_tpr)
plt.plot(fpr["micro"],tpr["micro"],label=f"micro AUC={aucv['micro']:.2f}",color='deeppink')
plt.plot(all_fpr,mean_tpr,label=f"macro AUC={aucv['macro']:.2f}",color='navy')
for i,c in enumerate(CATS):
    plt.plot(fpr[i],tpr[i],lw=1,label=f"{c} ({aucv[i]:.2f})")
plt.plot([0,1],[0,1],'k--',lw=0.8)
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC curves"); plt.legend(fontsize=8)
save("roc_curves.png")

# F Precision–Recall curves
plt.figure()
for i,c in enumerate(CATS):
    p,r,_ = precision_recall_curve(y_bin[:,i], probs[:,i])
    ap    = average_precision_score(y_bin[:,i], probs[:,i])
    plt.plot(r,p,lw=1,label=f"{c} AP={ap:.2f}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall curves")
plt.legend(fontsize=8)
save("precision_recall_curves.png")

# G Confidence violin
sns.violinplot(x=[CATS[i] for i in preds], y=conf,
               order=CATS, palette="Pastel1", cut=0)
plt.xlabel("Predicted class"); plt.ylabel("Predicted confidence")
plt.title("Per-class confidence distribution")
save("confidence_violin.png")

# -------------------------------------------------------------------------
# 5  Metrics text file
# -------------------------------------------------------------------------
with open(DIR/"email_calibration_metrics.txt","w") as f:
    f.write("Email-classification quantitative metrics\n")
    f.write("-"*42 + "\n")
    f.write(f"Accuracy:             {acc:.3f}\n")
    f.write(f"Log-loss:             {lloss:.3f}\n")
    f.write(f"Multiclass Brier:     {brier_v:.3f}\n")
    f.write(f"Expected Calib Err.:  {ece_v:.3f}\n")
    f.write(f"Slope:                {slope_v:.2f}\n")
    f.write(f"Intercept:            {int_v:.2f}\n\n")
    f.write("Classification report:\n")
    f.write(classification_report(y_te, preds, target_names=CATS, digits=3))
    f.write("\nClass distribution (test set):\n")
    for c in CATS:
        f.write(f"{c:11}: {(y_te == lab2int[c]).sum()}\n")
plots.append("email_calibration_metrics.txt")

# -------------------------------------------------------------------------
# 6  ZIP bundle
# -------------------------------------------------------------------------
with zipfile.ZipFile(DIR/"email_calibration_plots.zip","w") as z:
    for fn in plots:
        z.write(DIR/fn)

print("\nSaved:", *plots, sep="\n  • ")
print("\nArchived as  email_calibration_plots.zip")
