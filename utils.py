"""
utils.py - Shared utilities for prof.py, report1000.py, report500.py
EE 123 Project - Ved, Ching, Erick
"""

import os
import glob
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import scipy.signal
from tqdm import tqdm
from ching import build_advanced_feature_vector, preprocess_audio

# ─────────────────────────────────────────────
AUDIO_DIR = os.path.dirname(os.path.abspath(__file__))

PHYSICS_INDICES = [0, 1, 2, 3, 4, 125, 126]
PHYSICS_NAMES   = [
    "Glottal Kurtosis", "Glottal Std", "Spectral Flux Std",
    "Spectral Flux Max", "Modulation Energy", "Aliasing Mean", "Aliasing Std"
]
MFCC_START = 5
MFCC_END   = 125

LR_THRESHOLD  = 0.45
XGB_THRESHOLD = 0.50
AVG_THRESHOLD = 0.45

REAL_COLOR  = "#2980b9"
SPOOF_COLOR = "#e74c3c"
BG_COLOR    = "#f8f9fa"
DARK_COLOR  = "#1a1a2e"
# ─────────────────────────────────────────────


# ══════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════

def load_models():
    with open(os.path.join(AUDIO_DIR, "model_lr.pkl"),  "rb") as f:
        clf_lr,  scaler_lr,  lr_relative, phys_idx = pickle.load(f)
    with open(os.path.join(AUDIO_DIR, "model_xgb.pkl"), "rb") as f:
        clf_xgb, scaler_xgb, _           = pickle.load(f)
    return clf_lr, scaler_lr, lr_relative, clf_xgb, scaler_xgb


def predict_both(fv, clf_lr, scaler_lr, lr_relative, clf_xgb, scaler_xgb):
    physics  = fv[PHYSICS_INDICES]
    x_lr     = physics[lr_relative].reshape(1, -1)
    prob_lr  = clf_lr.predict_proba(scaler_lr.transform(x_lr))[0][1]
    x_xgb    = physics.reshape(1, -1)
    prob_xgb = clf_xgb.predict_proba(scaler_xgb.transform(x_xgb))[0][1]
    return float(prob_lr), float(prob_xgb)


# ══════════════════════════════════════════════
#  AUDIO HELPERS
# ══════════════════════════════════════════════

def load_audio(path):
    y, sr = preprocess_audio(path, target_sr=16000)
    return y, sr


def compute_stft(y, sr):
    freqs, times, Zxx = scipy.signal.stft(
        y, fs=sr, window="hamming", nperseg=400, noverlap=240, nfft=512)
    mag      = np.abs(Zxx)
    phase    = np.angle(Zxx)
    power    = mag ** 2
    centroid  = np.sum(freqs[:, None] * power, axis=0) / (np.sum(power, axis=0) + 1e-8)
    phase_var = np.var(phase, axis=0)
    return freqs, times, mag, phase_var, centroid


# ══════════════════════════════════════════════
#  DATASET EXTRACTION
# ══════════════════════════════════════════════

def extract_folder(folder, true_label, clf_lr, scaler_lr, lr_relative,
                   clf_xgb, scaler_xgb, desc=""):
    files   = sorted(glob.glob(os.path.join(folder, "*.wav")))
    results = []
    feats   = []
    errors  = 0
    for f in tqdm(files, desc=f"  {desc}", ncols=75):
        try:
            fv       = build_advanced_feature_vector(f)
            prob_lr, prob_xgb = predict_both(
                fv, clf_lr, scaler_lr, lr_relative, clf_xgb, scaler_xgb)
            results.append({"file": os.path.basename(f),
                            "true": true_label,
                            "lr": prob_lr, "xgb": prob_xgb})
            feats.append(fv[PHYSICS_INDICES])
        except Exception:
            errors += 1
    print(f"    {len(results)} processed, {errors} errors")
    return results, np.array(feats)


def extract_feats_only(folder, desc=""):
    files = sorted(glob.glob(os.path.join(folder, "*.wav")))
    feats = []
    for f in tqdm(files, desc=f"  {desc}", ncols=75):
        try:
            fv = build_advanced_feature_vector(f)
            feats.append(fv[PHYSICS_INDICES])
        except Exception:
            pass
    return np.array(feats)


# ══════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════

def evaluate_strategy(results, strategy):
    correct = real_c = spoof_c = real_t = spoof_t = 0
    for r in results:
        avg   = (r["lr"] + r["xgb"]) / 2
        if strategy == "OR":
            spoof = r["lr"] >= LR_THRESHOLD or r["xgb"] >= XGB_THRESHOLD
        elif strategy == "AND":
            spoof = r["lr"] >= LR_THRESHOLD and r["xgb"] >= XGB_THRESHOLD
        else:
            spoof = avg >= AVG_THRESHOLD
        verdict = "SPOOF" if spoof else "REAL"
        ok = verdict == r["true"]
        if r["true"] == "REAL":
            real_t += 1
            if ok: real_c += 1
        else:
            spoof_t += 1
            if ok: spoof_c += 1
        if ok: correct += 1
    total = len(results)
    return {
        "overall": correct / total * 100,
        "real_acc": real_c / real_t * 100 if real_t else 0,
        "spoof_acc": spoof_c / spoof_t * 100 if spoof_t else 0,
        "correct": correct, "total": total,
        "real_correct": real_c, "real_total": real_t,
        "spoof_correct": spoof_c, "spoof_total": spoof_t,
    }


def eval_lr_only(results):
    c = rc = sc = rt = st = 0
    for r in results:
        ok = ("SPOOF" if r["lr"] >= LR_THRESHOLD else "REAL") == r["true"]
        if r["true"] == "REAL": rt += 1; rc += ok
        else: st += 1; sc += ok
        c += ok
    return {"overall": c/len(results)*100,
            "real_acc": rc/rt*100, "spoof_acc": sc/st*100,
            "correct": c, "total": len(results),
            "real_correct": rc, "real_total": rt,
            "spoof_correct": sc, "spoof_total": st}


def eval_xgb_only(results):
    c = rc = sc = rt = st = 0
    for r in results:
        ok = ("SPOOF" if r["xgb"] >= XGB_THRESHOLD else "REAL") == r["true"]
        if r["true"] == "REAL": rt += 1; rc += ok
        else: st += 1; sc += ok
        c += ok
    return {"overall": c/len(results)*100,
            "real_acc": rc/rt*100, "spoof_acc": sc/st*100,
            "correct": c, "total": len(results),
            "real_correct": rc, "real_total": rt,
            "spoof_correct": sc, "spoof_total": st}


# ══════════════════════════════════════════════
#  SHARED PDF PAGES
# ══════════════════════════════════════════════

def make_cover(pdf, title, subtitle, lines):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    fig.patch.set_facecolor(DARK_COLOR)
    ax.text(0.5, 0.82, title,    ha="center", fontsize=38, fontweight="bold",
            color="white",   transform=ax.transAxes)
    ax.text(0.5, 0.70, subtitle, ha="center", fontsize=18, color="#a0a0c0",
            transform=ax.transAxes)
    ax.text(0.5, 0.60, "EE 123 Project  —  Ved, Ching, Erick",
            ha="center", fontsize=13, color="#7070a0", transform=ax.transAxes)
    for i, line in enumerate(lines):
        ax.text(0.5, 0.46 - i*0.08, line, ha="center", fontsize=11,
                color="#909090", transform=ax.transAxes)
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)


def sorted_feature_order(real_arr, spoof_arr):
    entries = []
    for i, name in enumerate(PHYSICS_NAMES):
        r, s = real_arr[:, i], spoof_arr[:, i]
        pooled = np.sqrt((r.std()**2 + s.std()**2) / 2) + 1e-8
        d = abs(r.mean() - s.mean()) / pooled
        entries.append((i, name, d, np.mean(r > s) * 100))
    entries.sort(key=lambda x: x[2], reverse=True)
    return entries


def make_feature_discriminability_page(pdf, real_feats, spoof_feats, title):
    order      = sorted_feature_order(real_feats, spoof_feats)
    names_s    = [e[1] for e in order]
    d_vals     = [e[2] for e in order]
    cons_vals  = [e[3] for e in order]
    idxs       = [e[0] for e in order]
    real_arr   = np.array(real_feats)
    spoof_arr  = np.array(spoof_feats)

    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(title, fontsize=16, fontweight="bold", color=DARK_COLOR, y=0.97)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.90, bottom=0.08, left=0.08, right=0.97)

    # Cohen's d bar chart
    ax1 = fig.add_subplot(gs[0, :])
    colors = ["#2ecc71" if d > 0.5 else "#e67e22" if d > 0.3 else "#e74c3c"
              for d in d_vals]
    bars = ax1.bar(names_s, d_vals, color=colors, alpha=0.85, edgecolor="white")
    ax1.axhline(0.5, color="#2ecc71", ls="--", lw=1.2, label="d=0.5 (medium)")
    ax1.axhline(0.3, color="#e67e22", ls="--", lw=1.2, label="d=0.3 (small)")
    ax1.set_title("Cohen's d — Feature Discriminability", fontweight="bold")
    ax1.set_ylabel("Cohen's d"); ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y"); ax1.set_facecolor("#f0f4f8")
    for bar, val in zip(bars, d_vals):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Real vs Spoof means
    ax2 = fig.add_subplot(gs[1, 0])
    x, w = np.arange(len(names_s)), 0.35
    ax2.bar(x-w/2, [real_arr[:,i].mean() for i in idxs], w,
            label="Real",  color=REAL_COLOR,  alpha=0.8)
    ax2.bar(x+w/2, [spoof_arr[:,i].mean() for i in idxs], w,
            label="Spoof", color=SPOOF_COLOR, alpha=0.8)
    ax2.set_title("Mean Feature Values: Real vs Spoof", fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(names_s, rotation=30, ha="right", fontsize=7)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_facecolor("#f0f4f8")

    # Consistency
    ax3 = fig.add_subplot(gs[1, 1])
    colors2 = ["#2ecc71" if c > 75 else "#e67e22" if c > 60 else "#e74c3c"
               for c in cons_vals]
    bars3 = ax3.bar(names_s, cons_vals, color=colors2, alpha=0.85)
    ax3.axhline(75, color="#2ecc71", ls="--", lw=1.2, label="75%")
    ax3.axhline(60, color="#e67e22", ls="--", lw=1.2, label="60%")
    ax3.set_title("Consistency: % Pairs where Real > Spoof", fontweight="bold")
    ax3.set_ylabel("Consistency (%)"); ax3.set_ylim(0, 105)
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3, axis="y")
    ax3.set_facecolor("#f0f4f8")
    ax3.set_xticklabels(names_s, rotation=30, ha="right", fontsize=7)
    for bar, val in zip(bars3, cons_vals):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f"{val:.0f}%", ha="center", va="bottom", fontsize=8)

    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)


def make_feature_stats_table(pdf, real_feats, spoof_feats,
                              title, n_real, n_spoof,
                              ref_feats=None, ref_spoof=None):
    """Aggregate feature stats table with raw counts and Cohen's d."""
    real_arr  = np.array(real_feats)
    spoof_arr = np.array(spoof_feats)
    order     = sorted_feature_order(real_arr, spoof_arr)

    fig, ax = plt.subplots(figsize=(18, 9))
    fig.patch.set_facecolor(BG_COLOR)
    ax.axis("off")
    fig.suptitle(title, fontsize=15, fontweight="bold", color=DARK_COLOR, y=0.97)

    col_labels = ["Feature", "Real Mean", "Real Std",
                  "Spoof Mean", "Spoof Std",
                  "Diff%", "Cohen's d",
                  f"Real>Spoof\n(out of {n_spoof} pairs)", "Consist%", "Keep?"]
    rows = []
    for orig_idx, name, d, consist in order:
        r = real_arr[:, orig_idx]
        s = spoof_arr[:, orig_idx]
        r_mean, r_std = r.mean(), r.std()
        s_mean, s_std = s.mean(), s.std()
        diff = (r_mean - s_mean) / (abs(r_mean) + 1e-8) * 100
        # count how many pairs real > spoof (use min length)
        min_n = min(len(r), len(s), n_spoof)
        count_gt = int(np.sum(r[:min_n] > s[:min_n]))
        keep = "✓ KEEP" if d > 0.3 and consist > 60 else "✗ DROP"
        rows.append([name,
                     f"{r_mean:.3f}", f"{r_std:.3f}",
                     f"{s_mean:.3f}", f"{s_std:.3f}",
                     f"{diff:+.1f}%", f"{d:.3f}",
                     f"{count_gt}/{min_n}", f"{consist:.1f}%",
                     keep])

    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.05, 2.3)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor(DARK_COLOR)
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    for row_idx, (orig_idx, name, d, consist) in enumerate(order, start=1):
        keep = rows[row_idx-1][-1]
        bg = "#d5f5e3" if d > 0.5 else "#fef9e7" if d > 0.3 else "#fde8e8"
        for j in range(len(col_labels)):
            tbl[row_idx, j].set_facecolor(bg)
        last_col = len(col_labels) - 1
        tbl[row_idx, last_col].set_text_props(
            fontweight="bold",
            color="#1e8449" if keep == "✓ KEEP" else "#c0392b")

    fig.text(0.5, 0.05,
             "Green = strong (d>0.5)  |  Yellow = moderate (d>0.3)  |  Red = weak/drop",
             ha="center", fontsize=10, color="#555555", style="italic")
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)


def make_ensemble_results_page(pdf, all_results, real_results, spoof_results,
                                title, n_real_total, n_spoof_total):
    """Strategy bar chart + probability distributions + raw count summary."""
    fig = plt.figure(figsize=(16, 13))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(title, fontsize=15, fontweight="bold", color=DARK_COLOR, y=0.97)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35,
                           top=0.91, bottom=0.06, left=0.08, right=0.97)

    strategies  = ["LR alone", "XGB alone", "OR", "AND", "AVG"]
    lr_s   = eval_lr_only(all_results)
    xgb_s  = eval_xgb_only(all_results)
    or_s   = evaluate_strategy(all_results, "OR")
    and_s  = evaluate_strategy(all_results, "AND")
    avg_s  = evaluate_strategy(all_results, "AVG")
    all_s  = [lr_s, xgb_s, or_s, and_s, avg_s]

    overall_vals  = [s["overall"]   for s in all_s]
    real_vals     = [s["real_acc"]  for s in all_s]
    spoof_vals    = [s["spoof_acc"] for s in all_s]

    # Bar chart
    ax1 = fig.add_subplot(gs[0, :])
    x, w = np.arange(len(strategies)), 0.25
    ax1.bar(x-w, real_vals,    w, label="Real accuracy",  color=REAL_COLOR,  alpha=0.8)
    ax1.bar(x,   spoof_vals,   w, label="Spoof accuracy", color=SPOOF_COLOR, alpha=0.8)
    ax1.bar(x+w, overall_vals, w, label="Overall",        color="#8e44ad",   alpha=0.8)
    ax1.set_title("Accuracy by Strategy", fontweight="bold", fontsize=13)
    ax1.set_xticks(x); ax1.set_xticklabels(strategies, fontsize=10)
    ax1.set_ylabel("Accuracy (%)"); ax1.set_ylim(0, 105)
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_facecolor("#f0f4f8")
    for xi, ov in zip(x+w, overall_vals):
        ax1.text(xi, ov+0.5, f"{ov:.1f}%", ha="center", va="bottom",
                 fontsize=8, fontweight="bold")

    # LR distribution
    ax2 = fig.add_subplot(gs[1, 0])
    lr_r = [r["lr"] for r in real_results]
    lr_s_probs = [r["lr"] for r in spoof_results]
    ax2.hist(lr_r,       bins=30, color=REAL_COLOR,  alpha=0.7, density=True,
             label=f"Real  (avg={np.mean(lr_r):.3f})")
    ax2.hist(lr_s_probs, bins=30, color=SPOOF_COLOR, alpha=0.7, density=True,
             label=f"Spoof (avg={np.mean(lr_s_probs):.3f})")
    ax2.axvline(LR_THRESHOLD, color="black", ls="--", lw=1.5,
                label=f"Threshold {LR_THRESHOLD}")
    ax2.set_title("LR Probability Distribution", fontweight="bold")
    ax2.set_xlabel("Spoof Probability"); ax2.set_ylabel("Density")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3); ax2.set_facecolor("#f0f4f8")

    # XGB distribution
    ax3 = fig.add_subplot(gs[1, 1])
    xgb_r = [r["xgb"] for r in real_results]
    xgb_s_probs = [r["xgb"] for r in spoof_results]
    ax3.hist(xgb_r,        bins=30, color=REAL_COLOR,  alpha=0.7, density=True,
             label=f"Real  (avg={np.mean(xgb_r):.3f})")
    ax3.hist(xgb_s_probs,  bins=30, color=SPOOF_COLOR, alpha=0.7, density=True,
             label=f"Spoof (avg={np.mean(xgb_s_probs):.3f})")
    ax3.axvline(XGB_THRESHOLD, color="black", ls="--", lw=1.5,
                label=f"Threshold {XGB_THRESHOLD}")
    ax3.set_title("XGBoost Probability Distribution", fontweight="bold")
    ax3.set_xlabel("Spoof Probability"); ax3.set_ylabel("Density")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3); ax3.set_facecolor("#f0f4f8")

    # Raw counts summary table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")
    col_labels = ["Strategy", "Real Correct", "Real Total", "Real%",
                  "Spoof Correct", "Spoof Total", "Spoof%",
                  "Overall Correct", "Overall Total", "Overall%"]
    rows = []
    for name, s in zip(strategies, all_s):
        rows.append([name,
                     str(s["real_correct"]),  str(s["real_total"]),
                     f"{s['real_acc']:.1f}%",
                     str(s["spoof_correct"]), str(s["spoof_total"]),
                     f"{s['spoof_acc']:.1f}%",
                     str(s["correct"]),       str(s["total"]),
                     f"{s['overall']:.1f}%"])
    tbl = ax4.table(cellText=rows, colLabels=col_labels,
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.8)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor(DARK_COLOR)
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # Highlight best overall row
    best_idx = int(np.argmax(overall_vals)) + 1
    for j in range(len(col_labels)):
        tbl[best_idx, j].set_facecolor("#d5f5e3")
    ax4.set_title("Raw Count Summary", fontweight="bold", pad=8)

    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)

    return {"lr": lr_s, "xgb": xgb_s, "or": or_s, "and": and_s, "avg": avg_s}


def make_waveform_spectrogram_page(pdf, pairs_data, title):
    """Average waveform energy, spectrogram stats, phase variance across pairs."""
    all_pv_r   = [d["pv_r"]   for d in pairs_data]
    all_pv_s   = [d["pv_s"]   for d in pairs_data]
    all_phys_r = np.array([d["phys_r"] for d in pairs_data])
    all_phys_s = np.array([d["phys_s"] for d in pairs_data])

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(title, fontsize=16, fontweight="bold", color=DARK_COLOR, y=0.98)

    order      = sorted_feature_order(all_phys_r, all_phys_s)
    idxs       = [e[0] for e in order]
    names_s    = [e[1] for e in order]
    mean_r     = np.mean(all_phys_r, axis=0)
    mean_s     = np.mean(all_phys_s, axis=0)
    pairs      = [f"P{i+1}" for i in range(len(pairs_data))]

    # Phase variance per pair
    ax = axes[0, 0]
    ax.plot(pairs, all_pv_r, "o-", color=REAL_COLOR,  lw=2, ms=6, label="Real")
    ax.plot(pairs, all_pv_s, "s-", color=SPOOF_COLOR, lw=2, ms=6, label="Spoof")
    ax.set_title("Mean Phase Variance per Pair", fontweight="bold")
    ax.set_ylabel("Phase Variance"); ax.legend()
    ax.grid(True, alpha=0.3); ax.set_facecolor("#f0f4f8")
    ax.tick_params(axis='x', rotation=30)

    # Avg physics features
    ax = axes[0, 1]
    x, w = np.arange(len(names_s)), 0.35
    ax.bar(x-w/2, [mean_r[i] for i in idxs], w,
           label="Real",  color=REAL_COLOR,  alpha=0.8)
    ax.bar(x+w/2, [mean_s[i] for i in idxs], w,
           label="Spoof", color=SPOOF_COLOR, alpha=0.8)
    ax.set_title("Avg Physics Features", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names_s, rotation=30, ha="right", fontsize=7)
    ax.legend(); ax.grid(True, alpha=0.3, axis="y"); ax.set_facecolor("#f0f4f8")

    # % diff per feature
    ax = axes[0, 2]
    mean_r_safe = np.where(np.abs(mean_r) < 1e-8, 1e-8, mean_r)
    pct = np.abs(mean_r - mean_s) / np.abs(mean_r_safe) * 100
    colors = ["#2ecc71" if p > 20 else "#e67e22" if p > 10 else "#e74c3c" for p in pct]
    bars = ax.bar([PHYSICS_NAMES[i] for i in idxs],
                  [pct[i] for i in idxs], color=colors, alpha=0.85)
    ax.set_title("Avg % Difference per Feature", fontweight="bold")
    ax.set_ylabel("% Difference")
    ax.set_xticklabels([PHYSICS_NAMES[i] for i in idxs],
                       rotation=30, ha="right", fontsize=7)
    ax.grid(True, alpha=0.3, axis="y"); ax.set_facecolor("#f0f4f8")
    for bar, val in zip(bars, [pct[i] for i in idxs]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=7)

    # Phase variance boxplot
    ax = axes[1, 0]
    ax.boxplot([all_pv_r, all_pv_s], labels=["Real", "Spoof"],
               patch_artist=True,
               boxprops=dict(facecolor="#d6eaf8"),
               medianprops=dict(color=DARK_COLOR, linewidth=2))
    ax.set_title("Phase Variance Distribution", fontweight="bold")
    ax.set_ylabel("Mean Phase Variance")
    ax.grid(True, alpha=0.3, axis="y"); ax.set_facecolor("#f0f4f8")

    # LR prob per pair
    ax = axes[1, 1]
    lr_r_probs = [d["prob_lr_r"] for d in pairs_data]
    lr_s_probs = [d["prob_lr_s"] for d in pairs_data]
    ax.plot(pairs, [p*100 for p in lr_r_probs], "o-",
            color=REAL_COLOR,  lw=2, ms=6, label="Real")
    ax.plot(pairs, [p*100 for p in lr_s_probs], "s-",
            color=SPOOF_COLOR, lw=2, ms=6, label="Spoof")
    ax.axhline(LR_THRESHOLD*100, color="gray", ls="--", lw=1.2,
               label=f"Threshold {LR_THRESHOLD*100:.0f}%")
    ax.set_title("LR Spoof Probability per Pair", fontweight="bold")
    ax.set_ylabel("Spoof %"); ax.set_ylim(0, 105)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_facecolor("#f0f4f8")
    ax.tick_params(axis='x', rotation=30)

    # XGB prob per pair
    ax = axes[1, 2]
    xgb_r_probs = [d["prob_xgb_r"] for d in pairs_data]
    xgb_s_probs = [d["prob_xgb_s"] for d in pairs_data]
    ax.plot(pairs, [p*100 for p in xgb_r_probs], "o-",
            color=REAL_COLOR,  lw=2, ms=6, label="Real")
    ax.plot(pairs, [p*100 for p in xgb_s_probs], "s-",
            color=SPOOF_COLOR, lw=2, ms=6, label="Spoof")
    ax.axhline(XGB_THRESHOLD*100, color="gray", ls="--", lw=1.2,
               label=f"Threshold {XGB_THRESHOLD*100:.0f}%")
    ax.set_title("XGB Spoof Probability per Pair", fontweight="bold")
    ax.set_ylabel("Spoof %"); ax.set_ylim(0, 105)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_facecolor("#f0f4f8")
    ax.tick_params(axis='x', rotation=30)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)