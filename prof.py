"""
prof.py - Evaluate Prof's 8 Sample Pairs
EE 123 Project - Ved, Ching, Erick

Evaluates the 8 real/spoof pairs from the professor with two approaches:
  1. Direct inference (using VCTK-trained models as-is)
  2. Normalized inference (z-score features relative to prof's own real files)

Audio paths:
  reals/r1.wav ... r8.wav    (real)
  try/try1.wav ... try8.wav  (spoof)

Output:
  prof_report.pdf

Usage:
    python prof.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from utils import (load_models, predict_both, load_audio, compute_stft,
                   make_cover, sorted_feature_order,
                   AUDIO_DIR, PHYSICS_INDICES, PHYSICS_NAMES,
                   LR_THRESHOLD, XGB_THRESHOLD, AVG_THRESHOLD,
                   REAL_COLOR, SPOOF_COLOR, BG_COLOR, DARK_COLOR)
from ching import build_advanced_feature_vector

OUTPUT_PDF = os.path.join(AUDIO_DIR, "prof_report.pdf")
REALS_DIR  = os.path.join(AUDIO_DIR, "reals")
TRY_DIR    = os.path.join(AUDIO_DIR, "try")
NUM_PAIRS  = 8


def get_pair_paths():
    pairs = []
    for i in range(1, NUM_PAIRS + 1):
        r = os.path.join(REALS_DIR, f"r{i}.wav")
        s = os.path.join(TRY_DIR,   f"try{i}.wav")
        if os.path.exists(r) and os.path.exists(s):
            pairs.append((i, r, s))
        else:
            print(f"  ⚠ Missing pair {i}")
    return pairs


def extract_all_features(pairs):
    print("  Extracting features from all pairs...")
    data = []
    for i, real_path, spoof_path in pairs:
        y_r, sr = load_audio(real_path)
        y_s, _  = load_audio(spoof_path)
        _, _, _, pv_r, cent_r = compute_stft(y_r, sr)
        _, _, _, pv_s, cent_s = compute_stft(y_s, sr)
        fv_r = build_advanced_feature_vector(real_path)
        fv_s = build_advanced_feature_vector(spoof_path)
        data.append({
            "i": i, "y_r": y_r, "y_s": y_s, "sr": sr,
            "pv_r": float(np.mean(pv_r)), "pv_s": float(np.mean(pv_s)),
            "cent_r": cent_r, "cent_s": cent_s,
            "fv_r": fv_r, "fv_s": fv_s,
            "phys_r": fv_r[PHYSICS_INDICES],
            "phys_s": fv_s[PHYSICS_INDICES],
        })
        print(f"    Pair {i}: done")
    return data


def run_direct_inference(data, clf_lr, scaler_lr, lr_relative, clf_xgb, scaler_xgb):
    """Predict using VCTK-trained models directly."""
    for d in data:
        lr_r, xgb_r = predict_both(d["fv_r"], clf_lr, scaler_lr, lr_relative,
                                    clf_xgb, scaler_xgb)
        lr_s, xgb_s = predict_both(d["fv_s"], clf_lr, scaler_lr, lr_relative,
                                    clf_xgb, scaler_xgb)
        d["direct_lr_r"]  = lr_r;  d["direct_xgb_r"]  = xgb_r
        d["direct_lr_s"]  = lr_s;  d["direct_xgb_s"]  = xgb_s
        d["direct_avg_r"] = (lr_r + xgb_r) / 2
        d["direct_avg_s"] = (lr_s + xgb_s) / 2


def run_normalized_inference(data, clf_lr, scaler_lr, lr_relative,
                              clf_xgb, scaler_xgb):
    """
    Normalize features relative to the prof's own real files before predicting.
    This corrects for amplitude/scale differences vs VCTK training data.

    Strategy:
      - Compute mean/std of physics features across all 8 REAL files
      - Z-score all 16 files relative to that distribution
      - Re-scale into VCTK range using VCTK scaler's mean/std
    """
    # Stack all real physics features
    real_physics = np.array([d["phys_r"] for d in data])  # (8, 7)
    spoof_physics = np.array([d["phys_s"] for d in data])  # (8, 7)

    # Fit a local scaler on the real files
    local_scaler = StandardScaler()
    local_scaler.fit(real_physics)

    # Normalize all files
    real_norm  = local_scaler.transform(real_physics)
    spoof_norm = local_scaler.transform(spoof_physics)

    # Now re-scale back using VCTK scaler statistics
    # scaler_xgb was fit on VCTK physics features
    # We apply: x_vctk_scale = z_local * vctk_std + vctk_mean
    vctk_mean = scaler_xgb.mean_
    vctk_std  = scaler_xgb.scale_

    real_rescaled  = real_norm  * vctk_std + vctk_mean
    spoof_rescaled = spoof_norm * vctk_std + vctk_mean

    # Also normalize the LR features (subset of physics)
    lr_relative_arr = np.array(lr_relative)
    vctk_lr_mean = scaler_lr.mean_
    vctk_lr_std  = scaler_lr.scale_

    for idx, d in enumerate(data):
        # XGB normalized
        x_xgb_r = real_rescaled[idx].reshape(1, -1)
        x_xgb_s = spoof_rescaled[idx].reshape(1, -1)

        # Scale through VCTK XGB scaler
        x_xgb_r_s = scaler_xgb.transform(x_xgb_r)
        x_xgb_s_s = scaler_xgb.transform(x_xgb_s)

        norm_xgb_r = float(clf_xgb.predict_proba(x_xgb_r_s)[0][1])
        norm_xgb_s = float(clf_xgb.predict_proba(x_xgb_s_s)[0][1])

        # LR normalized
        x_lr_r = real_rescaled[idx][lr_relative_arr].reshape(1, -1)
        x_lr_s = spoof_rescaled[idx][lr_relative_arr].reshape(1, -1)
        x_lr_r_s = scaler_lr.transform(x_lr_r)
        x_lr_s_s = scaler_lr.transform(x_lr_s)

        norm_lr_r = float(clf_lr.predict_proba(x_lr_r_s)[0][1])
        norm_lr_s = float(clf_lr.predict_proba(x_lr_s_s)[0][1])

        d["norm_lr_r"]  = norm_lr_r;  d["norm_xgb_r"]  = norm_xgb_r
        d["norm_lr_s"]  = norm_lr_s;  d["norm_xgb_s"]  = norm_xgb_s
        d["norm_avg_r"] = (norm_lr_r + norm_xgb_r) / 2
        d["norm_avg_s"] = (norm_lr_s + norm_xgb_s) / 2


# ══════════════════════════════════════════════
#  PDF PAGES
# ══════════════════════════════════════════════

def make_per_pair_page(pdf, d):
    """One page per pair: waveforms, spectrograms, phase var, centroid, physics."""
    i   = d["i"]
    sr  = d["sr"]
    y_r = d["y_r"]
    y_s = d["y_s"]

    freqs_r, times_r, mag_r, pv_r_arr, cent_r = compute_stft(y_r, sr)
    freqs_s, times_s, mag_s, pv_s_arr, cent_s = compute_stft(y_s, sr)

    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor(BG_COLOR)
    fig.text(0.5, 0.975, f"Pair {i}: r{i}.wav vs try{i}.wav",
             ha="center", fontsize=17, fontweight="bold", color=DARK_COLOR)
    fig.text(0.5, 0.952,
             f"Direct  → Real: LR={d['direct_lr_r']*100:.0f}% XGB={d['direct_xgb_r']*100:.0f}% "
             f"Avg={d['direct_avg_r']*100:.0f}%   |   "
             f"Spoof: LR={d['direct_lr_s']*100:.0f}% XGB={d['direct_xgb_s']*100:.0f}% "
             f"Avg={d['direct_avg_s']*100:.0f}%",
             ha="center", fontsize=10, color="#444")
    fig.text(0.5, 0.932,
             f"Normalized → Real: LR={d['norm_lr_r']*100:.0f}% XGB={d['norm_xgb_r']*100:.0f}% "
             f"Avg={d['norm_avg_r']*100:.0f}%   |   "
             f"Spoof: LR={d['norm_lr_s']*100:.0f}% XGB={d['norm_xgb_s']*100:.0f}% "
             f"Avg={d['norm_avg_s']*100:.0f}%",
             ha="center", fontsize=10, color="#1e8449")

    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.91, bottom=0.05, left=0.06, right=0.97)

    # Waveforms
    for col, (y, color, label) in enumerate([
            (y_r, REAL_COLOR,  f"REAL r{i}.wav"),
            (y_s, SPOOF_COLOR, f"SPOOF try{i}.wav")]):
        ax = fig.add_subplot(gs[0, col*2:(col+1)*2])
        t  = np.linspace(0, len(y)/sr, len(y))
        ax.plot(t, y, color=color, lw=0.4, alpha=0.9)
        ax.set_title(f"Waveform — {label}", fontweight="bold", color=color)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3); ax.set_facecolor("#f0f4f8")

    # Spectrograms
    for col, (times, freqs, mag, color, label) in enumerate([
            (times_r, freqs_r, mag_r, REAL_COLOR,  "REAL"),
            (times_s, freqs_s, mag_s, SPOOF_COLOR, "SPOOF")]):
        ax = fig.add_subplot(gs[1, col*2:(col+1)*2])
        mdb = 20 * np.log10(mag + 1e-8)
        im  = ax.pcolormesh(times, freqs, mdb, shading="auto", cmap="inferno",
                            vmin=mdb.max()-60, vmax=mdb.max())
        ax.set_title(f"Spectrogram — {label}", fontweight="bold", color=color)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Freq (Hz)")
        ax.set_ylim(0, 8000); plt.colorbar(im, ax=ax, label="dB", pad=0.01)

    # Phase variance
    ax_pv = fig.add_subplot(gs[2, :2])
    ax_pv.plot(times_r, pv_r_arr, color=REAL_COLOR,  lw=0.8, alpha=0.9,
               label=f"Real  (mean={np.mean(pv_r_arr):.3f})")
    ax_pv.plot(times_s, pv_s_arr, color=SPOOF_COLOR, lw=0.8, alpha=0.9,
               label=f"Spoof (mean={np.mean(pv_s_arr):.3f})")
    ax_pv.axhline(np.mean(pv_r_arr), color=REAL_COLOR,  ls="--", lw=0.8, alpha=0.5)
    ax_pv.axhline(np.mean(pv_s_arr), color=SPOOF_COLOR, ls="--", lw=0.8, alpha=0.5)
    ax_pv.set_title("Phase Variance over Time", fontweight="bold")
    ax_pv.set_xlabel("Time (s)"); ax_pv.set_ylabel("Phase Variance")
    ax_pv.legend(fontsize=8); ax_pv.grid(True, alpha=0.3); ax_pv.set_facecolor("#f0f4f8")

    # Spectral centroid
    ax_sc = fig.add_subplot(gs[2, 2:])
    ax_sc.plot(times_r, d["cent_r"], color=REAL_COLOR,  lw=0.8, alpha=0.9, label="Real")
    ax_sc.plot(times_s, d["cent_s"], color=SPOOF_COLOR, lw=0.8, alpha=0.9, label="Spoof")
    ax_sc.set_title("Spectral Centroid over Time", fontweight="bold")
    ax_sc.set_xlabel("Time (s)"); ax_sc.set_ylabel("Freq (Hz)")
    ax_sc.legend(fontsize=8); ax_sc.grid(True, alpha=0.3); ax_sc.set_facecolor("#f0f4f8")

    # Physics bar chart
    ax_bar = fig.add_subplot(gs[3, :2])
    x, w = np.arange(len(PHYSICS_NAMES)), 0.35
    ax_bar.bar(x-w/2, d["phys_r"], w, label="Real",  color=REAL_COLOR,  alpha=0.8)
    ax_bar.bar(x+w/2, d["phys_s"], w, label="Spoof", color=SPOOF_COLOR, alpha=0.8)
    ax_bar.set_title("Physics Features: Real vs Spoof", fontweight="bold")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(PHYSICS_NAMES, rotation=25, ha="right", fontsize=7)
    ax_bar.legend(fontsize=8); ax_bar.grid(True, alpha=0.3, axis="y")
    ax_bar.set_facecolor("#f0f4f8")

    # Feature table
    ax_tbl = fig.add_subplot(gs[3, 2:])
    ax_tbl.axis("off")
    rows = []
    for j, name in enumerate(PHYSICS_NAMES):
        rv, sv = d["phys_r"][j], d["phys_s"][j]
        diff = abs(rv - sv) / (abs(rv) + 1e-8) * 100
        rows.append([name, f"{rv:.4f}", f"{sv:.4f}",
                     f"{diff:.1f}%", "real>" if rv > sv else "spoof>"])
    rv_pv, sv_pv = d["pv_r"], d["pv_s"]
    rows.append(["Phase Var (mean)", f"{rv_pv:.4f}", f"{sv_pv:.4f}",
                 f"{abs(rv_pv-sv_pv)/(rv_pv+1e-8)*100:.1f}%",
                 "real>" if rv_pv > sv_pv else "spoof>"])
    tbl = ax_tbl.table(cellText=rows,
                       colLabels=["Feature", "Real", "Spoof", "Diff%", "Dir"],
                       loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.0, 1.35)
    for j in range(5):
        tbl[0, j].set_facecolor(DARK_COLOR)
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for row_idx, row in enumerate(rows, start=1):
        bg = "#d6eaf8" if row[4] == "real>" else "#fde8e8"
        for j in range(5):
            tbl[row_idx, j].set_facecolor(bg)
    ax_tbl.set_title("Feature Table", fontweight="bold", pad=8)

    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)


def make_verdict_table(pdf, data, method, lr_key, xgb_key, avg_key, title):
    """Verdict table for direct or normalized inference."""
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(BG_COLOR)
    ax.axis("off")
    fig.suptitle(title, fontsize=15, fontweight="bold", color=DARK_COLOR, y=0.97)

    rows = []
    real_correct = spoof_correct = 0

    for d in data:
        i = d["i"]
        for fname, true_label, prob_lr, prob_xgb, prob_avg in [
            (f"r{i}.wav",   "REAL",  d[lr_key+"_r"], d[xgb_key+"_r"], d[avg_key+"_r"]),
            (f"try{i}.wav", "SPOOF", d[lr_key+"_s"], d[xgb_key+"_s"], d[avg_key+"_s"]),
        ]:
            verdict = "SPOOF" if prob_avg >= AVG_THRESHOLD else "REAL"
            correct = verdict == true_label
            tick    = "✓" if correct else "✗"
            if correct:
                if true_label == "REAL": real_correct += 1
                else: spoof_correct += 1
            rows.append([fname, true_label,
                         f"{prob_lr*100:.1f}%", f"{prob_xgb*100:.1f}%",
                         f"{prob_avg*100:.1f}%", verdict, tick])

    # Summary rows
    rows.append(["─"*8]*7)
    total = real_correct + spoof_correct
    rows.append(["Real caught",  "", "", "", "", f"{real_correct}/8",  ""])
    rows.append(["Spoof caught", "", "", "", "", f"{spoof_correct}/8", ""])
    rows.append(["OVERALL",      "", "", "", "",
                 f"{total}/16 = {total/16*100:.0f}%", ""])

    col_labels = ["File", "True", "LR%", "XGB%", "Avg%", "Verdict (AVG)", "✓?"]
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 1.7)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor(DARK_COLOR)
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    for row_idx, row in enumerate(rows, start=1):
        if len(row[0]) < 3: continue
        bg = "#d6eaf8" if row[1] == "REAL" else "#fde8e8"
        for j in range(len(col_labels)):
            tbl[row_idx, j].set_facecolor(bg)

    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)
    return real_correct, spoof_correct


def make_normalization_comparison(pdf, data):
    """Side-by-side comparison of direct vs normalized probabilities."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle("Direct vs Normalized Inference — Spoof Probabilities",
                 fontsize=15, fontweight="bold", color=DARK_COLOR, y=0.97)

    pairs = [f"P{d['i']}" for d in data]

    for col, (title, lr_r_key, lr_s_key, xgb_r_key, xgb_s_key) in enumerate([
        ("Direct (VCTK scale)", "direct_lr_r", "direct_lr_s",
         "direct_xgb_r", "direct_xgb_s"),
        ("Normalized (prof scale)", "norm_lr_r", "norm_lr_s",
         "norm_xgb_r", "norm_xgb_s"),
    ]):
        ax = axes[col]
        lr_r   = [d[lr_r_key]*100   for d in data]
        lr_s   = [d[lr_s_key]*100   for d in data]
        xgb_r  = [d[xgb_r_key]*100  for d in data]
        xgb_s  = [d[xgb_s_key]*100  for d in data]

        x = np.arange(len(pairs))
        w = 0.2
        ax.bar(x-1.5*w, lr_r,  w, label="LR Real",   color=REAL_COLOR,  alpha=0.7)
        ax.bar(x-0.5*w, lr_s,  w, label="LR Spoof",  color=SPOOF_COLOR, alpha=0.7)
        ax.bar(x+0.5*w, xgb_r, w, label="XGB Real",  color=REAL_COLOR,  alpha=1.0)
        ax.bar(x+1.5*w, xgb_s, w, label="XGB Spoof", color=SPOOF_COLOR, alpha=1.0)
        ax.axhline(AVG_THRESHOLD*100, color="black", ls="--", lw=1.2,
                   label=f"Threshold {AVG_THRESHOLD*100:.0f}%")
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xticks(x); ax.set_xticklabels(pairs)
        ax.set_ylabel("Spoof Probability (%)"); ax.set_ylim(0, 105)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
        ax.set_facecolor("#f0f4f8")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)


def make_domain_mismatch_page(pdf, data, vctk_real_feats):
    """Bar chart comparing prof feature scale vs VCTK training scale."""
    phys_r = np.array([d["phys_r"] for d in data])
    phys_s = np.array([d["phys_s"] for d in data])
    vctk_r = np.array(vctk_real_feats)

    order   = sorted_feature_order(phys_r, phys_s)
    idxs    = [e[0] for e in order]
    names_s = [e[1] for e in order]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle("Domain Mismatch: Prof Samples vs VCTK Training Scale",
                 fontsize=15, fontweight="bold", color=DARK_COLOR, y=0.97)

    x, w = np.arange(len(names_s)), 0.35

    ax1 = axes[0]
    ax1.bar(x-w/2, [np.mean(vctk_r[:, i]) for i in idxs], w,
            label="VCTK Real (training)", color="#2980b9", alpha=0.85)
    ax1.bar(x+w/2, [np.mean(phys_r[:, i]) for i in idxs], w,
            label="Prof Real (test)",     color="#27ae60", alpha=0.85)
    ax1.set_title("Real Audio: VCTK vs Prof Feature Means", fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(names_s, rotation=30, ha="right", fontsize=8)
    ax1.legend(); ax1.grid(True, alpha=0.3, axis="y"); ax1.set_facecolor("#f0f4f8")
    ax1.set_ylabel("Mean Feature Value")

    ax2 = axes[1]
    # Scale ratio
    vctk_mean = np.array([np.mean(vctk_r[:, i]) for i in idxs])
    prof_mean = np.array([np.mean(phys_r[:, i]) for i in idxs])
    ratio = prof_mean / (vctk_mean + 1e-8)
    colors = ["#e74c3c" if r > 2 or r < 0.5 else "#e67e22" if r > 1.5 or r < 0.7
              else "#2ecc71" for r in ratio]
    bars = ax2.bar(names_s, ratio, color=colors, alpha=0.85)
    ax2.axhline(1.0, color="black", ls="--", lw=1.5, label="1.0 = same scale")
    ax2.axhline(2.0, color="#e74c3c", ls=":", lw=1.2, label="2x scale")
    ax2.axhline(0.5, color="#e74c3c", ls=":", lw=1.2, label="0.5x scale")
    ax2.set_title("Prof/VCTK Feature Scale Ratio\n(1.0 = identical, red = large mismatch)",
                  fontweight="bold")
    ax2.set_ylabel("Scale Ratio (Prof / VCTK)")
    ax2.set_xticklabels(names_s, rotation=30, ha="right", fontsize=8)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_facecolor("#f0f4f8")
    for bar, val in zip(bars, ratio):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                 f"{val:.2f}x", ha="center", va="bottom", fontsize=8)

    fig.text(0.5, 0.04,
             "Features with ratio far from 1.0 cause the model to predict incorrectly "
             "— the StandardScaler maps them outside the trained decision boundary.",
             ha="center", fontsize=10, color="#555", style="italic")

    plt.tight_layout(rect=[0, 0.07, 1, 0.94])
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  prof.py — Prof's 8-Pair Evaluation")
    print("  EE 123 Project — Ved, Ching, Erick")
    print("=" * 60)

    print("\nLoading models...")
    clf_lr, scaler_lr, lr_relative, clf_xgb, scaler_xgb = load_models()

    pairs = get_pair_paths()
    if not pairs:
        print("ERROR: No pairs found. Check reals/ and try/ folders.")
        return

    print(f"\n[1/4] Extracting features from {len(pairs)} pairs...")
    data = extract_all_features(pairs)

    print("\n[2/4] Running direct inference...")
    run_direct_inference(data, clf_lr, scaler_lr, lr_relative, clf_xgb, scaler_xgb)

    print("\n[3/4] Running normalized inference...")
    run_normalized_inference(data, clf_lr, scaler_lr, lr_relative, clf_xgb, scaler_xgb)

    # Print quick summary to terminal
    print("\n  Direct inference results (AVG threshold):")
    print(f"  {'File':<12} {'True':<8} {'LR%':>6} {'XGB%':>6} {'Avg%':>6} {'Verdict'}")
    print("  " + "-" * 50)
    for d in data:
        for fname, true, lr, xgb, avg in [
            (f"r{d['i']}.wav",   "REAL",  d["direct_lr_r"], d["direct_xgb_r"], d["direct_avg_r"]),
            (f"try{d['i']}.wav", "SPOOF", d["direct_lr_s"], d["direct_xgb_s"], d["direct_avg_s"]),
        ]:
            v = "SPOOF" if avg >= AVG_THRESHOLD else "REAL"
            t = "✓" if v == true else "✗"
            print(f"  {fname:<12} {true:<8} {lr*100:>5.1f}% {xgb*100:>5.1f}% "
                  f"{avg*100:>5.1f}% {v} {t}")

    print("\n  Normalized inference results (AVG threshold):")
    print(f"  {'File':<12} {'True':<8} {'LR%':>6} {'XGB%':>6} {'Avg%':>6} {'Verdict'}")
    print("  " + "-" * 50)
    for d in data:
        for fname, true, lr, xgb, avg in [
            (f"r{d['i']}.wav",   "REAL",  d["norm_lr_r"], d["norm_xgb_r"], d["norm_avg_r"]),
            (f"try{d['i']}.wav", "SPOOF", d["norm_lr_s"], d["norm_xgb_s"], d["norm_avg_s"]),
        ]:
            v = "SPOOF" if avg >= AVG_THRESHOLD else "REAL"
            t = "✓" if v == true else "✗"
            print(f"  {fname:<12} {true:<8} {lr*100:>5.1f}% {xgb*100:>5.1f}% "
                  f"{avg*100:>5.1f}% {v} {t}")

    # Load VCTK real feats for domain mismatch comparison
    from utils import extract_feats_only
    from utils import AUDIO_DIR as AD
    import os
    vctk_real_feats = extract_feats_only(
        os.path.join(AD, "Samples", "source-vctk1k"), "VCTK ref")

    print("\n[4/4] Generating PDF...")
    with PdfPages(OUTPUT_PDF) as pdf:

        make_cover(pdf, "Prof's 8-Pair Evaluation",
                   "Direct + Normalized Inference Analysis",
                   [f"{len(pairs)} matched real/spoof pairs",
                    "Models trained on VCTK 3000-pair dataset",
                    "Direct: VCTK scale  |  Normalized: Prof scale"])

        print("  → Per-pair analysis pages")
        for d in data:
            make_per_pair_page(pdf, d)

        print("  → Direct inference verdict table")
        rc_d, sc_d = make_verdict_table(
            pdf, data, "direct",
            "direct_lr", "direct_xgb", "direct_avg",
            "Direct Inference — Verdict Table (AVG strategy)")

        print("  → Normalized inference verdict table")
        rc_n, sc_n = make_verdict_table(
            pdf, data, "norm",
            "norm_lr", "norm_xgb", "norm_avg",
            "Normalized Inference — Verdict Table (AVG strategy)")

        print("  → Normalization comparison")
        make_normalization_comparison(pdf, data)

        print("  → Domain mismatch analysis")
        make_domain_mismatch_page(pdf, data, vctk_real_feats)

        d_info = pdf.infodict()
        d_info["Title"]  = "Prof 8-Pair Evaluation"
        d_info["Author"] = "Ved, Ching, Erick — EE 123"

    print(f"\n✅ Report saved to: {OUTPUT_PDF}")
    print(f"\n  Direct     : Real {rc_d}/8  Spoof {sc_d}/8  "
          f"Overall {rc_d+sc_d}/16 = {(rc_d+sc_d)/16*100:.0f}%")
    print(f"  Normalized : Real {rc_n}/8  Spoof {sc_n}/8  "
          f"Overall {rc_n+sc_n}/16 = {(rc_n+sc_n)/16*100:.0f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
