"""
single_clip.py - Single Clip Detailed Analysis
EE 123 Project - Ved, Ching, Erick

Takes a folder and file index, runs full analysis on that one clip
and generates a detailed PDF with waveform, spectrogram, phase variance,
spectral centroid, physics features, probability scores, and verdict.

Usage:
    python single_clip.py <set_letter> <index>

Examples:
    python single_clip.py A 0        # first clip in source-vctk1k
    python single_clip.py C 42       # 43rd clip in converted-vctk1k
    python single_clip.py D 10       # 11th clip in source-vctk500
    python single_clip.py F 0        # first clip in converted-vctk500

Set mapping:
    A -> Samples/source-vctk1k/      (real,  trained on)
    B -> Samples/target-vctk1k/      (real,  NOT trained on)
    C -> Samples/converted-vctk1k/   (spoof, trained on)
    D -> voice-conversion/source-vctk500/  (real,  NOT trained on)
    E -> voice-conversion/target-vctk500/  (real,  NOT trained on)
    F -> voice-conversion/converted-vctk500/ (spoof, NOT trained on)
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from utils import (load_models, predict_both, load_audio, compute_stft,
                   AUDIO_DIR, PHYSICS_INDICES, PHYSICS_NAMES,
                   LR_THRESHOLD, XGB_THRESHOLD, AVG_THRESHOLD,
                   REAL_COLOR, SPOOF_COLOR, BG_COLOR, DARK_COLOR)
from ching import build_advanced_feature_vector
import scipy.signal
import pywt

SAMPLES_DIR = os.path.join(AUDIO_DIR, "Samples")
VC_DIR      = os.path.join(AUDIO_DIR, "voice-conversion")

SET_MAP = {
    "A": (os.path.join(SAMPLES_DIR, "source-vctk1k"),              "REAL",  "trained on"),
    "B": (os.path.join(SAMPLES_DIR, "target-vctk1k"),              "REAL",  "NOT trained on"),
    "C": (os.path.join(SAMPLES_DIR, "converted-vctk1k"),           "SPOOF", "trained on"),
    "D": (os.path.join(VC_DIR,      "source-vctk500"),             "REAL",  "NOT trained on"),
    "E": (os.path.join(VC_DIR,      "target-vctk500"),             "REAL",  "NOT trained on"),
    "F": (os.path.join(VC_DIR,      "converted-vctk500"),          "SPOOF", "NOT trained on"),
}


def compute_dwt(y, wavelet="db4", levels=4):
    coeffs  = pywt.wavedec(y, wavelet=wavelet, level=levels)
    approx  = coeffs[0]
    details = coeffs[1:]
    approx_energy  = float(np.mean(approx ** 2))
    detail_energies = [float(np.mean(d ** 2)) for d in details]
    transient_score = float(np.std(details[0]) / (np.mean(np.abs(details[0])) + 1e-8))
    return approx_energy, detail_energies, transient_score


def make_single_clip_report(pdf, filepath, set_letter, true_label, trained,
                             clf_lr, scaler_lr, lr_relative, clf_xgb, scaler_xgb):
    filename = os.path.basename(filepath)
    y, sr    = load_audio(filepath)

    # STFT
    freqs, times, mag, phase_var, centroid = compute_stft(y, sr)

    # DWT
    approx_energy, detail_energies, transient_score = compute_dwt(y)

    # Feature vector
    fv = build_advanced_feature_vector(filepath)
    phys = fv[PHYSICS_INDICES]

    # Predictions
    prob_lr, prob_xgb = predict_both(fv, clf_lr, scaler_lr, lr_relative,
                                      clf_xgb, scaler_xgb)
    prob_avg = (prob_lr + prob_xgb) / 2

    v_lr  = "SPOOF" if prob_lr  >= LR_THRESHOLD  else "REAL"
    v_xgb = "SPOOF" if prob_xgb >= XGB_THRESHOLD else "REAL"
    v_avg = "SPOOF" if prob_avg >= AVG_THRESHOLD  else "REAL"

    color = REAL_COLOR if true_label == "REAL" else SPOOF_COLOR

    # ── Page 1: Waveform, Spectrogram, Phase Var, Centroid ──
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor(BG_COLOR)
    fig.text(0.5, 0.977,
             f"Set {set_letter} — {filename}",
             ha="center", fontsize=18, fontweight="bold", color=DARK_COLOR)
    fig.text(0.5, 0.955,
             f"True label: {true_label}  |  Training status: {trained}  |  "
             f"Duration: {len(y)/sr:.2f}s  |  Sample rate: {sr} Hz",
             ha="center", fontsize=11, color="#555")
    fig.text(0.5, 0.935,
             f"LR: {prob_lr*100:.1f}% → {v_lr}   "
             f"XGB: {prob_xgb*100:.1f}% → {v_xgb}   "
             f"AVG: {prob_avg*100:.1f}% → {v_avg}",
             ha="center", fontsize=12, fontweight="bold",
             color="#1e8449" if v_avg == true_label else "#c0392b")

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.91, bottom=0.06, left=0.07, right=0.97)

    # Waveform
    ax = fig.add_subplot(gs[0, :2])
    t = np.linspace(0, len(y)/sr, len(y))
    ax.plot(t, y, color=color, lw=0.5, alpha=0.9)
    ax.set_title(f"Waveform — {filename}", fontweight="bold", color=color)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3); ax.set_facecolor("#f0f4f8")

    # Verdict box
    ax_v = fig.add_subplot(gs[0, 2])
    ax_v.axis("off")
    verdict_color = "#1e8449" if v_avg == true_label else "#c0392b"
    ax_v.text(0.5, 0.82, "VERDICT (AVG)", ha="center", fontsize=12,
              fontweight="bold", transform=ax_v.transAxes, color=DARK_COLOR)
    ax_v.text(0.5, 0.62, v_avg, ha="center", fontsize=36,
              fontweight="bold", color=verdict_color, transform=ax_v.transAxes)
    ax_v.text(0.5, 0.44, f"Avg prob: {prob_avg*100:.1f}%",
              ha="center", fontsize=11, transform=ax_v.transAxes)
    tick = "✓ CORRECT" if v_avg == true_label else "✗ WRONG"
    tick_color = "#1e8449" if v_avg == true_label else "#c0392b"
    ax_v.text(0.5, 0.28, tick, ha="center", fontsize=13,
              fontweight="bold", color=tick_color, transform=ax_v.transAxes)
    ax_v.text(0.5, 0.12,
              f"LR: {prob_lr*100:.1f}%  XGB: {prob_xgb*100:.1f}%",
              ha="center", fontsize=9, color="#777", transform=ax_v.transAxes)

    # Spectrogram
    ax2 = fig.add_subplot(gs[1, :2])
    mag_db = 20 * np.log10(mag + 1e-8)
    im = ax2.pcolormesh(times, freqs, mag_db, shading="auto", cmap="inferno",
                         vmin=mag_db.max()-60, vmax=mag_db.max())
    ax2.set_title("Spectrogram (STFT Magnitude)", fontweight="bold")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Freq (Hz)")
    ax2.set_ylim(0, 8000)
    plt.colorbar(im, ax=ax2, label="dB", pad=0.01)

    # DWT detail energies
    ax3 = fig.add_subplot(gs[1, 2])
    levels = [f"D{i+1}" for i in range(len(detail_energies))]
    bars = ax3.bar(levels, detail_energies, color="#8e44ad", alpha=0.85)
    ax3.set_title(f"DWT Detail Energy\n(transient score: {transient_score:.3f})",
                  fontweight="bold")
    ax3.set_ylabel("Energy")
    ax3.grid(True, alpha=0.3, axis="y"); ax3.set_facecolor("#f0f4f8")
    for bar, val in zip(bars, detail_energies):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                 f"{val:.5f}", ha="center", va="bottom", fontsize=7)

    # Phase variance
    ax4 = fig.add_subplot(gs[2, :2])
    ax4.plot(times, phase_var, color="#e67e22", lw=0.8, alpha=0.9)
    ax4.axhline(np.mean(phase_var), color="red", ls="--", lw=1.2,
                label=f"Mean: {np.mean(phase_var):.3f}")
    ax4.set_title("Phase Variance per Frame", fontweight="bold")
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Phase Variance")
    ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3)
    ax4.set_facecolor("#f0f4f8")

    # Spectral centroid
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.plot(times, centroid, color="#27ae60", lw=0.8, alpha=0.9)
    ax5.set_title(f"Spectral Centroid\n(mean: {np.mean(centroid):.0f} Hz)",
                  fontweight="bold")
    ax5.set_xlabel("Time (s)"); ax5.set_ylabel("Freq (Hz)")
    ax5.grid(True, alpha=0.3); ax5.set_facecolor("#f0f4f8")

    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)

    # ── Page 2: Physics features + full feature table ──
    fig2 = plt.figure(figsize=(18, 12))
    fig2.patch.set_facecolor(BG_COLOR)
    fig2.suptitle(f"Physics Feature Analysis — Set {set_letter}: {filename}",
                  fontsize=15, fontweight="bold", color=DARK_COLOR, y=0.97)

    gs2 = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.45, wspace=0.35,
                            top=0.90, bottom=0.06, left=0.07, right=0.97)

    # Physics bar chart
    ax6 = fig2.add_subplot(gs2[0, 0])
    ax6.bar(PHYSICS_NAMES, phys, color=color, alpha=0.85)
    ax6.set_title("Physics Feature Values", fontweight="bold")
    ax6.set_xticklabels(PHYSICS_NAMES, rotation=30, ha="right", fontsize=7)
    ax6.grid(True, alpha=0.3, axis="y"); ax6.set_facecolor("#f0f4f8")

    # Model probability gauge
    ax7 = fig2.add_subplot(gs2[0, 1])
    ax7.axis("off")
    models = ["LR", "XGB", "AVG"]
    probs  = [prob_lr, prob_xgb, prob_avg]
    verdicts = [v_lr, v_xgb, v_avg]
    thresholds = [LR_THRESHOLD, XGB_THRESHOLD, AVG_THRESHOLD]
    y_pos = [0.80, 0.52, 0.24]
    for model, prob, verdict, thresh, yp in zip(models, probs, verdicts,
                                                 thresholds, y_pos):
        vc = "#1e8449" if verdict == true_label else "#c0392b"
        ax7.text(0.05, yp+0.12, f"{model} (threshold={thresh})",
                 fontsize=11, fontweight="bold", transform=ax7.transAxes,
                 color=DARK_COLOR)
        ax7.text(0.05, yp, f"Spoof probability: {prob*100:.2f}%  →  {verdict}",
                 fontsize=10, transform=ax7.transAxes, color=vc)
        # Mini bar
        bar_width = prob
        # draw a simple text bar instead of matplotlib transform tricks
        ax7.text(0.05, yp-0.04,
                 f"{'█' * int(prob * 40)}{' ' * (40 - int(prob * 40))} {prob*100:.1f}%",
                 fontsize=7, fontfamily="monospace",
                 color=SPOOF_COLOR if prob >= thresh else REAL_COLOR,
                 transform=ax7.transAxes)
    ax7.set_title("Model Probabilities", fontweight="bold", pad=10)

    # Full physics feature table
    ax8 = fig2.add_subplot(gs2[1, :])
    ax8.axis("off")

    # All features including MFCC summary
    mfcc_means = fv[5:65]    # first 20 MFCC means
    mfcc_stds  = fv[65:125]  # first 20 MFCC stds

    table_data = []
    for i, name in enumerate(PHYSICS_NAMES):
        table_data.append([name, f"{phys[i]:.5f}", "Physics feature"])
    table_data.append(["Phase Var (mean)", f"{np.mean(phase_var):.5f}", "STFT feature"])
    table_data.append(["DWT Approx Energy", f"{approx_energy:.5f}", "DWT feature"])
    table_data.append(["DWT Transient Score", f"{transient_score:.5f}", "DWT feature"])
    table_data.append(["MFCC1 mean", f"{fv[5]:.5f}", "MFCC feature"])
    table_data.append(["MFCC2 mean", f"{fv[6]:.5f}", "MFCC feature"])
    table_data.append(["MFCC3 mean", f"{fv[7]:.5f}", "MFCC feature"])
    table_data.append([f"... (20 MFCCs total)", "...", "MFCC features"])

    tbl = ax8.table(cellText=table_data,
                    colLabels=["Feature", "Value", "Type"],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.1, 1.6)
    for j in range(3):
        tbl[0, j].set_facecolor(DARK_COLOR)
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for row_idx in range(1, len(table_data)+1):
        ftype = table_data[row_idx-1][2]
        bg = "#d6eaf8" if "Physics" in ftype else \
             "#fef9e7" if "STFT" in ftype else \
             "#f9ebea" if "DWT" in ftype else "#e8f8f5"
        for j in range(3):
            tbl[row_idx, j].set_facecolor(bg)
    ax8.set_title("Complete Feature Table", fontweight="bold", pad=8)

    pdf.savefig(fig2, facecolor=fig2.get_facecolor())
    plt.close(fig2)

    return prob_lr, prob_xgb, prob_avg, v_lr, v_xgb, v_avg


def main():
    if len(sys.argv) < 3:
        print("Usage: python single_clip.py <set_letter> <index>")
        print("Example: python single_clip.py A 0")
        print(f"Valid sets: {list(SET_MAP.keys())}")
        sys.exit(1)

    set_letter = sys.argv[1].upper()
    try:
        index = int(sys.argv[2])
    except ValueError:
        print(f"ERROR: index must be an integer, got '{sys.argv[2]}'")
        sys.exit(1)

    if set_letter not in SET_MAP:
        print(f"ERROR: Invalid set '{set_letter}'. Valid sets: {list(SET_MAP.keys())}")
        sys.exit(1)

    folder, true_label, trained = SET_MAP[set_letter]

    if not os.path.exists(folder):
        print(f"ERROR: Folder not found: {folder}")
        sys.exit(1)

    files = sorted(glob.glob(os.path.join(folder, "*.wav")))
    if not files:
        print(f"ERROR: No wav files found in {folder}")
        sys.exit(1)

    if index >= len(files):
        print(f"ERROR: Index {index} out of range. Set {set_letter} has {len(files)} files (0-{len(files)-1})")
        sys.exit(1)

    filepath = files[index]
    filename = os.path.basename(filepath)
    output_pdf = os.path.join(AUDIO_DIR, f"clip_Set{set_letter}_{index}.pdf")

    print("=" * 60)
    print(f"  single_clip.py — Set {set_letter}, Index {index}")
    print("  EE 123 Project — Ved, Ching, Erick")
    print("=" * 60)
    print(f"\n  File       : {filename}")
    print(f"  True label : {true_label}")
    print(f"  Training   : {trained}")
    print(f"  Path       : {filepath}")

    print("\nLoading models...")
    clf_lr, scaler_lr, lr_relative, clf_xgb, scaler_xgb = load_models()

    print("\nAnalyzing clip...")
    with PdfPages(output_pdf) as pdf:
        prob_lr, prob_xgb, prob_avg, v_lr, v_xgb, v_avg = make_single_clip_report(
            pdf, filepath, set_letter, true_label, trained,
            clf_lr, scaler_lr, lr_relative, clf_xgb, scaler_xgb)

        d = pdf.infodict()
        d["Title"]  = f"Single Clip Analysis — Set {set_letter} Index {index}"
        d["Author"] = "Ved, Ching, Erick — EE 123"

    print(f"\n  Results:")
    print(f"    LR  : {prob_lr*100:.2f}% → {v_lr}  {'✓' if v_lr == true_label else '✗'}")
    print(f"    XGB : {prob_xgb*100:.2f}% → {v_xgb}  {'✓' if v_xgb == true_label else '✗'}")
    print(f"    AVG : {prob_avg*100:.2f}% → {v_avg}  {'✓' if v_avg == true_label else '✗'}")
    print(f"\n✅ Report saved to: {output_pdf}")
    print("=" * 60)


if __name__ == "__main__":
    main()
