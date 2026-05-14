"""
report500.py - Voice Conversion 500-Pair Dataset Analysis
EE 123 Project - Ved, Ching, Erick

Evaluates trained models on the voice-conversion dataset.
  Real  : voice-conversion/source-vctk500/ (500 files)
  Spoof : voice-conversion/converted-vctk500/ (500 files)

Also compares feature distributions vs VCTK training data.

Output:
  report500.pdf

Usage:
    python report500.py
"""

import os
import glob
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from utils import (load_models, extract_folder, extract_feats_only,
                   make_cover, make_feature_discriminability_page,
                   make_feature_stats_table, make_ensemble_results_page,
                   make_waveform_spectrogram_page,
                   load_audio, compute_stft, predict_both,
                   sorted_feature_order,
                   AUDIO_DIR, PHYSICS_INDICES, PHYSICS_NAMES,
                   REAL_COLOR, SPOOF_COLOR, BG_COLOR, DARK_COLOR)
from ching import build_advanced_feature_vector

OUTPUT_PDF    = os.path.join(AUDIO_DIR, "report500.pdf")
VC_DIR        = os.path.join(AUDIO_DIR, "voice-conversion")
VC_REAL_DIR   = os.path.join(VC_DIR, "source-vctk500")
VC_FAKE_DIR   = os.path.join(VC_DIR, "converted-vctk500")
SAMPLES_DIR   = os.path.join(AUDIO_DIR, "Samples")
VCTK_REAL_DIR = os.path.join(SAMPLES_DIR, "source-vctk1k")
VCTK_FAKE_DIR = os.path.join(SAMPLES_DIR, "converted-vctk1k")


def make_cross_dataset_comparison(pdf, vctk_real, vctk_spoof,
                                   vc_real, vc_spoof,
                                   stats_vc, stats_vctk_ref):
    """Side-by-side bar chart + table comparing VCTK vs VC accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle("Cross-Dataset Comparison: VCTK Training vs Voice Conversion",
                 fontsize=15, fontweight="bold", color=DARK_COLOR, y=0.97)

    methods = ["LR", "XGB", "OR", "AND", "AVG"]
    vctk_ov = [stats_vctk_ref[k]["overall"] for k in ["lr","xgb","or","and","avg"]]
    vc_ov   = [stats_vc[k]["overall"]       for k in ["lr","xgb","or","and","avg"]]

    x, w = np.arange(len(methods)), 0.35
    ax1 = axes[0]
    ax1.bar(x-w/2, vctk_ov, w, label="VCTK (in-dist)",  color="#2980b9", alpha=0.85)
    ax1.bar(x+w/2, vc_ov,   w, label="VC 500 (out-dist)",color="#e67e22", alpha=0.85)
    ax1.set_title("Overall Accuracy: VCTK vs Voice Conversion",
                  fontweight="bold", fontsize=12)
    ax1.set_xticks(x); ax1.set_xticklabels(methods, fontsize=11)
    ax1.set_ylabel("Accuracy (%)"); ax1.set_ylim(0, 105)
    ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_facecolor("#f0f4f8")
    for xi, (v1, v2) in enumerate(zip(vctk_ov, vc_ov)):
        ax1.text(xi-w/2, v1+0.5, f"{v1:.1f}%", ha="center", va="bottom",
                 fontsize=8, fontweight="bold", color="#2980b9")
        ax1.text(xi+w/2, v2+0.5, f"{v2:.1f}%", ha="center", va="bottom",
                 fontsize=8, fontweight="bold", color="#e67e22")

    ax2 = axes[1]
    ax2.axis("off")
    keys = ["lr","xgb","or","and","avg"]
    table_data = [["Method","Dataset","Real Correct","Real Tot","Real%",
                   "Spoof Correct","Spoof Tot","Spoof%","Overall"]]
    for k, name in zip(keys, methods):
        sv = stats_vctk_ref[k]
        table_data.append([name, "VCTK 1k",
                           str(sv["real_correct"]),  str(sv["real_total"]),
                           f"{sv['real_acc']:.1f}%",
                           str(sv["spoof_correct"]), str(sv["spoof_total"]),
                           f"{sv['spoof_acc']:.1f}%",
                           f"{sv['overall']:.1f}%"])
    table_data.append(["─"*4]*9)
    for k, name in zip(keys, methods):
        sv = stats_vc[k]
        table_data.append([name, "VC 500",
                           str(sv["real_correct"]),  str(sv["real_total"]),
                           f"{sv['real_acc']:.1f}%",
                           str(sv["spoof_correct"]), str(sv["spoof_total"]),
                           f"{sv['spoof_acc']:.1f}%",
                           f"{sv['overall']:.1f}%"])

    tbl = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.0, 1.7)
    for j in range(9):
        tbl[0,j].set_facecolor(DARK_COLOR)
        tbl[0,j].set_text_props(color="white", fontweight="bold")
    for row in range(1, 6):
        for j in range(9):
            tbl[row,j].set_facecolor("#d6eaf8")
    for row in range(7, 12):
        for j in range(9):
            tbl[row,j].set_facecolor("#fdebd0")
    ax2.set_title("Full Results Comparison Table",
                  fontweight="bold", fontsize=12, pad=15)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)


def make_feature_scale_page(pdf, vctk_real, vctk_spoof, vc_real, vc_spoof):
    """Compare feature scales between VCTK and VC datasets."""
    order   = sorted_feature_order(np.array(vctk_real), np.array(vctk_spoof))
    idxs    = [e[0] for e in order]
    names_s = [e[1] for e in order]

    vr = np.array(vctk_real);  vs = np.array(vctk_spoof)
    cr = np.array(vc_real);    cs = np.array(vc_spoof)

    x, w = np.arange(len(names_s)), 0.35
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle("Feature Scale: VCTK Training vs Voice Conversion Dataset",
                 fontsize=15, fontweight="bold", color=DARK_COLOR, y=0.97)

    ax1 = axes[0]
    ax1.bar(x-w/2, [vr[:,i].mean() for i in idxs], w,
            label="VCTK Real",  color=REAL_COLOR,  alpha=0.8)
    ax1.bar(x+w/2, [vs[:,i].mean() for i in idxs], w,
            label="VCTK Spoof", color=SPOOF_COLOR, alpha=0.8)
    ax1.set_title("VCTK Training Data (1000 pairs)", fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(names_s, rotation=30, ha="right", fontsize=8)
    ax1.legend(); ax1.grid(True, alpha=0.3, axis="y"); ax1.set_facecolor("#f0f4f8")
    ax1.set_ylabel("Mean Feature Value")

    ax2 = axes[1]
    ax2.bar(x-w/2, [cr[:,i].mean() for i in idxs], w,
            label="VC Real",  color=REAL_COLOR,  alpha=0.8)
    ax2.bar(x+w/2, [cs[:,i].mean() for i in idxs], w,
            label="VC Spoof", color=SPOOF_COLOR, alpha=0.8)
    ax2.set_title("Voice Conversion Dataset (500 pairs)", fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(names_s, rotation=30, ha="right", fontsize=8)
    ax2.legend(); ax2.grid(True, alpha=0.3, axis="y"); ax2.set_facecolor("#f0f4f8")
    ax2.set_ylabel("Mean Feature Value")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    print("=" * 60)
    print("  report500.py — Voice Conversion 500-Pair Analysis")
    print("  EE 123 Project — Ved, Ching, Erick")
    print("=" * 60)

    for d in [VC_REAL_DIR, VC_FAKE_DIR]:
        if not os.path.exists(d):
            print(f"ERROR: Missing folder: {d}")
            exit(1)

    print("\nLoading models...")
    clf_lr, scaler_lr, lr_relative, clf_xgb, scaler_xgb = load_models()

    print("\n[1/5] Extracting features from voice-conversion dataset...")
    print("  Real (source-vctk500):")
    vc_real_results, vc_real_feats = extract_folder(
        VC_REAL_DIR, "REAL", clf_lr, scaler_lr, lr_relative,
        clf_xgb, scaler_xgb, "vc_real")

    print("  Spoof (converted-vctk500):")
    vc_spoof_results, vc_spoof_feats = extract_folder(
        VC_FAKE_DIR, "SPOOF", clf_lr, scaler_lr, lr_relative,
        clf_xgb, scaler_xgb, "vc_spoof")

    all_vc = vc_real_results + vc_spoof_results
    n_real  = len(vc_real_results)
    n_spoof = len(vc_spoof_results)

    print("\n[2/5] Extracting VCTK features for comparison...")
    print("  VCTK Real:")
    vctk_real_feats  = extract_feats_only(VCTK_REAL_DIR,  "vctk_real")
    print("  VCTK Spoof:")
    vctk_spoof_feats = extract_feats_only(VCTK_FAKE_DIR, "vctk_spoof")

    # Re-run VCTK 1000 inference for reference stats
    print("\n[3/5] Running VCTK 1000-pair reference evaluation...")
    from utils import evaluate_strategy, eval_lr_only, eval_xgb_only
    vctk_real_dir = os.path.join(AUDIO_DIR, "Samples", "source-vctk1k")
    vctk_conv_dir = os.path.join(AUDIO_DIR, "Samples", "converted-vctk1k")
    print("  VCTK Real:")
    vctk_real_res, _ = extract_folder(
        vctk_real_dir, "REAL", clf_lr, scaler_lr, lr_relative,
        clf_xgb, scaler_xgb, "vctk_src")
    print("  VCTK Spoof:")
    vctk_spoof_res, _ = extract_folder(
        vctk_conv_dir, "SPOOF", clf_lr, scaler_lr, lr_relative,
        clf_xgb, scaler_xgb, "vctk_conv")
    all_vctk = vctk_real_res + vctk_spoof_res
    stats_vctk = {
        "lr":  eval_lr_only(all_vctk),
        "xgb": eval_xgb_only(all_vctk),
        "or":  evaluate_strategy(all_vctk, "OR"),
        "and": evaluate_strategy(all_vctk, "AND"),
        "avg": evaluate_strategy(all_vctk, "AVG"),
    }

    # Sample pairs for visual page
    print("\n[4/5] Sampling pairs for visual analysis...")
    vc_real_files  = sorted(glob.glob(os.path.join(VC_REAL_DIR,  "*.wav")))
    vc_spoof_files = sorted(glob.glob(os.path.join(VC_FAKE_DIR, "*.wav")))
    sample_n = min(20, len(vc_real_files), len(vc_spoof_files))
    random.seed(42)
    sampled_real  = random.sample(vc_real_files,  sample_n)
    sampled_spoof = random.sample(vc_spoof_files, sample_n)

    pairs_data = []
    for i, (rp, sp) in enumerate(zip(sampled_real, sampled_spoof)):
        y_r, sr = load_audio(rp)
        y_s, _  = load_audio(sp)
        _, _, _, pv_r, _ = compute_stft(y_r, sr)
        _, _, _, pv_s, _ = compute_stft(y_s, sr)
        fv_r = build_advanced_feature_vector(rp)
        fv_s = build_advanced_feature_vector(sp)
        lr_r, xgb_r = predict_both(fv_r, clf_lr, scaler_lr, lr_relative, clf_xgb, scaler_xgb)
        lr_s, xgb_s = predict_both(fv_s, clf_lr, scaler_lr, lr_relative, clf_xgb, scaler_xgb)
        pairs_data.append({
            "pv_r": float(np.mean(pv_r)), "pv_s": float(np.mean(pv_s)),
            "phys_r": fv_r[PHYSICS_INDICES], "phys_s": fv_s[PHYSICS_INDICES],
            "prob_lr_r": lr_r, "prob_lr_s": lr_s,
            "prob_xgb_r": xgb_r, "prob_xgb_s": xgb_s,
        })
        if (i+1) % 5 == 0:
            print(f"    {i+1}/{sample_n} sampled")

    print("\n[5/5] Generating PDF...")
    with PdfPages(OUTPUT_PDF) as pdf:

        print("  → Cover")
        make_cover(pdf,
                   "Voice Conversion 500-Pair",
                   "Out-of-Distribution Evaluation Report",
                   [f"Real: {n_real} files (source-vctk500)",
                    f"Spoof: {n_spoof} files (converted-vctk500)",
                    "Models trained on VCTK 1000 vs 1000 (balanced)"])

        print("  → Feature discriminability")
        make_feature_discriminability_page(
            pdf, vc_real_feats, vc_spoof_feats,
            "Feature Discriminability — Voice Conversion 500 Pairs")

        print("  → Feature stats table")
        make_feature_stats_table(
            pdf, vc_real_feats, vc_spoof_feats,
            "Voice Conversion 500-Pair Aggregate Feature Statistics",
            n_real, n_spoof)

        print("  → Feature scale comparison")
        make_feature_scale_page(pdf, vctk_real_feats, vctk_spoof_feats,
                                 vc_real_feats, vc_spoof_feats)

        print("  → Signal analysis summary")
        make_waveform_spectrogram_page(
            pdf, pairs_data,
            f"Signal Analysis Summary — {sample_n} Sampled VC Pairs")

        print("  → Ensemble results")
        stats_vc = make_ensemble_results_page(
            pdf, all_vc, vc_real_results, vc_spoof_results,
            "Voice Conversion 500-Pair — Ensemble Evaluation Results",
            n_real, n_spoof)

        print("  → Cross-dataset comparison")
        make_cross_dataset_comparison(pdf, vctk_real_feats, vctk_spoof_feats,
                                       vc_real_feats, vc_spoof_feats,
                                       stats_vc, stats_vctk)

        d = pdf.infodict()
        d["Title"]  = "Voice Conversion 500-Pair Report"
        d["Author"] = "Ved, Ching, Erick — EE 123"

    print(f"\nDone!")
    print(f"\n✅ Report saved to: {OUTPUT_PDF}")
    print("\n  Voice Conversion Results:")
    for name, s in stats_vc.items():
        print(f"    {name.upper():<5}: Real {s['real_correct']}/{s['real_total']} "
              f"({s['real_acc']:.1f}%)  "
              f"Spoof {s['spoof_correct']}/{s['spoof_total']} "
              f"({s['spoof_acc']:.1f}%)  "
              f"Overall {s['correct']}/{s['total']} ({s['overall']:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()