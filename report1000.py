"""
report1000.py - VCTK 1000-Pair Dataset Analysis
EE 123 Project - Ved, Ching, Erick

Pages (identical structure to report500.py):
  1. Cover
  2. Feature discriminability (Cohen's d)
  3. Feature stats table (raw counts)
  4. Averaged signal analysis (spectrogram, phase var, centroid, spectrum)
  5. Signal summary (phase var per pair, physics, % diff, LR/XGB prob)
  6. Ensemble results + raw counts

Usage:
    python report1000.py
Output:
    report1000.pdf
"""

import os
import glob
import random
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from utils import (load_models, extract_folder, make_cover,
                   make_feature_discriminability_page,
                   make_feature_stats_table,
                   make_ensemble_results_page,
                   make_waveform_spectrogram_page,
                   make_averaged_signal_page,
                   load_audio, compute_stft, predict_both,
                   AUDIO_DIR, PHYSICS_INDICES)
from ching import build_advanced_feature_vector

OUTPUT_PDF    = os.path.join(AUDIO_DIR, "report1000.pdf")
SAMPLES_DIR   = os.path.join(AUDIO_DIR, "Samples")
SOURCE_DIR    = os.path.join(SAMPLES_DIR, "source-vctk1k")
CONVERTED_DIR = os.path.join(SAMPLES_DIR, "converted-vctk1k")


def main():
    print("=" * 60)
    print("  report1000.py — VCTK 1000-Pair Analysis")
    print("  EE 123 Project — Ved, Ching, Erick")
    print("=" * 60)

    print("\nLoading models...")
    clf_lr, scaler_lr, lr_relative, clf_xgb, scaler_xgb = load_models()

    print("\n[1/5] Extracting features...")
    print("  Real (source-vctk1k):")
    real_results, real_feats = extract_folder(
        SOURCE_DIR, "REAL", clf_lr, scaler_lr, lr_relative,
        clf_xgb, scaler_xgb, "source")
    print("  Spoof (converted-vctk1k):")
    spoof_results, spoof_feats = extract_folder(
        CONVERTED_DIR, "SPOOF", clf_lr, scaler_lr, lr_relative,
        clf_xgb, scaler_xgb, "converted")

    all_results = real_results + spoof_results
    n_real  = len(real_results)
    n_spoof = len(spoof_results)
    print(f"\n  Total: {n_real} real + {n_spoof} spoof = {n_real+n_spoof} files")

    print("\n[2/5] Sampling pairs for signal analysis...")
    real_files  = sorted(glob.glob(os.path.join(SOURCE_DIR,    "*.wav")))
    spoof_files = sorted(glob.glob(os.path.join(CONVERTED_DIR, "*.wav")))
    sample_n = min(20, len(real_files), len(spoof_files))
    random.seed(42)
    sampled_real  = random.sample(real_files,  sample_n)
    sampled_spoof = random.sample(spoof_files, sample_n)

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
            print(f"    {i+1}/{sample_n} pairs sampled")

    print("\n[3/5] Generating PDF...")
    with PdfPages(OUTPUT_PDF) as pdf:

        print("  → Cover")
        make_cover(pdf,
                   "VCTK 1000-Pair Analysis",
                   "In-Distribution Evaluation Report",
                   [f"Real : {n_real} files (source-vctk1k)",
                    f"Spoof: {n_spoof} files (converted-vctk1k)",
                    "Models trained on source vs converted (1000 vs 1000, balanced)"])

        print("  → Feature discriminability")
        make_feature_discriminability_page(
            pdf, real_feats, spoof_feats,
            "Feature Discriminability — VCTK 1000 Pairs")

        print("  → Feature stats table")
        make_feature_stats_table(
            pdf, real_feats, spoof_feats,
            "VCTK 1000-Pair Aggregate Feature Statistics",
            n_real, n_spoof)

        print("  → Averaged signal analysis")
        make_averaged_signal_page(
            pdf, real_files, spoof_files,
            "Averaged Signal Analysis — VCTK 1000 Pairs (50 files sampled per class)")

        print("  → Signal summary (20 sampled pairs)")
        make_waveform_spectrogram_page(
            pdf, pairs_data,
            f"Signal Analysis Summary — {sample_n} Sampled Pairs")

        print("  → Ensemble results")
        stats = make_ensemble_results_page(
            pdf, all_results, real_results, spoof_results,
            "VCTK 1000-Pair — Ensemble Evaluation Results",
            n_real, n_spoof)

        d = pdf.infodict()
        d["Title"]  = "VCTK 1000-Pair Analysis Report"
        d["Author"] = "Ved, Ching, Erick — EE 123"

    print(f"\n[4/5] Done!")
    print(f"\n✅ Report saved to: {OUTPUT_PDF}")
    print("\n  Results summary:")
    for name, s in stats.items():
        print(f"    {name.upper():<5}: Real {s['real_correct']}/{s['real_total']} "
              f"({s['real_acc']:.1f}%)  "
              f"Spoof {s['spoof_correct']}/{s['spoof_total']} "
              f"({s['spoof_acc']:.1f}%)  "
              f"Overall {s['correct']}/{s['total']} ({s['overall']:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
