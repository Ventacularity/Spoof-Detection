"""
feature_analysis.py - Full Feature Analysis Across 1000 Pairs
EE 123 Project - Ved, Ching, Erick

For each of the 1000 source/converted pairs, extracts all 7 physics features
and compares real vs spoof values. Finds which features are most consistent
discriminators at scale.

Results saved to: feature_analysis.txt

Usage:
    python feature_analysis.py
"""

import os
import glob
import numpy as np
from ching import build_advanced_feature_vector

AUDIO_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(AUDIO_DIR, "Samples")
SOURCE_DIR    = os.path.join(SAMPLES_DIR, "source-vctk1k")
CONVERTED_DIR = os.path.join(SAMPLES_DIR, "converted-vctk1k")
OUTPUT_FILE = os.path.join(AUDIO_DIR, "feature_analysis.txt")

PHYSICS_START = 120
FEATURE_NAMES = [
    "Glottal Kurtosis",
    "Glottal Std",
    "Spectral Flux Std",
    "Modulation Energy",
    "Rolloff Std",
    "ZCR Std",
    "Aliasing Mean"
]


def get_number(filename):
    """Extract the leading number from a filename like vctk_real_42.wav -> 42"""
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return -1


def get_src_number(filename):
    """Extract source number from converted filename like vctk_real_14_to_vctk_real_1047.wav -> 14"""
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_to_")
    if len(parts) == 2:
        src_parts = parts[0].split("_")
        for part in reversed(src_parts):
            if part.isdigit():
                return int(part)
    return -1


def main():
    print("=" * 60)
    print("   Feature Analysis -- 1000 Pairs")
    print("   EE 123 Project -- Ved, Ching, Erick")
    print("=" * 60)

    # Build lookup: number -> filepath for source files
    source_files = sorted(glob.glob(os.path.join(SOURCE_DIR, "*.wav")))
    converted_files = sorted(glob.glob(os.path.join(CONVERTED_DIR, "*.wav")))

    source_map = {get_number(f): f for f in source_files}

    print(f"\n  Found {len(source_files)} source files")
    print(f"  Found {len(converted_files)} converted files")
    print(f"\n  Extracting features from all pairs...")
    print(f"  (Progress printed every 100 pairs)\n")

    # Storage for per-pair results
    real_features = []   # shape: (N, 7)
    spoof_features = []  # shape: (N, 7)
    pairs_used = 0
    pairs_skipped = 0

    for i, conv_path in enumerate(converted_files):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(converted_files)} pairs processed...")

        src_num = get_src_number(conv_path)
        if src_num not in source_map:
            pairs_skipped += 1
            continue

        src_path = source_map[src_num]

        try:
            fv_real = build_advanced_feature_vector(src_path)
            fv_spoof = build_advanced_feature_vector(conv_path)

            real_physics = fv_real[PHYSICS_START:]
            spoof_physics = fv_spoof[PHYSICS_START:]

            real_features.append(real_physics)
            spoof_features.append(spoof_physics)
            pairs_used += 1

        except Exception as e:
            pairs_skipped += 1

    real_features = np.array(real_features)
    spoof_features = np.array(spoof_features)

    print(f"\n  Pairs successfully processed: {pairs_used}")
    print(f"  Pairs skipped (errors):       {pairs_skipped}")
    print(f"\n  Writing analysis to: {OUTPUT_FILE}")

    # ── Analysis ──
    with open(OUTPUT_FILE, "w") as f:
        f.write("SPARC Spoof Detection — Full Feature Analysis\n")
        f.write("EE 123 Project — Ved, Ching, Erick\n")
        f.write(f"Pairs analyzed: {pairs_used}\n")
        f.write("=" * 70 + "\n\n")

        f.write("SECTION 1: AVERAGE VALUES (real vs spoof)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Feature':<22} {'Real Avg':>10} {'Spoof Avg':>10} {'Diff%':>8} {'Direction'}\n")
        f.write("-" * 70 + "\n")

        separability = []

        for i, name in enumerate(FEATURE_NAMES):
            real_avg = float(np.mean(real_features[:, i]))
            spoof_avg = float(np.mean(spoof_features[:, i]))
            diff_pct = (real_avg - spoof_avg) / (abs(real_avg) + 1e-8) * 100
            direction = "real > spoof" if real_avg > spoof_avg else "spoof > real"
            f.write(f"{name:<22} {real_avg:>10.4f} {spoof_avg:>10.4f} {diff_pct:>7.1f}% {direction}\n")
            separability.append((abs(diff_pct), name, diff_pct))

        f.write("\n\n")
        f.write("SECTION 2: CONSISTENCY (how often is real > spoof per pair)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Feature':<22} {'Real>Spoof':>12} {'Spoof>Real':>12} {'Consistency%':>14}\n")
        f.write("-" * 70 + "\n")

        consistency_scores = []

        for i, name in enumerate(FEATURE_NAMES):
            real_greater = int(np.sum(real_features[:, i] > spoof_features[:, i]))
            spoof_greater = pairs_used - real_greater
            consistency = max(real_greater, spoof_greater) / pairs_used * 100
            dominant = "real > spoof" if real_greater > spoof_greater else "spoof > real"
            f.write(f"{name:<22} {real_greater:>12} {spoof_greater:>12} {consistency:>13.1f}%  ({dominant})\n")
            consistency_scores.append((consistency, name, real_greater, spoof_greater))

        f.write("\n\n")
        f.write("SECTION 3: DISTRIBUTION OVERLAP\n")
        f.write("(How much do real and spoof distributions overlap? Lower = better separator)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Feature':<22} {'Real Mean':>10} {'Real Std':>10} {'Spoof Mean':>11} {'Spoof Std':>10} {'Overlap%':>10}\n")
        f.write("-" * 70 + "\n")

        overlap_scores = []

        for i, name in enumerate(FEATURE_NAMES):
            r_mean = float(np.mean(real_features[:, i]))
            r_std = float(np.std(real_features[:, i]))
            s_mean = float(np.mean(spoof_features[:, i]))
            s_std = float(np.std(spoof_features[:, i]))

            # Estimate overlap using Cohen's d
            pooled_std = np.sqrt((r_std**2 + s_std**2) / 2)
            cohens_d = abs(r_mean - s_mean) / (pooled_std + 1e-8)
            # Rough overlap estimate: lower d = more overlap
            overlap_pct = max(0, 100 - cohens_d * 30)

            f.write(f"{name:<22} {r_mean:>10.4f} {r_std:>10.4f} {s_mean:>11.4f} {s_std:>10.4f} {overlap_pct:>9.1f}%\n")
            overlap_scores.append((cohens_d, name))

        f.write("\n\n")
        f.write("SECTION 4: FEATURE RANKING (best to worst discriminators)\n")
        f.write("-" * 70 + "\n")
        f.write("Ranked by Cohen's d (higher = better separator)\n\n")

        overlap_scores.sort(reverse=True)
        for rank, (d, name) in enumerate(overlap_scores, 1):
            stars = "★" * min(5, max(1, int(d * 2)))
            f.write(f"  #{rank}  {name:<22}  Cohen's d = {d:.3f}  {stars}\n")

        f.write("\n\n")
        f.write("SECTION 5: RECOMMENDATION\n")
        f.write("-" * 70 + "\n")
        f.write("Features to USE (Cohen's d > 0.3, consistency > 60%):\n")
        consistency_map = {name: (c, rg, sg) for c, name, rg, sg in consistency_scores}
        for d, name in overlap_scores:
            c, rg, sg = consistency_map[name]
            if d > 0.3 and c > 60:
                direction = "real > spoof" if rg > sg else "spoof > real"
                f.write(f"  + {name:<22}  d={d:.3f}  consistency={c:.1f}%  ({direction})\n")

        f.write("\nFeatures to DROP (weak separator):\n")
        for d, name in overlap_scores:
            c, rg, sg = consistency_map[name]
            if d <= 0.3 or c <= 60:
                f.write(f"  - {name:<22}  d={d:.3f}  consistency={c:.1f}%\n")

    print("\n  Done! Open feature_analysis.txt to see results.")
    print("=" * 60)


if __name__ == "__main__":
    main()