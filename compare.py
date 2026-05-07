"""
compare.py - Side-by-Side Real vs Spoof Analysis
EE 123 Project - Ved, Ching, Erick

Compares 8 matched real/spoof pairs and saves a full PDF report.
Each pair gets its own page with:
  - Waveforms (real vs spoof)
  - Spectrograms (real vs spoof)
  - Phase variance over time
  - Spectral centroid trajectory
  - Physics feature bar chart comparison
  - Summary stats table

Usage:
    python compare.py

Output:
    spoof_analysis_report.pdf
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
import librosa
import soundfile as sf
import scipy.signal
import pywt
from scipy.stats import kurtosis
from ching import build_advanced_feature_vector, preprocess_audio

# ─────────────────────────────────────────────
AUDIO_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_PAIRS = 8
OUTPUT_PDF = os.path.join(AUDIO_DIR, "spoof_analysis_report.pdf")
SR = 16000
# ─────────────────────────────────────────────

PHYSICS_NAMES = [
    "Glottal Kurtosis", "Glottal Std", "Spectral Flux Std",
    "Modulation Energy", "Rolloff Std", "ZCR Std", "Aliasing Mean"
]

def load_audio(filepath):
    y, sr = preprocess_audio(filepath, target_sr=SR)
    return y, sr

def compute_stft(y, sr):
    freqs, times, Zxx = scipy.signal.stft(
        y, fs=sr, window='hamming', nperseg=400, noverlap=240, nfft=512
    )
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)
    power = magnitude ** 2
    centroid = np.sum(freqs[:, None] * power, axis=0) / (np.sum(power, axis=0) + 1e-8)
    phase_var = np.var(phase, axis=0)
    return freqs, times, magnitude, phase_var, centroid

def mfcc_similarity(y1, y2, sr):
    """
    Rough word/content similarity using MFCC cosine distance.
    Returns a 0-100 confidence score that both clips have similar content.
    """
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr, n_mfcc=13)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr, n_mfcc=13)
    # Match lengths
    min_len = min(mfcc1.shape[1], mfcc2.shape[1])
    mfcc1 = mfcc1[:, :min_len]
    mfcc2 = mfcc2[:, :min_len]
    # Cosine similarity per frame, averaged
    dot = np.sum(mfcc1 * mfcc2, axis=0)
    norm = (np.linalg.norm(mfcc1, axis=0) * np.linalg.norm(mfcc2, axis=0)) + 1e-8
    cos_sim = dot / norm
    return float(np.mean(cos_sim)) * 100

def make_cover_page(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    fig.patch.set_facecolor('#1a1a2e')

    ax.text(0.5, 0.75, "SPARC Spoof Detection", ha='center', va='center',
            fontsize=36, fontweight='bold', color='white', transform=ax.transAxes)
    ax.text(0.5, 0.63, "Real vs Voice-Converted Audio Analysis", ha='center', va='center',
            fontsize=18, color='#a0a0c0', transform=ax.transAxes)
    ax.text(0.5, 0.50, "EE 123 Project  —  Ved, Ching, Erick", ha='center', va='center',
            fontsize=14, color='#7070a0', transform=ax.transAxes)

    ax.text(0.5, 0.35, "8 Matched Pairs: r1-r8 (Real)  vs  try1-try8 (SPARC Voice Converted)",
            ha='center', va='center', fontsize=11, color='#909090', transform=ax.transAxes)

    ax.text(0.5, 0.20,
            "Each page shows one pair. Plots include waveform, spectrogram,\n"
            "phase variance, spectral centroid, and physics feature comparison.",
            ha='center', va='center', fontsize=10, color='#707070',
            transform=ax.transAxes, linespacing=1.8)

    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)

def make_pair_page(pdf, pair_idx, real_path, spoof_path):
    pair_num = pair_idx + 1
    print(f"  Processing pair {pair_num}/8: {os.path.basename(real_path)} vs {os.path.basename(spoof_path)}")

    # Load audio
    y_real, sr = load_audio(real_path)
    y_spoof, _ = load_audio(spoof_path)

    # Content similarity
    content_score = mfcc_similarity(y_real, y_spoof, sr)

    # STFT features
    freqs_r, times_r, mag_r, phase_var_r, cent_r = compute_stft(y_real, sr)
    freqs_s, times_s, mag_s, phase_var_s, cent_s = compute_stft(y_spoof, sr)

    # Physics features from ching.py
    fvec_real = build_advanced_feature_vector(real_path)
    fvec_spoof = build_advanced_feature_vector(spoof_path)
    physics_real = fvec_real[120:]   # 7 physics features
    physics_spoof = fvec_spoof[120:]

    # ── LAYOUT ──
    fig = plt.figure(figsize=(22, 17))
    fig.patch.set_facecolor('#f8f9fa')

    # Title bar
    fig.text(0.5, 0.97,
             f"Pair {pair_num}:  {os.path.basename(real_path)}  vs  {os.path.basename(spoof_path)}",
             ha='center', va='top', fontsize=16, fontweight='bold', color='#1a1a2e')
    fig.text(0.5, 0.945,
             f"Content Similarity Score: {content_score:.1f}%  "
             f"({'High — same content' if content_score > 60 else 'Low — may be different content'})",
             ha='center', va='top', fontsize=12,
             color='#2ecc71' if content_score > 60 else '#e74c3c')

    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.92, bottom=0.06, left=0.06, right=0.97)

    real_color = '#2980b9'
    spoof_color = '#e74c3c'

    # ── ROW 0: Waveforms ──
    ax_wr = fig.add_subplot(gs[0, :2])
    t_r = np.linspace(0, len(y_real)/sr, len(y_real))
    ax_wr.plot(t_r, y_real, color=real_color, linewidth=0.4, alpha=0.9)
    ax_wr.set_title(f'Waveform — REAL (r{pair_num}.wav)', fontweight='bold', color=real_color)
    ax_wr.set_xlabel('Time (s)'); ax_wr.set_ylabel('Amplitude')
    ax_wr.grid(True, alpha=0.3); ax_wr.set_facecolor('#f0f4f8')

    ax_ws = fig.add_subplot(gs[0, 2:])
    t_s = np.linspace(0, len(y_spoof)/sr, len(y_spoof))
    ax_ws.plot(t_s, y_spoof, color=spoof_color, linewidth=0.4, alpha=0.9)
    ax_ws.set_title(f'Waveform — SPOOF (try{pair_num}.wav)', fontweight='bold', color=spoof_color)
    ax_ws.set_xlabel('Time (s)'); ax_ws.set_ylabel('Amplitude')
    ax_ws.grid(True, alpha=0.3); ax_ws.set_facecolor('#f0f4f8')

    # ── ROW 1: Spectrograms ──
    def plot_spec(ax, times, freqs, mag, title, color):
        mag_db = 20 * np.log10(mag + 1e-8)
        im = ax.pcolormesh(times, freqs, mag_db, shading='auto', cmap='inferno',
                           vmin=mag_db.max()-60, vmax=mag_db.max())
        ax.set_title(title, fontweight='bold', color=color)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Frequency (Hz)')
        ax.set_ylim(0, 8000)
        plt.colorbar(im, ax=ax, label='dB', pad=0.01)

    ax_sr = fig.add_subplot(gs[1, :2])
    plot_spec(ax_sr, times_r, freqs_r, mag_r, f'Spectrogram — REAL', real_color)

    ax_ss = fig.add_subplot(gs[1, 2:])
    plot_spec(ax_ss, times_s, freqs_s, mag_s, f'Spectrogram — SPOOF', spoof_color)

    # ── ROW 2: Phase Variance + Spectral Centroid ──
    ax_pv = fig.add_subplot(gs[2, :2])
    ax_pv.plot(times_r, phase_var_r, color=real_color, linewidth=0.8, label=f'Real (mean={np.mean(phase_var_r):.2f})', alpha=0.9)
    ax_pv.plot(times_s, phase_var_s, color=spoof_color, linewidth=0.8, label=f'Spoof (mean={np.mean(phase_var_s):.2f})', alpha=0.9)
    ax_pv.axhline(np.mean(phase_var_r), color=real_color, linestyle='--', linewidth=0.8, alpha=0.5)
    ax_pv.axhline(np.mean(phase_var_s), color=spoof_color, linestyle='--', linewidth=0.8, alpha=0.5)
    ax_pv.set_title('Phase Variance over Time\n(Real should be higher/messier)', fontweight='bold')
    ax_pv.set_xlabel('Time (s)'); ax_pv.set_ylabel('Phase Variance')
    ax_pv.legend(fontsize=8); ax_pv.grid(True, alpha=0.3)
    ax_pv.set_facecolor('#f0f4f8')

    ax_sc = fig.add_subplot(gs[2, 2:])
    ax_sc.plot(times_r, cent_r, color=real_color, linewidth=0.8, label='Real', alpha=0.9)
    ax_sc.plot(times_s, cent_s, color=spoof_color, linewidth=0.8, label='Spoof', alpha=0.9)
    ax_sc.set_title('Spectral Centroid over Time\n(SPARC spoofs tend to be smoother)', fontweight='bold')
    ax_sc.set_xlabel('Time (s)'); ax_sc.set_ylabel('Frequency (Hz)')
    ax_sc.legend(fontsize=8); ax_sc.grid(True, alpha=0.3)
    ax_sc.set_facecolor('#f0f4f8')

    # ── ROW 3: Physics Feature Comparison + Summary Table ──
    ax_bar = fig.add_subplot(gs[3, :2])
    x = np.arange(len(PHYSICS_NAMES))
    width = 0.35
    bars_r = ax_bar.bar(x - width/2, physics_real, width, label='Real', color=real_color, alpha=0.8)
    bars_s = ax_bar.bar(x + width/2, physics_spoof, width, label='Spoof', color=spoof_color, alpha=0.8)
    ax_bar.set_title('Physics Features: Real vs Spoof', fontweight='bold')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(PHYSICS_NAMES, rotation=25, ha='right', fontsize=7)
    ax_bar.legend(fontsize=8)
    ax_bar.grid(True, alpha=0.3, axis='y')
    ax_bar.set_facecolor('#f0f4f8')

    # ── Summary Table ──
    ax_tbl = fig.add_subplot(gs[3, 2:])
    ax_tbl.axis('off')

    table_data = []
    for i, name in enumerate(PHYSICS_NAMES):
        r_val = physics_real[i] if i < len(physics_real) else 0
        s_val = physics_spoof[i] if i < len(physics_spoof) else 0
        diff_pct = abs(r_val - s_val) / (abs(r_val) + 1e-8) * 100
        table_data.append([name, f"{r_val:.4f}", f"{s_val:.4f}", f"{diff_pct:.1f}%"])

    # Add phase variance summary
    table_data.append(["Phase Var (mean)",
                        f"{np.mean(phase_var_r):.4f}",
                        f"{np.mean(phase_var_s):.4f}",
                        f"{abs(np.mean(phase_var_r)-np.mean(phase_var_s))/(np.mean(phase_var_r)+1e-8)*100:.1f}%"])

    col_labels = ["Feature", "Real", "Spoof", "Diff%"]
    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)

    # Color header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#1a1a2e')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    ax_tbl.set_title('Feature Comparison Table', fontweight='bold', pad=10)

    # Legend patches
    real_patch = mpatches.Patch(color=real_color, label='Real Audio')
    spoof_patch = mpatches.Patch(color=spoof_color, label='SPARC Voice Converted (Spoof)')
    fig.legend(handles=[real_patch, spoof_patch], loc='lower center',
               ncol=2, fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))

    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)


def make_summary_page(pdf, all_phase_real, all_phase_spoof, all_physics_real, all_physics_spoof):
    """Final summary page showing aggregate differences across all 8 pairs."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor('#f8f9fa')
    fig.suptitle("Summary: Aggregate Differences Across All 8 Pairs",
                 fontsize=18, fontweight='bold', color='#1a1a2e', y=0.98)

    real_color = '#2980b9'
    spoof_color = '#e74c3c'

    # ── 1. Phase variance per pair ──
    ax1 = axes[0, 0]
    pairs = [f"Pair {i+1}" for i in range(NUM_PAIRS)]
    ax1.plot(pairs, all_phase_real, 'o-', color=real_color, label='Real', linewidth=2, markersize=6)
    ax1.plot(pairs, all_phase_spoof, 's-', color=spoof_color, label='Spoof', linewidth=2, markersize=6)
    ax1.set_title('Mean Phase Variance per Pair', fontweight='bold')
    ax1.set_ylabel('Phase Variance')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=30)
    ax1.set_facecolor('#f0f4f8')

    # ── 2. Average physics features across all pairs ──
    ax2 = axes[0, 1]
    mean_real = np.mean(all_physics_real, axis=0)
    mean_spoof = np.mean(all_physics_spoof, axis=0)
    x = np.arange(len(PHYSICS_NAMES))
    width = 0.35
    ax2.bar(x - width/2, mean_real, width, label='Real (avg)', color=real_color, alpha=0.8)
    ax2.bar(x + width/2, mean_spoof, width, label='Spoof (avg)', color=spoof_color, alpha=0.8)
    ax2.set_title('Average Physics Features (All 8 Pairs)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(PHYSICS_NAMES, rotation=30, ha='right', fontsize=7)
    ax2.legend(); ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_facecolor('#f0f4f8')

    # ── 3. Phase variance box plot ──
    ax3 = axes[1, 0]
    ax3.boxplot([all_phase_real, all_phase_spoof],
                labels=['Real', 'Spoof'],
                patch_artist=True,
                boxprops=dict(facecolor='#d6eaf8'),
                medianprops=dict(color='#1a1a2e', linewidth=2))
    ax3.set_title('Phase Variance Distribution\n(Real should be higher)', fontweight='bold')
    ax3.set_ylabel('Mean Phase Variance')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_facecolor('#f0f4f8')

    # ── 4. % difference per feature ──
    ax4 = axes[1, 1]
    mean_real_safe = np.where(np.abs(mean_real) < 1e-8, 1e-8, mean_real)
    pct_diff = np.abs(mean_real - mean_spoof) / np.abs(mean_real_safe) * 100
    colors = ['#2ecc71' if p > 20 else '#e67e22' if p > 10 else '#e74c3c' for p in pct_diff]
    bars = ax4.bar(PHYSICS_NAMES, pct_diff, color=colors, alpha=0.85)
    ax4.set_title('Avg % Difference per Feature\n(Green = strong discriminator)', fontweight='bold')
    ax4.set_ylabel('% Difference')
    ax4.set_xticklabels(PHYSICS_NAMES, rotation=30, ha='right', fontsize=7)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_facecolor('#f0f4f8')
    for bar, val in zip(bars, pct_diff):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.0f}%', ha='center', va='bottom', fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    print("=" * 55)
    print("   compare.py — SPARC Pair Analysis Report")
    print("   EE 123 Project — Ved, Ching, Erick")
    print("=" * 55)
    print(f"\nGenerating PDF: {OUTPUT_PDF}\n")

    all_phase_real = []
    all_phase_spoof = []
    all_physics_real = []
    all_physics_spoof = []

    with PdfPages(OUTPUT_PDF) as pdf:

        # Cover page
        make_cover_page(pdf)

        # One page per pair
        for i in range(NUM_PAIRS):
            real_path = os.path.join(AUDIO_DIR, f"r{i+1}.wav")
            spoof_path = os.path.join(AUDIO_DIR, f"try{i+1}.wav")

            if not os.path.exists(real_path):
                print(f"  ⚠️  Missing: {real_path} — skipping pair {i+1}")
                continue
            if not os.path.exists(spoof_path):
                print(f"  ⚠️  Missing: {spoof_path} — skipping pair {i+1}")
                continue

            # Collect summary data
            y_real, sr = load_audio(real_path)
            y_spoof, _ = load_audio(spoof_path)
            _, _, _, pv_r, _ = compute_stft(y_real, sr)
            _, _, _, pv_s, _ = compute_stft(y_spoof, sr)
            all_phase_real.append(float(np.mean(pv_r)))
            all_phase_spoof.append(float(np.mean(pv_s)))

            fv_r = build_advanced_feature_vector(real_path)
            fv_s = build_advanced_feature_vector(spoof_path)
            all_physics_real.append(fv_r[120:])
            all_physics_spoof.append(fv_s[120:])

            make_pair_page(pdf, i, real_path, spoof_path)

        # Summary page
        print("\n  Generating summary page...")
        make_summary_page(pdf, all_phase_real, all_phase_spoof,
                          all_physics_real, all_physics_spoof)

        # PDF metadata
        d = pdf.infodict()
        d['Title'] = 'SPARC Spoof Detection Analysis'
        d['Author'] = 'Ved, Ching, Erick — EE 123'
        d['Subject'] = 'Real vs Voice Converted Audio Comparison'

    print(f"\n✅ Done! Report saved to:\n   {OUTPUT_PDF}")
    print(f"   Open it to view all 8 pair comparisons + summary.")


if __name__ == "__main__":
    main()