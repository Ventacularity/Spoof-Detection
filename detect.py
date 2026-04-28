"""
detect.py - AI Audio Spoof Detector
EE 123 Project - Ved, Ching, Erick

Usage:
    python detect.py <audio_file.wav>

Example:
    python detect.py sample_audio/sample1.wav

The script will:
1. Load and preprocess the audio
2. Run STFT analysis (phase variance, spectral centroid)
3. Run DWT analysis (multi-resolution features)
4. Build a feature vector and compute a spoof score
5. Display spectrograms and feature plots
6. Print a final verdict: REAL or SPOOF
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import soundfile as sf
import pywt
from scipy.signal import stft as scipy_stft

# ─────────────────────────────────────────────
# THRESHOLD: tune this once you have spoof samples.
# Lower = more sensitive to spoofs.
# Range is roughly 0.0 (definitely spoof) to 1.0 (definitely real)
SPOOF_THRESHOLD = 0.5
# ─────────────────────────────────────────────


def load_audio(filepath):
    """Load a wav file and return signal + sample rate."""
    print(f"\n[1/5] Loading audio: {filepath}")
    wav, sr = sf.read(filepath)

    # Convert stereo to mono if needed
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    duration = len(wav) / sr
    print(f"      Sample rate : {sr} Hz")
    print(f"      Duration    : {duration:.2f} seconds")
    print(f"      Samples     : {len(wav)}")
    return wav, sr


def compute_stft(wav, sr, n_fft=512, hop_length=160, win_length=400):
    """
    Compute Short-Time Fourier Transform.
    Window size ~25ms, hop ~10ms at 16kHz — standard speech settings.
    Returns frequencies, times, complex STFT matrix.
    """
    print("\n[2/5] Computing STFT (time-frequency analysis)...")

    freqs, times, Zxx = scipy_stft(
        wav,
        fs=sr,
        window='hamming',
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft
    )

    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)

    # ── Spectral Centroid ──
    # Weighted average frequency per frame — where the "center of mass" of
    # energy sits. SPARC-converted voices tend to have smoother centroid
    # trajectories than real voices.
    power = magnitude ** 2
    centroid = np.sum(freqs[:, None] * power, axis=0) / (np.sum(power, axis=0) + 1e-8)
    centroid_mean = float(np.mean(centroid))
    centroid_std = float(np.std(centroid))

    # ── Phase Variance ──
    # Real voices have naturally messy, high phase variance across frames.
    # SPARC's neural vocoder produces unnaturally smooth/stable phase —
    # a telltale artifact of synthesis.
    phase_var_per_frame = np.var(phase, axis=0)  # variance across freq bins per frame
    phase_variance = float(np.mean(phase_var_per_frame))

    # ── High-Frequency Energy Ratio ──
    # Real voices have irregular breathiness above ~4kHz.
    # Neural vocoders tend to be too clean in that range.
    freq_4k_idx = np.searchsorted(freqs, 4000)
    low_energy = np.mean(magnitude[:freq_4k_idx, :])
    high_energy = np.mean(magnitude[freq_4k_idx:, :])
    hf_ratio = float(high_energy / (low_energy + 1e-8))

    print(f"      Spectral centroid : {centroid_mean:.1f} Hz (std: {centroid_std:.1f})")
    print(f"      Phase variance    : {phase_variance:.4f}")
    print(f"      HF energy ratio   : {hf_ratio:.4f}")

    return {
        'freqs': freqs,
        'times': times,
        'magnitude': magnitude,
        'phase': phase,
        'centroid': centroid,
        'centroid_mean': centroid_mean,
        'centroid_std': centroid_std,
        'phase_variance': phase_variance,
        'hf_ratio': hf_ratio
    }


def compute_dwt(wav, wavelet='db4', levels=4):
    """
    Compute Discrete Wavelet Transform for multi-resolution analysis.
    Uses Daubechies-4 wavelet — good match for speech signal structure.

    Low-frequency approximation coefficients reflect slow, stable vocal
    tract properties — reliable identity features.
    High-frequency detail coefficients capture fast transients (consonants,
    stop bursts) that vocoders often smear.
    """
    print("\n[3/5] Computing DWT (multi-resolution analysis)...")

    coeffs = pywt.wavedec(wav, wavelet=wavelet, level=levels)
    approx = coeffs[0]       # lowest frequency approximation
    details = coeffs[1:]     # detail coefficients at each level

    # ── Approximation Energy ──
    # Captures slow vocal tract resonance structure
    approx_energy = float(np.mean(approx ** 2))

    # ── Detail Energy per Level ──
    # Higher levels = finer time detail. Vocoders smooth over transients,
    # so detail energy at fine scales tends to be lower in spoofed audio.
    detail_energies = [float(np.mean(d ** 2)) for d in details]

    # ── Transient Irregularity ──
    # Real speech has bursty, irregular energy in fine detail coefficients.
    # Measure how "spiky" the finest detail level is.
    finest_detail = details[0]
    transient_score = float(np.std(finest_detail) / (np.mean(np.abs(finest_detail)) + 1e-8))

    print(f"      Approx energy     : {approx_energy:.4f}")
    print(f"      Detail energies   : {[f'{e:.4f}' for e in detail_energies]}")
    print(f"      Transient score   : {transient_score:.4f}")

    return {
        'approx': approx,
        'details': details,
        'approx_energy': approx_energy,
        'detail_energies': detail_energies,
        'transient_score': transient_score
    }


def build_feature_vector(stft_feats, dwt_feats):
    """
    Concatenate all extracted features into a single vector.
    These are the dimensions your classifier/threshold operates on.
    """
    print("\n[4/5] Building feature vector...")

    feature_vector = np.array([
        stft_feats['centroid_mean'],
        stft_feats['centroid_std'],
        stft_feats['phase_variance'],
        stft_feats['hf_ratio'],
        dwt_feats['approx_energy'],
        dwt_feats['transient_score'],
        *dwt_feats['detail_energies']
    ])

    print(f"      Feature vector ({len(feature_vector)} dims): {np.round(feature_vector, 4)}")
    return feature_vector


def compute_spoof_score(stft_feats, dwt_feats):
    """
    Combine features into a single spoof score between 0 and 1.
    0 = likely spoof, 1 = likely real.

    Each sub-score is normalized so higher = more "real-like".
    Weights are tunable — adjust once you have labeled spoof samples.

    NOTE: These weights and normalizations are initial estimates.
    Once your teammates generate SPARC spoof samples, compare real vs spoof
    feature values and retune these ranges accordingly.
    """

    # Phase variance: real voices ~2.0-3.5, spoofed ~0.5-1.5 (estimated)
    phase_score = np.clip(stft_feats['phase_variance'] / 2.5, 0, 1)

    # Transient irregularity: real ~3.0+, spoofed ~1.0-2.0 (estimated)
    transient_score = np.clip(dwt_feats['transient_score'] / 3.0, 0, 1)

    # HF ratio: real voices have more irregular high-freq content
    hf_score = np.clip(stft_feats['hf_ratio'] / 0.3, 0, 1)

    # Weighted combination — phase variance is our strongest signal
    weights = {'phase': 0.5, 'transient': 0.3, 'hf': 0.2}
    final_score = (
        weights['phase'] * phase_score +
        weights['transient'] * transient_score +
        weights['hf'] * hf_score
    )

    return float(final_score), {
        'phase_score': phase_score,
        'transient_score': transient_score,
        'hf_score': hf_score
    }


def plot_results(wav, sr, stft_feats, dwt_feats, spoof_score, sub_scores, filepath):
    """
    Generate a full diagnostic plot with:
    - Waveform
    - Spectrogram (magnitude)
    - Phase variance over time
    - Spectral centroid trajectory
    - DWT detail energies by level
    - Final spoof score gauge
    """
    print("\n[5/5] Generating plots...")

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Spoof Detection Analysis\n{filepath}", fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    time_axis = np.linspace(0, len(wav) / sr, len(wav))

    # ── 1. Waveform ──
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(time_axis, wav, color='steelblue', linewidth=0.5)
    ax1.set_title('Waveform')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)

    # ── 2. Spectrogram ──
    ax2 = fig.add_subplot(gs[1, :2])
    mag_db = 20 * np.log10(stft_feats['magnitude'] + 1e-8)
    im = ax2.pcolormesh(
        stft_feats['times'], stft_feats['freqs'], mag_db,
        shading='auto', cmap='inferno',
        vmin=mag_db.max() - 60, vmax=mag_db.max()
    )
    ax2.set_title('Spectrogram (STFT Magnitude)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_ylim(0, 8000)
    plt.colorbar(im, ax=ax2, label='dB')

    # ── 3. Phase Variance over Time ──
    ax3 = fig.add_subplot(gs[2, :2])
    phase_var_per_frame = np.var(stft_feats['phase'], axis=0)
    ax3.plot(stft_feats['times'], phase_var_per_frame, color='darkorange', linewidth=0.8)
    ax3.axhline(y=stft_feats['phase_variance'], color='red', linestyle='--',
                label=f"Mean: {stft_feats['phase_variance']:.3f}")
    ax3.set_title('Phase Variance per Frame (key spoof indicator)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Phase Variance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ── 4. Spectral Centroid ──
    ax4 = fig.add_subplot(gs[0, 2])
    ax4.plot(stft_feats['times'], stft_feats['centroid'], color='mediumseagreen', linewidth=0.8)
    ax4.set_title('Spectral Centroid')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Frequency (Hz)')
    ax4.grid(True, alpha=0.3)

    # ── 5. DWT Detail Energies by Level ──
    ax5 = fig.add_subplot(gs[1, 2])
    levels = [f'D{i+1}' for i in range(len(dwt_feats['detail_energies']))]
    bars = ax5.bar(levels, dwt_feats['detail_energies'], color='mediumpurple')
    ax5.set_title('DWT Detail Energy by Level\n(D1=finest, D4=coarsest)')
    ax5.set_ylabel('Energy')
    ax5.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, dwt_feats['detail_energies']):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    # ── 6. Spoof Score Gauge ──
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')

    verdict = "REAL" if spoof_score >= SPOOF_THRESHOLD else "SPOOF"
    verdict_color = "green" if verdict == "REAL" else "red"
    confidence = abs(spoof_score - SPOOF_THRESHOLD) / SPOOF_THRESHOLD * 100

    ax6.text(0.5, 0.85, "VERDICT", ha='center', va='center',
             fontsize=13, fontweight='bold', transform=ax6.transAxes)
    ax6.text(0.5, 0.65, verdict, ha='center', va='center',
             fontsize=32, fontweight='bold', color=verdict_color,
             transform=ax6.transAxes)
    ax6.text(0.5, 0.45, f"Score: {spoof_score:.3f} / 1.000",
             ha='center', va='center', fontsize=11, transform=ax6.transAxes)
    ax6.text(0.5, 0.30, f"Threshold: {SPOOF_THRESHOLD}",
             ha='center', va='center', fontsize=9, color='gray', transform=ax6.transAxes)
    ax6.text(0.5, 0.15, f"Sub-scores:", ha='center', va='center',
             fontsize=9, transform=ax6.transAxes)
    ax6.text(0.5, 0.05,
             f"Phase: {sub_scores['phase_score']:.2f}  "
             f"Transient: {sub_scores['transient_score']:.2f}  "
             f"HF: {sub_scores['hf_score']:.2f}",
             ha='center', va='center', fontsize=8, color='gray', transform=ax6.transAxes)

    plt.savefig('detection_report.png', dpi=150, bbox_inches='tight')
    print("      Saved plot to: detection_report.png")
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python detect.py <audio_file.wav>")
        sys.exit(1)

    filepath = sys.argv[1]

    print("=" * 55)
    print("   EE 123 Spoof Detector — Ved, Ching, Erick")
    print("=" * 55)

    # Run pipeline
    wav, sr = load_audio(filepath)
    stft_feats = compute_stft(wav, sr)
    dwt_feats = compute_dwt(wav)
    feature_vector = build_feature_vector(stft_feats, dwt_feats)
    spoof_score, sub_scores = compute_spoof_score(stft_feats, dwt_feats)

    # Final verdict
    verdict = "REAL" if spoof_score >= SPOOF_THRESHOLD else "SPOOF"
    print("\n" + "=" * 55)
    print(f"   SPOOF SCORE : {spoof_score:.4f} (threshold: {SPOOF_THRESHOLD})")
    print(f"   VERDICT     : {verdict}")
    print("=" * 55)

    # Plot everything
    plot_results(wav, sr, stft_feats, dwt_feats, spoof_score, sub_scores, filepath)


if __name__ == "__main__":
    main()