# --- JUPYTER NOTEBOOK CELL: THE OPTIMIZED DEFENSE PIPELINE ---

import librosa
import numpy as np
import scipy.signal
from scipy.stats import kurtosis
import warnings
import pandas as pd
from scipy.spatial.distance import cosine

warnings.filterwarnings('ignore')

def preprocess_audio(audio_path, target_sr=16000):
    """Trims silence to prevent skewed statistical means."""
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    return y_trimmed, sr

def extract_glottal_residual_sharpness(y):
    """DEFENSE 1: LPC Residual Kurtosis"""
    y_pre = librosa.effects.preemphasis(y)
    lpc_coeffs = librosa.lpc(y_pre, order=16)
    residual = scipy.signal.lfilter(lpc_coeffs, [1.0], y_pre)
    
    res_kurtosis = float(kurtosis(residual))
    res_std = float(np.std(residual))
    return res_kurtosis, res_std

def extract_spectral_flux_dynamics(y):
    """DEFENSE 2: Spectral Flux Variance (The Consonant Punch)"""
    S = np.abs(librosa.stft(y))
    flux = np.maximum(0, np.diff(S, axis=1))
    total_flux_per_frame = np.sum(flux, axis=0)
    
    flux_std = float(np.std(total_flux_per_frame))
    flux_max = float(np.max(total_flux_per_frame)) 
    return flux_std, flux_max

def extract_modulation_energy(y, sr):
    """DEFENSE 3: Amplitude Modulation Spectrum (The Jaw Rhythm)"""
    analytic_signal = scipy.signal.hilbert(y)
    amplitude_envelope = np.abs(analytic_signal)
    
    env_fft = np.abs(np.fft.rfft(amplitude_envelope))
    freqs = np.fft.rfftfreq(len(amplitude_envelope), 1/sr)
    
    muscle_band = np.where((freqs >= 2) & (freqs <= 10))[0]
    
    if len(muscle_band) == 0:
        return 0.0
    return float(np.mean(env_fft[muscle_band]))

def extract_vocal_tract_formants(y, sr):
    """DEFENSE 4: MFCCs + Temporal Dynamics (Deltas)"""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    
    combined_mfccs = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
    mfcc_means = np.mean(combined_mfccs, axis=1)
    mfcc_stds = np.std(combined_mfccs, axis=1)
    return np.hstack([mfcc_means, mfcc_stds]) 

def extract_aliasing_signature(y, sr):
    """DEFENSE 5: High-Frequency Sibilance (RMS Aliasing Energy)"""
    stft_matrix = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    
    fold_back_indices = np.where((freqs >= 5000) & (freqs <= 8000))[0]
    
    # NEW: Calculate Root Mean Square to measure true acoustic power
    band_rms_per_frame = np.sqrt(np.mean(stft_matrix[fold_back_indices, :]**2, axis=0))
    
    return float(np.mean(band_rms_per_frame)), float(np.std(band_rms_per_frame))

def build_advanced_feature_vector(audio_path):
    """Compiles the highly targeted feature array."""
    y, sr = preprocess_audio(audio_path, target_sr=16000)
    
    # Extract Features
    glottal_kurtosis, glottal_std = extract_glottal_residual_sharpness(y)
    flux_std, flux_max = extract_spectral_flux_dynamics(y)
    mod_energy = extract_modulation_energy(y, sr)
    
    formant_features = extract_vocal_tract_formants(y, sr)
    alias_mean, alias_std = extract_aliasing_signature(y, sr)
    
    # Assemble Vector
    feature_vector = np.hstack((
        [glottal_kurtosis, glottal_std],           # Indices 0, 1
        [flux_std, flux_max],                      # Indices 2, 3
        [mod_energy],                              # Index 4
        formant_features,                          # Indices 5 to 124
        [alias_mean, alias_std]                    # Indices -2, -1
    ))
    
    return feature_vector

def compare_advanced_audio(file_path_1, file_path_2, label_1="Real", label_2="Suspect"):
    """Compares two files using separated Dual-Scoring with Weighted Physics."""
    print(f"Extracting [{label_1}]...")
    fp_1 = build_advanced_feature_vector(file_path_1)
    
    print(f"Extracting [{label_2}]...\n")
    fp_2 = build_advanced_feature_vector(file_path_2)

    # --- DUAL SCORING SYSTEM ---
    
    # 1. VOCAL TIMBRE MATCH (The MFCCs: Indices 5 to -2)
    fp_1_timbre = fp_1[5:-2]
    fp_2_timbre = fp_2[5:-2]
    timbre_similarity = (1 - cosine(fp_1_timbre, fp_2_timbre)) * 100

    # 2. PHYSICAL PLAUSIBILITY (The Custom Metrics)
    # Explicitly select the 5 metrics we are scoring:
    # 0: Kurtosis, 2: Flux Var, 4: Mod Energy, -2: Alias Mean, -1: Alias Std
    physical_indices = [0, 2, 4, -2, -1]
    fp_1_physical = fp_1[physical_indices]
    fp_2_physical = fp_2[physical_indices]
    
    # Calculate percentage deviations
    percent_deviations = np.abs(fp_1_physical - fp_2_physical) / (np.abs(fp_1_physical) + 1e-8)
    capped_deviations = np.clip(percent_deviations, 0, 1)
    
    # --- WEIGHTED SCORING ---
    metric_weights = np.array([
        0.3,  # Glottal Sharpness (Kurtosis) - Keep low (28% deviation)
        1.5,  # Spectral Flux Variance - SUPERCHARGED (35% deviation)
        1.5,  # Modulation Rhythm Energy - SUPERCHARGED (33% deviation)
        0.5,  # Aliasing Power (RMS Mean) - Lowered (AI is good at faking this in 48kHz)
        0.5   # Aliasing Power (RMS Std) - Lowered (AI is good at faking this in 48kHz)
    ])
    
    # Calculate the weighted average
    weighted_deviation = np.average(capped_deviations, weights=metric_weights)
    physical_plausibility = (1 - weighted_deviation) * 100

    # --- BUILD DISPLAY TABLE ---
    comparison_data = {
        "Metric": [
            "Glottal Sharpness (Kurtosis)", 
            "Spectral Flux (Variance)",
            "Modulation Rhythm Energy",
            "Aliasing Power (RMS Mean)",
            "Aliasing Power (RMS Std)"
        ],
        "Weight": [
            f"{w*100}%" for w in metric_weights
        ],
        label_1: [fp_1[0], fp_1[2], fp_1[4], fp_1[-2], fp_1[-1]],
        label_2: [fp_2[0], fp_2[2], fp_2[4], fp_2[-2], fp_2[-1]],
    }
    
    comparison_data["Difference"] = [
        abs(fp_1[0] - fp_2[0]),
        abs(fp_1[2] - fp_2[2]),
        abs(fp_1[4] - fp_2[4]),
        abs(fp_1[-2] - fp_2[-2]),
        abs(fp_1[-1] - fp_2[-1])
    ]

    df = pd.DataFrame(comparison_data)
    df_styled = df.style.format({label_1: '{:.5f}', label_2: '{:.5f}', "Difference": '{:.5f}'})
    
    print("=" * 65)
    print(f" VOCAL TIMBRE MATCH (Clone Accuracy)  : {timbre_similarity:.2f}%")
    print(f" PHYSICAL PLAUSIBILITY (Human Physics): {physical_plausibility:.2f}%")
    print("=" * 65 + "\n")
    
    if physical_plausibility < 75.0:
        print(" ⚠️ WARNING: HIGH PROBABILITY OF SYNTHETIC AUDIO DETECTED")
        print(" The vocal timbre matches, but the physical artifacts fail.")
        print("=" * 65 + "\n")
    
    print(df)
    return fp_1, fp_2
# --- EXECUTION ---
if __name__ == "__main__":
    # Replace these paths with your actual 16kHz audio files
    real_audio_path = "SPARC-audio-clips/resyn/ground-truth/source3.wav" 
    suspect_audio_path = "SPARC-audio-clips/resyn/ground-truth/source3_1.wav"
    
    try:
        fp_real, fp_suspect = compare_advanced_audio(real_audio_path, suspect_audio_path)
    except FileNotFoundError:
        print("Error: Please check your audio file paths.")