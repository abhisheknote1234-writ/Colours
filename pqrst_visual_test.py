import argparse
import socket
import time
from collections import deque
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, iirnotch


@dataclass
class AnalysisResult:
    quality: str
    r_peak_sharpness: str
    r_peak_consistency: str
    p_wave_presence: str
    qrs_morphology: str
    t_wave_presence: str
    baseline_stability: str
    noise_type: str
    rr_consistency: str
    snr_db: float
    missed_beats: int
    false_positives: int
    suggestions: list[str]


def bandpass(signal, fs, low=0.5, high=35.0, order=2):
    nyq = 0.5 * fs
    high = min(high, nyq * 0.95)
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def notch_filter(signal, fs, f0=50.0, q=30.0):
    if f0 >= fs * 0.5:
        return signal
    b, a = iirnotch(w0=f0, Q=q, fs=fs)
    return filtfilt(b, a, signal)


def robust_r_peaks(signal, fs):
    distance = max(1, int(0.25 * fs))
    prominence = max(np.std(signal) * 0.6, 1e-6)
    peaks, props = find_peaks(signal, distance=distance, prominence=prominence)
    return peaks, props


def slope_amplitude_r_peaks(signal, fs, amp_factor=0.6, slope_factor=0.6):
    """Detect peaks requiring both amplitude and slope criteria."""
    grad = np.gradient(signal)
    amp_th = np.mean(signal) + amp_factor * np.std(signal)
    slope_th = slope_factor * np.std(grad)

    distance = max(1, int(0.25 * fs))
    candidates, _ = find_peaks(signal, distance=distance, height=amp_th)

    selected = []
    win = max(1, int(0.03 * fs))
    for idx in candidates:
        left = max(0, idx - win)
        right = min(len(signal), idx + win + 1)
        local_slope = np.max(np.abs(grad[left:right]))
        if local_slope >= slope_th:
            selected.append(idx)

    return np.array(selected, dtype=int), amp_th, slope_th


def coefficient_of_variation(x):
    if len(x) < 2:
        return np.inf
    mean_x = np.mean(x)
    if abs(mean_x) < 1e-9:
        return np.inf
    return float(np.std(x) / abs(mean_x))


def estimate_snr_db(filtered, baseline):
    signal_power = np.var(filtered)
    noise_power = np.var(filtered - baseline)
    return float(10 * np.log10(max(signal_power, 1e-9) / max(noise_power, 1e-9)))


def match_peaks(reference, detected, tolerance_samples):
    ref_used = np.zeros(len(reference), dtype=bool)
    det_used = np.zeros(len(detected), dtype=bool)

    for j, d in enumerate(detected):
        nearest_idx = np.argmin(np.abs(reference - d)) if len(reference) else None
        if nearest_idx is not None and abs(reference[nearest_idx] - d) <= tolerance_samples and not ref_used[nearest_idx]:
            ref_used[nearest_idx] = True
            det_used[j] = True

    missed = int(np.sum(~ref_used))
    false_pos = int(np.sum(~det_used))
    return missed, false_pos


def classify_quality(r_detectability, snr_db, rr_cv, baseline_drift, missed, false_pos):
    if r_detectability > 0.85 and snr_db > 12 and rr_cv < 0.12 and baseline_drift < 0.10 and missed <= 1 and false_pos <= 1:
        return "Excellent (clean diagnostic-level signal)"
    if r_detectability > 0.70 and snr_db > 8 and rr_cv < 0.20 and baseline_drift < 0.18 and missed <= 3 and false_pos <= 3:
        return "Good (usable for HR/HRV)"
    if r_detectability > 0.45 and snr_db > 4:
        return "Acceptable (R-peaks detectable but morphology weak)"
    return "Poor (distorted, unreliable)"


def analyze_segment(signal, fs, mains_freq):
    filtered = bandpass(signal, fs)
    filtered = notch_filter(filtered, fs, f0=mains_freq)

    b_lp, a_lp = butter(2, min(0.7 / (0.5 * fs), 0.99), btype="low")
    baseline = filtfilt(b_lp, a_lp, filtered)
    baseline_drift = float(np.std(baseline) / max(np.std(filtered), 1e-6))

    reference_peaks, ref_props = robust_r_peaks(filtered, fs)
    detected_peaks, amp_th, slope_th = slope_amplitude_r_peaks(filtered, fs)

    if len(detected_peaks) < 2:
        return AnalysisResult(
            quality="Poor (distorted, unreliable)",
            r_peak_sharpness="Low (insufficient clear peaks)",
            r_peak_consistency="Low (too few R-peaks)",
            p_wave_presence="Not assessable",
            qrs_morphology="Not assessable",
            t_wave_presence="Not assessable",
            baseline_stability="Unstable",
            noise_type="Likely high artifact/noise",
            rr_consistency="Inconsistent",
            snr_db=estimate_snr_db(filtered, baseline),
            missed_beats=max(0, len(reference_peaks) - len(detected_peaks)),
            false_positives=max(0, len(detected_peaks) - len(reference_peaks)),
            suggestions=["Improve electrode contact and reduce movement before HRV use."],
        ), filtered, detected_peaks, amp_th, slope_th

    rr_sec = np.diff(detected_peaks) / fs
    rr_cv = coefficient_of_variation(rr_sec)

    ref_prom = ref_props.get("prominences", np.array([0.0]))
    r_sharpness_metric = float(np.mean(ref_prom) / max(np.std(filtered), 1e-6))
    r_detectability = min(1.0, r_sharpness_metric / 6.0)

    p_scores, qrs_scores, t_scores = [], [], []
    for pk in detected_peaks[1:-1]:
        p_start = int(pk - 0.22 * fs)
        p_end = int(pk - 0.08 * fs)
        qrs_start = int(pk - 0.04 * fs)
        qrs_end = int(pk + 0.06 * fs)
        t_start = int(pk + 0.10 * fs)
        t_end = int(pk + 0.40 * fs)

        if p_start >= 0 and p_end > p_start:
            p_seg = filtered[p_start:p_end]
            p_scores.append(float(np.max(p_seg) - np.min(p_seg)))

        if qrs_start >= 0 and qrs_end < len(filtered) and qrs_end > qrs_start:
            qrs_seg = filtered[qrs_start:qrs_end]
            qrs_scores.append(float(np.max(qrs_seg) - np.min(qrs_seg)))

        if t_start >= 0 and t_end < len(filtered) and t_end > t_start:
            t_seg = filtered[t_start:t_end]
            t_scores.append(float(np.max(t_seg) - np.min(t_seg)))

    p_rel = float(np.mean(p_scores) / max(np.mean(qrs_scores), 1e-6)) if p_scores and qrs_scores else 0.0
    t_rel = float(np.mean(t_scores) / max(np.mean(qrs_scores), 1e-6)) if t_scores and qrs_scores else 0.0

    fft = np.fft.rfft(filtered)
    freqs = np.fft.rfftfreq(len(filtered), d=1.0 / fs)
    power = np.abs(fft) ** 2
    total_power = float(np.sum(power) + 1e-9)
    mains_band = float(np.sum(power[(freqs > mains_freq - 2) & (freqs < mains_freq + 2)]) / total_power)
    high_freq = float(np.sum(power[freqs > 35]) / total_power)

    if mains_band > 0.08:
        noise_type = f"{int(mains_freq)} Hz powerline interference likely"
    elif high_freq > 0.20:
        noise_type = "Muscle artifact likely"
    elif baseline_drift > 0.20:
        noise_type = "Motion/baseline drift artifact likely"
    else:
        noise_type = "Low noise / no dominant artifact"

    snr_db = estimate_snr_db(filtered, baseline)

    missed, false_pos = match_peaks(reference_peaks, detected_peaks, tolerance_samples=max(1, int(0.06 * fs)))

    quality = classify_quality(r_detectability, snr_db, rr_cv, baseline_drift, missed, false_pos)

    suggestions = []
    if rr_cv > 0.30 or missed > 3 or false_pos > 3:
        suggestions.append("R-peaks are inconsistent: lower motion, improve lead contact, verify threshold tuning.")
    if p_rel < 0.04:
        suggestions.append("P-wave is weak/invisible: improve electrode placement and reduce high-frequency noise.")
    if baseline_drift > 0.18:
        suggestions.append("Baseline drift present: secure electrodes/cable and improve grounding.")
    if mains_band > 0.08:
        suggestions.append(f"Strong {int(mains_freq)} Hz noise: improve shielding/ground, keep leads away from mains adapters.")
    if not suggestions:
        suggestions.append("Signal is adequate for HRV/interval analysis under current settings.")

    result = AnalysisResult(
        quality=quality,
        r_peak_sharpness="High" if r_sharpness_metric > 4.5 else "Moderate" if r_sharpness_metric > 2.5 else "Low",
        r_peak_consistency="Consistent" if rr_cv < 0.15 else "Moderately variable" if rr_cv < 0.30 else "Inconsistent",
        p_wave_presence="Visible" if p_rel > 0.09 else "Weak/unclear" if p_rel > 0.04 else "Not clearly visible",
        qrs_morphology="QRS distinguishable" if np.mean(qrs_scores) > 0.15 else "QRS morphology weak",
        t_wave_presence="Present with smooth recovery" if t_rel > 0.10 else "T-wave weak/unclear",
        baseline_stability="Stable" if baseline_drift < 0.12 else "Mild drift" if baseline_drift < 0.22 else "Unstable drift",
        noise_type=noise_type,
        rr_consistency=f"RR CV={rr_cv:.3f}",
        snr_db=snr_db,
        missed_beats=missed,
        false_positives=false_pos,
        suggestions=suggestions,
    )
    return result, filtered, detected_peaks, amp_th, slope_th


def load_signal(path, value_col):
    df = pd.read_csv(path)
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found. Available: {list(df.columns)}")
    return df[value_col].astype(float).to_numpy()


def _collect_udp_segment(sample_rate, duration_s, udp_ip, udp_port):
    target_samples = int(sample_rate * duration_s)
    values = deque(maxlen=target_samples)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((udp_ip, udp_port))
    sock.settimeout(2.0)

    print(f"📡 Listening UDP on {udp_ip}:{udp_port} for {duration_s:.1f}s ({target_samples} samples target)")
    start_t = time.time()
    try:
        while len(values) < target_samples:
            try:
                data, _ = sock.recvfrom(1024)
            except socket.timeout:
                # continue waiting until enough samples
                continue
            msg = data.decode("utf-8", errors="ignore").strip()
            try:
                v = float(msg)
                values.append(v)
            except ValueError:
                continue

            if time.time() - start_t > max(5.0, duration_s * 4):
                # fail-safe in case incoming stream is too sparse
                break
    finally:
        sock.close()

    if len(values) < int(sample_rate * 2):
        raise RuntimeError("Not enough UDP samples collected. Ensure stream is active.")

    print(f"✅ Collected {len(values)} samples from UDP")
    return np.array(values, dtype=float)


def main():
    parser = argparse.ArgumentParser(description="Analyze ECG segment (5-10s) for PQRST quality")
    parser.add_argument("--file", help="CSV file containing ECG samples")
    parser.add_argument("--value-col", default="value", help="Column name with ECG values")
    parser.add_argument("--sample-rate", type=float, default=250.0, help="Sampling rate in Hz")
    parser.add_argument("--start", type=float, default=5.0, help="Segment start in seconds (file mode)")
    parser.add_argument("--end", type=float, default=10.0, help="Segment end in seconds (file mode)")
    parser.add_argument("--mains-freq", type=float, default=50.0, help="Powerline frequency: 50 or 60 Hz")
    parser.add_argument("--plot", action="store_true", help="Show segment with detected R-peaks")
    parser.add_argument("--udp", action="store_true", help="Use live UDP input instead of --file")
    parser.add_argument("--udp-ip", default="0.0.0.0", help="UDP bind IP in --udp mode")
    parser.add_argument("--udp-port", type=int, default=5007, help="UDP bind port in --udp mode")
    parser.add_argument("--duration", type=float, default=5.0, help="UDP capture duration in seconds")
    args = parser.parse_args()

    if args.udp:
        segment = _collect_udp_segment(args.sample_rate, args.duration, args.udp_ip, args.udp_port)
        segment_start = 0.0
    else:
        if not args.file:
            raise SystemExit("Use --file <csv> or enable --udp mode")
        raw = load_signal(args.file, args.value_col)
        i0 = int(args.start * args.sample_rate)
        i1 = int(args.end * args.sample_rate)
        if i0 < 0 or i1 <= i0 or i1 > len(raw):
            raise ValueError("Invalid segment bounds for provided signal length")
        segment = raw[i0:i1]
        segment_start = args.start

    result, filtered, peaks, amp_th, slope_th = analyze_segment(segment, args.sample_rate, args.mains_freq)

    rr_ms = np.diff(peaks) / args.sample_rate * 1000.0 if len(peaks) >= 2 else np.array([])

    print("\n=== ECG Segment Analysis ===")
    print(f"Overall Quality: {result.quality}")
    print(f"R-peak sharpness: {result.r_peak_sharpness}")
    print(f"R-peak consistency: {result.r_peak_consistency}")
    print(f"P-wave before QRS: {result.p_wave_presence}")
    print(f"QRS morphology: {result.qrs_morphology}")
    print(f"T-wave recovery: {result.t_wave_presence}")
    print(f"Baseline stability: {result.baseline_stability}")
    print(f"Noise type: {result.noise_type}")
    print(f"RR interval consistency: {result.rr_consistency}")
    print(f"SNR estimate (dB): {result.snr_db:.2f}")
    print(f"Missed beats (vs robust reference): {result.missed_beats}")
    print(f"False positives (vs robust reference): {result.false_positives}")
    print(f"Amplitude threshold: {amp_th:.4f}")
    print(f"Slope threshold: {slope_th:.4f}")

    if len(rr_ms):
        rr_text = ", ".join(f"{v:.1f}" for v in rr_ms)
        print(f"RR intervals (ms): {rr_text}")
    else:
        print("RR intervals (ms): Not enough peaks")

    print("Suggestions:")
    for sug in result.suggestions:
        print(f"- {sug}")

    if args.plot:
        t = np.arange(len(filtered)) / args.sample_rate + segment_start
        plt.figure(figsize=(12, 4))
        plt.plot(t, filtered, label="Filtered ECG", linewidth=1.3, color="#00FFFF")
        plt.axhline(amp_th, color="orange", linestyle="--", linewidth=1, label="Amp threshold")
        if len(peaks) > 0:
            plt.plot(t[peaks], filtered[peaks], "ro", label="Detected R-peaks", markersize=4)
        plt.title("ECG Segment: R-peak Overlay + PQRST Quality")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
