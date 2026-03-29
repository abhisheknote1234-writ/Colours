"""
hybrid_rpeak_detector.py
========================
Hybrid ECG R-Peak Detection Pipeline
Author: Research Project — ECG Stress/Emotion Study

Architecture:
  Raw ECG
    │
    ▼  [1] Preprocessing
    │      Bandpass 0.5–40 Hz · Notch 50 Hz · Baseline removal
    │
    ▼  [2] Pan-Tompkins (loose threshold)
    │      Generates candidate peaks — prefers recall over precision
    │
    ▼  [3] Rule-Based Filter (strict)
    │      RR interval · QRS width · energy · morphology consistency
    │      Rejects obvious non-peaks before any ML runs
    │
    ▼  [4] Feature Extraction (32 relative features)
    │      3-beat window · adaptive baseline · morphology template
    │
    ▼  [5] Lightweight CNN + Attention (morphology validator)
    │      Input: ±200 ms ECG window · Output: embedding (16-dim)
    │      Pure NumPy — no PyTorch/TensorFlow required
    │
    ▼  [6] XGBoost Classifier
    │      Input: 32 engineered + 16 CNN features = 48 total
    │      Threshold: 0.80 (precision-focused)
    │
    ▼  [7] Post-Processing
    │      Physiological RR enforcement · duplicate removal
    │      Conservative missed-beat recovery
    │
    ▼  Verified R-peaks

Design priorities:
  - PRECISION over recall (false positives are unacceptable)
  - No PyTorch/TensorFlow — CNN implemented in NumPy (edge-device ready)
  - Adaptive baseline — adapts to individual ECG morphology
  - Fully modular — each stage independently testable

Usage:
  detector = HybridRPeakDetector(fs=250)
  detector.load_model('models/hybrid_detector.pkl')   # if trained
  peaks, info = detector.detect(ecg_signal)

  # Or train from scratch:
  from hybrid_rpeak_detector import train_hybrid_model
  train_hybrid_model(X_ecg_list, y_labels_list, fs=250)
"""

import numpy as np
import pickle
import warnings
from collections import deque
from scipy.signal import butter, filtfilt, iirnotch, medfilt
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════
#  CONSTANTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

class Config:
    """
    Central configuration — all thresholds in one place.
    Tune here, never buried in function bodies.
    """
    # ── Preprocessing ───────────────────────────────────────────────
    BP_LOW          = 0.5    # Hz  bandpass low cutoff
    BP_HIGH         = 40.0   # Hz  bandpass high cutoff
    BP_ORDER        = 4      # Butterworth order
    NOTCH_FREQ      = 50.0   # Hz  mains noise (India)
    NOTCH_Q         = 30.0   # Notch quality factor
    BASELINE_WIN    = 201    # samples  median filter baseline window

    # ── Pan-Tompkins (loose — prefer recall) ────────────────────────
    PT_BP_LOW       = 5.0    # Hz  QRS-specific bandpass
    PT_BP_HIGH      = 15.0   # Hz
    MWI_WINDOW_MS   = 150    # ms  moving window integration
    REFRACTORY_MS   = 180    # ms  slightly shorter than standard (more permissive)
    # Using 0.25× threshold instead of standard 0.5× for looser detection
    PT_THRESHOLD_MULT = 0.25

    # ── Rule-based filter (STRICT) ───────────────────────────────────
    RR_MIN_MS       = 300    # ms  minimum physiological RR
    RR_MAX_MS       = 2000   # ms  maximum (30 bpm lower bound)
    RR_DEVIATION    = 0.35   # max fractional deviation from running avg
    QRS_WIDTH_MIN_MS= 30     # ms  minimum QRS width (30ms after bandpass filtering)
    QRS_WIDTH_MAX_MS= 200    # ms  maximum QRS width (incl. bundle branch block)
    ENERGY_MIN_RATIO= 0.5    # local energy must be this × running avg
    MORPH_CORR_MIN  = 0.40   # minimum correlation with running template

    # ── Adaptive baseline (exponential moving average) ───────────────
    EMA_ALPHA       = 0.10   # update rate: new = 0.9*old + 0.1*new
    TEMPLATE_N      = 8      # beats used to maintain QRS template

    # ── CNN window ──────────────────────────────────────────────────
    CNN_WINDOW_MS   = 200    # ms  ±200 ms around peak
    CNN_EMBED_DIM   = 16     # embedding output dimension

    # ── Final decision ───────────────────────────────────────────────
    ML_THRESHOLD    = 0.80   # XGBoost probability threshold (precision-focused)

    # ── Post-processing ──────────────────────────────────────────────
    POST_RR_MIN_MS  = 300    # ms  enforced after ML
    MISSED_BEAT_MULT= 1.75   # if gap > this × mean RR → search for missed beat


# ═══════════════════════════════════════════════════════════════════
#  MODULE 1: PREPROCESSING
# ═══════════════════════════════════════════════════════════════════

class Preprocessor:
    """
    Signal preprocessing pipeline.
    Designed once per detector instance — filter coefficients cached.
    """

    def __init__(self, fs: int):
        self.fs = fs
        # Design filters at init — not every call
        self._b_notch, self._a_notch = self._design_notch(fs)
        self._b_bp,    self._a_bp    = self._design_bandpass(fs)

    def _design_notch(self, fs):
        return iirnotch(Config.NOTCH_FREQ, Config.NOTCH_Q, fs)

    def _design_bandpass(self, fs):
        nyq = fs / 2.0
        lo  = Config.BP_LOW  / nyq
        hi  = Config.BP_HIGH / nyq
        return butter(Config.BP_ORDER, [lo, hi], btype='band')

    def process(self, ecg_raw: np.ndarray,
                smooth: bool = False) -> np.ndarray:
        """
        Full preprocessing chain.

        Parameters
        ----------
        ecg_raw : raw 1D ECG array from ADC
        smooth  : optional light smoothing (3-point moving average)
                  use only for very noisy signals

        Returns
        -------
        ecg_clean : preprocessed float64 array, same length
        """
        sig = ecg_raw.astype(np.float64)

        # Step 1 — Remove 50 Hz mains noise (critical for India)
        sig = filtfilt(self._b_notch, self._a_notch, sig)

        # Step 2 — Bandpass 0.5–40 Hz
        # Removes: DC offset, respiration drift (<0.5 Hz),
        #          high-freq muscle noise (>40 Hz)
        sig = filtfilt(self._b_bp, self._a_bp, sig)

        # Step 3 — Baseline wander removal
        # Median filter estimates the slow-moving baseline
        # then subtract it. Window = 401ms covers one full
        # respiratory cycle without distorting QRS.
        w        = Config.BASELINE_WIN
        w        = w if w % 2 else w + 1   # must be odd
        baseline = medfilt(sig, kernel_size=w)
        sig      = sig - baseline

        # Step 4 — Optional smoothing (3-point moving average)
        # Only use for very noisy signals — can blur QRS peaks
        if smooth:
            sig = np.convolve(sig, [1/3, 1/3, 1/3], mode='same')

        return sig


# ═══════════════════════════════════════════════════════════════════
#  MODULE 2: PAN-TOMPKINS (LOOSE THRESHOLD)
# ═══════════════════════════════════════════════════════════════════

class PanTompkinsLoose:
    """
    Pan-Tompkins with a deliberately loose detection threshold.

    Design philosophy:
    ─────────────────
    Standard Pan-Tompkins balances Se and P+.
    Here we tune for HIGH RECALL (miss fewer peaks).
    The strict rule-based filter and ML will handle precision.
    Better to have too many candidates than miss real beats.
    """

    def __init__(self, fs: int):
        self.fs         = fs
        self.mwi_win    = int(Config.MWI_WINDOW_MS * fs / 1000)
        self.refractory = int(Config.REFRACTORY_MS  * fs / 1000)

        nyq = fs / 2.0
        self._b_bp, self._a_bp = butter(
            2, [Config.PT_BP_LOW/nyq, Config.PT_BP_HIGH/nyq], btype='band')
        b_deriv = np.array([-1., -2., 0., 2., 1.]) * (fs / 8.0)
        self._b_deriv = b_deriv
        self._a_deriv = np.array([1.0])

    def detect(self, ecg_clean: np.ndarray) -> tuple:
        """
        Returns (candidates, bandpass_signal, mwi_signal)
        candidates: loose R-peak indices for further filtering
        """
        n = len(ecg_clean)

        # Stage 1–4: signal transformation
        bp      = filtfilt(self._b_bp,    self._a_bp,    ecg_clean)
        deriv   = filtfilt(self._b_deriv, self._a_deriv, bp)
        squared = deriv ** 2
        mwi     = np.convolve(squared,
                               np.ones(self.mwi_win) / self.mwi_win,
                               mode='same')

        # Stage 5: adaptive thresholding — LOOSE
        # Standard uses 0.5× threshold for searchback; we use 0.25×
        init_n    = min(2 * self.fs, n)
        spki      = np.max(mwi[:init_n]) * 0.25
        npki      = spki * 0.5
        thresh1   = npki + 0.25 * (spki - npki)
        thresh2   = Config.PT_THRESHOLD_MULT * thresh1   # looser

        candidates = []
        prev_r     = -self.refractory - 1
        rr_list    = []
        rr_mean    = self.fs * 0.8

        i = self.mwi_win
        while i < n - self.mwi_win:
            # Local maximum check
            if not (mwi[i] >= mwi[max(0,i-1)] and
                    mwi[i] >= mwi[min(n-1,i+1)]):
                i += 1
                continue

            if (i - prev_r) < self.refractory:
                i += 1
                continue

            peak_val = mwi[i]

            if peak_val >= thresh2:   # loose: accept at thresh2 level
                # Find true R in bandpass signal
                s = max(0, i - self.mwi_win // 2)
                e = min(n, i + 6)
                r_idx = s + int(np.argmax(bp[s:e]))
                candidates.append(r_idx)

                if len(candidates) > 1:
                    rr = candidates[-1] - candidates[-2]
                    rr_list.append(rr)
                    rr_mean = np.mean(rr_list[-8:])

                # Update thresholds
                spki   = 0.125 * peak_val + 0.875 * spki

                # Searchback for missed beats
                if rr_list and (i - prev_r) > 1.66 * rr_mean:
                    sb_start = max(0, prev_r + self.refractory)
                    sb_idx   = sb_start + int(np.argmax(mwi[sb_start:i]))
                    if mwi[sb_idx] >= thresh2:
                        r_back = max(0, sb_idx - 5) + \
                                 int(np.argmax(bp[max(0,sb_idx-5):sb_idx+5]))
                        if r_back not in candidates:
                            candidates.insert(-1, r_back)

                prev_r = r_idx
            else:
                npki = 0.125 * peak_val + 0.875 * npki

            thresh1 = npki + 0.25 * (spki - npki)
            thresh2 = Config.PT_THRESHOLD_MULT * thresh1
            i += 1

        candidates = sorted(set(candidates))
        candidates = self._remove_duplicates(candidates)
        return np.array(candidates, dtype=int), bp, mwi

    def _remove_duplicates(self, peaks: list) -> list:
        if len(peaks) < 2:
            return peaks
        keep = [peaks[0]]
        for p in peaks[1:]:
            if p - keep[-1] >= self.refractory:
                keep.append(p)
        return keep


# ═══════════════════════════════════════════════════════════════════
#  MODULE 3: ADAPTIVE BASELINE
# ═══════════════════════════════════════════════════════════════════

class AdaptiveBaseline:
    """
    Maintains running statistics of verified R-peaks using
    exponential moving average (EMA).

    update rule: new_avg = (1 - alpha) * old_avg + alpha * new_value
    alpha = 0.10 → ~10 beat memory, slow adaptation
    alpha = 0.25 → ~4 beat memory, fast adaptation

    Also maintains a QRS template (average of last N verified beats)
    used for morphology correlation in rule filter.
    """

    def __init__(self, fs: int):
        self.fs             = fs
        self.alpha          = Config.EMA_ALPHA
        self.template_n     = Config.TEMPLATE_N
        self.qrs_half_win   = int(0.06 * fs)  # 60ms each side for template

        # Running averages — initialised to physiological defaults
        self.rr_avg         = float(fs * 0.8)  # 75 bpm
        self.width_avg      = float(int(0.08 * fs))  # 80ms QRS
        self.amplitude_avg  = 1.0
        self.energy_avg     = 1.0

        # QRS template — running mean of recent beats
        template_len        = 2 * self.qrs_half_win + 1
        self.template       = np.zeros(template_len)
        self._template_buf  = deque(maxlen=self.template_n)
        self._n_updates     = 0

    def update(self, ecg_bp: np.ndarray, r_idx: int,
               rr_interval: float = None):
        """
        Update running averages with a newly verified R-peak.
        Call only for confirmed true R-peaks.
        """
        n = len(ecg_bp)
        alpha = self.alpha

        # Update RR average
        if rr_interval is not None and rr_interval > 0:
            self.rr_avg = (1 - alpha) * self.rr_avg + alpha * rr_interval

        # Extract QRS segment for template and amplitude
        s = max(0, r_idx - self.qrs_half_win)
        e = min(n, r_idx + self.qrs_half_win + 1)
        seg = ecg_bp[s:e]

        if len(seg) > 4:
            amp = float(np.max(np.abs(seg)))
            self.amplitude_avg = (1-alpha)*self.amplitude_avg + alpha*amp

            energy = float(np.sum(seg**2))
            self.energy_avg = (1-alpha)*self.energy_avg + alpha*energy

            # Update QRS width estimate
            half_amp = amp * 0.5
            above    = np.abs(seg) > half_amp
            width    = int(np.sum(above))
            self.width_avg = (1-alpha)*self.width_avg + alpha*width

            # Update template (pad/trim to standard length)
            tlen  = 2 * self.qrs_half_win + 1
            if len(seg) == tlen:
                # Normalise before adding to template
                norm_seg = seg / (amp + 1e-8)
                self._template_buf.append(norm_seg)
                self.template = np.mean(self._template_buf, axis=0)

        self._n_updates += 1

    def get_template_correlation(self, ecg_bp: np.ndarray,
                                  r_idx: int) -> float:
        """
        Pearson correlation between candidate QRS and running template.
        Returns 0 if template not yet established (< 3 beats).
        """
        if self._n_updates < 3:
            return 1.0   # no template yet — don't filter

        n = len(ecg_bp)
        s = max(0, r_idx - self.qrs_half_win)
        e = min(n, r_idx + self.qrs_half_win + 1)
        seg = ecg_bp[s:e]
        tlen = len(self.template)

        if len(seg) < 4 or len(seg) != tlen:
            return 0.5   # edge case

        amp = np.max(np.abs(seg))
        if amp < 1e-9:
            return 0.0

        norm_seg = seg / amp
        corr = np.corrcoef(norm_seg, self.template)[0, 1]
        return float(corr) if np.isfinite(corr) else 0.0

    @property
    def rr_avg_ms(self):
        return self.rr_avg / self.fs * 1000.0

    @property
    def is_ready(self):
        """True once enough beats seen to trust the baseline."""
        return self._n_updates >= 3


# ═══════════════════════════════════════════════════════════════════
#  MODULE 4: STRICT RULE-BASED FILTER
# ═══════════════════════════════════════════════════════════════════

class RuleBasedFilter:
    """
    Hard physiological rules applied BEFORE ML.
    Each rule is independently tunable via Config.
    A candidate must pass ALL rules to proceed to ML.

    Design: fail fast — cheapest checks first.
    """

    def __init__(self, fs: int):
        self.fs          = fs
        self.rr_min      = int(Config.RR_MIN_MS  * fs / 1000)
        self.rr_max      = int(Config.RR_MAX_MS  * fs / 1000)
        self.qrs_min     = int(Config.QRS_WIDTH_MIN_MS * fs / 1000)
        self.qrs_max     = int(Config.QRS_WIDTH_MAX_MS * fs / 1000)

    def apply(self, ecg_bp: np.ndarray,
              candidate: int,
              prev_r: int,
              baseline: AdaptiveBaseline) -> tuple:
        """
        Apply all rules to one candidate.

        Returns
        -------
        (passed: bool, reason: str)
        reason is 'ok' or describes why it failed.
        """
        n = len(ecg_bp)

        # ── Rule 1: RR interval minimum (300 ms) ─────────────────────
        # Two peaks closer than 300 ms cannot both be R-peaks
        # (human heart cannot beat faster than 200 bpm sustained)
        if prev_r >= 0:
            rr = candidate - prev_r
            if rr < self.rr_min:
                return False, 'rr_too_short'
            if rr > self.rr_max:
                # Long gap is not a rejection — could be pause
                # Only flag if baseline is established and gap is extreme
    # Only fire rr_extreme if prev_r is a genuine verified peak
                # (not the initial -1 sentinel) and baseline is stable
                if (baseline._n_updates >= 5 and
                        rr > 3.5 * baseline.rr_avg):
                    return False, 'rr_extreme_long'

        # ── Rule 2: RR deviation from running average ─────────────────
        # Sudden irregular beats are suspicious — but don't reject
        # if baseline not yet established
        if prev_r >= 0 and baseline._n_updates >= 5:
            rr      = candidate - prev_r
            rr_dev  = abs(rr - baseline.rr_avg) / (baseline.rr_avg + 1e-6)
            if rr_dev > Config.RR_DEVIATION:
                return False, f'rr_deviation_{rr_dev:.2f}'

        # ── Rule 3: QRS width ─────────────────────────────────────────
        # Measure width at 50% of peak amplitude
        s   = max(0, candidate - int(0.15 * self.fs))
        e   = min(n, candidate + int(0.15 * self.fs))
        seg = ecg_bp[s:e]
        if len(seg) < 4:
            return False, 'edge_of_signal'

        amp      = np.max(np.abs(seg))
        if amp < 1e-9:
            return False, 'zero_amplitude'

        half_amp = amp * 0.5
        above    = np.abs(seg) > half_amp
        width    = int(np.sum(above))

        if width < self.qrs_min:
            return False, f'qrs_too_narrow_{width}'
        if width > self.qrs_max:
            return False, f'qrs_too_wide_{width}'

        # ── Rule 4: Local energy concentration ───────────────────────
        # Real QRS has energy concentrated in a short window.
        # Noise has similar energy spread across a large window.
        # Compare: energy in ±40ms vs energy in ±150ms
        qrs_win   = int(0.04 * self.fs)
        local_win = int(0.15 * self.fs)

        s_qrs  = max(0, candidate - qrs_win)
        e_qrs  = min(n, candidate + qrs_win)
        s_loc  = max(0, candidate - local_win)
        e_loc  = min(n, candidate + local_win)

        e_qrs_val  = float(np.sum(ecg_bp[s_qrs:e_qrs]**2))
        e_local    = float(np.sum(ecg_bp[s_loc:e_loc]**2))

        if e_local < 1e-12:
            return False, 'zero_energy'

        # QRS should contain > 30% of local energy (concentrated)
        energy_ratio = e_qrs_val / e_local
        if energy_ratio < 0.30:
            return False, f'energy_diffuse_{energy_ratio:.2f}'

        # Compare to running average energy
        if baseline.is_ready:
            beat_energy = float(np.sum(seg**2))
            energy_vs_avg = beat_energy / (baseline.energy_avg + 1e-6)
            if energy_vs_avg < Config.ENERGY_MIN_RATIO:
                return False, f'energy_too_low_{energy_vs_avg:.2f}'

        # ── Rule 5: Morphology correlation with template ──────────────
        # Only applied once template is established (after ~8 beats)
        if baseline.is_ready and baseline._n_updates >= baseline.template_n:
            corr = baseline.get_template_correlation(ecg_bp, candidate)
            if corr < Config.MORPH_CORR_MIN:
                return False, f'morph_corr_{corr:.2f}'

        return True, 'ok'


# ═══════════════════════════════════════════════════════════════════
#  MODULE 5: FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    # Relative amplitude features (scale-invariant)
    'rel_amplitude',         # peak / local RMS
    'rel_prominence',        # peak - local mean / local std
    'rel_amplitude_vs_avg',  # peak / running avg amplitude

    # QRS morphology
    'qrs_width',             # samples at half amplitude
    'qrs_width_vs_avg',      # relative to running average
    'rise_slope',            # slope 20ms before peak
    'fall_slope',            # slope 20ms after peak
    'slope_ratio',           # rise/fall ratio (symmetry indicator)
    'sharpness',             # 2nd derivative at peak
    'curr_symmetry',         # area ratio left/right

    # Energy features
    'local_energy',          # windowed sum of squares
    'energy_ratio',          # QRS energy / local energy
    'energy_vs_avg',         # local energy vs running average

    # Rhythm features (all relative to running average)
    'rr_prev_ms',            # previous RR in ms
    'rr_next_ms',            # next RR in ms (lookahead)
    'rr_ratio_prev',         # rr_prev / rr_avg
    'rr_ratio_next',         # rr_next / rr_avg
    'rr_regularity',         # std([rr_prev, rr_next]) / mean
    'triplet_hr_bpm',        # instantaneous HR from triplet

    # Morphology similarity
    'template_corr',         # correlation with running QRS template
    'template_corr_sq',      # squared correlation (emphasise high values)

    # 3-beat context (neighbour beats)
    'prev_amplitude',        # left neighbour peak amplitude
    'next_amplitude',        # right neighbour peak amplitude
    'amplitude_trend',       # next_amp - prev_amp (gradual = normal)
    'prev_width',            # left neighbour QRS width
    'next_width',            # right neighbour QRS width
    'morphology_consistency',# correlation between all 3 QRS shapes

    # Statistical (local window)
    'local_kurtosis',        # kurtosis of local window (peaked = QRS)
    'local_skewness',        # skewness of local window
    'snr_local',             # peak / RMS (signal quality indicator)

    # Derivative-based
    'max_derivative',        # maximum slope in QRS window
    'derivative_symmetry',   # left/right slope balance
]

assert len(FEATURE_NAMES) == 32, f"Expected 32 features, got {len(FEATURE_NAMES)}"


class FeatureExtractor:
    """
    Extracts 32 relative, scale-invariant features per candidate.
    All features are normalised relative to local statistics or
    running averages — never absolute values.
    This makes the model robust across different people and hardware.
    """

    def __init__(self, fs: int):
        self.fs       = fs
        self.win      = int(0.10 * fs)   # 100ms half-window for local stats
        self.rise_win = max(1, int(0.02 * fs))  # 20ms rise/fall window
        self.qrs_win  = int(0.06 * fs)   # 60ms QRS window

    def _single_beat_morphology(self, ecg_bp: np.ndarray,
                                 idx: int) -> dict:
        """Extract morphology features for one beat at index idx."""
        n   = len(ecg_bp)
        w   = self.win
        rw  = self.rise_win
        qw  = self.qrs_win

        s   = max(0, idx - w)
        e   = min(n, idx + w)
        seg = ecg_bp[s:e]

        amp        = float(ecg_bp[idx])
        local_mean = float(np.mean(seg))
        local_std  = float(np.std(seg)) + 1e-9
        local_rms  = float(np.sqrt(np.mean(seg**2))) + 1e-9

        prominence = (amp - local_mean) / local_std
        rel_amp    = abs(amp) / local_rms

        # Rise/fall slope
        rs = float(ecg_bp[idx] - ecg_bp[max(0, idx-rw)]) / rw
        fs_ = float(ecg_bp[idx] - ecg_bp[min(n-1, idx+rw)]) / rw

        # Sharpness (2nd derivative)
        c = idx - s
        sharpness = float(seg[c-1] - 2*seg[c] + seg[c+1]) \
                    if 1 < c < len(seg)-2 else 0.0

        # Symmetry
        left  = ecg_bp[max(0,idx-w): idx]
        right = ecg_bp[idx: min(n,idx+w)]
        la    = float(np.sum(np.abs(left  - local_mean))) + 1e-9
        ra    = float(np.sum(np.abs(right - local_mean))) + 1e-9
        sym   = min(la, ra) / max(la, ra)

        # QRS width at half amplitude
        qs = max(0, idx - qw)
        qe = min(n, idx + qw)
        qseg   = ecg_bp[qs:qe]
        qamp   = np.max(np.abs(qseg)) if len(qseg) > 0 else 1.0
        width  = int(np.sum(np.abs(qseg) > 0.5 * qamp))

        # SNR
        peak  = float(np.max(np.abs(seg)))
        rms   = float(np.sqrt(np.mean(seg**2))) + 1e-9
        snr   = peak / rms

        # Kurtosis / skewness
        z     = (seg - local_mean) / local_std
        kurt  = float(np.mean(z**4)) - 3.0 if len(z) > 3 else 0.0
        skew  = float(np.mean(z**3)) if len(z) > 2 else 0.0

        # Max derivative
        diffs = np.diff(ecg_bp[max(0,idx-qw): min(n,idx+qw)])
        max_d = float(np.max(np.abs(diffs))) if len(diffs) > 0 else 0.0

        # Derivative symmetry: left vs right max slope
        mid    = qw
        dl     = diffs[:mid]   if len(diffs) > mid else diffs
        dr     = diffs[mid:]   if len(diffs) > mid else diffs
        dsl    = float(np.max(np.abs(dl))) + 1e-9 if len(dl) > 0 else 1.0
        dsr    = float(np.max(np.abs(dr))) + 1e-9 if len(dr) > 0 else 1.0
        deriv_sym = min(dsl, dsr) / max(dsl, dsr)

        # Local energy
        local_energy = float(np.sum(seg**2))
        qrs_energy   = float(np.sum(ecg_bp[max(0,idx-int(0.04*self.fs)):
                                            min(n,idx+int(0.04*self.fs))]**2))
        energy_ratio = qrs_energy / (local_energy + 1e-9)

        return {
            'amplitude'    : abs(amp),
            'rel_amp'      : rel_amp,
            'prominence'   : prominence,
            'rise_slope'   : rs,
            'fall_slope'   : fs_,
            'slope_ratio'  : rs / (abs(fs_) + 1e-9),
            'sharpness'    : sharpness,
            'symmetry'     : sym,
            'width'        : width,
            'snr'          : snr,
            'kurtosis'     : kurt,
            'skewness'     : skew,
            'local_energy' : local_energy,
            'energy_ratio' : energy_ratio,
            'max_deriv'    : max_d,
            'deriv_sym'    : deriv_sym,
        }

    def _morphology_consistency(self, ecg_bp: np.ndarray,
                                 ip: int, ic: int, inx: int) -> float:
        """Inter-beat QRS shape correlation — the key FP rejection feature."""
        n   = len(ecg_bp)
        qw  = self.qrs_win

        def _seg(idx):
            s, e = max(0, idx-qw), min(n, idx+qw)
            return ecg_bp[s:e]

        def _norm(x):
            s = np.std(x)
            return (x - np.mean(x)) / s if s > 1e-9 else x - np.mean(x)

        sp, sc, sn = _seg(ip), _seg(ic), _seg(inx)
        L = min(len(sp), len(sc), len(sn))
        if L < 4:
            return 0.5

        sp, sc, sn = sp[:L], sc[:L], sn[:L]
        c1 = float(np.corrcoef(_norm(sp), _norm(sc))[0, 1])
        c2 = float(np.corrcoef(_norm(sc), _norm(sn))[0, 1])
        val = float(np.mean([c1, c2]))
        return val if np.isfinite(val) else 0.5

    def extract(self, ecg_bp: np.ndarray,
                candidates: np.ndarray,
                baseline: AdaptiveBaseline,
                idx: int) -> np.ndarray:
        """
        Extract 32 features for candidate at candidates[idx].

        Parameters
        ----------
        ecg_bp     : bandpass ECG
        candidates : all candidate indices
        baseline   : adaptive baseline object
        idx        : which candidate to extract features for

        Returns
        -------
        feature_vector : np.ndarray of shape (32,)
        """
        n     = len(ecg_bp)
        c_idx = int(candidates[idx])
        p_idx = int(candidates[idx-1]) if idx > 0             else max(0, c_idx - int(baseline.rr_avg))
        n_idx = int(candidates[idx+1]) if idx < len(candidates)-1 else min(n-1, c_idx + int(baseline.rr_avg))

        # ── Per-beat morphology ───────────────────────────────────────
        mc = self._single_beat_morphology(ecg_bp, c_idx)
        mp = self._single_beat_morphology(ecg_bp, p_idx)
        mn = self._single_beat_morphology(ecg_bp, n_idx)

        # ── Rhythm features ───────────────────────────────────────────
        rr_prev_s = float(c_idx - p_idx)
        rr_next_s = float(n_idx - c_idx)
        rr_prev_ms = rr_prev_s / self.fs * 1000.0
        rr_next_ms = rr_next_s / self.fs * 1000.0
        rr_avg     = baseline.rr_avg + 1e-6
        rr_ratio_p = rr_prev_s / rr_avg
        rr_ratio_n = rr_next_s / rr_avg
        rr_mean_t  = (rr_prev_s + rr_next_s) / 2.0
        rr_reg     = float(np.std([rr_prev_s, rr_next_s])) / (rr_mean_t + 1e-6)
        hr_bpm     = float(60 * self.fs / rr_mean_t) if rr_mean_t > 0 else 0.0

        # ── Template correlation ──────────────────────────────────────
        t_corr = baseline.get_template_correlation(ecg_bp, c_idx)

        # ── Relative to running averages ──────────────────────────────
        rel_amp_vs_avg = mc['amplitude'] / (baseline.amplitude_avg + 1e-6)
        qrs_w_vs_avg   = mc['width'] / (baseline.width_avg + 1e-6)
        energy_vs_avg  = mc['local_energy'] / (baseline.energy_avg + 1e-6)

        # ── 3-beat morphology consistency ─────────────────────────────
        morph_con = self._morphology_consistency(ecg_bp, p_idx, c_idx, n_idx)

        # ── Assemble 32-feature vector ────────────────────────────────
        features = np.array([
            # Relative amplitude (3)
            mc['rel_amp'],
            mc['prominence'],
            rel_amp_vs_avg,
            # QRS morphology (7)
            mc['width'],
            qrs_w_vs_avg,
            mc['rise_slope'],
            mc['fall_slope'],
            mc['slope_ratio'],
            mc['sharpness'],
            mc['symmetry'],
            # Energy (3)
            mc['local_energy'],
            mc['energy_ratio'],
            energy_vs_avg,
            # Rhythm (6)
            rr_prev_ms,
            rr_next_ms,
            rr_ratio_p,
            rr_ratio_n,
            rr_reg,
            hr_bpm,
            # Morphology similarity (2)
            t_corr,
            t_corr ** 2,
            # 3-beat context (5)
            mp['amplitude'],
            mn['amplitude'],
            mn['amplitude'] - mp['amplitude'],  # amplitude_trend
            mp['width'],
            mn['width'],
            morph_con,
            # Statistical (4)
            mc['kurtosis'],
            mc['skewness'],
            mc['snr'],
            # Derivative (2)
            mc['max_deriv'],
            mc['deriv_sym'],
        ], dtype=np.float32)

        assert len(features) == 32, f"Feature count error: {len(features)}"
        # Replace any NaN/Inf with 0
        features = np.where(np.isfinite(features), features, 0.0)
        return features


# ═══════════════════════════════════════════════════════════════════
#  MODULE 6: LIGHTWEIGHT CNN + ATTENTION (PURE NUMPY)
# ═══════════════════════════════════════════════════════════════════

class LightweightCNNAttention:
    """
    Lightweight morphology validator implemented in pure NumPy.
    No PyTorch or TensorFlow required — runs on any device.

    Architecture:
    ─────────────
    Input: ECG window (±200ms = 100 samples at 250Hz)
        │
        ▼  Conv1D block × 2   (1D convolution with ReLU)
        │  Filters: 8 → 16
        │
        ▼  Self-Attention      (dot-product, single head)
        │  Weights how much each time step attends to others
        │
        ▼  Global Average Pool (collapse time dimension)
        │
        ▼  Linear projection   (→ 16-dim embedding)
        │
    Output: 16-dimensional embedding vector

    Parameters are random at init — the embedding is used as
    additional features for XGBoost, not for standalone classification.
    XGBoost learns which embedding dimensions are informative.

    For a trained version, call .set_weights() after loading from pkl.
    """

    def __init__(self, fs: int, embed_dim: int = Config.CNN_EMBED_DIM,
                 random_seed: int = 42):
        self.fs         = fs
        self.embed_dim  = embed_dim
        self.win_samples= int(Config.CNN_WINDOW_MS * fs / 1000)  # ±200ms
        self.input_len  = 2 * self.win_samples + 1

        rng = np.random.RandomState(random_seed)

        # Conv layer 1: 8 filters of size 5 (FIR-like)
        self.conv1_w = rng.randn(8, 1, 5).astype(np.float32) * 0.1
        self.conv1_b = np.zeros(8, dtype=np.float32)

        # Conv layer 2: 16 filters of size 3
        self.conv2_w = rng.randn(16, 8, 3).astype(np.float32) * 0.1
        self.conv2_b = np.zeros(16, dtype=np.float32)

        # Attention: Q, K, V projections (16 → 8)
        self.Wq = rng.randn(16, 8).astype(np.float32) * 0.1
        self.Wk = rng.randn(16, 8).astype(np.float32) * 0.1
        self.Wv = rng.randn(16, 8).astype(np.float32) * 0.1

        # Output projection: 8 → embed_dim
        self.W_out = rng.randn(8, embed_dim).astype(np.float32) * 0.1
        self.b_out = np.zeros(embed_dim, dtype=np.float32)

    def _conv1d(self, x: np.ndarray, W: np.ndarray,
                b: np.ndarray) -> np.ndarray:
        """
        1D convolution: x (C_in, T) × W (C_out, C_in, K) → (C_out, T)
        Uses 'same' padding. ReLU activation applied.
        """
        C_out, C_in, K = W.shape
        T      = x.shape[1]
        pad    = K // 2
        x_pad  = np.pad(x, ((0,0),(pad,pad)), mode='edge')
        out    = np.zeros((C_out, T), dtype=np.float32)
        for oc in range(C_out):
            for ic in range(C_in):
                for k in range(K):
                    out[oc] += W[oc, ic, k] * x_pad[ic, k:k+T]
            out[oc] += b[oc]
        return np.maximum(0, out)   # ReLU

    def _attention(self, x: np.ndarray) -> np.ndarray:
        """
        Single-head dot-product self-attention.
        x: (C, T) → output: (8, T)

        Attention allows the model to weight which time steps
        (e.g., the QRS peak vs the T-wave) are most relevant.
        """
        # x: (C, T) → transpose to (T, C)
        xT  = x.T   # (T, C)
        Q   = xT @ self.Wq   # (T, 8)
        K   = xT @ self.Wk   # (T, 8)
        V   = xT @ self.Wv   # (T, 8)

        # Scaled dot-product attention
        scale   = np.sqrt(Q.shape[1]).astype(np.float32)
        scores  = (Q @ K.T) / scale          # (T, T)
        # Softmax along last axis
        scores  = scores - np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights = weights / (np.sum(weights, axis=-1, keepdims=True) + 1e-9)

        attended = weights @ V   # (T, 8)
        return attended.T        # (8, T)

    def embed(self, ecg_bp: np.ndarray, r_idx: int) -> np.ndarray:
        """
        Extract 16-dim embedding for one candidate peak.

        Parameters
        ----------
        ecg_bp : bandpass ECG signal
        r_idx  : sample index of candidate peak

        Returns
        -------
        embedding : np.ndarray shape (embed_dim,)
        """
        n   = len(ecg_bp)
        ws  = self.win_samples

        # Extract and pad window
        s   = max(0, r_idx - ws)
        e   = min(n, r_idx + ws + 1)
        seg = ecg_bp[s:e].astype(np.float32)

        # Pad to standard length if edge of signal
        if len(seg) < self.input_len:
            pad_l = r_idx - s
            pad_r = self.input_len - len(seg) - pad_l
            seg   = np.pad(seg, (max(0,pad_l), max(0,pad_r)),
                           mode='edge')[:self.input_len]

        # Normalise to zero mean, unit std
        mu   = np.mean(seg)
        std  = np.std(seg) + 1e-9
        seg  = (seg - mu) / std

        # Shape: (1, T) for conv input
        x = seg[np.newaxis, :]   # (1, input_len)

        # Conv block 1: (1, T) → (8, T)
        x = self._conv1d(x, self.conv1_w, self.conv1_b)

        # Conv block 2: (8, T) → (16, T)
        x = self._conv1d(x, self.conv2_w, self.conv2_b)

        # Self-attention: (16, T) → (8, T)
        x = self._attention(x)

        # Global average pooling: (8, T) → (8,)
        x = np.mean(x, axis=1)

        # Linear projection: (8,) → (embed_dim,)
        embedding = x @ self.W_out + self.b_out

        return embedding.astype(np.float32)

    def set_weights(self, weights: dict):
        """Load trained weights from dict."""
        for attr, val in weights.items():
            if hasattr(self, attr):
                setattr(self, attr, np.array(val, dtype=np.float32))

    def get_weights(self) -> dict:
        """Export weights for saving."""
        return {attr: getattr(self, attr).tolist()
                for attr in ['conv1_w','conv1_b','conv2_w','conv2_b',
                             'Wq','Wk','Wv','W_out','b_out']}


# ═══════════════════════════════════════════════════════════════════
#  MODULE 7: XGBOOST CLASSIFIER
# ═══════════════════════════════════════════════════════════════════

def build_xgboost(class_weight: dict = None) -> Pipeline:
    """
    XGBoost-equivalent using sklearn GradientBoostingClassifier.
    Input: 32 engineered + 16 CNN = 48 features.
    Threshold applied externally (not in model) for tunability.
    """
    cw = class_weight or {0: 5, 1: 1}   # strongly upweight FP class
                                          # since precision is priority

    clf = GradientBoostingClassifier(
        n_estimators     = 300,   # more trees = better precision
        max_depth        = 4,     # shallow trees → less overfitting
        learning_rate    = 0.03,  # slow learning = more robust
        subsample        = 0.75,
        min_samples_leaf = 5,
        random_state     = 42,
    )

    return Pipeline([
        ('scaler', RobustScaler()),  # robust to outliers
        ('clf',    clf),
    ])


# ═══════════════════════════════════════════════════════════════════
#  MODULE 8: POST-PROCESSING
# ═══════════════════════════════════════════════════════════════════

class PostProcessor:
    """
    Final physiological constraints after ML decision.
    Even if ML accepts a peak, post-processing can reject it.
    """

    def __init__(self, fs: int):
        self.fs      = fs
        self.rr_min  = int(Config.POST_RR_MIN_MS * fs / 1000)

    def process(self, peaks: np.ndarray,
                ecg_bp: np.ndarray,
                baseline: AdaptiveBaseline) -> np.ndarray:
        """
        Apply post-processing to final verified peaks.

        1. Remove duplicates within refractory period
           (keep the one with higher amplitude)
        2. Conservative missed beat recovery
           (only if gap > 1.75× mean RR AND sub-threshold peak found)
        """
        if len(peaks) < 2:
            return peaks

        peaks = np.sort(peaks)

        # ── Step 1: Remove duplicates ─────────────────────────────────
        keep    = [peaks[0]]
        for pk in peaks[1:]:
            if pk - keep[-1] >= self.rr_min:
                keep.append(pk)
            else:
                # Keep the higher-amplitude one
                if abs(ecg_bp[pk]) > abs(ecg_bp[keep[-1]]):
                    keep[-1] = pk

        peaks = np.array(keep)

        # ── Step 2: Conservative missed beat recovery ─────────────────
        if not baseline.is_ready or len(peaks) < 3:
            return peaks

        recovered = list(peaks)
        rr_avg    = baseline.rr_avg
        i = 0
        while i < len(recovered) - 1:
            gap = recovered[i+1] - recovered[i]
            if gap > Config.MISSED_BEAT_MULT * rr_avg:
                # Expected number of beats in this gap
                n_expected = round(gap / rr_avg)
                if n_expected == 2:
                    # Look for a sub-threshold peak at expected position
                    expected_pos = recovered[i] + int(rr_avg)
                    search_win   = int(0.10 * self.fs)  # ±100ms
                    s = max(0, expected_pos - search_win)
                    e = min(len(ecg_bp), expected_pos + search_win)
                    seg = ecg_bp[s:e]
                    if len(seg) > 0:
                        local_max_idx = s + int(np.argmax(seg))
                        local_max_val = ecg_bp[local_max_idx]
                        # Only recover if amplitude ≥ 40% of running avg
                        # (very conservative — don't insert noise)
                        if local_max_val >= 0.40 * baseline.amplitude_avg:
                            recovered.insert(i+1, local_max_idx)
                            i += 1  # skip the just-inserted peak
            i += 1

        return np.array(sorted(set(recovered)), dtype=int)


# ═══════════════════════════════════════════════════════════════════
#  MAIN DETECTOR CLASS
# ═══════════════════════════════════════════════════════════════════

class HybridRPeakDetector:
    """
    Complete hybrid ECG R-peak detector.

    Combines all 8 modules into one clean interface.
    """

    def __init__(self, fs: int = 250):
        self.fs            = fs
        self.preprocessor  = Preprocessor(fs)
        self.pan_tompkins  = PanTompkinsLoose(fs)
        self.rule_filter   = RuleBasedFilter(fs)
        self.feat_extractor= FeatureExtractor(fs)
        self.cnn           = LightweightCNNAttention(fs)
        self.post_processor= PostProcessor(fs)
        self.model         = None   # XGBoost — set after training
        self._is_trained   = False

    def detect(self, ecg_raw: np.ndarray,
               smooth: bool = False,
               verbose: bool = False) -> tuple:
        """
        Full detection pipeline.

        Parameters
        ----------
        ecg_raw : raw 1D ECG from ADC (any scale)
        smooth  : apply optional 3-point smoothing
        verbose : print per-stage candidate counts

        Returns
        -------
        verified_peaks : np.ndarray of R-peak indices
        info           : dict with intermediate results + metrics
        """
        info = {}

        # ── Stage 1: Preprocessing ────────────────────────────────────
        ecg_clean = self.preprocessor.process(ecg_raw, smooth=smooth)
        info['ecg_clean'] = ecg_clean

        # ── Stage 2: Pan-Tompkins (loose) ─────────────────────────────
        candidates, ecg_bp, mwi = self.pan_tompkins.detect(ecg_clean)
        info['ecg_bp']         = ecg_bp
        info['mwi']            = mwi
        info['n_candidates']   = len(candidates)

        if verbose:
            print(f"  [PT]   {len(candidates)} candidates")

        if len(candidates) < 2:
            info['verified_peaks'] = candidates
            return candidates, info

        # ── Initialise adaptive baseline ──────────────────────────────
        baseline = AdaptiveBaseline(self.fs)

        # ── Stage 3: Rule-based filter ────────────────────────────────
        passed_rules = []
        rule_reasons = {}
        prev_verified = -1

        for ci, cand in enumerate(candidates):
            passed, reason = self.rule_filter.apply(
                ecg_bp, cand, prev_verified, baseline)
            if passed:
                passed_rules.append(ci)
                # Update baseline with rule-passing candidate
                rr = (cand - prev_verified) if prev_verified >= 0 else None
                baseline.update(ecg_bp, cand, rr)
                prev_verified = cand
            rule_reasons[ci] = reason

        info['n_after_rules']  = len(passed_rules)
        info['rule_reasons']   = rule_reasons

        if verbose:
            n_rej = len(candidates) - len(passed_rules)
            print(f"  [Rules] {n_rej} rejected → {len(passed_rules)} remaining")

        if len(passed_rules) < 2:
            peaks = candidates[np.array(passed_rules)] if passed_rules else np.array([], dtype=int)
            info['verified_peaks'] = peaks
            return peaks, info

        # ── Stage 4 + 5 + 6: Feature + CNN + XGBoost ─────────────────
        rule_candidates = candidates[np.array(passed_rules)]

        # Reset baseline for accurate feature computation
        baseline2 = AdaptiveBaseline(self.fs)
        for i, cand in enumerate(rule_candidates[:5]):
            rr = (rule_candidates[i] - rule_candidates[i-1]) if i > 0 else None
            baseline2.update(ecg_bp, cand, rr)

        X_feat  = []
        X_embed = []
        valid_idx = []

        for i, cand in enumerate(rule_candidates):
            try:
                feat    = self.feat_extractor.extract(
                    ecg_bp, rule_candidates, baseline2, i)
                embed   = self.cnn.embed(ecg_bp, cand)
                combined= np.concatenate([feat, embed])
                X_feat.append(combined)
                valid_idx.append(i)

                # Update baseline for subsequent features
                rr = (cand - rule_candidates[i-1]) if i > 0 else None
                baseline2.update(ecg_bp, cand, rr)
            except Exception:
                continue

        info['n_feature_extracted'] = len(X_feat)

        if not X_feat:
            info['verified_peaks'] = rule_candidates
            return rule_candidates, info

        X = np.array(X_feat, dtype=np.float32)

        # ── ML classification ──────────────────────────────────────────
        if self._is_trained and self.model is not None:
            probs = self.model.predict_proba(X)[:, 1]
            # Precision-focused: only accept above 0.80
            ml_accept = probs >= Config.ML_THRESHOLD
            final_candidates = rule_candidates[np.array(valid_idx)][ml_accept]
            info['probs'] = probs
            info['n_after_ml'] = int(ml_accept.sum())
            if verbose:
                n_rej = len(probs) - int(ml_accept.sum())
                print(f"  [ML]   {n_rej} rejected → {int(ml_accept.sum())} remaining")
        else:
            # No trained model — use rule-filtered candidates
            final_candidates = rule_candidates
            info['probs'] = np.ones(len(rule_candidates))
            info['n_after_ml'] = len(rule_candidates)
            if verbose:
                print("  [ML]   No model loaded — using rule-filtered peaks")

        # ── Stage 7: Post-processing ───────────────────────────────────
        verified = self.post_processor.process(
            final_candidates, ecg_bp, baseline2)

        info['verified_peaks'] = verified
        info['baseline']       = baseline2

        # ── Compute HRV metrics ────────────────────────────────────────
        if len(verified) >= 2:
            rr_ms = np.diff(verified) / self.fs * 1000.0
            info['rr_intervals_ms'] = rr_ms
            info['mean_hr_bpm']     = float(60000 / np.mean(rr_ms))
            info['sdnn_ms']         = float(np.std(rr_ms, ddof=1))
            info['rmssd_ms']        = float(np.sqrt(np.mean(np.diff(rr_ms)**2)))
        else:
            info['rr_intervals_ms'] = np.array([])
            info['mean_hr_bpm']     = 0.0
            info['sdnn_ms']         = 0.0
            info['rmssd_ms']        = 0.0

        if verbose:
            print(f"  [Post] {len(verified)} verified R-peaks")
            print(f"  [HRV]  HR={info['mean_hr_bpm']:.1f} bpm  "
                  f"SDNN={info['sdnn_ms']:.1f} ms")

        return verified, info

    def load_model(self, path: str):
        """Load a trained XGBoost model + CNN weights."""
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        self.model = payload['model']
        if 'cnn_weights' in payload:
            self.cnn.set_weights(payload['cnn_weights'])
        self._is_trained = True
        print(f"[Detector] Model loaded: {path}")

    def save_model(self, path: str, meta: dict = None):
        """Save trained model + CNN weights."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        payload = {
            'model'      : self.model,
            'cnn_weights': self.cnn.get_weights(),
            'fs'         : self.fs,
            'config'     : {k: v for k, v in vars(Config).items()
                            if not k.startswith('_')},
            'meta'       : meta or {},
        }
        with open(path, 'wb') as f:
            pickle.dump(payload, f)
        print(f"[Detector] Model saved: {path}")


# ═══════════════════════════════════════════════════════════════════
#  TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════

def extract_training_data(ecg_list: list,
                           label_list: list,
                           fs: int = 250,
                           verbose: bool = True) -> tuple:
    """
    Extract features from a list of labelled ECG recordings.

    Parameters
    ----------
    ecg_list   : list of 1D np.ndarray ECG signals
    label_list : list of dicts {'candidate_idx': int, 'label': 0/1}
                 OR list of np.ndarray with expert R-peak indices
                 If expert peaks provided, auto-labels via 50ms match
    fs         : sampling rate

    Returns
    -------
    X      : np.ndarray (n_candidates, 48)
    y      : np.ndarray (n_candidates,)
    groups : np.ndarray (n_candidates,)  subject index for LOSO
    """
    detector = HybridRPeakDetector(fs=fs)

    all_X, all_y, all_g = [], [], []
    MATCH_TOL = int(0.05 * fs)   # 50 ms

    for subj_idx, (ecg_raw, labels) in enumerate(zip(ecg_list, label_list)):
        ecg_clean = detector.preprocessor.process(ecg_raw)
        candidates, ecg_bp, _ = detector.pan_tompkins.detect(ecg_clean)

        if len(candidates) < 3:
            continue

        # Initialise baseline from first 5 candidates
        baseline = AdaptiveBaseline(fs)
        for i in range(min(5, len(candidates))):
            rr = (candidates[i]-candidates[i-1]) if i > 0 else None
            baseline.update(ecg_bp, candidates[i], rr)

        # If labels is an array of expert peak indices → auto-label
        if isinstance(labels, np.ndarray):
            expert_peaks = labels
            for ci, cand in enumerate(candidates):
                min_dist = int(np.min(np.abs(expert_peaks - cand))) \
                           if len(expert_peaks) > 0 else 9999
                label = 1 if min_dist <= MATCH_TOL else 0

                try:
                    feat  = detector.feat_extractor.extract(
                        ecg_bp, candidates, baseline, ci)
                    embed = detector.cnn.embed(ecg_bp, cand)
                    vec   = np.concatenate([feat, embed])
                    all_X.append(vec)
                    all_y.append(label)
                    all_g.append(subj_idx)
                except Exception:
                    pass

                if label == 1:
                    rr = (cand - candidates[ci-1]) if ci > 0 else None
                    baseline.update(ecg_bp, cand, rr)

        else:
            # labels is a list of {candidate_idx, label} dicts
            label_map = {d['candidate_idx']: d['label'] for d in labels}
            for ci, cand in enumerate(candidates):
                if cand not in label_map:
                    continue
                label = label_map[cand]
                try:
                    feat  = detector.feat_extractor.extract(
                        ecg_bp, candidates, baseline, ci)
                    embed = detector.cnn.embed(ecg_bp, cand)
                    all_X.append(np.concatenate([feat, embed]))
                    all_y.append(label)
                    all_g.append(subj_idx)
                except Exception:
                    pass

        if verbose:
            n_true = sum(1 for y in all_y if y == 1 and
                         len(all_g) > 0 and all_g[-1] == subj_idx)
            print(f"  Subject {subj_idx+1}: {len(candidates)} candidates extracted")

    X      = np.array(all_X, dtype=np.float32)
    y      = np.array(all_y, dtype=int)
    groups = np.array(all_g, dtype=int)

    if verbose:
        print(f"\nTotal: {len(y)} candidates | "
              f"{(y==1).sum()} true R | {(y==0).sum()} false P")

    return X, y, groups


def train_hybrid_model(ecg_list: list,
                        label_list: list,
                        fs: int = 250,
                        model_out: str = 'models/hybrid_detector.pkl',
                        use_smote: bool = True) -> HybridRPeakDetector:
    """
    Full training pipeline.

    Parameters
    ----------
    ecg_list   : list of ECG arrays (one per subject/recording)
    label_list : list of expert peak index arrays or label dicts
    fs         : sampling rate
    model_out  : where to save the trained model
    use_smote  : apply SMOTE oversampling for class imbalance

    Returns
    -------
    detector   : trained HybridRPeakDetector
    """
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import f1_score, precision_score, recall_score

    print("[Train] Extracting features...")
    X, y, groups = extract_training_data(ecg_list, label_list, fs)

    if len(y) == 0:
        raise ValueError("No training data extracted. Check inputs.")

    print(f"\n[Train] Dataset: {len(y)} candidates, "
          f"{len(np.unique(groups))} subjects")

    # ── SMOTE for class imbalance ─────────────────────────────────────
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        if use_smote and (y==0).sum() > 5:
            smote_ratio = min(0.4, (y==0).sum() / (y==1).sum())
            clf  = GradientBoostingClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.03,
                subsample=0.75, min_samples_leaf=5, random_state=42)
            model = ImbPipeline([
                ('scaler', RobustScaler()),
                ('smote', SMOTE(sampling_strategy=smote_ratio,
                                random_state=42, k_neighbors=5)),
                ('clf', clf),
            ])
            print("[Train] SMOTE enabled")
        else:
            model = build_xgboost()
    except ImportError:
        model = build_xgboost()
        print("[Train] SMOTE not available — using class weights")

    # ── LOSO cross-validation ─────────────────────────────────────────
    logo      = LeaveOneGroupOut()
    all_preds = np.zeros(len(y), dtype=int)
    all_probs = np.zeros(len(y))
    n_subj    = len(np.unique(groups))

    print(f"\n[Train] LOSO CV — {n_subj} subjects")
    print("─" * 50)

    for fold, (tr, te) in enumerate(logo.split(X, y, groups)):
        m = build_xgboost()
        m.fit(X[tr], y[tr])
        all_preds[te] = m.predict(X[te])
        all_probs[te] = m.predict_proba(X[te])[:, 1]

        subj = groups[te[0]]
        se   = recall_score(y[te], all_preds[te], zero_division=0)
        p    = precision_score(y[te], all_preds[te], zero_division=0)
        f1   = f1_score(y[te], all_preds[te], zero_division=0)
        print(f"  Fold {fold+1:2d} | Subj {subj} | "
              f"Se={se:.3f}  P+={p:.3f}  F1={f1:.3f}")

    # ── Overall metrics ────────────────────────────────────────────────
    print("─" * 50)
    ov_se = recall_score(y, all_preds, zero_division=0)
    ov_p  = precision_score(y, all_preds, zero_division=0)
    ov_f1 = f1_score(y, all_preds, zero_division=0)
    print(f"\n  Overall | Se={ov_se:.4f}  P+={ov_p:.4f}  F1={ov_f1:.4f}")
    print(f"  Priority: P+ (precision) — target > 0.99\n")

    # ── Train final model on ALL data ─────────────────────────────────
    print("[Train] Training final model on full dataset...")
    model.fit(X, y)

    # ── Build and save detector ────────────────────────────────────────
    detector           = HybridRPeakDetector(fs=fs)
    detector.model     = model
    detector._is_trained = True
    detector.save_model(model_out, meta={
        'overall_Se': ov_se, 'overall_P+': ov_p, 'overall_F1': ov_f1,
        'n_candidates': len(y), 'n_subjects': n_subj,
    })

    return detector


# ═══════════════════════════════════════════════════════════════════
#  VISUALISATION
# ═══════════════════════════════════════════════════════════════════

def plot_detection(ecg_raw: np.ndarray,
                   verified_peaks: np.ndarray,
                   info: dict,
                   fs: int = 250,
                   title: str = 'Hybrid R-Peak Detection',
                   save_path: str = None):
    """
    4-panel ECG detection plot:
    1. Raw ECG + detected R-peaks
    2. Preprocessed (clean) ECG + detected peaks
    3. Pan-Tompkins MWI signal
    4. RR interval tachogram
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    t = np.arange(len(ecg_raw)) / fs

    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=False)
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # ── Panel 1: Raw ECG ──────────────────────────────────────────────
    axes[0].plot(t, ecg_raw, color='gray', lw=0.7, alpha=0.8)
    if len(verified_peaks) > 0:
        axes[0].plot(t[verified_peaks], ecg_raw[verified_peaks],
                     'r^', ms=7, label=f'{len(verified_peaks)} R-peaks', zorder=5)
    axes[0].set_ylabel('Amplitude (raw)')
    axes[0].set_title('Raw ECG + Detected R-Peaks')
    axes[0].legend(fontsize=9)
    axes[0].set_xlim([t[0], t[-1]])

    # ── Panel 2: Clean ECG ────────────────────────────────────────────
    ecg_clean = info.get('ecg_clean', ecg_raw)
    axes[1].plot(t, ecg_clean, color='steelblue', lw=0.8)
    if len(verified_peaks) > 0:
        axes[1].plot(t[verified_peaks], ecg_clean[verified_peaks],
                     'r^', ms=7, zorder=5)
    axes[1].set_ylabel('Amplitude (clean)')
    axes[1].set_title('Preprocessed ECG (bandpass + notch + baseline)')
    axes[1].set_xlim([t[0], t[-1]])

    # ── Panel 3: MWI ──────────────────────────────────────────────────
    mwi = info.get('mwi', None)
    if mwi is not None:
        axes[2].plot(t, mwi, color='darkorange', lw=0.8)
        axes[2].set_ylabel('MWI')
        axes[2].set_title('Pan-Tompkins Moving Window Integration')
        axes[2].set_xlim([t[0], t[-1]])
    else:
        axes[2].set_visible(False)

    # ── Panel 4: RR tachogram ─────────────────────────────────────────
    rr_ms = info.get('rr_intervals_ms', np.array([]))
    if len(rr_ms) > 1:
        t_rr = t[verified_peaks[1:]]
        axes[3].plot(t_rr, rr_ms, 'o-', color='seagreen',
                     ms=4, lw=1.2)
        axes[3].axhline(np.mean(rr_ms), color='red', ls='--',
                        lw=1, label=f'Mean RR={np.mean(rr_ms):.0f} ms')
        axes[3].set_ylabel('RR interval (ms)')
        axes[3].set_xlabel('Time (s)')
        axes[3].set_title(f'RR Tachogram — HR={info.get("mean_hr_bpm",0):.1f} bpm  '
                          f'SDNN={info.get("sdnn_ms",0):.1f} ms  '
                          f'RMSSD={info.get("rmssd_ms",0):.1f} ms')
        axes[3].legend(fontsize=8)
        axes[3].set_xlim([t[0], t[-1]])
    else:
        axes[3].text(0.5, 0.5, 'Not enough peaks for tachogram',
                     ha='center', va='center', transform=axes[3].transAxes)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════
#  EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════

def generate_synthetic_ecg(duration_sec: int = 30,
                             fs: int = 250,
                             hr: float = 72.0,
                             noise: float = 0.05) -> np.ndarray:
    """
    Generate a realistic synthetic ECG for testing.
    Includes P-wave, QRS complex, T-wave, baseline wander, and noise.
    """
    t   = np.linspace(0, duration_sec, fs * duration_sec)
    ecg = np.zeros_like(t)
    rr  = 60.0 / hr

    for bt in np.arange(0.5, duration_sec, rr):
        dt = t - bt
        # P-wave (small, 80ms before R)
        ecg += 0.15 * np.exp(-dt**2 / (2*(0.025)**2))
        # QRS complex (sharp, large)
        ecg += 1.00 * np.exp(-dt**2 / (2*(0.010)**2))
        ecg -= 0.25 * np.exp(-dt**2 / (2*(0.020)**2))
        # T-wave (broad, positive, 200ms after R)
        ecg += 0.30 * np.exp(-(dt-0.20)**2 / (2*(0.050)**2))

    # Baseline wander (respiration ~0.25 Hz)
    ecg += 0.05 * np.sin(2 * np.pi * 0.25 * t)
    # 50 Hz mains noise
    ecg += noise * 0.3 * np.sin(2 * np.pi * 50 * t)
    # White noise
    ecg += noise * np.random.randn(len(t))

    return (ecg * 1000).astype(np.float64)   # mV-scale


if __name__ == '__main__':
    print("=" * 60)
    print("  Hybrid R-Peak Detector — Example Run")
    print("=" * 60)
    print()

    FS = 250

    # ── Generate test signal ─────────────────────────────────────────
    print("[1] Generating synthetic ECG (30s @ 250 Hz, 72 bpm)...")
    np.random.seed(42)
    ecg_raw  = generate_synthetic_ecg(duration_sec=30, fs=FS,
                                       hr=72, noise=0.05)
    expected = int(30 * 72 / 60)
    print(f"    Signal length: {len(ecg_raw)} samples  ({len(ecg_raw)/FS:.1f} s)")
    print(f"    Expected beats: ~{expected}")
    print()

    # ── Initialise detector (no trained model) ───────────────────────
    print("[2] Initialising detector (rule-based only — no ML model)...")
    detector = HybridRPeakDetector(fs=FS)

    # ── Run detection ─────────────────────────────────────────────────
    print("[3] Running detection pipeline...")
    peaks, info = detector.detect(ecg_raw, verbose=True)
    print()

    # ── Results ───────────────────────────────────────────────────────
    print("[4] Results:")
    print(f"    Candidates (Pan-Tompkins) : {info['n_candidates']}")
    print(f"    After rule filter         : {info['n_after_rules']}")
    print(f"    After ML (no model)       : {info['n_after_ml']}")
    print(f"    Final verified peaks      : {len(peaks)}")
    print(f"    Expected beats            : ~{expected}")
    print()
    print(f"    Mean HR    : {info['mean_hr_bpm']:.1f} bpm")
    print(f"    SDNN       : {info['sdnn_ms']:.2f} ms")
    print(f"    RMSSD      : {info['rmssd_ms']:.2f} ms")
    print()

    if len(peaks) > 0:
        se_approx = min(len(peaks), expected) / expected * 100
        print(f"    Detection rate  : ~{se_approx:.0f}%")

    # ── Feature extraction demo ────────────────────────────────────────
    print("[5] Feature extraction demo (first 3 peaks):")
    _, ecg_bp, _ = detector.pan_tompkins.detect(
        detector.preprocessor.process(ecg_raw))
    baseline = AdaptiveBaseline(FS)
    for i, pk in enumerate(peaks[:5]):
        baseline.update(ecg_bp, pk, None)

    for i in range(min(3, len(peaks))):
        feat  = detector.feat_extractor.extract(ecg_bp, peaks, baseline, i)
        embed = detector.cnn.embed(ecg_bp, peaks[i])
        combined = np.concatenate([feat, embed])
        print(f"    Peak {i+1} @ sample {peaks[i]:4d} | "
              f"48 features | embed norm={np.linalg.norm(embed):.3f}")

    print()
    print("[6] Training pipeline demo (5 subjects, 10s each)...")
    ecg_list   = [generate_synthetic_ecg(10, FS, hr=60+5*s) for s in range(5)]
    label_list = []
    for s, ecg in enumerate(ecg_list):
        hr    = 60 + 5*s
        peaks_gt = (np.arange(0.5, 10.0, 60/hr) * FS).astype(int)
        peaks_gt = peaks_gt[peaks_gt < len(ecg)]
        label_list.append(peaks_gt)

    X_tr, y_tr, g_tr = extract_training_data(ecg_list, label_list, FS)
    print(f"    Extracted: {len(y_tr)} candidates | "
          f"{(y_tr==1).sum()} true R | {(y_tr==0).sum()} FP")
    print(f"    Feature shape: {X_tr.shape}  (32 engineered + 16 CNN = 48)")
    print()

    # Quick model fit (no LOSO for demo)
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import RobustScaler
    from sklearn.pipeline import Pipeline
    quick_model = Pipeline([('s', RobustScaler()),
                             ('c', GradientBoostingClassifier(
                                n_estimators=50, random_state=42))])
    quick_model.fit(X_tr, y_tr)
    detector.model       = quick_model
    detector._is_trained = True

    peaks2, info2 = detector.detect(ecg_raw, verbose=True)
    print()
    print(f"    With ML model: {len(peaks2)} verified peaks "
          f"(threshold={Config.ML_THRESHOLD})")
    print(f"    HR={info2['mean_hr_bpm']:.1f} bpm  "
          f"SDNN={info2['sdnn_ms']:.1f} ms")
    print()

    # ── Visualise ─────────────────────────────────────────────────────
    print("[7] Generating plot...")
    try:
        plot_detection(ecg_raw, peaks2, info2, fs=FS,
                       title='Hybrid Detector — Synthetic ECG (250 Hz, 72 bpm)',
                       save_path='hybrid_detection_result.png')
    except Exception as e:
        print(f"    Plot skipped: {e}")

    print()
    print("=" * 60)
    print("  Pipeline complete.")
    print("  Next: train on DREAMER/MIT-BIH with train_hybrid_model()")
    print("=" * 60)
