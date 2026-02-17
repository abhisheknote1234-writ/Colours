import csv
import json
import os
import socket
import threading
import time
from collections import deque
from datetime import datetime

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

# --- CONFIGURATION ---
UDP_IP = "0.0.0.0"
UDP_PORT = int(os.getenv("ECG_UDP_PORT", "5007"))
MAX_HISTORY = 800
HRV_HISTORY = 160
RECORD_FILE = "ecg_research_session.csv"
SESSION_META_FILE = "ecg_research_session_meta.json"
PROCESSED_METRICS_FILE = "processed_metrics_log.csv"
SMOOTHING_LEVEL = 5

# --- DETECTION / QUALITY TUNING ---
MOTION_THRESHOLD = 600
MIN_RR_MS = 300
MAX_RR_MS = 2000
BANDPASS_LOW_HZ = 5.0
BANDPASS_HIGH_HZ = 20.0
INTEGRATION_WINDOW_SEC = 0.12
REFINE_WINDOW_SEC = 0.08
SQI_ARTIFACT_THRESHOLD = 0.30
BASELINE_CALIBRATION_SEC = 60.0
BASELINE_MIN_SAMPLES = 30
HRV_SLOPE_WINDOW = 20
HRV_VARIANCE_WINDOW = 20


class AnalyticECGResearch:
    def __init__(self):
        # Streaming state
        self.data_buffer = deque(maxlen=MAX_HISTORY)
        self.timestamps = deque(maxlen=MAX_HISTORY)

        # Recording + messages
        self.is_recording = False
        self.recorded_rows = []
        self.bookmarks = []
        self.current_note = ""
        self.message_history = deque(maxlen=6)
        self.pending_events = deque()

        # Physiology state
        self.rr_history = deque(maxlen=35)
        self.hrv35_trend = deque(maxlen=HRV_HISTORY)
        self.bpm = 0.0
        self.hrv7 = 0.0
        self.hrv35 = 0.0
        self.hrv35_slope = 0.0
        self.hrv35_variance = 0.0
        self.latest_rr = 0.0
        self.last_beat_time = 0.0

        # Quality/artifact state
        self.is_artifact = False
        self.artifact_counter = 0
        self.noise_floor = 0.0
        self.signal_level = 1.0
        self.sample_rate_hz = 250.0
        self.sqi = 0.0
        self.snr_db = 0.0

        # Adaptive baseline state
        self.session_started_at = None
        self.baseline_sqi_mean = None
        self.baseline_sqi_std = None
        self.baseline_amp_mean = None
        self.baseline_amp_std = None
        self.baseline_ready = False
        self.baseline_sqi_samples = []
        self.baseline_amp_samples = []

        # Runtime
        self.bound_udp_port = UDP_PORT
        self.ani = None

        # IO locks/writers
        self.metrics_log_lock = threading.Lock()
        self.metrics_log_fp = None
        self.metrics_log_writer = None
        self._init_metrics_log_file()

        self._setup_ui()
        self.thread = threading.Thread(target=self.udp_listener, daemon=True)
        self.thread.start()

    def _init_metrics_log_file(self):
        file_exists = os.path.exists(PROCESSED_METRICS_FILE)
        self.metrics_log_fp = open(PROCESSED_METRICS_FILE, "a", newline="", encoding="utf-8", buffering=1)
        self.metrics_log_writer = csv.DictWriter(
            self.metrics_log_fp,
            fieldnames=[
                "Timestamp",
                "BPM",
                "HRV7",
                "HRV35",
                "HRV35_Slope",
                "HRV35_Variance",
                "SQI",
                "SNR_dB",
                "ArtifactFlag",
                "EventType",
                "EventContent",
            ],
        )
        if not file_exists:
            self.metrics_log_writer.writeheader()

    def _queue_event(self, event_type: str, event_content: str):
        with self.metrics_log_lock:
            self.pending_events.append({"timestamp": time.time(), "event_type": event_type, "event_content": event_content})

    def _consume_pending_event(self):
        with self.metrics_log_lock:
            if not self.pending_events:
                return "", ""
            event = self.pending_events.popleft()
        return event["event_type"], event["event_content"]

    def _log_processed_metrics(self, timestamp: float):
        event_type, event_content = self._consume_pending_event()
        if self.metrics_log_writer is None:
            return
        with self.metrics_log_lock:
            self.metrics_log_writer.writerow(
                {
                    "Timestamp": f"{timestamp:.6f}",
                    "BPM": f"{self.bpm:.4f}",
                    "HRV7": f"{self.hrv7:.4f}",
                    "HRV35": f"{self.hrv35:.4f}",
                    "HRV35_Slope": f"{self.hrv35_slope:.6f}",
                    "HRV35_Variance": f"{self.hrv35_variance:.6f}",
                    "SQI": f"{self.sqi:.6f}",
                    "SNR_dB": f"{self.snr_db:.6f}",
                    "ArtifactFlag": int(self.is_artifact),
                    "EventType": event_type,
                    "EventContent": event_content,
                }
            )
            self.metrics_log_fp.flush()

    def _setup_ui(self):
        plt.style.use("default")
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title("ECG Research Dashboard")

        gs = self.fig.add_gridspec(
            nrows=3,
            ncols=2,
            width_ratios=[4.2, 1.8],
            height_ratios=[3.2, 2.0, 1.2],
        )

        self.ax_ecg = self.fig.add_subplot(gs[0, 0])
        self.ax_hrv = self.fig.add_subplot(gs[1, 0], sharex=None)
        self.ax_metrics = self.fig.add_subplot(gs[0:2, 1])
        self.ax_chat = self.fig.add_subplot(gs[2, :])

        self.line_ecg, = self.ax_ecg.plot([], [], color="tab:blue", linewidth=1.4)
        self.peaks_plot, = self.ax_ecg.plot([], [], "o", color="red", markersize=4)
        self.ax_ecg.set_xlim(0, MAX_HISTORY)
        self.ax_ecg.set_ylim(-300, 300)
        self.ax_ecg.set_title("ECG Waveform", fontsize=13, fontweight="bold", pad=10)
        self.ax_ecg.set_ylabel("Amplitude", fontsize=10)
        self.ax_ecg.grid(True, alpha=0.25)

        self.line_hrv35, = self.ax_hrv.plot([], [], color="tab:orange", linewidth=1.4)
        self.ax_hrv.set_xlim(0, HRV_HISTORY)
        self.ax_hrv.set_ylim(0, 180)
        self.ax_hrv.set_title("HRV35 Trend", fontsize=12, fontweight="bold", pad=8)
        self.ax_hrv.set_ylabel("RMSSD (ms)", fontsize=10)
        self.ax_hrv.set_xlabel("Samples", fontsize=10)
        self.ax_hrv.grid(True, alpha=0.25)

        self.ax_metrics.set_axis_off()
        self.ax_metrics.set_title("Metrics", fontsize=13, fontweight="bold", loc="left", pad=8)
        self.text_status = self.ax_metrics.text(0.02, 0.96, "", fontsize=9, va="top", transform=self.ax_metrics.transAxes)

        self.metric_value_text = {}
        metric_rows = [
            ("BPM", 0.86),
            ("HRV7", 0.76),
            ("HRV35", 0.66),
            ("HRV35 Slope", 0.56),
            ("HRV35 Variance", 0.46),
            ("SQI", 0.36),
            ("Artifact", 0.26),
        ]
        for label, y in metric_rows:
            self.ax_metrics.text(0.03, y, label, fontsize=10, fontweight="bold", ha="left", va="center", transform=self.ax_metrics.transAxes)
            self.metric_value_text[label] = self.ax_metrics.text(
                0.97,
                y,
                "--",
                fontsize=10,
                ha="right",
                va="center",
                transform=self.ax_metrics.transAxes,
            )

        self.ax_chat.set_axis_off()
        self.ax_chat.set_title("Chat", fontsize=11, fontweight="bold", loc="left", pad=6)
        self.text_chat = self.ax_chat.text(0.01, 0.08, "", fontsize=10, transform=self.ax_chat.transAxes, va="bottom", family="DejaVu Sans")

        self.fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.06, wspace=0.18, hspace=0.28)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def _finalize_baseline_from_samples(self):
        if len(self.baseline_sqi_samples) >= BASELINE_MIN_SAMPLES:
            sqi_arr = np.array(self.baseline_sqi_samples, dtype=float)
            amp_arr = np.array(self.baseline_amp_samples, dtype=float)
            self.baseline_sqi_mean = float(np.mean(sqi_arr))
            self.baseline_sqi_std = float(np.std(sqi_arr))
            self.baseline_amp_mean = float(np.mean(amp_arr))
            self.baseline_amp_std = float(np.std(amp_arr))
        else:
            self.baseline_sqi_mean = SQI_ARTIFACT_THRESHOLD
            self.baseline_sqi_std = 0.0
            self.baseline_amp_mean = MOTION_THRESHOLD
            self.baseline_amp_std = 0.0
        self.baseline_ready = True

    def _handle_phone_message(self, message: str):
        text = message.split(":", 1)[1].strip() if ":" in message else ""
        self.message_history.appendleft(f"📱 {text}")
        self._queue_event("MESSAGE", text)

    def _handle_scale_reply(self, message: str):
        normalized = message.replace("SURVEY_REPLY:", "SCALE_REPLY:", 1)
        parts = normalized.split(":", 2)
        score = parts[1].strip() if len(parts) > 1 else ""
        text = parts[2].strip() if len(parts) > 2 else ""
        if text:
            display = f"SCALE {score} | {text}"
            event_content = text
        else:
            display = f"SCALE {score}"
            event_content = normalized
        self.message_history.appendleft(f"PHONE: {display}")
        self._queue_event("SCALE", event_content)

    def udp_listener(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        bind_ok = False
        bind_error = None
        for port in range(UDP_PORT, UDP_PORT + 15):
            try:
                sock.bind((UDP_IP, port))
                self.bound_udp_port = port
                bind_ok = True
                break
            except OSError as error:
                bind_error = error

        if not bind_ok:
            print(f"UDP bind failed: {bind_error}")
            return

        while True:
            try:
                data, _ = sock.recvfrom(1024)
                message = data.decode("utf-8").strip()

                if message == "SESSION_START":
                    self.text_status.set_text("Session sync received")
                    self._queue_event("MESSAGE", "SESSION_START")
                    continue

                if message.startswith("PHONE_MESSAGE:"):
                    self._handle_phone_message(message)
                    continue

                if message.startswith("SCALE_REPLY:") or message.startswith("SURVEY_REPLY:"):
                    self._handle_scale_reply(message)
                    continue

                if message.startswith("LAPTOP_MESSAGE:"):
                    msg = message.split(":", 1)[1]
                    self.message_history.appendleft(f"💻 {msg}")
                    self._queue_event("MESSAGE", msg)
                    continue

                if message.startswith("SCALE_REQUEST:"):
                    self.message_history.appendleft(f"📤 {message}")
                    self._queue_event("SCALE", message)
                    continue

                if message == "BASELINE_START":
                    self.session_started_at = time.time()
                    self.baseline_ready = False
                    self.baseline_sqi_samples = []
                    self.baseline_amp_samples = []
                    self.baseline_sqi_mean = None
                    self.baseline_sqi_std = None
                    self.baseline_amp_mean = None
                    self.baseline_amp_std = None
                    self.text_status.set_text("Baseline calibration synced")
                    self._queue_event("MESSAGE", "BASELINE_START")
                    continue

                if message == "BASELINE_COMPLETE":
                    if not self.baseline_ready:
                        self._finalize_baseline_from_samples()
                    self.text_status.set_text("Baseline complete (synced)")
                    self._queue_event("MESSAGE", "BASELINE_COMPLETE")
                    continue

                val = float(message)
                self.data_buffer.append(val)
                self.timestamps.append(time.time())
            except Exception:
                pass

    @staticmethod
    def calc_rmssd(rr_list):
        if len(rr_list) < 2:
            return 0.0
        diff = np.diff(np.array(rr_list, dtype=float))
        return float(np.sqrt(np.mean(diff**2)))

    @staticmethod
    def calculate_slope(data):
        if len(data) < 5:
            return 0.0
        y = np.array(data, dtype=float)
        x = np.arange(len(y))
        return float(np.polyfit(x, y, 1)[0])

    @staticmethod
    def calculate_variance(data):
        if len(data) < 5:
            return 0.0
        return float(np.var(np.array(data, dtype=float)))

    @staticmethod
    def estimate_sampling_rate(times):
        if len(times) < 3:
            return 250.0
        dt = np.diff(np.array(times, dtype=float))
        median_dt = np.median(dt)
        if median_dt <= 0:
            return 250.0
        return float(np.clip(1.0 / median_dt, 100.0, 1000.0))

    @staticmethod
    def bandpass_filter(signal, fs):
        nyq = fs * 0.5
        low = BANDPASS_LOW_HZ
        high = min(BANDPASS_HIGH_HZ, nyq * 0.9)
        if len(signal) < 16 or low >= high:
            return signal
        b, a = butter(2, [low / nyq, high / nyq], btype="band")
        return filtfilt(b, a, signal)

    def estimate_signal_quality(self, filtered, integrated, peak_indices):
        if len(filtered) < 3:
            self.sqi = 0.0
            self.snr_db = 0.0
            return

        peak_energy = np.mean(integrated[peak_indices]) if len(peak_indices) else 0.0
        baseline = np.percentile(integrated, 25)
        dynamic = max(np.percentile(integrated, 95) - baseline, 1e-6)
        self.sqi = float(np.clip((peak_energy - baseline) / dynamic, 0.0, 1.0))

        signal_power = np.mean(filtered**2)
        noise_component = filtered - np.mean(filtered)
        noise_power = max(np.var(noise_component) * (1.0 - self.sqi + 0.1), 1e-8)
        self.snr_db = float(10.0 * np.log10(max(signal_power, 1e-8) / noise_power))

    def detect_r_peaks_advanced(self, raw, times):
        self.sample_rate_hz = self.estimate_sampling_rate(times)
        filtered = self.bandpass_filter(raw, self.sample_rate_hz)

        diff = np.diff(filtered, prepend=filtered[0])
        squared = diff**2
        win = max(3, int(INTEGRATION_WINDOW_SEC * self.sample_rate_hz))
        integrated = np.convolve(squared, np.ones(win) / win, mode="same")

        baseline = np.percentile(integrated, 25)
        top = np.percentile(integrated, 95)
        self.noise_floor = 0.95 * self.noise_floor + 0.05 * baseline
        self.signal_level = 0.90 * self.signal_level + 0.10 * max(top, baseline + 1e-6)
        threshold = self.noise_floor + 0.35 * (self.signal_level - self.noise_floor)

        refractory_samples = max(int((MIN_RR_MS / 1000.0) * self.sample_rate_hz), 1)
        candidates, _ = find_peaks(
            integrated,
            height=threshold,
            distance=refractory_samples,
            prominence=max(np.std(integrated) * 0.5, 1e-6),
        )

        search_radius = max(1, int(REFINE_WINDOW_SEC * self.sample_rate_hz))
        refined = []
        for idx in candidates:
            left = max(0, idx - search_radius)
            right = min(len(filtered), idx + search_radius + 1)
            local_idx = left + int(np.argmax(filtered[left:right]))
            refined.append(local_idx)

        refined = np.unique(np.array(refined, dtype=int)) if len(refined) else np.array([], dtype=int)
        self.estimate_signal_quality(filtered, integrated, refined)
        return filtered, refined

    def process_data(self):
        if len(self.data_buffer) < 50:
            return False, np.array([]), np.array([], dtype=int)

        raw = np.array(self.data_buffer, dtype=float)
        times = np.array(self.timestamps, dtype=float)

        filtered, peaks = self.detect_r_peaks_advanced(raw, times)

        kernel = np.ones(SMOOTHING_LEVEL) / SMOOTHING_LEVEL
        smoothed = np.convolve(filtered, kernel, mode="valid")
        smooth_offset = SMOOTHING_LEVEL // 2
        peaks = peaks - smooth_offset
        peaks = peaks[(peaks >= 0) & (peaks < len(smoothed))]

        rr_interval_candidate = None
        if len(peaks) > 1:
            smooth_times = times[-len(smoothed):]
            newest_peak_time = smooth_times[peaks[-1]]
            if self.last_beat_time != 0 and newest_peak_time > self.last_beat_time:
                rr_interval_candidate = (newest_peak_time - self.last_beat_time) * 1000.0

        current_amp = float(np.max(raw) - np.min(raw))

        elapsed = 0.0
        if self.session_started_at is not None:
            elapsed = time.time() - self.session_started_at

        is_calibrating = self.is_recording and not self.baseline_ready and elapsed < BASELINE_CALIBRATION_SEC
        if is_calibrating:
            self.baseline_sqi_samples.append(self.sqi)
            self.baseline_amp_samples.append(current_amp)

        if self.is_recording and not self.baseline_ready and elapsed >= BASELINE_CALIBRATION_SEC:
            self._finalize_baseline_from_samples()

        if self.baseline_ready and self.baseline_sqi_mean is not None and self.baseline_amp_mean is not None:
            sqi_threshold = self.baseline_sqi_mean - 3.0 * self.baseline_sqi_std
            amp_threshold = self.baseline_amp_mean + 3.0 * self.baseline_amp_std
        else:
            sqi_threshold = SQI_ARTIFACT_THRESHOLD
            amp_threshold = MOTION_THRESHOLD

        sqi_check = self.sqi < sqi_threshold
        range_check = current_amp > amp_threshold
        rr_jump_check = False
        if rr_interval_candidate is not None and len(self.rr_history) >= 1:
            rr_jump_check = abs(rr_interval_candidate - self.rr_history[-1]) > 250.0

        artifact_condition = sqi_check or (range_check and rr_jump_check)
        if is_calibrating:
            artifact_condition = False

        if artifact_condition:
            self.artifact_counter += 1
        else:
            self.artifact_counter = max(0, self.artifact_counter - 1)

        self.is_artifact = self.artifact_counter >= 3

        beat_found = False
        if not self.is_artifact and len(peaks) > 1:
            smooth_times = times[-len(smoothed):]
            newest_peak_time = smooth_times[peaks[-1]]

            if newest_peak_time > self.last_beat_time:
                beat_found = True
                rr_interval = (newest_peak_time - self.last_beat_time) * 1000.0

                if MIN_RR_MS < rr_interval < MAX_RR_MS and self.last_beat_time != 0:
                    self.latest_rr = rr_interval
                    self.rr_history.append(rr_interval)
                    self.bpm = 60000.0 / rr_interval

                    rr = list(self.rr_history)
                    self.hrv7 = self.calc_rmssd(rr[-7:])
                    self.hrv35 = self.calc_rmssd(rr[-35:])
                    self.hrv35_trend.append(self.hrv35)
                    self.hrv35_slope = self.calculate_slope(list(self.hrv35_trend)[-HRV_SLOPE_WINDOW:])
                    self.hrv35_variance = self.calculate_variance(list(self.hrv35_trend)[-HRV_VARIANCE_WINDOW:])
                    self._log_processed_metrics(newest_peak_time)

                    if self.is_recording:
                        self.recorded_rows.append(
                            {
                                "Time": newest_peak_time,
                                "SampleRateHz": self.sample_rate_hz,
                                "BPM": self.bpm,
                                "RR_ms": self.latest_rr,
                                "HRV_7": self.hrv7,
                                "HRV_35": self.hrv35,
                                "HRV35_Slope": self.hrv35_slope,
                                "HRV35_Variance": self.hrv35_variance,
                                "SQI": self.sqi,
                                "SNR_dB": self.snr_db,
                                "Artifact": self.is_artifact,
                                "Note": self.current_note,
                                "Bookmark": False,
                            }
                        )

                self.last_beat_time = newest_peak_time

        return beat_found, smoothed, peaks

    def update_plot(self, _frame):
        if len(self.data_buffer) <= 50:
            return (
                self.line_ecg,
                self.peaks_plot,
                self.line_hrv35,
                self.text_status,
                self.text_chat,
                *self.metric_value_text.values(),
            )

        _, smoothed_line, peak_indices = self.process_data()

        if len(smoothed_line) > 0:
            self.line_ecg.set_data(range(len(smoothed_line)), smoothed_line)
            ymin, ymax = np.percentile(smoothed_line, [2, 98])
            pad = max(30.0, (ymax - ymin) * 0.25)
            self.ax_ecg.set_ylim(ymin - pad, ymax + pad)
            if self.is_artifact:
                self.line_ecg.set_color("gray")
                self.peaks_plot.set_data([], [])
            else:
                self.line_ecg.set_color("tab:blue")
                if len(peak_indices) > 0:
                    self.peaks_plot.set_data(peak_indices, smoothed_line[peak_indices])

        if len(self.hrv35_trend) > 0:
            self.line_hrv35.set_data(range(len(self.hrv35_trend)), self.hrv35_trend)
            if self.hrv35 > 180:
                self.ax_hrv.set_ylim(0, self.hrv35 + 20)

        self.metric_value_text["BPM"].set_text(f"{self.bpm:.0f}")
        self.metric_value_text["HRV7"].set_text(f"{self.hrv7:.1f}")
        self.metric_value_text["HRV35"].set_text(f"{self.hrv35:.1f}")
        self.metric_value_text["HRV35 Slope"].set_text(f"{self.hrv35_slope:.4f}")
        self.metric_value_text["HRV35 Variance"].set_text(f"{self.hrv35_variance:.2f}")
        self.metric_value_text["SQI"].set_text(f"{self.sqi:.2f}")
        if self.is_artifact:
            self.metric_value_text["Artifact"].set_text("ARTIFACT")
            self.metric_value_text["Artifact"].set_color("tab:red")
        else:
            self.metric_value_text["Artifact"].set_text("CLEAN")
            self.metric_value_text["Artifact"].set_color("tab:green")

        self.text_status.set_text("r: start/stop recording | b: bookmark")

        chat_lines = []
        for msg in list(self.message_history):
            line = msg.replace("📱", "PHONE:").replace("💻", "LAPTOP:").replace("📤", "REQUEST:")
            chat_lines.append(line)
        self.text_chat.set_text("\n".join(chat_lines))

        return (
            self.line_ecg,
            self.peaks_plot,
            self.line_hrv35,
            self.text_status,
            self.text_chat,
            *self.metric_value_text.values(),
        )

    def on_key(self, event):
        if event.key == "h":
            print("Controls: r=start/stop recording | b=bookmark")

        if event.key == "b":
            if not self.current_note:
                self.current_note = "Manual bookmark"
            bookmark = {
                "Time": time.time(),
                "SampleRateHz": self.sample_rate_hz,
                "BPM": self.bpm,
                "RR_ms": self.latest_rr,
                "HRV_7": self.hrv7,
                "HRV_35": self.hrv35,
                "HRV35_Slope": self.hrv35_slope,
                "HRV35_Variance": self.hrv35_variance,
                "SQI": self.sqi,
                "SNR_dB": self.snr_db,
                "Artifact": self.is_artifact,
                "Note": self.current_note,
                "Bookmark": True,
            }
            self.bookmarks.append(bookmark)
            if self.is_recording:
                self.recorded_rows.append(bookmark)
            self.text_status.set_text("Bookmark saved")

        if event.key == "r":
            self.is_recording = not self.is_recording
            if self.is_recording:
                self.session_started_at = time.time()
                self.recorded_rows = []
                self.bookmarks = []
                self.rr_history.clear()
                self.hrv35_trend.clear()
                self.artifact_counter = 0
                self.baseline_ready = False
                self.baseline_sqi_samples = []
                self.baseline_amp_samples = []
                self.baseline_sqi_mean = None
                self.baseline_sqi_std = None
                self.baseline_amp_mean = None
                self.baseline_amp_std = None
                self.text_status.set_text("Recording started")
            else:
                self.save_session()

    def save_session(self):
        if not self.recorded_rows:
            return

        df = pd.DataFrame(self.recorded_rows)
        df.to_csv(RECORD_FILE, index=False)

        meta = {
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "session_started_at_epoch": self.session_started_at,
            "rows": int(len(df)),
            "bookmarks": int(sum(df.get("Bookmark", pd.Series(dtype=bool)).fillna(False))),
            "udp_ip": UDP_IP,
            "udp_port": self.bound_udp_port,
            "max_history": MAX_HISTORY,
            "detector": {
                "bandpass_low_hz": BANDPASS_LOW_HZ,
                "bandpass_high_hz": BANDPASS_HIGH_HZ,
                "min_rr_ms": MIN_RR_MS,
                "max_rr_ms": MAX_RR_MS,
                "integration_window_sec": INTEGRATION_WINDOW_SEC,
                "refine_window_sec": REFINE_WINDOW_SEC,
            },
        }
        with open(SESSION_META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def start(self):
        self.ani = animation.FuncAnimation(
            self.fig,
            self.update_plot,
            interval=30,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()


if __name__ == "__main__":
    app = AnalyticECGResearch()
    app.start()
