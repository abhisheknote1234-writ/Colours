import argparse
import csv
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# --- CONFIGURATION ---
LAPTOP_IP = "0.0.0.0"  # Listen on all local interfaces
PHONE_IP = "192.168.1.XX"  # Change to your phone IP; auto-learn from incoming packets if left as-is
DATA_PORT = 5005  # Laptop listens for ECG + replies here
COMMAND_PORT = 5006  # Phone listens for outbound commands here
FORWARD_ECG_IP = "127.0.0.1"
FORWARD_ECG_PORT = 5007  # Forward ECG packets here for local dashboard
FORWARD_ECG_PORT_SPAN = 15  # Must match dashboard fallback bind window
RAW_LOG_CSV_FILE = "raw_ecg_log.csv"
RECV_BUFFER = 2048
ECG_PRINT_EVERY = 50  # reduce console overhead for long runs


@dataclass
class BridgeStats:
    packets_total: int = 0
    ecg_packets: int = 0
    scale_replies: int = 0
    phone_messages: int = 0
    forwarded_ecg_packets: int = 0
    parse_errors: int = 0


class ResearchBridge:
    """Receives ECG/replies from phone, forwards ECG locally, and supports two-way messaging."""

    def __init__(
        self,
        laptop_ip: str,
        phone_ip: str,
        data_port: int,
        command_port: int,
        forward_ecg_ip: str,
        forward_ecg_port: int,
        forward_ecg_port_span: int,
        raw_log_csv_file: str,
    ):
        self.laptop_ip = laptop_ip
        self.phone_ip = phone_ip
        self.data_port = data_port
        self.command_port = command_port
        self.forward_ecg_ip = forward_ecg_ip
        self.forward_ecg_port = forward_ecg_port
        self.forward_ecg_port_span = max(1, int(forward_ecg_port_span))
        self.raw_log_csv_file = Path(raw_log_csv_file)

        self.running = False
        self.stats = BridgeStats()
        self.last_phone_addr: Optional[Tuple[str, int]] = None
        self.baseline_start_time: Optional[float] = None

        self.data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.data_sock.bind((self.laptop_ip, self.data_port))

        self.command_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.forward_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.listener_thread = threading.Thread(target=self._listener_loop, daemon=True)

        self._log_lock = threading.Lock()
        self._raw_log_fp = None
        self._raw_log_writer = None
        self._ecg_print_counter = 0
        self._init_raw_log_file()

    def _init_raw_log_file(self):
        file_exists = self.raw_log_csv_file.exists()
        self._raw_log_fp = self.raw_log_csv_file.open("a", newline="", encoding="utf-8", buffering=1)
        self._raw_log_writer = csv.DictWriter(
            self._raw_log_fp,
            fieldnames=["Timestamp", "RawECG", "EventType", "Direction", "EventContent"],
        )
        if not file_exists:
            self._raw_log_writer.writeheader()

    def _log_raw_event(
        self,
        timestamp: float,
        direction: str,
        event_type: str,
        event_content: str,
        raw_ecg: str = "",
    ):
        if self._raw_log_writer is None:
            return
        with self._log_lock:
            self._raw_log_writer.writerow(
                {
                    "Timestamp": f"{timestamp:.6f}",
                    "RawECG": raw_ecg,
                    "EventType": event_type,
                    "Direction": direction,
                    "EventContent": event_content,
                }
            )

    def start(self):
        self.running = True
        self.listener_thread.start()
        print(f"✅ ResearchBridge listening on {self.laptop_ip}:{self.data_port}")
        print(f"📤 Command target port: {self.command_port}")
        high = self.forward_ecg_port + self.forward_ecg_port_span - 1
        print(f"🔁 Forwarding ECG to {self.forward_ecg_ip}:{self.forward_ecg_port}-{high}")
        print(f"🗂️ Raw ECG/event log file: {self.raw_log_csv_file}")

    def stop(self):
        self.running = False
        for s in (self.data_sock, self.command_sock, self.forward_sock):
            try:
                s.close()
            except OSError:
                pass
        try:
            if self._raw_log_fp:
                self._raw_log_fp.close()
        except OSError:
            pass
        print("🛑 ResearchBridge stopped")

    def _resolve_phone_ip(self) -> Optional[str]:
        if self.phone_ip and "XX" not in self.phone_ip:
            return self.phone_ip
        if self.last_phone_addr:
            return self.last_phone_addr[0]
        return None

    def _forward_to_dashboard(self, payload: str):
        data = payload.encode("utf-8")
        for port in range(self.forward_ecg_port, self.forward_ecg_port + self.forward_ecg_port_span):
            self.forward_sock.sendto(data, (self.forward_ecg_ip, port))

    def _listener_loop(self):
        while self.running:
            try:
                data, addr = self.data_sock.recvfrom(RECV_BUFFER)
            except OSError:
                break

            self.last_phone_addr = addr
            payload = data.decode("utf-8", errors="replace").strip()
            ts = time.time()
            self.stats.packets_total += 1

            if payload.startswith("SCALE_REPLY:") or payload.startswith("SURVEY_REPLY:"):
                self.stats.scale_replies += 1
                normalized = payload.replace("SURVEY_REPLY:", "SCALE_REPLY:", 1)
                self._log_raw_event(ts, "phone_to_laptop", "SCALE", normalized)
                self._forward_to_dashboard(normalized)
                print(f"[{ts:.3f}] 📝 {normalized}")
                continue

            if payload.startswith("PHONE_MESSAGE:"):
                self.stats.phone_messages += 1
                self._log_raw_event(ts, "phone_to_laptop", "MESSAGE", payload)
                self._forward_to_dashboard(payload)
                print(f"[{ts:.3f}] 💬 {payload}")
                continue

            if payload == "SESSION_START":
                self._log_raw_event(ts, "phone_to_laptop", "MESSAGE", payload)
                print(f"[{ts:.3f}] 🎬 Initializing session log...")
                self._forward_to_dashboard(payload)
                continue
            if payload == "BASELINE_START":
                self.baseline_start_time = time.time()
                self._log_raw_event(ts, "phone_to_laptop", "MESSAGE", payload)
                print(f"[{ts:.3f}] ⏳ 60s Calibration Syncing... Resetting buffers.")
                self._forward_to_dashboard(payload)
                continue
            if payload == "BASELINE_COMPLETE":
                elapsed = None if self.baseline_start_time is None else (time.time() - self.baseline_start_time)
                self._log_raw_event(ts, "phone_to_laptop", "MESSAGE", payload)
                if elapsed is None:
                    print(f"[{ts:.3f}] ✅ Calibration complete. Locking thresholds.")
                else:
                    print(f"[{ts:.3f}] ✅ Calibration complete after {elapsed:.1f}s. Locking thresholds.")
                self._forward_to_dashboard(payload)
                continue

            try:
                ecg_value = float(payload)
                self.stats.ecg_packets += 1
                self._log_raw_event(ts, "phone_to_laptop", "ECG", "", raw_ecg=payload)
                self._forward_to_dashboard(payload)
                self.stats.forwarded_ecg_packets += 1
                self._ecg_print_counter += 1
                if self._ecg_print_counter % ECG_PRINT_EVERY == 0:
                    print(f"[{ts:.3f}] ❤️ ECG {ecg_value} (every {ECG_PRINT_EVERY}th shown)")
            except ValueError:
                self.stats.parse_errors += 1
                self._log_raw_event(ts, "phone_to_laptop", "UNPARSED", payload)
                print(f"[{ts:.3f}] ⚠️ Unparsed packet from {addr[0]}:{addr[1]} -> {payload}")

    def _send_to_phone(self, payload: str, event_type: str):
        ip = self._resolve_phone_ip()
        if not ip:
            print("❌ Phone IP unknown. Set PHONE_IP or wait for an incoming packet from phone.")
            return
        ts = time.time()
        self.command_sock.sendto(payload.encode("utf-8"), (ip, self.command_port))
        self._log_raw_event(ts, "laptop_to_phone", event_type, payload)
        self._forward_to_dashboard(payload)
        print(f"📨 Sent to {ip}:{self.command_port} -> {payload}")

    def send_scale_request(self, label: str, question_text: str):
        self._send_to_phone(f"SCALE_REQUEST:{label}:{question_text}", "SCALE")

    def send_message(self, message_text: str):
        self._send_to_phone(f"LAPTOP_MESSAGE:{message_text}", "MESSAGE")

    def send_ping(self):
        self._send_to_phone("PING", "MESSAGE")

    def add_personal_note(self, note_text: str):
        self._log_raw_event(time.time(), "laptop_to_phone", "MESSAGE", f"PERSONAL_NOTE:{note_text}")
        print(f"🧷 Personal note saved: {note_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UDP research bridge for phone/laptop ECG workflow")
    parser.add_argument("--interactive", action="store_true", help="Enable command prompt for scale/message/ping/stats/note")
    parser.add_argument("--forward-port", type=int, default=FORWARD_ECG_PORT, help="First local UDP port to forward ECG packets")
    parser.add_argument("--forward-span", type=int, default=FORWARD_ECG_PORT_SPAN, help="Number of consecutive local ports to forward to")
    parser.add_argument("--log-file", default=RAW_LOG_CSV_FILE, help="CSV file for raw ECG + message timeline")
    args = parser.parse_args()

    bridge = ResearchBridge(
        laptop_ip=LAPTOP_IP,
        phone_ip=PHONE_IP,
        data_port=DATA_PORT,
        command_port=COMMAND_PORT,
        forward_ecg_ip=FORWARD_ECG_IP,
        forward_ecg_port=args.forward_port,
        forward_ecg_port_span=args.forward_span,
        raw_log_csv_file=args.log_file,
    )

    try:
        bridge.start()
        if args.interactive:
            print("ℹ️ Commands: scale <label> <question>, message <text>, note <text>, ping, stats, quit")
            while True:
                cmd = input("bridge> ").strip()
                if not cmd:
                    continue
                if cmd in {"quit", "exit"}:
                    break
                if cmd == "ping":
                    bridge.send_ping()
                    continue
                if cmd == "stats":
                    print(bridge.stats)
                    continue
                if cmd.startswith("scale "):
                    parts = cmd.split(" ", 2)
                    if len(parts) < 3:
                        print("Usage: scale <label> <question text>")
                        continue
                    _, label, question = parts
                    bridge.send_scale_request(label, question)
                    continue
                if cmd.startswith("message "):
                    text = cmd[len("message "):].strip()
                    if not text:
                        print("Usage: message <text>")
                        continue
                    bridge.send_message(text)
                    continue
                if cmd.startswith("note "):
                    text = cmd[len("note "):].strip()
                    if not text:
                        print("Usage: note <text>")
                        continue
                    bridge.add_personal_note(text)
                    continue
                print("Unknown command. Use: scale <label> <question>, message <text>, note <text>, ping, stats, quit")
        else:
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        bridge.stop()
