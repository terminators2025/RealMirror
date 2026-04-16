# Copyright 2025 ZTE Corporation.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

import threading
import time
from collections import deque
from typing import Optional
from utility.logger import Logger


class MetricsCollector:

    def __init__(self, report_interval: float = 1.0):
        self.report_interval = report_interval
        self.lock = threading.Lock()

        self.request_count = 0
        self.batch_count = 0
        self.batch_sizes = deque(maxlen=1000)
        self.queue_latencies = deque(maxlen=1000)
        self.inference_times = deque(maxlen=1000)
        self.e2e_latencies = deque(maxlen=1000)

        self.window_start_time = time.time()
        self.last_report_time = time.time()

        self._stop_event = threading.Event()
        self._report_thread: Optional[threading.Thread] = None

    def start(self):
        self._report_thread = threading.Thread(target=self._report_loop, daemon=True)
        self._report_thread.start()
        Logger.info(f"MetricsCollector started with {self.report_interval}s interval")

    def stop(self):
        self._stop_event.set()
        if self._report_thread:
            self._report_thread.join()
        Logger.info("MetricsCollector stopped")

    def record_batch(self, batch_size: int):
        with self.lock:
            self.batch_count += 1
            self.request_count += batch_size
            self.batch_sizes.append(batch_size)

    def record_queue_latency(self, latency_ms: float):
        with self.lock:
            self.queue_latencies.append(latency_ms)

    def record_inference_time(self, time_ms: float):
        with self.lock:
            self.inference_times.append(time_ms)

    def record_e2e_latency(self, latency_ms: float):
        with self.lock:
            self.e2e_latencies.append(latency_ms)

    def _report_loop(self):
        while not self._stop_event.is_set():
            time.sleep(self.report_interval)
            self._report_metrics()

    def _report_metrics(self):
        with self.lock:
            current_time = time.time()
            window_duration = current_time - self.window_start_time

            if self.request_count == 0:
                return

            avg_batch_size = sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0
            qps = self.request_count / window_duration if window_duration > 0 else 0

            queue_lat_stats = self._compute_stats(self.queue_latencies, "Queue")

            inference_stats = self._compute_stats(self.inference_times, "Inference")

            e2e_stats = self._compute_stats(self.e2e_latencies, "E2E")

            Logger.info(
                f"[METRICS] Window: {window_duration:.1f}s | "
                f"Requests: {self.request_count} | "
                f"QPS: {qps:.2f} | "
                f"Batches: {self.batch_count} | "
                f"Avg Batch Size: {avg_batch_size:.1f}"
            )

            if queue_lat_stats:
                Logger.info(f"[METRICS] {queue_lat_stats}")

            if inference_stats:
                Logger.info(f"[METRICS] {inference_stats}")

            if e2e_stats:
                Logger.info(f"[METRICS] {e2e_stats}")

            self.request_count = 0
            self.batch_count = 0
            self.batch_sizes.clear()
            self.queue_latencies.clear()
            self.inference_times.clear()
            self.e2e_latencies.clear()
            self.window_start_time = current_time

    def _compute_stats(self, data: deque, label: str) -> Optional[str]:
        if not data:
            return None

        data_list = list(data)
        data_sorted = sorted(data_list)
        count = len(data_list)

        avg = sum(data_list) / count
        min_val = data_sorted[0]
        max_val = data_sorted[-1]
        p50 = data_sorted[int(count * 0.5)]
        p95 = data_sorted[int(count * 0.95)]
        p99 = data_sorted[int(count * 0.99)]

        return (
            f"{label} Latency (ms): "
            f"Avg={avg:.2f}, "
            f"Min={min_val:.2f}, "
            f"P50={p50:.2f}, "
            f"P95={p95:.2f}, "
            f"P99={p99:.2f}, "
            f"Max={max_val:.2f}"
        )
