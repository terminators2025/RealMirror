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
import json
import math
import os
from typing import Any, Dict, Optional

from utility.logger import Logger

from .base_controller import BaseController


class PicoController(BaseController):

    def __init__(self, data_file_path: str, latency_measurement=None):
        self.data_file_path = data_file_path
        self.previous_states: Dict[str, Dict[str, Any]] = {
            "right": {"buttons": {}, "axes": []},
            "left": {"buttons": {}, "axes": []},
        }
        self.latency_measurement = latency_measurement
        self._last_logged_seq = -1

    def read_data(self) -> Optional[Dict[str, Any]]:

        if not os.path.exists(self.data_file_path):
            return None

        try:
            with open(self.data_file_path, "r") as f:
                data = json.load(f)

            if not isinstance(data, dict) or "data" not in data:
                return None

            # Record latency measurement if enabled
            if self.latency_measurement:
                latency_point = self.latency_measurement.record_received_data(data)
                if latency_point and latency_point.seq != self._last_logged_seq:
                    # Log latency info occasionally (every 100 packets)
                    if latency_point.seq % 100 == 0:
                        Logger.info(
                            f"Latency [seq={latency_point.seq}]: "
                            f"Server={latency_point.server_processing_latency:.2f}ms, "
                            f"File->Python={latency_point.file_to_python_latency:.2f}ms, "
                            f"Total={latency_point.total_estimated_latency:.2f}ms"
                        )
                    self._last_logged_seq = latency_point.seq

            result = {"left": None, "right": None}
            for controller_data in data.get("data", []):
                handedness = controller_data.get("handedness")
                if handedness in ["left", "right"]:
                    result[handedness] = controller_data

            return result

        except json.JSONDecodeError:
            return None
        except Exception as e:
            Logger.error(f"Failed to read PICO data: {e}")
            return None

    def get_buttons(self, controller_data: Dict) -> Dict[str, bool]:

        if not controller_data or "buttons" not in controller_data:
            return {}

        buttons = controller_data.get("buttons", [])
        result = {}

        button_mapping = {
            0: "Trigger",
            1: "Squeeze",
            3: "Thumbstick",
            4: "button_x/a",
            5: "button_y/b",
        }

        for idx, name in button_mapping.items():
            if idx < len(buttons):
                result[name] = buttons[idx].get("pressed", False)

        return result

    def get_axes(self, controller_data: Dict) -> list:

        if not controller_data:
            return []
        return list(controller_data.get("axes") or [])

    def get_position(self, controller_data: Dict) -> Optional[list]:

        if not controller_data:
            return None

        pose_data = controller_data.get("pose", {})
        position = pose_data.get("position")

        if position and len(position) == 3:
            try:
                values = [float(p) for p in position]
            except (TypeError, ValueError):
                return None
            if not all(math.isfinite(v) for v in values):
                Logger.warning("Rejecting PICO position with NaN/Inf: %s", values)
                return None
            return values
        return None

    def get_orientation(self, controller_data: Dict) -> Optional[list]:

        if not controller_data:
            return None

        pose_data = controller_data.get("pose", {})
        orientation = pose_data.get("orientation")

        if orientation and len(orientation) == 4:
            try:
                values = [float(o) for o in orientation]
            except (TypeError, ValueError):
                return None
            if not all(math.isfinite(v) for v in values):
                Logger.warning("Rejecting PICO orientation with NaN/Inf: %s", values)
                return None
            return values
        return None

    def get_latency_stats(self) -> Optional[Dict]:
        """Get current latency statistics if measurement is enabled"""
        if self.latency_measurement:
            stats = self.latency_measurement.get_statistics()
            return stats if isinstance(stats, dict) else None
        return None

    def print_latency_stats(self):
        """Print latency statistics to console"""
        if self.latency_measurement:
            self.latency_measurement.print_statistics()
        else:
            Logger.info("Latency measurement is not enabled")

    def export_latency_data(self, filepath: str):
        """Export latency data to JSON file"""
        if self.latency_measurement:
            self.latency_measurement.export_to_json(filepath)
        else:
            Logger.warning("Latency measurement is not enabled, cannot export data")
