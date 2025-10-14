# Copyright 2025 ZTE Corporation.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

import omni.ui as ui
from typing import Optional, Tuple, Set
from utility.logger import Logger


class DataCollectionVisualizer:
    """
    Data Collection Position Visualizer.
    Provides a UI window to display current and historical data collection positions.
    """

    def __init__(
        self,
        window_width: int = 400,
        window_height: int = 400,
        grid_size: int = 20,
        dot_radius: int = 8,
    ):
        """
        Initializes the visualizer.

        Args:
            window_width: Window width in pixels.
            window_height: Window height in pixels.
            grid_size: Pixel size of each grid point.
            dot_radius: Radius of the dot in pixels.
        """
        self.window_width = window_width
        self.window_height = window_height
        self.grid_size = grid_size
        self.dot_radius = dot_radius

        # Window and canvas objects
        self.window: Optional[ui.Window] = None
        self.canvas: Optional[ui.VStack] = None
        self.position_label: Optional[ui.Label] = None

        # Position tracking
        self.current_position: Tuple[int, int] = (0, 0)
        self.collection_history: Set[Tuple[int, int]] = set()

        # Create window
        self._create_window()

    def _build_header(self):
        """Builds the header section of the window."""
        # Create title
        ui.Label(
            "Collection Position History (XY Plane)",
            alignment=ui.Alignment.CENTER,
            style={
                "font_size": 18,
                "color": 0xFFFFFFFF,
                "font_weight": "bold",
            },
        )

        # Create description text
        with ui.HStack(height=30):
            ui.Spacer(width=10)
            ui.Label(
                "O Current Position",
                style={
                    "font_size": 14,
                    "color": 0xFF0000FF,
                    "font_weight": "bold",
                },
            )
            ui.Spacer()
            ui.Label(
                "* Collected Positions",
                style={
                    "font_size": 14,
                    "color": 0xFF00FF00,
                    "font_weight": "bold",
                },
            )
            ui.Spacer(width=10)

        # Add current position display
        self.position_label = ui.Label(
            f"Current: {self.current_position}",
            alignment=ui.Alignment.CENTER,
            style={
                "font_size": 16,
                "color": 0xFFFFFFFF,
                "font_weight": "bold",
            },
        )

    def _build_grid_view(self):
        """Builds the grid view section of the window."""
        # Create a scrolling frame to accommodate grids that may be out of view
        with ui.ScrollingFrame(
            height=self.window_height - 120,
            style={"background_color": 0xFF202020},
        ):
            # Create grid container
            self.canvas = ui.VStack(
                alignment=ui.Alignment.CENTER, spacing=0
            )

            # Initial draw
            self._update_grid_display()

    def _create_window(self):
        """Create the visualization window."""
        try:
            self.window = ui.Window(
                "Data Collection Position Tracker",
                width=self.window_width,
                height=self.window_height,
                position_x=100,
                position_y=100,
            )

            with self.window.frame:
                with ui.VStack():
                    self._build_header()
                    self._build_grid_view()

            Logger.info("Data collection position visualization window created.")

        except Exception as e:
            Logger.error(f"Failed to create visualization window: {e}")
            self.window = None

    def _get_cell_display_info(self, x: int, y: int, styles: dict) -> Tuple[str, dict]:
        """Gets the character and style for a grid cell."""
        pos = (x, y)
        if pos == self.current_position:
            return "O", styles["current_pos"]
        if pos in self.collection_history:
            return "*", styles["history_pos"]
        if x == 0 and y == 0:
            return "+", styles["origin"]
        if x == 0:
            return "|", styles["y_axis"]
        if y == 0:
            return "-", styles["x_axis"]
        return ".", styles["empty"]

    def _update_grid_display(self):
        """Update the grid display."""
        if not self.canvas:
            return

        styles = {
            "current_pos": {"font_size": 16, "color": 0xFF0000FF, "margin": 2, "font_weight": "bold"},
            "history_pos": {"font_size": 16, "color": 0xFF00FF00, "margin": 2, "font_weight": "bold"},
            "origin": {"font_size": 14, "color": 0xFF808080, "margin": 2, "font_weight": "bold"},
            "y_axis": {"font_size": 16, "color": 0xFF404040, "margin": 0, "font_weight": "bold"},
            "x_axis": {"font_size": 16, "color": 0xFF404040, "margin": 0, "font_weight": "bold"},
            "empty": {"font_size": 8, "color": 0xFF303030, "margin": 2, "font_weight": "bold"},
            "y_label": {"font_size": 10, "color": 0xFF606060, "font_weight": "bold"},
            "y_label_zero": {"font_size": 10, "color": 0xFFFFFF00, "font_weight": "bold"},
            "x_label": {"font_size": 12, "color": 0xFF606060, "margin": 2, "font_weight": "bold"},
            "x_label_zero": {"font_size": 12, "color": 0xFFFFFF00, "margin": 2, "font_weight": "bold"},
            "x_label_empty": {"font_size": 12, "margin": 2, "font_weight": "bold"},
        }

        self.canvas.clear()

        with self.canvas:
            all_positions = list(self.collection_history) + [self.current_position]
            if not all_positions:
                ui.Label("No data", style={"color": 0xFF808080})
                return

            min_x = min(pos[0] for pos in all_positions) - 2
            max_x = max(pos[0] for pos in all_positions) + 2
            min_y = min(pos[1] for pos in all_positions) - 2
            max_y = max(pos[1] for pos in all_positions) + 2

            # Draw grid
            for y in range(max_y, min_y - 1, -1):
                with ui.HStack(spacing=0):
                    ui.Spacer(width=10)
                    style = styles["y_label_zero"] if y == 0 else styles["y_label"]
                    ui.Label(f"{y:3d} ", style=style)
                    for x in range(min_x, max_x + 1):
                        char, style = self._get_cell_display_info(x, y, styles)
                        ui.Label(char, style=style)
                    ui.Spacer()

            # X-axis labels
            with ui.HStack(spacing=0):
                ui.Spacer(width=40)
                for x in range(min_x, max_x + 1):
                    if x == 0:
                        label, style = f" {x} ", styles["x_label_zero"]
                    elif abs(x) % 5 == 0:
                        label, style = f" {x} ", styles["x_label"]
                    else:
                        label, style = "   ", styles["x_label_empty"]
                    ui.Label(label, style=style)
                ui.Spacer()

            ui.Spacer(height=10)

            # Statistics
            with ui.HStack():
                ui.Spacer()
                ui.Label(
                    f"Current Position: ({self.current_position[0]}, {self.current_position[1]})",
                    style={
                        "font_size": 14,
                        "color": 0xFFFFFFFF,
                        "font_weight": "bold",
                    },
                )
                ui.Spacer()
                ui.Label(
                    f"Total Collected: {len(self.collection_history)} points",
                    style={
                        "font_size": 14,
                        "color": 0xFF00FF00,
                        "font_weight": "bold",
                    },
                )
                ui.Spacer()

    def update_position(self, dx: int, dy: int):
        """
        Updates the current collection position.

        Args:
            dx: Displacement in the X-axis direction.
            dy: Displacement in the Y-axis direction.
        """
        # Update position
        new_x = self.current_position[0] + dx
        new_y = self.current_position[1] + dy
        self.current_position = (new_x, new_y)

        # Update label
        if self.position_label:
            self.position_label.text = f"Current: {self.current_position}"

        # Redraw
        self._update_grid_display()

        Logger.info(f"Visualizer position updated to: ({new_x}, {new_y})")

    def mark_collection_point(self):
        """
        Marks the current position as a collected point.
        """
        self.collection_history.add(self.current_position)
        self._update_grid_display()
        Logger.info(f"Position {self.current_position} has been marked as a collection point.")

    def reset_current_position(self):
        """
        Resets the current position to the origin.
        """
        self.current_position = (0, 0)
        if self.position_label:
            self.position_label.text = f"Current: {self.current_position}"
        self._update_grid_display()
        Logger.info("Current position has been reset to the origin.")

    def clear_history(self):
        """
        Clears all historical collection points.
        """
        self.collection_history.clear()
        self._update_grid_display()
        Logger.info("Historical collection points have been cleared.")

    def destroy(self):
        """
        Destroys the visualization window.
        """
        if self.window:
            try:
                self.window.destroy()
                self.window = None
                self.canvas = None
                self.position_label = None
                Logger.info("Visualization window has been closed.")
            except Exception as e:
                Logger.error(f"Error while closing visualization window: {e}")

    def is_valid(self) -> bool:
        """
        Checks if the window is valid.

        Returns:
            Whether the window is valid.
        """
        return self.window is not None

    def get_current_position(self) -> Tuple[int, int]:
        """
        Gets the current position.

        Returns:
            The current position coordinates.
        """
        return self.current_position

    def get_collection_count(self) -> int:
        """
        Gets the number of collected points.

        Returns:
            The number of collected points.
        """
        return len(self.collection_history)
