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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
from typing import List, Dict, Optional, Tuple
import math
import os
from utility.logger import Logger


def load_area_points_from_txt(file_path: str) -> Optional[np.ndarray]:
    """
    Loads area boundary points from a text file.

    Args:
        file_path: Path to the text file containing area points.

    Returns:
        A numpy array of area points, or None if it fails.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Area definition file not found: {file_path}")
    try:
        points = np.loadtxt(file_path, comments='#', usecols=(1, 2))
        Logger.info(f"Successfully loaded {len(points)} coordinate points from {file_path}.")
        return points
    except Exception as e:
        Logger.error(f"Failed to load area definition file: {e}")
        return None


def generate_uniform_grid_points(points: np.ndarray, num_samples: int) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    Generates uniformly distributed grid points within a given area.

    Args:
        points: The points defining the area boundary.
        num_samples: The number of sample points to generate.

    Returns:
        (Array of sample points, bounding box dictionary) or (None, None) on failure.
    """
    if points is None or len(points) == 0:
        return None, None

    # Calculate bounding box
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    bbox = {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}

    width = x_max - x_min
    height = y_max - y_min

    if width == 0 or height == 0:
        # If the area is a line, generate random points
        rand_x = np.random.uniform(x_min, x_max, size=num_samples)
        rand_y = np.random.uniform(y_min, y_max, size=num_samples)
        combined_points = np.vstack([rand_x, rand_y]).T
        Logger.info(f"Area is zero, randomly generated {len(combined_points)} sample points on the boundary.")
        return combined_points, bbox

    # Calculate the optimal grid size
    aspect_ratio = width / height
    ideal_ny = math.sqrt(num_samples / aspect_ratio)
    ideal_nx = math.sqrt(num_samples * aspect_ratio)

    # Try different grid size combinations
    pairs = [
        (math.floor(ideal_nx), math.floor(ideal_ny)),
        (math.floor(ideal_nx), math.ceil(ideal_ny)),
        (math.ceil(ideal_nx), math.floor(ideal_ny)),
        (math.ceil(ideal_nx), math.ceil(ideal_ny)),
    ]

    nx_grid, ny_grid = 1, 1
    max_grid_points = 0

    for nx, ny in pairs:
        if nx < 1 or ny < 1:
            continue
        num_grid_points = nx * ny
        if max_grid_points < num_grid_points <= num_samples:
            max_grid_points = num_grid_points
            nx_grid, ny_grid = nx, ny
        elif num_grid_points == max_grid_points and max_grid_points > 0:
            current_aspect = nx_grid / ny_grid if ny_grid > 0 else float('inf')
            new_aspect = nx / ny if ny > 0 else float('inf')
            if abs(new_aspect - aspect_ratio) < abs(current_aspect - aspect_ratio):
                nx_grid, ny_grid = nx, ny

    # Generate grid points
    if nx_grid * ny_grid == 1 and num_samples > 1:
        grid_points = np.array([]).reshape(0, 2)
    else:
        x_coords = np.linspace(x_min, x_max, num=nx_grid)
        y_coords = np.linspace(y_min, y_max, num=ny_grid)
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    # If grid points are not enough, randomly generate the rest
    num_random_points = num_samples - len(grid_points)
    random_points = np.array([]).reshape(0, 2)

    if num_random_points > 0:
        Logger.info(
            f"Grid generated {len(grid_points)} points ({nx_grid}x{ny_grid}), now randomly sampling the remaining {num_random_points} points."
        )
        rand_x = np.random.uniform(x_min, x_max, size=num_random_points)
        rand_y = np.random.uniform(y_min, y_max, size=num_random_points)
        random_points = np.vstack([rand_x, rand_y]).T

    # Combine all points
    final_points = np.vstack([grid_points, random_points])

    Logger.info(
        f"Generated {len(final_points)} sample points in the area [{x_min:.2f}, {x_max:.2f}] x [{y_min:.2f}, {y_max:.2f}]."
    )
    return final_points, bbox


def generate_results_plot(
    points: np.ndarray, results: np.ndarray, bbox: Dict, output_path: str, model_type: str = "N/A"
):
    """
    Generates a beautiful, publication-ready success/failure distribution scatter plot
    based on evaluation results, with a success point heatmap overlay.
    """
    if points is None or results is None or len(points) == 0 or len(results) == 0:
        Logger.error("No valid data, cannot generate results plot.")
        return

    points_arr = np.array(points)
    results_arr = np.array(results).astype(bool)

    successful_points = points_arr[results_arr]
    failed_points = points_arr[~results_arr]

    # --- Aesthetic Parameters ---
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12

    success_color = '#3498db'  # Soft Blue
    failure_color = '#e74c3c'  # Soft Red

    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
    ax.set_facecolor('white')

    # --- 1. Draw Success Rate Heatmap Background ---
    if len(successful_points) > 0:
        heatmap_res = 200
        y_range = np.linspace(bbox['y_min'], bbox['y_max'], heatmap_res)
        x_range = np.linspace(bbox['x_min'], bbox['x_max'], heatmap_res)
        success_grid = np.zeros((heatmap_res, heatmap_res))

        y_indices = np.clip(np.searchsorted(y_range, successful_points[:, 1]) - 1, 0, heatmap_res - 1)
        x_indices = np.clip(np.searchsorted(x_range, successful_points[:, 0]) - 1, 0, heatmap_res - 1)

        success_grid[x_indices, y_indices] = 1

        # --- Adaptive Sigma Adjustment ---
        # Dynamically adjust the degree of Gaussian blur based on the number of successful points
        # Fewer points -> larger sigma (wider blur), more points -> smaller sigma (focused blur)
        num_successful = len(successful_points)

        # Define input range (number of points) and output range (sigma divisor)
        # These values can be adjusted as needed to change the blur effect
        points_range = [1, 400]  # Expected min and max range of points
        divisor_range = [2.5, 20]  # Corresponding sigma divisor range (smaller value = more blur)

        # Use linear interpolation to calculate the optimal divisor for the current number of points
        divisor = np.interp(num_successful, points_range, divisor_range)
        sigma = heatmap_res / divisor
        Logger.info(
            f"Based on {num_successful} successful points, sigma was automatically adjusted to {sigma:.2f} (divisor is {divisor:.2f})"
        )
        # --- End of Adaptive Adjustment ---

        # Use mode='constant' to avoid numerical anomalies caused by boundary reflections
        density_map = gaussian_filter(success_grid, sigma=sigma, mode='constant')

        if np.max(density_map) > 0:
            density_map /= np.max(density_map)

        im = ax.imshow(
            density_map,
            extent=[bbox['y_min'], bbox['y_max'], bbox['x_min'], bbox['x_max']],
            origin='lower',
            cmap='RdYlBu',
            alpha=0.5,
            zorder=0,
            aspect='auto',
            vmin=0,
            vmax=1,
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Success Probability', rotation=270, labelpad=20, fontsize=14)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
        cbar.ax.tick_params(labelsize=10)

    # --- 2. Draw Grid Lines and Bounding Box ---
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgray', zorder=1)
    rect = patches.Rectangle(
        (bbox['y_min'], bbox['x_min']),
        bbox['y_max'] - bbox['y_min'],
        bbox['x_max'] - bbox['x_min'],
        linewidth=1.5,
        edgecolor='black',
        linestyle='--',
        facecolor='none',
        label='Sampling Area',
        zorder=2,
    )
    ax.add_patch(rect)

    # --- 3. Draw Scatter Plot ---
    if len(failed_points) > 0:
        ax.scatter(
            failed_points[:, 1],
            failed_points[:, 0],
            c=failure_color,
            marker='x',  # type: ignore
            s=50,
            label=f'Failure ({len(failed_points)})',
            zorder=3,
        )

    if len(successful_points) > 0:
        ax.scatter(
            successful_points[:, 1],
            successful_points[:, 0],
            facecolors='none',
            edgecolors=success_color,
            marker='o',  # type: ignore
            s=50,
            alpha=0.8,
            linewidths=1.0,
            label=f'Success ({len(successful_points)})',
            zorder=4,
        )

    # --- 4. Set Title, Labels, and Legend ---
    ax.set_title(f"Success/Failure Distribution of {model_type.upper()} Model", pad=20)
    ax.set_xlabel("Y Coordinate (meters)")
    ax.set_ylabel("X Coordinate (meters)")

    # Set axis limits, adding some padding
    x_pad = (bbox['x_max'] - bbox['x_min']) * 0.05
    y_pad = (bbox['y_max'] - bbox['y_min']) * 0.05
    ax.set_xlim(bbox['y_max'] + y_pad, bbox['y_min'] - y_pad)  # Invert the Y-axis (the X-axis of the plot)
    ax.set_ylim(bbox['x_min'] - x_pad, bbox['x_max'] + x_pad)

    ax.set_aspect('equal', adjustable='box')

    ax.legend(
        loc='upper right', bbox_to_anchor=(1.2, -0.025), ncol=3, fancybox=True, handletextpad=0.1, columnspacing=0.6
    )

    # --- 5. Save the Image ---
    try:
        plt.savefig(output_path, format='svg', bbox_inches='tight', pad_inches=0.1)
        Logger.info(f"Evaluation results plot saved to: {output_path}")
    except Exception as e:
        Logger.error(f"Failed to save results plot: {e}")
    finally:
        plt.close(fig)
