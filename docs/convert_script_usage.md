# HDF5 to LeRobot Dataset Converter - User Guide

## Overview

The `script/convert.py` tool converts teleoperation data collected by Isaac Sim from HDF5 format to LeRobot dataset format, enabling seamless integration with the training pipeline.

## Key Features

- **HDF5 to LeRobot Format Conversion**: Transforms `teleop.py` collected data into LeRobot training-ready format
- **Batch Processing**: Convert multiple HDF5 files in a single run
- **Parallel Processing**: Utilize multi-process parallelization for efficient episode processing
- **Arc-to-Gear Conversion** (Optional): Convert hand joint values from radians to gear positions (0-2000 range)
- **Image Preprocessing**: Automatically resize camera images to 256×256 pixels
- **Statistics Calculation**: Compute min/max/mean/std statistics for each episode
- **Metadata Generation**: Automatically generate all required LeRobot metadata files

## Basic Usage

```bash
python3 script/convert.py <output_dir> --data_pairs <data_pairs.json> [--arc2gear]
```

### Command-Line Arguments

| Argument | Required | Description | Default |
|----------|----------|-------------|---------|
| `output_dir` | Yes | Output directory path for the converted LeRobot dataset | - |
| `--data_pairs` | Yes | Path to data pairs JSON file containing HDF5 file paths and task names | - |
| `--arc2gear` | No | Enable arc-to-gear conversion for hand joints | False |
| `--robot_config` | No | Custom robot configuration file path | `comm_config/configs/a2_robot_config.json` |
| `--fps` | No | Dataset frame rate | 15 |
| `--max_workers` | No | Maximum number of parallel worker processes | 24 |

## Data Pairs File Format

Create a JSON file specifying the HDF5 files to convert and their corresponding task descriptions:

```json
[
    {
        "h5_path": "/path/to/recorded_data_Task1.hdf5",
        "task_name": "Put the red potato chips on the left into the yellow basket on the right"
    },
    {
        "h5_path": "/path/to/recorded_data_Task2.hdf5",
        "task_name": "Transfer the cup from left to right"
    }
]
```

**Important Notes:**
- Each entry must contain both `h5_path` (absolute path to HDF5 file) and `task_name` (natural language task description)
- Task names should be descriptive and meaningful for training purposes
- Multiple HDF5 files can reference the same task name for multi-episode tasks

## Usage Examples

### Example 1: Single Task with Arc-to-Gear Conversion

```bash
python3 script/convert.py /path/to/output \
    --data_pairs Task1_data_pair.json \
    --arc2gear
```

This command converts a single task's HDF5 data with hand joint arc-to-gear conversion enabled.

### Example 2: Multiple Tasks Without Gear Conversion

```bash
python3 script/convert.py /path/to/output \
    --data_pairs multi_task_data_pair.json
```

This command converts multiple tasks while keeping joint values in radians.

### Example 3: Custom Configuration

```bash
python3 script/convert.py /path/to/output \
    --data_pairs data.json \
    --robot_config /path/to/custom_robot_config.json \
    --fps 30 \
    --max_workers 16
```

This command uses a custom robot configuration, higher frame rate, and reduced parallelism.

## Output Structure

The converted dataset follows the LeRobot format structure:

```
output_dir/
├── data/
│   ├── episode_000000.parquet    # Episode data in Parquet format
│   ├── episode_000001.parquet
│   └── ...
└── meta/
    ├── info.json                  # Dataset summary information
    ├── episodes.jsonl             # Per-episode metadata
    ├── episodes_stats.jsonl       # Per-episode statistics
    └── tasks.jsonl                # Task index to task name mapping
```

### Metadata Files Description

| File | Description |
|------|-------------|
| `info.json` | Overall dataset information (total episodes, frames, tasks, feature definitions, etc.) |
| `episodes.jsonl` | Basic information for each episode (index, task, length) |
| `episodes_stats.jsonl` | Statistical information for each episode (min/max/mean/std for each feature) |
| `tasks.jsonl` | Mapping from task indices to task names |

## Technical Details

### Joint Ordering

The script organizes 26 degrees of freedom (DOF) in the following order:

1. **Left Arm**: 7 joints (idx13-idx19)
2. **Right Arm**: 7 joints (idx20-idx26)
3. **Left Hand**: 6 joints (thumb_0, thumb_1, index, middle, ring, pinky)
4. **Right Hand**: 6 joints (thumb_0, thumb_1, index, middle, ring, pinky)

### Arc-to-Gear Conversion

When `--arc2gear` is enabled:

- **Arm Joints**: Remain in radian values (unchanged)
- **Hand Joints**:
  - **thumb_0** (thumb swing joint): Fixed value of **1195**
  - **Other finger joints**: Linear mapping from radians to 0-2000 range, rounded to nearest integer

**Conversion Formula:**
```python
gear_value = int(round((radian_value / max_radian) * 2000))
```

### Camera Image Processing

All three camera streams are processed uniformly:
- `left_wrist_camera_bgr`
- `right_wrist_camera_bgr`
- `head_camera_bgr`

**Processing Steps:**
1. Read raw image data from HDF5
2. Resize to 256×256 pixels using Pillow's LANCZOS interpolation
3. Save as uint8 format in Parquet files

### Task Index Management

Task indices are assigned based on first occurrence order:
- Same task names receive the same index
- Indices start from 0 and increment sequentially
- Consistent indexing maintained even with duplicate tasks across episodes

### Parallel Processing

The converter uses `ProcessPoolExecutor` for efficient parallel processing:
- Each episode is processed in a separate worker process
- `multiprocessing.Manager` facilitates inter-process communication
- Results are automatically collected and metadata is merged

## Performance Considerations

### Memory Management
- Each episode's memory is released immediately after processing (`gc.collect()`)
- Parquet format provides efficient columnar storage with compression
- Large datasets are processed incrementally to avoid memory overflow

### Parallelism
- Default: 24 worker processes
- Adjust `--max_workers` based on available CPU cores and memory
- Optimal worker count typically equals CPU core count for I/O-bound workloads

### I/O Optimization
- Parquet format enables fast read/write operations
- Column-wise storage improves training data loading performance
- Compressed storage reduces disk space requirements

## Troubleshooting

### Issue: Import Errors

**Solution:** Ensure all dependencies are installed:
```bash
pip install h5py numpy datasets pillow
```

### Issue: Robot Configuration File Not Found

**Solution:** 
- Verify the configuration file path exists
- Use `--robot_config` argument to specify the correct path
- Ensure the config file is a valid JSON with required fields

### Issue: HDF5 File Format Mismatch

**Solution:** Verify the HDF5 file was generated by `teleop.py` and contains the required data structure:

Required HDF5 Groups:
- `data/demo_*/actions`
- `data/demo_*/states/articulation/robot/joint_position`
- `data/demo_*/initial_state/articulation/robot/joint_position`
- `data/demo_*/obs/left_wrist_camera_bgr`
- `data/demo_*/obs/right_wrist_camera_bgr`
- `data/demo_*/obs/head_camera_bgr`

### Issue: Out of Memory Errors

**Solution:**
- Reduce `--max_workers` parameter to decrease parallel load
- Process HDF5 files in smaller batches
- Ensure sufficient system RAM (recommended: 32GB+ for large datasets)

### Issue: Slow Conversion Speed

**Solution:**
- Increase `--max_workers` (up to CPU core count)
- Use SSD storage for both input HDF5 and output directories
- Close unnecessary applications to free system resources

## Important Notes

### Data Consistency
- Ensure all HDF5 files use the same robot configuration
- Verify consistent camera configurations across episodes
- Check that joint ordering matches the robot configuration

### Storage Space
- Converted datasets are typically larger than original HDF5 files (due to uncompressed images)
- Reserve sufficient disk space (approximately 2-3x the size of HDF5 files)
- Consider using external storage for large-scale datasets

### Task Naming
- Use descriptive, natural language task names
- Consistent naming helps with multi-task learning
- Task names are embedded in the dataset for VLA model training

### Arc-to-Gear Selection
- Choose based on training requirements and robot hardware interface
- **Important:** Once training begins, maintain consistency (always use or never use `--arc2gear`)
- Switching between modes requires retraining from scratch

## Integration with Training Pipeline

After conversion, the dataset can be directly used with LeRobot training scripts:

```bash
# Example training command (adapt to your training framework)
bash script/train.sh -d /path/to/output -p act -o task1_act
```

The converted LeRobot format ensures compatibility with:
- ACT (Action Chunking Transformer)
- Diffusion Policy
- SmolVLA (Vision-Language-Action models)
- Other LeRobot-compatible training frameworks

## Additional Resources

For implementation details and source code, see:
- Conversion script: `script/convert.py`
- Robot configuration: `comm_config/configs/a2_robot_config.json`
- Training scripts: `script/train.sh`

For questions or issues, please refer to the project repository or contact the development team.
