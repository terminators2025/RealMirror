# Model Inference Script - User Guide

## Overview

The `script/infer.py` tool enables interactive validation of trained VLA models in Isaac Sim environments. This is essential for testing model behavior, debugging training issues, and verifying task performance before running comprehensive benchmark evaluations.

## Key Features

- **Multi-Model Support**: Compatible with ACT, Diffusion Policy, and SmolVLA architectures
- **Interactive Testing**: Real-time model inference with keyboard controls
- **Visual Debugging**: Observe robot actions and scene interactions in Isaac Sim
- **Flexible Configuration**: Supports multiple tasks and model configurations
- **Object Manipulation**: Manually adjust object positions during inference
- **State Management**: Reset and restart inference sessions on demand
- **Arc-to-Gear Conversion**: Support for models trained with gear position outputs

## Basic Usage

```bash
$isaac_sim_dir/python.sh script/infer.py \
    --task <task_name> \
    --model-type <model_type> \
    --model-path <path_to_model> \
    [--arc2gear] \
    [--headless]
```

### Command-Line Arguments

| Argument | Required | Description | Default |
|----------|----------|-------------|---------|
| `--task` | Yes | Task configuration name (e.g., `Task1_Kitchen_Cleanup`) | - |
| `--model-type` | Yes | Model architecture: `act`, `diffusion`, or `smolvla` | - |
| `--model-path` | Yes | Path to trained model checkpoint directory | - |
| `--arc2gear` | No | Enable arc-to-gear conversion for hand joints | False |
| `--headless` | No | Run without GUI (for automated testing) | False |
| `--task-name` | No | Natural language task description (required for SmolVLA) | None |
| `--num-actions` | No | Number of actions to use from model output chunk | 5 |
| `--max-steps` | No | Maximum number of simulation steps before auto-exit | 1000 |
| `--resolution` | No | UI window resolution (e.g., `1920x1080`, `1280x720`) | `1920x1080` |

## Usage Examples

### Example 1: Test ACT Model with Arc-to-Gear Conversion

```bash
$isaac_sim_dir/python.sh script/infer.py \
    --task Task1_Kitchen_Cleanup \
    --model-type act \
    --model-path outputs/task1_act/checkpoints/last/pretrained_model \
    --arc2gear
```

This runs inference for Task 1 using an ACT model trained with gear position outputs.

### Example 2: Test Diffusion Policy Model

```bash
$isaac_sim_dir/python.sh script/infer.py \
    --task Task2_Cup_to_Cup_Transfer \
    --model-type diffusion \
    --model-path outputs/task2_diffusion/checkpoints/last/pretrained_model
```

This runs inference for Task 2 using a Diffusion Policy model without gear conversion.

### Example 3: Test SmolVLA Model with Task Description

```bash
$isaac_sim_dir/python.sh script/infer.py \
    --task Task4_Can_Stacking \
    --model-type smolvla \
    --model-path outputs/task4_smolvla/checkpoints/last/pretrained_model \
    --task-name "Stack the cans in a stable tower" \
    --arc2gear
```

SmolVLA models require natural language task descriptions for vision-language-action inference.

### Example 4: Headless Mode for Automated Testing

```bash
$isaac_sim_dir/python.sh script/infer.py \
    --task Task3_Assembly_Line_Sorting \
    --model-type act \
    --model-path outputs/task3_act/checkpoints/last/pretrained_model \
    --arc2gear \
    --headless \
    --max-steps 500
```

This runs inference without GUI for automated testing or CI/CD pipelines.

### Example 5: Custom Resolution for Recording

```bash
$isaac_sim_dir/python.sh script/infer.py \
    --task Task5_Air_Fryer_Manipulation \
    --model-type diffusion \
    --model-path outputs/task5_diffusion/checkpoints/last/pretrained_model \
    --resolution 1280x720
```

Use lower resolution to improve rendering performance during video recording.

## Interactive Controls

Once the inference script is running, use the following keyboard controls:

### Primary Controls

| Key | Function | Description |
|-----|----------|-------------|
| **S** | Start/Stop Inference | Toggle model inference on/off |
| **R** | Reset Environment | Reset robot and objects to initial state |
| **Q** | Quit | Exit the application |

### Object Manipulation

| Key | Function | Description |
|-----|----------|-------------|
| **↑** (Up Arrow) | Move Forward | Move active object forward in world coordinates |
| **↓** (Down Arrow) | Move Backward | Move active object backward in world coordinates |
| **←** (Left Arrow) | Move Left | Move active object left in world coordinates |
| **→** (Right Arrow) | Move Right | Move active object right in world coordinates |
| **Numpad 0** | Reset Pose | Reset active object to initial pose |
| **Numpad 1** | Select Object 1 | Set first task object as active |
| **Numpad 2** | Select Object 2 | Set second task object as active |
| **Numpad 3** | Select Object 3 | Set third task object as active |

### Control Workflow

1. **Start Application**: Script loads model and scene automatically
2. **Wait for Stabilization**: Scene stabilizes for ~50 physics steps
3. **Press 'S'**: Begin model inference (robot starts executing actions)
4. **Observe Behavior**: Watch robot interact with objects
5. **Press 'S' Again**: Pause inference to examine current state
6. **Press 'R'**: Reset environment to retry inference
7. **Adjust Objects** (Optional): Use arrow keys and numpad to reposition objects
8. **Press 'Q'**: Exit when finished

## Technical Details

### Model Loading

The inference script supports three model architectures:

1. **ACT (Action Chunking Transformer)**:
   - Loads from `pretrained_model` directory
   - Expects PyTorch checkpoint with model state dict
   - Supports chunked action prediction

2. **Diffusion Policy**:
   - Loads from `pretrained_model` directory
   - Uses diffusion denoising for action generation
   - Supports temporal action sequences

3. **SmolVLA (Vision-Language-Action)**:
   - Requires base VLM model (SmolVLM2-500M-Video-Instruct)
   - Loads fine-tuned weights from checkpoint
   - Requires natural language task description via `--task-name`

### Observation Processing

**Robot State (26-DOF)**:
- 7 left arm joints (radians)
- 7 right arm joints (radians)
- 6 left hand joints (radians or gear positions)
- 6 right hand joints (radians or gear positions)

**Camera Images**:
- `head_camera`: RGB image from robot head
- `left_wrist_camera`: RGB image from left wrist
- `right_wrist_camera`: RGB image from right wrist
- All images resized to 256×256 pixels

### Action Application

**Action Expansion (26-DOF → 38-DOF)**:

Models predict 26-DOF actions, which are expanded to 38-DOF for simulation:

| Master Joints (26-DOF) | Simulation Joints (38-DOF) | Expansion |
|------------------------|----------------------------|-----------|
| 14 arm joints | 14 arm joints | 1-to-1 |
| 12 hand master joints | 24 hand joints (with mimics) | 1-to-2 or 1-to-3 |

**Hand Joint Mimic Rules**:
- `thumb_swing`: 1→1 (direct mapping)
- `thumb_1`: 1→3 (thumb_1/2/3_joint)
- `index_1`: 1→2 (index_1/2_joint)
- `middle_1`: 1→2 (middle_1/2_joint)
- `ring_1`: 1→2 (ring_1/2_joint)
- `little_1`: 1→2 (little_1/2_joint)

### Arc-to-Gear Conversion

When `--arc2gear` is enabled:

**Forward Conversion (State → Model Input)**:
```python
# Arm joints: remain in radians
arm_state = state[:14]  # unchanged

# Hand joints: radians → gear positions (0-2000)
hand_state = gear_to_radian(state[14:])
```

**Inverse Conversion (Model Output → Simulation)**:
```python
# Arm joints: remain in radians
arm_actions = actions[:14]  # unchanged

# Hand joints: gear positions → radians
hand_actions = gear_to_radian(actions[14:])
```

**Gear Conversion Formula**:
```python
gear_value = int(round((radian_value / max_radian) * 2000))
radian_value = (gear_value / 2000) * max_radian
```

Special case: `thumb_swing` joints fixed at gear position 1195.

### Inference Loop

1. **Get Observation**:
   - Read robot joint positions (26-DOF)
   - Capture camera images (3 cameras)
   - Apply arc-to-gear conversion if enabled

2. **Run Model Inference**:
   - Process observation through model
   - Get action chunk (e.g., 5 actions)

3. **Apply Actions**:
   - Expand 26-DOF to 38-DOF
   - Apply inverse gear conversion if enabled
   - Execute actions using Isaac Sim ArticulationController

4. **Step Simulation**:
   - Advance physics simulation
   - Render scene
   - Repeat from step 1

## Workflow Integration

### Typical Model Validation Workflow

1. **Train Model**: Complete training using `script/train.sh`
   ```bash
   bash script/train.sh -d datasets/Task1_Kitchen_Cleanup -p act -o task1_act
   ```

2. **Test Model**: Run inference to validate behavior
   ```bash
   $isaac_sim_dir/python.sh script/infer.py \
       --task Task1_Kitchen_Cleanup \
       --model-type act \
       --model-path outputs/task1_act/checkpoints/last/pretrained_model \
       --arc2gear
   ```

3. **Observe Performance**:
   - Press 'S' to start inference
   - Watch robot execute task
   - Check for smooth motions and successful grasps
   - Verify task completion

4. **Iterate if Needed**:
   - Press 'R' to reset and retry
   - Adjust object positions to test robustness
   - Note failure modes for debugging

5. **Run Benchmark**: If satisfied, proceed with full evaluation
   ```bash
   $isaac_sim_dir/python.sh script/eval.py \
       --task Task1_Kitchen_Cleanup \
       --model-type act \
       --arc2gear \
       --num-rollouts 400
   ```

## Troubleshooting

### Issue: Model Fails to Load

**Symptoms:**
- Error: "Failed to load inference model"
- Model weights not found

**Solutions:**
1. Verify model path points to correct checkpoint directory
2. Check that checkpoint contains required files:
   - ACT/Diffusion: `pretrained_model/` directory with PyTorch checkpoint
   - SmolVLA: Fine-tuned weights + base VLM model
3. Ensure model was trained with correct architecture
4. Check file permissions on model directory

### Issue: Inference Not Starting (Pressing 'S' Has No Effect)

**Symptoms:**
- Pressing 'S' key does nothing
- No robot motion observed

**Solutions:**
1. Check console for error messages
2. Verify cameras initialized successfully
3. Ensure robot joint indices mapped correctly
4. Try pressing 'R' to reset, then 'S' to start
5. Check that model output shape matches expected 26-DOF

### Issue: Robot Moves Erratically or Incorrectly

**Symptoms:**
- Robot joints move in unexpected ways
- Actions seem inverted or scaled incorrectly

**Solutions:**
1. **Arc-to-Gear Mismatch**: Ensure `--arc2gear` matches training configuration
   - If trained with gear conversion: use `--arc2gear`
   - If trained without gear conversion: omit `--arc2gear`
2. Check joint position limits in robot configuration
3. Verify observation normalization matches training
4. Inspect model output range (should be in valid joint limits)

### Issue: SmolVLA Model Not Working

**Symptoms:**
- SmolVLA inference fails or produces no output
- Error about missing task description

**Solutions:**
1. Always provide `--task-name` with natural language description
   ```bash
   --task-name "Place the red object into the basket"
   ```
2. Ensure SmolVLM2-500M-Video-Instruct base model is available
3. Check that model was fine-tuned correctly from base VLM
4. Verify task description matches training format

### Issue: Cameras Not Working

**Symptoms:**
- Error: "Failed to get camera images"
- Black or missing camera views

**Solutions:**
1. Verify camera configuration in task JSON matches robot setup
2. Check camera prims exist in USD scene
3. Ensure camera resolution settings are valid
4. Try lower resolution: `--resolution 1280x720`
5. Check GPU memory availability

### Issue: Performance is Slow or Laggy

**Symptoms:**
- Low frame rate during inference
- Simulation stuttering

**Solutions:**
1. Reduce UI resolution: `--resolution 1280x720`
2. Close unnecessary applications
3. Use headless mode for pure performance: `--headless`
4. Reduce `--num-actions` (fewer actions per inference call)
5. Check GPU utilization and temperature
6. Verify system meets hardware requirements

### Issue: Objects Not Responding to Arrow Keys

**Symptoms:**
- Arrow keys don't move objects
- Object manipulation has no effect

**Solutions:**
1. Ensure you've selected an active object (Numpad 1-3)
2. Check that task has `task_related_objects` defined
3. Verify object manager initialized correctly
4. Try pressing Numpad 0 to reset object pose first

## Advanced Usage

### Custom Inference Configuration

Modify inference parameters programmatically:

```python
from inference import InferenceConfig

config = InferenceConfig(
    model_type="act",
    model_path="path/to/model",
    use_gear_conversion=True,
    num_actions_in_chunk=10,  # Use more actions per inference
    task_name="Custom task description"
)
```

### Saving Inference Results

Enable image saving for debugging:

```python
# In run_inference_step() method
actions = self.inference_engine.run_inference(
    state,
    left_wrist_image,
    right_wrist_image,
    head_image,
    save_images=True  # Saves images to disk
)
```

Images will be saved to `debug_images/` directory with timestamps.

### Testing Different Initial Conditions

Modify `reset_environment()` to test various starting configurations:

```python
def reset_environment(self):
    # Custom initial joint positions
    custom_positions = {
        "idx13_left_arm_joint1": 0.5,
        "idx20_right_arm_joint1": -0.5,
        # ... more joints
    }
    
    for joint_name, angle in custom_positions.items():
        idx = self.robot.get_dof_index(joint_name)
        if idx != -1:
            current_positions[idx] = angle
    
    self.robot.set_joint_positions(current_positions)
```

### Batch Testing Multiple Models

Create a shell script to test multiple checkpoints:

```bash
#!/bin/bash

for checkpoint in outputs/task1_act/checkpoints/*/; do
    echo "Testing checkpoint: $checkpoint"
    $isaac_sim_dir/python.sh script/infer.py \
        --task Task1_Kitchen_Cleanup \
        --model-type act \
        --model-path "$checkpoint/pretrained_model" \
        --arc2gear \
        --headless \
        --max-steps 500
done
```

### Logging and Metrics

Add custom logging for performance analysis:

```python
import time

start_time = time.time()
actions = self.inference_engine.run_inference(...)
inference_time = time.time() - start_time

Logger.info(f"Inference time: {inference_time*1000:.2f}ms")
Logger.info(f"Actions shape: {actions.shape}")
Logger.info(f"Action range: [{actions.min():.3f}, {actions.max():.3f}]")
```

## Performance Considerations

### System Requirements

- **GPU**: NVIDIA RTX 5090/5880 (as per RealMirror requirements)
- **RAM**: 16GB+ recommended
- **Isaac Sim**: 5.0.0 with proper CUDA/driver configuration

### Optimization Tips

1. **Model Inference Speed**:
   - ACT: ~10-20ms per inference (fastest)
   - Diffusion: ~50-100ms per inference (moderate)
   - SmolVLA: ~100-200ms per inference (slowest due to VLM)

2. **Rendering Performance**:
   - Use `--headless` for maximum speed
   - Lower resolution improves frame rate
   - Disable unnecessary camera views if possible

3. **Action Chunking**:
   - Larger `--num-actions` reduces inference calls
   - Too large may cause delayed reactions
   - Optimal: 5-10 actions per chunk

## Important Notes

### Model Compatibility

- **Arc-to-Gear**: Must match training configuration exactly
- **Image Resolution**: Models expect 256×256 RGB images
- **Joint Order**: Fixed 26-DOF order (14 arms + 12 hands)
- **Observation Format**: Normalized joint positions + camera images

### Known Limitations

1. **Real-time Performance**: Inference may not run at true real-time speed (30 Hz) depending on model complexity
2. **Object Physics**: Simulation physics may differ slightly from training environment
3. **Camera Calibration**: Camera positions fixed by task configuration
4. **Action Smoothing**: No temporal smoothing applied to model outputs

### Best Practices

1. **Always test with same `--arc2gear` setting used during training**
2. **Start with short inference sessions** (low `--max-steps`) to verify functionality
3. **Use 'R' key frequently** to reset and retry different scenarios
4. **Test robustness** by manually moving objects during inference
5. **Compare with replay** to distinguish model issues from data issues
6. **Monitor GPU memory** when testing large models (especially SmolVLA)

## Related Documentation

For related workflows and tools, see:
- Training: README section 2.6.3 and `script/train.sh`
- Evaluation: README section 2.4.2 and `script/eval.py`
- Data Collection: README section 2.5 and `script/teleop.py`
- Data Replay: `docs/replay.md` and `script/replay.py`

## Additional Resources

For implementation details and source code, see:
- Inference script: `script/infer.py`
- Inference engine: `inference/inference_engine.py`
- Model loader: `inference/model_loader.py`
- Unit converter: `inference/unit_converter.py`
- Task configurations: `tasks/Task*.json`

For questions or issues, please refer to the project repository or contact the development team.
