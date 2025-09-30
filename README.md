![teaser.png](./docs/image/Teaser.png)

<div align="center">
  <a href="https://terminators2025.github.io/RealMirror.github.io/">
    <img src="https://img.shields.io/badge/GitHub-grey?logo=GitHub" alt="GitHub">
  </a>
</div>

# <h1 style="color: blue;">*Open Source Plan</h1>
- [ ] **Mid-October**: Open-source inference and evaluation code
- [ ] **Mid-October**: Open-source A2 robot model, all model weights, scene assets, training code and training data 
- [ ] **Late-October**: Open-source complete data collection code, teleoperation code and toolchains

# 1. Overview
The emerging field of Vision-Language-Action (VLA) for humanoid robots faces several fundamental challenges, including the high cost of data acquisition, the lack of a standardized benchmark, and the significant gap between simulation and the real world. To overcome these obstacles, we propose RealMirror, a comprehensive, open-source embodied AI VLA platform. RealMirror builds an efficient, low-cost data collection, model training, and inference system that enables end-to-end VLA research without requiring a real robot. To facilitate model evolution and fair comparison, we also introduce a dedicated VLA benchmark for humanoid robots, featuring multiple scenarios, extensive trajectories, and various VLA models. Furthermore, by integrating generative models and 3D Gaussian Splatting to reconstruct realistic environments and robot models, we successfully demonstrate zero-shot Sim2Real transfer, where models trained exclusively on simulation data can perform tasks on a real robot seamlessly, without any fine-tuning. In conclusion, with the unification of these critical components, RealMirror provides a robust framework that significantly accelerates the development of VLA models for humanoid robots.

# 2. Quick Start

### 2.1 Requirements

| Tested |
| :----|
| <ul><li>Ubuntu 22.04</li><li>NVIDIA Isaac Sim 5.0.0</li><li>Hardware<ul><li>GPU: GeForce RTX 5090/5880</li><li>Driver:575.64.03 + CUDA 12.9</li></ul></li></ul> |

### 2.2 Prepare Benchmark Data

To get started with the RealMirror benchmark, you need to download the pre-collected dataset and trained models:

1. **Download the Dataset**
   - Access the benchmark data from our Google Drive: [RealMirror Benchmark Dataset](https://drive.google.com/drive/folders/xxxxx)
   - The dataset includes both simulation assets and pre-trained models for all benchmark tasks
   - Total download size: approximately XX GB

2. **Extract and Setup**
   - After downloading, extract the compressed archive to your local directory
   - We recommend extracting to a dedicated folder, for example: `~/RealMirror_Benchmark_data/`
   - Ensure the extracted directory structure matches the following organization:

- **Data Structure**

```
RealMirror_Benchmark_data
├── assets
│   ├── robot
│   │   └── AgiBotA2
│   └── scenes
│       ├── Task1_Kitchen_Cleanup
│       ├── Task2_Cup_to_Cup_Transfer
│       ├── Task3_Assembly_Line_Sorting
│       ├── Task4_Can_Stacking
│       └── Task5_Air_Fryer_Manipulation
└── models
    ├── Task1_Kitchen_Cleanup
    │   ├── ACT
    │   ├── Diffusion
    │   └── SmolVLA
    ├── Task2_Cup_to_Cup_Transfer
    │   ├── ACT
    │   ├── Diffusion
    │   └── SmolVLA
    ├── Task3_Assembly_Line_Sorting
    │   ├── ACT
    │   ├── Diffusion
    │   └── SmolVLA
    ├── Task4_Can_Stacking
    │   ├── ACT
    │   ├── Diffusion
    │   └── SmolVLA
    └── Task5_Air_Fryer_Manipulation
        ├── ACT
        ├── Diffusion
        └── SmolVLA
```

### 2.3 Installation

1. Donwload RealMirror source code:
```bash
git clone https://github.com/terminators2025/RealMirror.git
cd RealMirror
```
2. Fetch lerobot by git submodule command:
```bash
git submodule init
git submodule update
cd thirdparty/lerobot

# Checkout and apply patch of lerobot
git checkout fe88c5942cce222c5463350d999a463b9016cf8c
git apply ../../patches/lerobot_customizations.patch
```
3. Install LeRobot in editable mode into Isaac Sim's Python environment
```bash
$isaac_sim_dir/python.sh -m pip install -e .
```
4.  Install RealMirror in editable mode into Isaac Sim's Python site-packages library
```bash
#Install RealMirror To Issac Sim's python Site lib
$isaac_sim_dir/python.sh -m pip install -e .
```

### 2.4 Usage

### 2.4 Benchmark Tasks


#### 2.4.1 Benchmark Task Config
The benchmark configuration is defined in a JSON file, such as `Task1_Kitchen_Cleanup.json`. This file specifies all the necessary parameters for the robot, scene, and evaluation.


The following task names are supported

<table>
  <tr>
    <th colspan="2"><strong>Task Name</strong></th>
  </tr>
  <tr>
    <td>Task1_Kitchen_Cleanup</td>
    <td>Task2_Cup_to_Cup_Transfer</td>
  </tr>
    <td>Task3_Assembly_Line_Sorting</td>
    <td>Task4_Can_Stacking</td>
  <tr>
  </tr>
    <tr>
    <td>Task5_Air_Fryer_Manipulation</td>
  </tr>
</table>

- **`robot`**: This section defines the robot to be used in the task.
  - `robot_type`: Specifies the type of robot (e.g., "A2").
  - `robot_cfg`: Points to the robot's specific configuration file (e.g., "a2_robot_config.json").

- **`scene`**: This is an array that contains the settings for the simulation scene.
  - `task_related_objects`: An array of objects that are relevant to the task. Each object includes:
    - `name`: A unique identifier for the object.
    - `description`: A natural language instruction describing the task goal for this object.
    - `cylinder_prim_path`: The path to the object's primitive in the USD scene.
    - `target_prim_path`: The path to the target location where the object should be placed.
  - `scene_id`: A unique identifier for the scene.
  - `scene_usd`: The absolute path to the USD file that defines the scene layout.
  - `prime_path`: The base path for all primitives within the scene.
  - `camera_configs`: Defines the properties for each camera used in the simulation, including:
    - `head_camera`, `left_wrist_camera`, `right_wrist_camera`: Each camera has settings for name, path, image resolution (`height`, `width`), and other camera-specific parameters.
  - `eval_cfg`: Contains configuration for the evaluation process.
    - `model`: Specifies the paths for the pre-trained models.
      - `model_root_dir`: The root directory where all models are stored.
      - `checkpoints_dir`: Subdirectories for different model architectures like ACT, Diffusion, and SmolVLA.

##### Important Path Configuration

After extracting the benchmark data, you need to update the configuration files to point to the correct paths on your system:

**1. Robot Configuration (`comm_config/configs/a2_robot_config.json`)**

Update the following paths to match your extracted benchmark data location:

```json
{
    "usd_path": "/path/to/your/RealMirror_Benchmark_data/assets/robot/AgiBotA2/model_no_col2.usd",
    "urdf_path": "/path/to/your/RealMirror_Benchmark_data/assets/robot/AgiBotA2/model_no_col2.usd",
    "robot_descriptor_paths": {
        "right": "/path/to/your/RealMirror_Benchmark_data/assets/robot/AgiBotA2/arm_descriptor/a2_right_arm_robot_descriptor.yaml",
        "left": "/path/to/your/RealMirror_Benchmark_data/assets/robot/AgiBotA2/arm_descriptor/a2_left_arm_robot_descriptor.yaml"
    }
}
```

**2. Task Configuration Files (`tasks/Task*.json`)**

For each task configuration file (Task1_Kitchen_Cleanup.json, Task2_Cup_to_Cup_Transfer.json, etc.), update:

- **Scene USD path**: Point to the corresponding scene asset
  ```json
  "scene_usd": "/path/to/your/RealMirror_Benchmark_data/assets/scenes/Task1_Kitchen_Cleanup/kitchen.usd"
  ```
  
- **Model root directory**: Point to the models directory
  ```json
  "model_root_dir": "/path/to/your/RealMirror_Benchmark_data/models"
  ```

Example for Task1_Kitchen_Cleanup.json:
```json
{
    "scene": [{
        "scene_usd": "/path/to/your/RealMirror_Benchmark_data/assets/scenes/Task1_Kitchen_Cleanup/kitchen.usd",
        "eval_cfg": {
            "model": {
                "model_root_dir": "/path/to/your/RealMirror_Benchmark_data/models"
            }
        }
    }]
}
```

Replace `/path/to/your/` with the actual path where you extracted the RealMirror_Benchmark_data directory.


#### 2.4.2 Run Benchmark
Execute the following command under RealMirror root directory outside:

```bash
# Task1 - Kitchen Cleanup (ACT model)
$isaac_sim_dir/python.sh script/eval.py \
    --task Task1_Kitchen_Cleanup \
    --model-type act \
    --area-file data/eval/data_area/data_area_task1.txt \
    --arc2gear --num-rollouts 400 --max-horizon 400 --headless

# For Task2
$isaac_sim_dir/python.sh script/eval.py \
    --task Task2_Cup_to_Cup_Transfer \
    --model-type act/diffusion/smolvla \
    --area-file data/eval/data_area/data_area_task2.txt \
    --arc2gear --num-rollouts 200 --max-horizon 320 --headless

# For Task3
$isaac_sim_dir/python.sh script/eval.py \
    --task Task3_Assembly_Line_Sorting \
    --model-type act \
    --arc2gear --num-rollouts 100 --max-horizon 3000 --headless

# For Task4
$isaac_sim_dir/python.sh script/eval.py \
    --task Task4_Can_Stacking \
    --model-type act \
    --arc2gear --use-stability-check \
    --area-file data/eval/data_area/data_area_task4.txt \
    --num-rollouts 400 --max-horizon 500 --stability-frames 100 --headless 

# For Task5
$isaac_sim_dir/python.sh script/eval.py \
    --task Task5_Air_Fryer_Manipulation \
    --model-type act \
    --area-file data/eval/data_area/data_area_task5.txt \
    --arc2gear --num-rollouts 400 --max-horizon 500 --headless
```

Note： Add --headless flag for run without GUI and  saves the results to the runs/eval directory.
