# Docker Scripts Documentation

This directory contains scripts for building and managing Docker containers used in the RealMirror project. Below is a guide to each script's purpose and usage.

## Table of Contents
- [docker_build.sh](#docker_buildsh)
- [launch_docker.sh](#launch_dockersh)
- [remove_docker.sh](#remove_dockersh)

## docker_build.sh

Builds Docker images for teleoperation and training environments.

### Usage
```bash
./docker_build.sh [OPTIONS]
```

### Options
- `-h, --help`: Show help information
- `-t, --type TYPE`: Specify build type (`teleop` | `train`) (default: `teleop`)
- `-g, --tag TAG`: Specify image tag (default: `latest`)
- `-b, --base-image IMAGE`: [Only for teleop] Specify Isaac Sim base image (default: `nvcr.io/nvidia/isaac-sim`)
- `-v, --isaac-version VERSION`: [Only for teleop] Specify Isaac Sim version (default: `5.0.0`)
- `--build-arg KEY=VALUE`: Pass build arguments to docker build (can be used multiple times)
- `--no-cache`: Build without using cache
- `--push`: Push image to registry after build

### Examples
1. Build teleop image (default):
   ```bash
   ./docker_build.sh
   ```

2. Build train image:
   ```bash
   ./docker_build.sh -t train
   ```

3. Build with custom tag:
   ```bash
   ./docker_build.sh -t train -g v1.0.0
   ```

4. Build without cache:
   ```bash
   ./docker_build.sh -t train --no-cache
   ```

5. Pass extra build arguments:
   ```bash
   ./docker_build.sh -t teleop --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
   ```

## launch_docker.sh

Launches Docker containers with proper configurations for teleoperation or training.

### Usage
```bash
./launch_docker.sh [OPTIONS]
```

### Options
- `-h, --help`: Show help information
- `-i, --image IMAGE`: Specify Docker image name (default: `real_mirror_teleop:latest`)
- `-n, --name NAME`: Specify container name (default: `real-mirror-teleop`)
- `-w, --workdir WORKDIR`: Specify working directory (default: `/workspace/zbot`)
- `-v, --volume HOST:CONTAINER`: Add extra mount directories (can be used multiple times)

### Examples
1. Launch with default configuration:
   ```bash
   ./launch_docker.sh
   ```

2. Launch with custom image and container name:
   ```bash
   ./launch_docker.sh -i real_mirror_training:latest -n my-container
   ```

3. Mount extra directories:
   ```bash
   ./launch_docker.sh -v /data/datasets:/datasets -v /data/models:/models
   ```

4. Specify working directory:
   ```bash
   ./launch_docker.sh -w /workspace/custom_dir
   ```

## remove_docker.sh

Removes Docker containers with optional confirmation prompts.

### Usage
```bash
./remove_docker.sh [OPTIONS] [CONTAINER_NAME...]
```

### Options
- `-h, --help`: Show help information
- `-n, --name NAME`: Specify the container name to delete (default: `real-mirror-default`)
- `-a, --all`: Delete all `real-mirror-*` containers
- `-f, --force`: Force deletion without confirmation

### Examples
1. Remove default container:
   ```bash
   ./remove_docker.sh
   ```

2. Remove specific container:
   ```bash
   ./remove_docker.sh -n my-container
   ```

3. Remove multiple containers:
   ```bash
   ./remove_docker.sh container1 container2 container3
   ```

4. Remove all real-mirror containers:
   ```bash
   ./remove_docker.sh --all
   ```

5. Force removal without confirmation:
   ```bash
   ./remove_docker.sh -n my-container --force
   ```