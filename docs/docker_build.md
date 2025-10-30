**Build Train Docker Image**
```bash
docker buildx build -f docker/train/Dockerfile --build-arg CUDA_VERSION=12.8.1 --build-arg UBUNTU_VERSION=22.04 -t real_mirror_train:latest .
```
Key flags:
    --build-arg CUDA_VERSION=12.4.1
    Passes the build argument CUDA_VERSION for use by ARG CUDA_VERSION in the Dockerfile.
    Lets you dynamically set the CUDA version in the base image tag (e.g., nvcr.io/nvidia/cuda:12.4.1-...).
    --build-arg UBUNTU_VERSION=22.04
    Passes the build argument UBUNTU_VERSION for use by ARG UBUNTU_VERSION in the Dockerfile.
