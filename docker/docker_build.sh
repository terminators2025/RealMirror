#!/bin/bash
set -e

# Default values
DEFAULT_TYPE="teleop"
DEFAULT_TAG="latest"
DEFAULT_BASE_IMAGE="nvcr.io/nvidia/isaac-sim"
DEFAULT_ISAAC_VERSION="5.0.0"
BUILD_ARGS=()
NO_CACHE=false

# Help information
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -h, --help                      Show help information
    -t, --type TYPE                 Specify build type: teleop | train (default: ${DEFAULT_TYPE})
                                      - teleop: Based on Isaac Sim image for teleoperation and evaluation
                                      - train: Based on CUDA image for model training
    -g, --tag TAG                   Specify image tag (default: ${DEFAULT_TAG})
    -b, --base-image IMAGE          [Only teleop] Specify Isaac Sim base image (default: ${DEFAULT_BASE_IMAGE})
    -v, --isaac-version VERSION     [Only teleop] Specify Isaac Sim version (default: ${DEFAULT_ISAAC_VERSION})
    --build-arg KEY=VALUE           Pass build arguments to docker build (can be used multiple times)
                                      teleop can override: PIP_INDEX_URL, HTTP_PROXY, HTTPS_PROXY, NO_PROXY
                                      train can override: CUDA_VERSION, UBUNTU_VERSION, PYTHON_VERSION
    --no-cache                      Build without using cache
    --push                          Push image to registry after build

Examples:
    # Build teleop image (default)
    $0

    # Build train image
    $0 -t train

    # Specify custom tag
    $0 -t train -g v1.0.0

    # Specify different Isaac Sim version
    $0 -t teleop -v 4.5.0

    # Use custom base image
    $0 -b my-registry/isaac-sim -v 5.0.0

    # Build without cache
    $0 -t train --no-cache

    # Pass extra build arguments (teleop image)
    $0 -t teleop --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
    $0 -t teleop --build-arg HTTP_PROXY=http://proxy.example.com:8080

    # Pass extra build arguments (train image)
    $0 -t train --build-arg PYTHON_VERSION=3.11
    $0 -t train --build-arg CUDA_VERSION=12.6.0 --build-arg UBUNTU_VERSION=22.04

    # Build and push
    $0 -t train -g v1.0.0 --push

    # Full train image build
    $0 -t train -g v1.0.0 --build-arg PYTHON_VERSION=3.11 --no-cache
EOF
}

# Parse command line arguments
TYPE="${DEFAULT_TYPE}"
TAG="${DEFAULT_TAG}"
BASE_IMAGE="${DEFAULT_BASE_IMAGE}"
ISAAC_VERSION="${DEFAULT_ISAAC_VERSION}"
PUSH_IMAGE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -t|--type)
            TYPE="$2"
            shift 2
            ;;
        -g|--tag)
            TAG="$2"
            shift 2
            ;;
        -b|--base-image)
            BASE_IMAGE="$2"
            shift 2
            ;;
        -v|--isaac-version)
            ISAAC_VERSION="$2"
            shift 2
            ;;
        --build-arg)
            BUILD_ARGS+=("--build-arg" "$2")
            shift 2
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --push)
            PUSH_IMAGE=true
            shift
            ;;
        *)
            echo "Error: Unknown option '$1'"
            echo "Use '$0 --help' for usage information"
            exit 1
            ;;
    esac
done

# Validate build type
if [[ "$TYPE" != "teleop" && "$TYPE" != "train" ]]; then
    echo "Error: Invalid build type '$TYPE'"
    echo "Supported types: teleop, train"
    exit 1
fi

# Set build context and Dockerfile path
SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
DOCKERFILE="${SCRIPT_DIR}/${TYPE}/Dockerfile"
IMAGE_NAME="real_mirror_${TYPE}:${TAG}"

# Check if Dockerfile exists
if [ ! -f "$DOCKERFILE" ]; then
    echo "Error: Dockerfile not found: $DOCKERFILE"
    exit 1
fi

# Print build information
echo "========================================"
echo "Docker Image Build Configuration:"
echo "========================================"
echo "Build Type:     ${TYPE}"
echo "Image Name:     ${IMAGE_NAME}"
if [ "$TYPE" = "teleop" ]; then
    echo "Base Image:     ${BASE_IMAGE}:${ISAAC_VERSION}"
elif [ "$TYPE" = "train" ]; then
    echo "Base Image:     CUDA (defined in Dockerfile)"
fi
echo "Dockerfile:   ${DOCKERFILE}"
echo "Build Context:   ${PROJECT_ROOT}"
echo "Use Cache:     $([ "$NO_CACHE" = true ] && echo "No" || echo "Yes")"
if [ ${#BUILD_ARGS[@]} -gt 0 ]; then
    echo "Build Args:     ${BUILD_ARGS[*]}"
fi
echo "========================================"

# Confirm build
read -p "Confirm to start build? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Build cancelled"
    exit 0
fi

# Build image
echo ""
echo "========================================"
echo "Starting image build..."
echo "========================================"

BUILD_CMD=(
    docker build
    -f "$DOCKERFILE"
    -t "$IMAGE_NAME"
)

# Add different build arguments based on type
if [ "$TYPE" = "teleop" ]; then
    # teleop uses Isaac Sim base image
    BUILD_CMD+=(
        --build-arg "ISAACSIM_BASE_IMAGE_ARG=${BASE_IMAGE}"
        --build-arg "ISAACSIM_VERSION_ARG=${ISAAC_VERSION}"
    )
    # Pass proxy and PyPI source arguments (using default values in Dockerfile unless overridden with --build-arg)
    # These arguments have default values in Dockerfile, here just showing how to override them
elif [ "$TYPE" = "train" ]; then
    # train uses CUDA base image, default arguments are defined in Dockerfile
    # If need to override, can pass through --build-arg
    :
fi

# Add extra build arguments
BUILD_CMD+=("${BUILD_ARGS[@]}")

if [ "$NO_CACHE" = true ]; then
    BUILD_CMD+=(--no-cache)
fi

BUILD_CMD+=("$PROJECT_ROOT")

# Execute build
echo "Executing command: ${BUILD_CMD[*]}"
echo ""

if "${BUILD_CMD[@]}"; then
    echo ""
    echo "========================================"
    echo "✓ Image built successfully!"
    echo "========================================"
    echo "Image Name: ${IMAGE_NAME}"
    echo ""
    
    # Display image information
    echo "Image Information:"
    docker images "${IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    echo ""
    
    # Push image (if specified)
    if [ "$PUSH_IMAGE" = true ]; then
        echo "========================================"
        echo "Pushing image to registry..."
        echo "========================================"
        if docker push "$IMAGE_NAME"; then
            echo "✓ Image pushed successfully!"
        else
            echo "✗ Image push failed"
            exit 1
        fi
        echo ""
    fi
    
    # Show usage tips
    echo "Using the image:"
    echo "  # Start container"
    echo "  ./docker/launch_docker.sh -i ${IMAGE_NAME} -n real-mirror-${TYPE}"
    echo ""
    echo "  # Run directly"
    echo "  docker run -it --rm --gpus all ${IMAGE_NAME}"
    echo ""
    echo "  # Push to registry"
    echo "  docker push ${IMAGE_NAME}"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "✗ Image build failed"
    echo "========================================"
    exit 1
fi
