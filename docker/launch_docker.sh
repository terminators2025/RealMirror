#!/bin/bash

# Default values
DEFAULT_IMAGE="real_mirror_teleop:latest"
DEFAULT_CONTAINER_NAME="real-mirror-teleop"
DEFAULT_WORKDIR="/workspace/zbot"
EXTRA_VOLUMES=()
EXTRA_PORTS=()
ENABLE_DESKTOP=false
VNC_PASSWORD="realmirror"

# Help information
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -h, --help                  Show help information
    -i, --image IMAGE           Specify Docker image name (default: ${DEFAULT_IMAGE})
    -n, --name NAME             Specify container name (default: ${DEFAULT_CONTAINER_NAME})
    -w, --workdir WORKDIR       Specify working directory (default: ${DEFAULT_WORKDIR})
    -v, --volume HOST:CONTAINER Add extra mount directories (can be used multiple times)
    -p, --port HOST:CONTAINER   Add extra port mappings (can be used multiple times)
    -d, --desktop               Enable desktop environment with VNC (5901) and XRDP (3389)
    --vnc-password PASSWORD     Set VNC password (default: realmirror)

Examples:
    # Use default configuration
    $0

    # Specify image and container name
    $0 -i real_mirror_training:latest -n my-container

    # Add extra mount directories
    $0 -v /data/datasets:/datasets -v /data/models:/models

    # Specify working directory
    $0 -w /workspace/custom_dir

    # Enable desktop environment
    $0 -d --vnc-password mypassword

    # Combined usage with custom ports
    $0 -i my_image:v1.0 -n my-robot -w /workspace -v /data:/data -p 8080:8080 -d
EOF
}

# Parse command line arguments
IMAGE="${DEFAULT_IMAGE}"
CONTAINER_NAME="${DEFAULT_CONTAINER_NAME}"
WORKDIR="${DEFAULT_WORKDIR}"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -i|--image)
            IMAGE="$2"
            shift 2
            ;;
        -n|--name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        -w|--workdir)
            WORKDIR="$2"
            shift 2
            ;;
        -v|--volume)
            EXTRA_VOLUMES+=("-v" "$2")
            shift 2
            ;;
        -p|--port)
            EXTRA_PORTS+=("-p" "$2")
            shift 2
            ;;
        -d|--desktop)
            ENABLE_DESKTOP=true
            shift
            ;;
        --vnc-password)
            VNC_PASSWORD="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown option '$1'"
            echo "Use '$0 --help' for usage information"
            exit 1
            ;;
    esac
done

# Add desktop ports if enabled
if [ "$ENABLE_DESKTOP" = true ]; then
    EXTRA_PORTS+=("-p" "5901:5901" "-p" "3389:3389")
fi

# Print configuration information
echo "========================================"
echo "Docker Container Configuration:"
echo "========================================"
echo "Image Name:     ${IMAGE}"
echo "Container Name: ${CONTAINER_NAME}"
echo "Workdir:        ${WORKDIR}"
if [ ${#EXTRA_VOLUMES[@]} -gt 0 ]; then
    echo "Extra Volumes:  ${EXTRA_VOLUMES[*]}"
fi
if [ ${#EXTRA_PORTS[@]} -gt 0 ]; then
    echo "Extra Ports:    ${EXTRA_PORTS[*]}"
fi
if [ "$ENABLE_DESKTOP" = true ]; then
    echo "Desktop:        Enabled (VNC: 5901, XRDP: 3389)"
    echo "VNC Password:   ${VNC_PASSWORD}"
fi
echo "========================================"

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Warning: Container '${CONTAINER_NAME}' already exists"
    read -p "Delete and recreate? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Operation cancelled"
        exit 0
    fi
    echo "Removing existing container..."
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

# Build docker run command
DOCKER_CMD="docker run -d --restart unless-stopped -it"

DOCKER_CMD="$DOCKER_CMD --name ${CONTAINER_NAME}"
DOCKER_CMD="$DOCKER_CMD --gpus all"
DOCKER_CMD="$DOCKER_CMD --network host"
DOCKER_CMD="$DOCKER_CMD --privileged"
DOCKER_CMD="$DOCKER_CMD -v ${WORKDIR}:${WORKDIR}"
DOCKER_CMD="$DOCKER_CMD -w ${WORKDIR}"

# Add extra volumes
for vol in "${EXTRA_VOLUMES[@]}"; do
    DOCKER_CMD="$DOCKER_CMD $vol"
done

# Add extra ports
for port in "${EXTRA_PORTS[@]}"; do
    DOCKER_CMD="$DOCKER_CMD $port"
done

# Add Isaac Sim EULA acceptance
DOCKER_CMD="$DOCKER_CMD -e ACCEPT_EULA=Y"

# Add desktop environment variables
if [ "$ENABLE_DESKTOP" = true ]; then
    DOCKER_CMD="$DOCKER_CMD -e VNC_PASSWORD=${VNC_PASSWORD}"
fi

DOCKER_CMD="$DOCKER_CMD ${IMAGE}"

# Determine startup command
# if [ "$ENABLE_DESKTOP" = true ]; then
#     DOCKER_CMD="$DOCKER_CMD /usr/local/bin/start-desktop.sh"
# else
    # DOCKER_CMD="$DOCKER_CMD bash"
# fi

# Start container
echo "Starting container..."
echo "$DOCKER_CMD"
eval $DOCKER_CMD

if [ $? -eq 0 ]; then
    echo "Container started successfully!"
    echo "Container Name: ${CONTAINER_NAME}"
    echo ""
    if [ "$ENABLE_DESKTOP" = true ]; then
        echo "Desktop Access:"
        echo "  VNC:  localhost:5901 (password: ${VNC_PASSWORD})"
        echo "  XRDP: localhost:3389 (username: root, password: ${VNC_PASSWORD})"
        echo ""
    fi
    echo "Access container:"
    echo "  docker exec -it ${CONTAINER_NAME} bash"
    echo ""
    echo "View logs:"
    echo "  docker logs ${CONTAINER_NAME}"
    echo ""
    echo "Stop container:"
    echo "  docker stop ${CONTAINER_NAME}"
    echo ""
    echo "Remove container:"
    echo "  docker rm ${CONTAINER_NAME}"
else
    echo "Failed to start container"
    exit 1
fi