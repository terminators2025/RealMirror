#!/bin/bash
set -e

# 默认值
DEFAULT_CONTAINER_NAME="real-mirror-default"

# 帮助信息
show_help() {
    cat << EOF
Usage: $0 [OPTIONS] [CONTAINER_NAME...]

Options:
    -h, --help          Show this help message
    -n, --name NAME     Specify the container name to delete (default: ${DEFAULT_CONTAINER_NAME})
    -a, --all           Delete all real-mirror-* containers
    -f, --force         Force deletion without confirmation

Examples:
    # Delete default container
    $0

    # Delete a specific container
    $0 -n my-container

    # Delete multiple containers
    $0 container1 container2 container3

    # Delete all real-mirror containers
    $0 --all

    # Force delete without confirmation
    $0 -n my-container --force

EOF
}

# 删除容器函数
remove_container() {
    local container_name="$1"
    local force="$2"
    
    # 检查容器是否存在
    if ! docker ps -a --format '{{.Names}}' | grep -q "^${container_name}$"; then
        echo "Container '${container_name}' does not exist, skipping"
        return 0
    fi
    
    # 获取容器状态
    local status=$(docker inspect -f '{{.State.Status}}' "${container_name}" 2>/dev/null || echo "not found")
    
    if [ "$force" != "true" ]; then
        echo "========================================"
        echo "Container Information:"
        echo "  Name: ${container_name}" 
        echo "  Status: ${status}"
        echo "========================================"
        read -p "Confirm deletion of this container? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Operation canceled for '${container_name}'"
            return 0
        fi
    fi
    
    echo "Deleting container '${container_name}'..."
    
    # 如果容器正在运行，先停止
    if [ "$status" = "running" ]; then
        echo "  Stopping container..."
        docker stop "${container_name}" >/dev/null 2>&1 || true
    fi
    
    # 删除容器
    echo "  Removing container..."
    docker rm "${container_name}" >/dev/null 2>&1 || true
    
    echo "✓ Container '${container_name}' deleted"
}

# 解析命令行参数
CONTAINER_NAMES=()
FORCE_MODE=false
DELETE_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -n|--name)
            CONTAINER_NAMES+=("$2")
            shift 2
            ;;
        -a|--all)
            DELETE_ALL=true
            shift
            ;;
        -f|--force)
            FORCE_MODE=true
            shift
            ;;
        -*)
            echo "Error: Unknown option '$1'"
            echo "Use '$0 --help' for usage information"
            exit 1
            ;;
        *)
            # 位置参数作为容器名称
            CONTAINER_NAMES+=("$1")
            shift
            ;;
    esac
done

# 如果指定了 --all，获取所有 real-mirror-* 容器
if [ "$DELETE_ALL" = true ]; then
    echo "Searching for all real-mirror-* containers..."
    mapfile -t CONTAINER_NAMES < <(docker ps -a --format '{{.Names}}' | grep '^real-mirror-' || true)
    
    if [ ${#CONTAINER_NAMES[@]} -eq 0 ]; then
        echo "No real-mirror-* containers found"
        exit 0
    fi
    
    echo "Found ${#CONTAINER_NAMES[@]} containers:"
    printf '  - %s\n' "${CONTAINER_NAMES[@]}"
    
    if [ "$FORCE_MODE" != "true" ]; then
        read -p "Confirm deletion of all these containers? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Operation canceled"
            exit 0
        fi
        # No further confirmation during batch deletion
        FORCE_MODE=true
    fi
fi

# 如果没有指定容器名称，使用默认值
if [ ${#CONTAINER_NAMES[@]} -eq 0 ]; then
    CONTAINER_NAMES=("${DEFAULT_CONTAINER_NAME}")
fi

# 删除所有指定的容器
echo "========================================"
echo "Starting container deletion..."
echo "========================================"

for container in "${CONTAINER_NAMES[@]}"; do
    remove_container "$container" "$FORCE_MODE"
done

echo "========================================"
echo "Completed!"
echo "========================================"