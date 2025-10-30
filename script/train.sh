#!/bin/bash

# Exit immediately if a command exits with a non-zero status, and treat unset variables as an error.
set -euo pipefail

# --- Default Values ---
WANDB_ENABLE="false"
DATASET_REPO_ID="local-dummy"
CUDA_DEVICE="0"
VERBOSE_MODE="false"
DATASET_ROOT=""
POLICY_TYPE=""
OUTPUT_DIR_SUFFIX=""

# --- Helper Functions ---

# Function to display usage information
usage() {
    cat <<EOF
Usage: $0 -d <dataset_path> -p <policy_type> -o <output_directory_name> [-g <CUDA_device_id_or_list>] [-v] [-h] [-- <python_args>]

Optimized training script for lerobot policies.

Required Arguments:
  -d <dataset_path>          Path to the root of the dataset.
  -p <policy_type>           The type of policy to train. Available: act | diffusion | smolvla.
  -o <output_directory_name> Suffix for the output directory.

Optional Arguments:
  -g <CUDA_device_id_or_list>  CUDA device ID(s) to use (e.g., "0" or "0,1"). Defaults to "0".
  -v                         Enable verbose/debug mode.
  -h, --help                 Show this help message and exit.

Passing Additional Arguments:
  To pass arguments directly to the underlying 'lerobot.scripts.train' command,
  place them after a '--' separator. All arguments after '--' will be ignored
  by this script and passed directly to the Python script.

  Example:
  $0 -d ./data -p act -o my_run -- --training.batch_size=32 --optimizer.lr=0.0001
EOF
    exit 1
}

# Function to report an error and exit
die() {
    echo "Error: $1" >&2
    echo >&2
    usage
}

# --- Main Logic ---
main() {
    # Check for a standalone --help argument.
    for arg in "$@"; do
      if [[ "$arg" == "--help" ]]; then
        usage
      fi
    done

    # Parse command-line options
    while getopts "d:p:o:g:vh" opt; do
      case ${opt} in
        d) DATASET_ROOT=$OPTARG ;;
        p) POLICY_TYPE=$OPTARG ;;
        o) OUTPUT_DIR_SUFFIX=$OPTARG ;;
        g) CUDA_DEVICE=$OPTARG ;;
        v) VERBOSE_MODE="true" ;;
        h) usage ;;
        ?|:) die "Invalid option or missing argument." ;;
      esac
    done
    shift $((OPTIND -1))

    # --- Parameter Validation ---
    if [[ -z "${DATASET_ROOT}" || -z "${POLICY_TYPE}" || -z "${OUTPUT_DIR_SUFFIX}" ]]; then
        die "Missing required arguments."
    fi

    case "${POLICY_TYPE}" in
        "act"|"diffusion"|"smolvla")
            # Valid policy, continue
            ;;
        *) 
            die "Invalid policy type '${POLICY_TYPE}'. Choose from: act, diffusion, smolvla."
            ;;
    esac

    # --- Environment and Command Setup ---
    if [[ "${VERBOSE_MODE}" == "true" ]]; then
        export ACCELERATE_LOG_LEVEL="DEBUG"
        echo "Enabling detailed debug mode (ACCELERATE_LOG_LEVEL=DEBUG)..."
    fi

    # Special handling for smolvla: it does not support multi-GPU training.
    if [[ "${POLICY_TYPE}" == "smolvla" && "${CUDA_DEVICE}" == *","* ]]; then
        echo "Warning: smolvla policy does not support multi-GPU training. Forcing single-GPU mode."
        # Use the first device from the list
        CUDA_DEVICE=$(echo "${CUDA_DEVICE}" | cut -d',' -f1)
    fi

    local train_cmd_prefix
    local OUTPUT_DIR

    # Decide whether to use accelerate for multi-GPU or python for single-GPU
    if [[ "${CUDA_DEVICE}" == *","* ]]; then
        echo "Detected multi-GPU devices (${CUDA_DEVICE}), using accelerate launch."
        train_cmd_prefix=(accelerate launch --main_process_port 0 --dynamo_backend no -m)
        OUTPUT_DIR="runs/train/${OUTPUT_DIR_SUFFIX}_multi_gpu"
    else
        echo "Detected single-GPU device (${CUDA_DEVICE}), using direct python command."
        train_cmd_prefix=(python -m)
        OUTPUT_DIR="runs/train/${OUTPUT_DIR_SUFFIX}"
    fi

    # --- Training Execution ---
    echo "Starting training for ${POLICY_TYPE} policy..."
    echo "Dataset Path: ${DATASET_ROOT}"
    echo "Output Directory: ${OUTPUT_DIR}"

    # Set environment variables for the training command
    export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}
    export TOKENIZERS_PARALLELISM=False

    # Execute the training command with all arguments
    "${train_cmd_prefix[@]}" lerobot.scripts.train \
        --policy.type "${POLICY_TYPE}" \
        --dataset.root "${DATASET_ROOT}" \
        --dataset.repo_id "${DATASET_REPO_ID}" \
        --output_dir "${OUTPUT_DIR}" \
        --wandb.enable "${WANDB_ENABLE}" \
        "$@"

    echo
    echo "${POLICY_TYPE} policy training finished."
    echo "----------------------------------------"
}

# Run the main function, passing all script arguments to it
main "$@"
