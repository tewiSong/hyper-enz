#!/bin/bash

#SBATCH --job-name=hyperenz_brenda
#SBATCH --output=hyperenz_brenda.%j.out
#SBATCH --error=hyperenz_brenda.%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --constraint=v100|a100|p6000

echo "Date: $(date)"

echo ""
echo "GPU Information:"
nvidia-smi || true
echo ""

# Activate conda environment if not already active
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "hyper-enz" ]; then
  echo "Activating conda environment: hyper-enz"
  source ~/anaconda3/etc/profile.d/conda.sh
  conda activate hyper-enz
fi
echo "Active environment: ${CONDA_DEFAULT_ENV}"

echo ""
echo "Package versions:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || true
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || true
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')" || true

# Threading controls
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export PYTHONHASHSEED=42

# Paths and config
PROJECT_ROOT="/home/songt/hyper-enz"
cd "$PROJECT_ROOT" || exit 1

# Configuration
SAVE_PATH="brenda/run1"          # change per run
CONFIG_NAME="hyper_graph_noise_07"  # see config/hypergraph_brenda_07_all_data.yml
INIT_CKPT=""                     # set to models/<path>/hit10 to resume; leave empty to start fresh

# Data checks (required preprocessed files)
GRAPH_INFO="${PROJECT_ROOT}/pre_handle_data/all/brenda_bigger_lhf_no_e_add_edge_type_add_ne_reaction_graph_info.pkl"
TRAIN_INFO="${PROJECT_ROOT}/pre_handle_data/all/brenda_bigger_lhf_no_e_add_edge_type_add_ne_reaction_train_info.pkl"
if [ ! -f "$GRAPH_INFO" ] || [ ! -f "$TRAIN_INFO" ]; then
  echo "Missing preprocessed data files under pre_handle_data/all/. Aborting."
  exit 1
fi

echo "Starting training..."
set -e
if [ -z "$INIT_CKPT" ]; then
  python ${PROJECT_ROOT}/hyper_graph_brenda.py \
    --cuda \
    --train \
    --save_path "$SAVE_PATH" \
    --configName "$CONFIG_NAME"
else
  python ${PROJECT_ROOT}/hyper_graph_brenda.py \
    --cuda \
    --train \
    --save_path "$SAVE_PATH" \
    --configName "$CONFIG_NAME" \
    --init "$INIT_CKPT"
fi

EXIT_CODE=${PIPESTATUS[0]}
echo "=================================="
if [ $EXIT_CODE -eq 0 ]; then
  echo "Training completed successfully."
else
  echo "Training failed with exit code: $EXIT_CODE"
fi

echo "End time: $(date)"
echo "Total runtime: $SECONDS seconds"

echo "Final GPU status:"
nvidia-smi || true

exit $EXIT_CODE


