#!/bin/bash

#SBATCH --job-name=hyperenz_reactzyme
#SBATCH --output=hyperenz_reactzyme.%j.out
#SBATCH --error=hyperenz_reactzyme.%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

echo "Hyper-Enz: ReactZyme Matching (Experimental)"
echo "============================================"
echo "Date: $(date)"
echo "User: $(whoami)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"

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

PROJECT_ROOT="/home/songt/hyper-enz"
cd "$PROJECT_ROOT" || exit 1

# Configuration
SAVE_PATH="reactzyme/run1"
CONFIG_NAME="hyper_graph_noise_07"

# Dependencies check (this pipeline is incomplete in repo)
MISSING=0
if [ ! -f "${PROJECT_ROOT}/pre_handle_data/react_zyme/c_emb_table.pt" ]; then
  echo "Missing: pre_handle_data/react_zyme/c_emb_table.pt"; MISSING=1
fi
if [ ! -f "${PROJECT_ROOT}/pre_handle_data/react_zyme/e_emb_table.pt" ]; then
  echo "Missing: pre_handle_data/react_zyme/e_emb_table.pt"; MISSING=1
fi
if [ ! -d "${PROJECT_ROOT}/util_react" ]; then
  echo "Missing module directory: util_react/ (react_zyme_datasets.py)"; MISSING=1
fi
if [ $MISSING -ne 0 ]; then
  echo "Required assets/modules for ReactZyme pipeline are missing. Aborting."
  exit 2
fi

echo "Starting training..."
set -e
python ${PROJECT_ROOT}/hyper_graph_react_zyme.py \
  --cuda \
  --train \
  --save_path "$SAVE_PATH" \
  --configName "$CONFIG_NAME"

EXIT_CODE=${PIPESTATUS[0]}
echo "============================================"
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


