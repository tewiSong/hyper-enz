# Hyper-Enzyme: Hypergraph Convolution Transformer for Enzymatic Reaction Prediction

Overview

This project implements a hypergraph-based transformer to model enzymatic reactions and supports two evaluation modes:

1) Link prediction on incomplete reaction equations using a hypergraph convolution transformer.
2) Enzyme–substrate matching: given an enzyme–substrate pair, retrieve the most relevant reaction equation.


Environment

Tested with Python 3.9. Please use a dedicated conda environment and appropriate CUDA toolkit.

1) Create and activate the environment

```bash
conda create -n hyper-enz python=3.9 -y
conda activate hyper-enz
```

2) Install PyTorch + CUDA (choose the version matching your system)

```bash
# Example for CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

3) Install PyTorch Geometric and dependencies (match PyTorch/CUDA)

```bash
# Replace cu128 with your CUDA version if different
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-$(python -c "import torch;print(torch.__version__.split('+')[0])")+cu128.html
pip install torch-geometric
```

4) Install remaining Python packages

```bash
pip install numpy scipy pyyaml tensorboard transformers rdkit-pypi torchstat
```


Running Experiments

Note: Pass --cuda to enable GPU. Checkpoints and logs are written under ./models/<save_path>.

1) BRENDA Link Prediction (hyper_graph_brenda.py)

Train

```bash
python hyper_graph_brenda.py \
  --cuda \
  --train \
  --save_path brenda/run1 \
  --configName hyper_graph_noise_07
```

Evaluate best checkpoint on test set

```bash
python hyper_graph_brenda.py \
  --cuda \
  --test \
  --save_path brenda/run1 \
  --configName hyper_graph_noise_07
```

Metrics printed include MRR, MR, HITS@1/3/10 on relation (reaction type) ranking.

2) Enzyme–Substrate Matching (ReactZyme-style) — in progress

The entry script hyper_graph_react_zyme.py and core/HyperGraphReactZyme.py implement the matching head and evaluation, but the dataset builder util_react/react_zyme_datasets.py and required precomputed embeddings referenced by the script (pre_handle_data/react_zyme/c_emb_table.pt and e_emb_table.pt) are not included here. If you provide those assets and the missing util_react module, you can run:

```bash
python hyper_graph_react_zyme.py \
  --cuda \
  --train \
  --save_path reactzyme/run1 \
  --configName hyper_graph_noise_07
```

Without these assets, this pipeline will not run. See Caveats below.

GPU and logging notes

- Use the --cuda flag to run on GPU; select devices via CUDA_VISIBLE_DEVICES if needed.
- TensorBoard logs are under models/<save_path>/log. Start TensorBoard with:
  - tensorboard --logdir models/<save_path>/log --port 6006

Configuration

- Use config/hypergraph_brenda_07_all_data.yml. Example configs are hyper_graph_noise_0X.
- Key fields: batch_size, dim, lr, decay, max_step, dropout, neibor_size, test_neibor_size, gamma_s/gamma_t for auxiliary losses.

