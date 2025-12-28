# Hyper-Enzyme: Hypergraph Convolution Transformer for Enzymatic Reaction Prediction

## Overview

This project implements a hypergraph-based transformer to model enzymatic reactions and supports two evaluation modes:

1) Link prediction on incomplete reaction equations using a hypergraph convolution transformer.
2) Enzyme–substrate matching: given an enzyme–substrate pair, retrieve the most relevant reaction equation.


## Environment

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


## Running Experiments

Note: Pass --cuda to enable GPU. Checkpoints and logs are written under ./models/<save_path>.

1) BRENDA Link Prediction (hyper_graph_brenda.py)

Train

```bash
python hyper_graph_brenda.py \
  --cuda \
  --train \
  --save_path brenda/run1 \
  --configName hyper_graph_noise_04
```

Evaluate best checkpoint on test set

```bash
python hyper_graph_brenda.py \
  --cuda \
  --test \
  --save_path brenda/run1 \
  --configName hyper_graph_noise_04
```

Metrics printed include MRR, MR, HITS@1/3/10 on relation (reaction type) ranking.


Without these assets, this pipeline will not run. See Caveats below.

## Data Preprocessing

- Default: The repo already includes preprocessed graph files under `pre_handle_data/all/`. Training loads them directly; no preprocessing is required.
- What the trainer loads: see `util/brenda_datasets.py` which reads
  - `pre_handle_data/all/brenda_bigger_lhf_no_e_add_edge_type_add_ne_reaction_graph_info.pkl`
  - `pre_handle_data/all/brenda_bigger_lhf_no_e_add_edge_type_add_ne_reaction_train_info.pkl`

Rebuild the preprocessed data (NE-enhanced version used by training)
- Inputs required under `brenda_07/all/`:
  - `train.json`, `valid.json`, `test.json`, and `e2noereaction.json`
- Commands (run from repo root):
  - `cd util`
  - `python pre_handle_breand_data_ne.py`
- Expected outputs:
  - `pre_handle_data/all/brenda_bigger_lhf_no_e_add_edge_type_add_ne_reaction_graph_info.pkl`
  - `pre_handle_data/all/brenda_bigger_lhf_no_e_add_edge_type_add_ne_reaction_train_info.pkl`

Optional: build a non-NE version
- Command: `cd util && python pre_handle_breand_data.py`
- Outputs go to: `pre_handle_data/brenda_07/`
- Note: The main training code currently loads the NE version under `pre_handle_data/all/`. To switch, update the file paths in `util/brenda_datasets.py` accordingly.

Baselines (no preprocessing needed)
- `baselines/train_lightgcl.py` reads `brenda_07/*.json` directly via `baselines/data/json_brenda.py`.

Notes
- Some SMILES utilities under `util/smiles/transormer_util.py` contain hardcoded absolute paths pointing to an external machine; they are not required for the main training/evaluation pipeline here.

### GPU and logging notes

- Use the --cuda flag to run on GPU; select devices via CUDA_VISIBLE_DEVICES if needed.
- TensorBoard logs are under models/<save_path>/log. Start TensorBoard with:
  - tensorboard --logdir models/<save_path>/log --port 6006

### Configuration

- Use config/hypergraph_brenda_07_all_data.yml. Example configs are hyper_graph_noise_0X.
- Key fields: batch_size, dim, lr, decay, max_step, dropout, neibor_size, test_neibor_size, gamma_s/gamma_t for auxiliary losses.

## Citation
If you compare with, build on, or use aspects of the Hyper-Enz, please cite the following:

```bash
@inproceedings{song2026hyperenz,
  title     = {Improving Enzyme Prediction with Chemical Reaction Equations via Hypergraph-Enhanced Knowledge Graph Embeddings},
  author    = {Song, Tengwei and Yin, Long and Han, Zhen and Xu, Zhiqiang},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '26)},
  year      = {2026},
  month     = aug,
  address   = {Jeju Island, South Korea},
  publisher = {ACM}
}
```
