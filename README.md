# Edge Training with Masked PPO

This repository contains the simulation code for memory-aware split inference of ResNet50 on CIFAR-10. A PPO policy selects feasible split points under changing bandwidth, available memory, alpha, and RTT conditions. The experiments compare PPO with delay-only and memory-constrained search baselines.

## Contents

- `config.py`: experiment parameters and repository-relative paths.
- `prepare_data.py`: profile split points and train the CIFAR-10 accuracy curve.
- `train_ppo.py`: train the PPO policy with memory action masking.
- `run_compare.py`: compare the three decision methods.
- `run_ablation.py`: run ablation experiments.
- `draw_figures.py`: generate the paper figures and tables from experiment CSV files.
- `figure1/`, `figure2/`, `figure3/`: figure-generation scripts and source CSV inputs.
- `ANALYSIS.md`, `DESIGN.md`: experiment analysis and design notes.

Generated results are written to `outputs/`, which is intentionally ignored by Git. The CIFAR-10 binary dataset is also ignored because it is large; place the extracted `cifar-10-batches-py/` directory in the repository root before running `prepare_data.py`.

## Setup

Use Python 3.10+ and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

For GPU acceleration, install the PyTorch build appropriate for your CUDA version from the official PyTorch instructions.

## Run

Run the stages from the repository root:

```bash
python prepare_data.py
python train_ppo.py
python run_compare.py
python draw_figures.py
```

Set `QUICK_MODE = True` in `config.py` for a short validation run. The default `False` setting runs the full paper configuration.

The code uses `config.py` paths relative to the repository, so it can be cloned and run on another machine without editing a machine-specific drive path.

## License

This project is released under the MIT License. See `LICENSE`.
