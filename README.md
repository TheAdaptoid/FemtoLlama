# FemtoLlama

## Overview

FemtoLlama is a lightweight implementation of transformer-based language models designed for educational purposes and small-scale experiments.

## Usage

### Set up environment

1. Install `uv` (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
    Or via pip:
   ```bash
   pip install uv
   ```

2. Install dependencies:
   ```bash
   uv venv
   uv sync
   ```

### Pre-training

Open the `pre-training.ipynb` notebook in Jupyter and run all cells to prepare the training data.

### Training

Run the training script with the desired configuration. For example:

```bash
uv run python train.py --mix control --batch-size 32 --eval-steps 500 --epochs 5
```

Available mixes: `control`, `more_heads`, `more_layers`, `more_dimensions`, `all`. Adjust batch size, eval steps, and epochs as needed.

### Evaluation

Run the model comparison script to evaluate trained models:

```bash
uv run python compare_models.py
```

This will generate plots and print a summary of model performance.