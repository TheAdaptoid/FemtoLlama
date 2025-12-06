"""
Training script for FemtoLlama models on next-token prediction.

Usage:
    python train.py --mix control --batch-size 32 --eval-steps 500 --epochs 5
    python train.py --mix all --batch-size 16 --eval-steps 1000
    python train.py --mix more_heads --batch-size 64 --epochs 10
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from architecture import FemtoLlama

# Configuration
VOCAB_SIZE: int = 10_000
CONTEXT_WINDOW: int = 512
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SequenceDataset(Dataset):
    """Dataset for next-token prediction from numpy arrays."""

    def __init__(self, sequences_path: str):
        """
        Initialize dataset.

        Args:
            sequences_path: Path to .npy file with sequences of shape
                (N, seq_len+1).
        """
        self.sequences = np.load(sequences_path, mmap_mode="r")
        self.length = len(self.sequences)

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return self.length

    def __getitem__(self, idx):
        """Return (input_ids, target_id) for next-token prediction."""
        seq = self.sequences[idx]  # shape (seq_len+1,)
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        target_id = torch.tensor(seq[-1], dtype=torch.long)
        return input_ids, target_id


def create_dataloaders(batch_size: int, num_workers: int = 0):
    """Create train and eval dataloaders."""
    train_dataset = SequenceDataset("data/train_sequences.npy")
    eval_dataset = SequenceDataset("data/test_sequences.npy")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, eval_loader


def load_model_configs() -> dict:
    """Load model configurations from JSON."""
    with open("artifacts/model_configs.json", "r") as f:
        return json.load(f)


def build_model(mix_name: str, configs: dict) -> FemtoLlama:
    """Build FemtoLlama model for the given mix."""
    if mix_name not in configs:
        raise ValueError(f"Unknown mix: {mix_name}. Available: {list(configs.keys())}")

    cfg = configs[mix_name]
    model = FemtoLlama(
        vocab_size=VOCAB_SIZE,
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        context_window_len=CONTEXT_WINDOW,
    )
    return model.to(DEVICE)


def save_checkpoint(model: nn.Module, path: str):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(model: nn.Module, path: str):
    """Load model checkpoint."""
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        print(f"Checkpoint loaded: {path}")
    else:
        print(f"Checkpoint not found: {path}")


def evaluate(model: nn.Module, eval_loader: DataLoader) -> float:
    """Evaluate model on eval set and return average loss."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for input_ids, target_ids in eval_loader:
            input_ids = input_ids.to(DEVICE)
            target_ids = target_ids.to(DEVICE)

            logits = model(input_ids)  # shape (batch, seq_len, vocab_size)
            loss = criterion(logits[:, -1, :], target_ids)
            total_loss += loss.item()
            num_batches += 1

    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0


def train_model(
    mix_name: str,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    model: nn.Module,
    num_epochs: int,
    eval_steps: int,
    save_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train model and return arrays of losses.

    Returns:
        (train_losses, val_losses): Arrays of losses at eval steps.
    """
    optimizer = AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    csv_path = os.path.join(log_dir, f"femtollama_{mix_name}_train.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "epoch", "train_loss", "val_loss", "lr", "timestamp"])
    csv_file.flush()

    train_losses = []
    val_losses = []
    step_losses = []
    global_step = 0
    best_val_loss = float("inf")
    start_time = time.time()

    model.train()

    for epoch in range(num_epochs):
        print(f"\n[{mix_name}] Epoch {epoch + 1}/{num_epochs}")
        epoch_start = time.time()

        for input_ids, target_ids in train_loader:
            input_ids = input_ids.to(DEVICE)
            target_ids = target_ids.to(DEVICE)

            # Forward pass
            logits = model(input_ids)  # shape (batch, seq_len, vocab_size)
            loss = criterion(logits[:, -1, :], target_ids)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_losses.append(loss.item())
            global_step += 1

            # Evaluate and log every eval_steps
            if global_step % eval_steps == 0:
                val_loss = evaluate(model, eval_loader)
                avg_train_loss = np.mean(step_losses)
                elapsed = time.time() - start_time
                lr = optimizer.param_groups[0]["lr"]
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                train_losses.append(avg_train_loss)
                val_losses.append(val_loss)

                print(
                    f"  Step {global_step}: train_loss={avg_train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, lr={lr:.2e}, elapsed={elapsed:.1f}s"
                )

                csv_writer.writerow(
                    [
                        global_step,
                        epoch + 1,
                        f"{avg_train_loss:.6f}",
                        f"{val_loss:.6f}",
                        f"{lr:.2e}",
                        timestamp,
                    ]
                )
                csv_file.flush()

                # Save checkpoint if val_loss improves
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(
                        save_dir, f"femtollama_{mix_name}_best.pth"
                    )
                    save_checkpoint(model, best_path)

                step_losses = []

        # Save epoch checkpoint
        epoch_path = os.path.join(
            save_dir, f"femtollama_{mix_name}_epoch_{epoch + 1}.pth"
        )
        save_checkpoint(model, epoch_path)

        epoch_time = time.time() - epoch_start
        print(f"  Epoch {epoch + 1} completed in {epoch_time:.1f}s")

    csv_file.close()

    # Save loss arrays
    train_losses_arr: np.ndarray = np.array(train_losses)
    val_losses_arr: np.ndarray = np.array(val_losses)
    np.save(
        os.path.join(log_dir, f"femtollama_{mix_name}_train_losses.npy"),
        train_losses_arr,
    )
    np.save(
        os.path.join(log_dir, f"femtollama_{mix_name}_val_losses.npy"),
        val_losses_arr,
    )

    print(f"\nTraining complete for {mix_name}")
    print(f"  CSV log: {csv_path}")

    return train_losses_arr, val_losses_arr


def plot_results(
    mix_name: str,
    train_losses: np.ndarray,
    val_losses: np.ndarray,
    save_dir: str,
) -> None:
    """Generate and save matplotlib plots for training results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return

    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    steps = np.arange(1, len(train_losses) + 1)
    epochs = np.arange(1, len(train_losses) + 1)

    # Loss vs Step
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, label="Train Loss", marker="o", alpha=0.7)
    plt.plot(steps, val_losses, label="Val Loss", marker="s", alpha=0.7)
    plt.xlabel("Evaluation Step")
    plt.ylabel("Loss")
    plt.title(f"{mix_name}: Loss vs Step")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    step_png = os.path.join(plots_dir, f"{mix_name}_loss_vs_step.png")
    step_pdf = os.path.join(plots_dir, f"{mix_name}_loss_vs_step.pdf")
    plt.savefig(step_png, dpi=300)
    plt.savefig(step_pdf)
    plt.close()
    print(f"  Plot saved: {step_png}, {step_pdf}")

    # Loss vs Epoch
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o", alpha=0.7)
    plt.plot(epochs, val_losses, label="Val Loss", marker="s", alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{mix_name}: Loss vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    epoch_png = os.path.join(plots_dir, f"{mix_name}_loss_vs_epoch.png")
    epoch_pdf = os.path.join(plots_dir, f"{mix_name}_loss_vs_epoch.pdf")
    plt.savefig(epoch_png, dpi=300)
    plt.savefig(epoch_pdf)
    plt.close()
    print(f"  Plot saved: {epoch_png}, {epoch_pdf}")

    # Perplexity vs Epoch
    perplexity = np.exp(val_losses)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, perplexity, marker="o", alpha=0.7, color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title(f"{mix_name}: Perplexity vs Epoch")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    ppl_png = os.path.join(plots_dir, f"{mix_name}_perplexity_vs_epoch.png")
    ppl_pdf = os.path.join(plots_dir, f"{mix_name}_perplexity_vs_epoch.pdf")
    plt.savefig(ppl_png, dpi=300)
    plt.savefig(ppl_pdf)
    plt.close()
    print(f"  Plot saved: {ppl_png}, {ppl_pdf}")


def main() -> None:
    """Train FemtoLlama models on next-token prediction task."""
    parser = argparse.ArgumentParser(
        description="Train FemtoLlama models on next-token prediction."
    )
    parser.add_argument(
        "--mix",
        type=str,
        default="control",
        help=(
            "Model mix to train: control, more_heads, more_layers, "
            "more_dimensions, or all"
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="artifacts",
        help="Directory to save checkpoints, logs, and plots",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers",
    )

    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Vocab size: {VOCAB_SIZE}")
    print(f"Context window: {CONTEXT_WINDOW}")

    # Load configs
    configs = load_model_configs()
    mixes = configs.keys() if args.mix == "all" else [args.mix]

    # Create dataloaders once
    train_loader, eval_loader = create_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Train each model mix
    for mix in mixes:
        print(f"\n{'=' * 60}")
        print(f"Training {mix}")
        print(f"{'=' * 60}")

        model = build_model(mix, configs)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model {mix} built. Params: {num_params:,}")

        train_losses, val_losses = train_model(
            mix,
            train_loader,
            eval_loader,
            model,
            num_epochs=args.epochs,
            eval_steps=args.eval_steps,
            save_dir=args.save_dir,
        )

        # Generate plots
        print(f"\nGenerating plots for {mix}...")
        plot_results(mix, train_losses, val_losses, args.save_dir)

    print(f"\n{'=' * 60}")
    print("All training complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
