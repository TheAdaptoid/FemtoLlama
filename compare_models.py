"""
Compare trained FemtoLlama model variants.

Generates combined plots for train/val loss and perplexity, computes next-token
prediction accuracy on the test set, and measures inference latency.

Usage:
    python compare_models.py

Outputs written to `artifacts/plots/` and summary printed to stdout.
"""

import json
import os
import time
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from architecture import FemtoLlama

# Constants
ARTIFACTS_DIR = "artifacts"
LOGS_DIR = os.path.join(ARTIFACTS_DIR, "logs")
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots")
MODEL_CONFIGS = os.path.join(ARTIFACTS_DIR, "model_configs.json")
TEST_SEQS = os.path.join("data", "test_sequences.npy")

# Fixed color scheme: blue, orange, green, red
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
MIX_ORDER = ["control", "more_heads", "more_layers", "more_dimensions"]


class TestDataset(torch.utils.data.Dataset):
    """Simple dataset returning (input_ids, target_id) pairs from sequences .npy."""

    def __init__(self, path: str):
        self.array = np.load(path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.array)

    def __getitem__(self, idx):
        seq = self.array[idx]
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        target = torch.tensor(seq[-1], dtype=torch.long)
        return input_ids, target


def load_logs(mix: str) -> pd.DataFrame:
    """
    Load CSV log for a mix into a DataFrame.

    Expects `artifacts/logs/femtollama_{mix}_train.csv`.
    """
    path = os.path.join(LOGS_DIR, f"femtollama_{mix}_train.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return df


def plot_losses(all_dfs: Dict[str, pd.DataFrame], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Combined Train Loss vs Step
    plt.figure(figsize=(10, 6))
    for i, mix in enumerate(MIX_ORDER):
        if mix not in all_dfs:
            continue
        df = all_dfs[mix]
        plt.plot(df["step"], df["train_loss"], label=mix, color=COLORS[i])
    plt.xlabel("Step")
    plt.ylabel("Train Loss")
    plt.title("Train Loss vs Step (all mixes)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(out_dir, "compare_train_loss.png")
    out_pdf = os.path.join(out_dir, "compare_train_loss.pdf")
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()

    # Combined Val Loss vs Step
    plt.figure(figsize=(10, 6))
    for i, mix in enumerate(MIX_ORDER):
        if mix not in all_dfs:
            continue
        df = all_dfs[mix]
        if "val_loss" in df.columns:
            plt.plot(df["step"], df["val_loss"], label=mix, color=COLORS[i])
    plt.xlabel("Step")
    plt.ylabel("Val Loss")
    plt.title("Validation Loss vs Step (all mixes)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(out_dir, "compare_val_loss.png")
    out_pdf = os.path.join(out_dir, "compare_val_loss.pdf")
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()

    # Perplexity (exp(val_loss)) vs Step
    plt.figure(figsize=(10, 6))
    for i, mix in enumerate(MIX_ORDER):
        if mix not in all_dfs:
            continue
        df = all_dfs[mix]
        if "val_loss" in df.columns:
            ppl = np.exp(df["val_loss"].astype(float))
            plt.plot(df["step"], ppl, label=mix, color=COLORS[i])
    plt.xlabel("Step")
    plt.ylabel("Perplexity")
    plt.title("Perplexity vs Step (all mixes)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(out_dir, "compare_perplexity.png")
    out_pdf = os.path.join(out_dir, "compare_perplexity.pdf")
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()

    print(f"Saved comparison plots to {out_dir}")


def infer_vocab_from_checkpoint(checkpoint_path: str) -> int:
    """
    Load checkpoint state_dict (CPU) and infer vocab size from token_embedding weight.

    This avoids guessing vocab size when constructing the model.
    """
    sd = torch.load(checkpoint_path, map_location="cpu")
    # If the checkpoint is a dict with keys like 'model_state_dict' or similar,
    # try to find nested state_dict
    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    # find a key that ends with 'token_embedding.weight' or 'token_embedding.weight'
    for k, v in sd.items():
        if k.endswith("token_embedding.weight") or k.endswith("token_embedding.weight"):
            return int(v.shape[0])
        # some saves might use 'token_embedding.weight' without suffix
        if k.endswith("embedding.weight") and len(v.shape) == 2:
            return int(v.shape[0])

    # fallback: try tokenizer.json
    tok_path = os.path.join(ARTIFACTS_DIR, "tokenizer.json")
    if os.path.exists(tok_path):
        import json as _json

        with open(tok_path, "r") as f:
            tok = _json.load(f)
        if isinstance(tok, dict) and "model" in tok and "vocab" in tok["model"]:
            return len(tok["model"]["vocab"])

    # last fallback
    return 100_000


def build_and_load_model(mix: str, checkpoint_path: str, configs: Dict) -> nn.Module:
    """Instantiate model with inferred vocab and load checkpoint weights."""
    vocab_size = infer_vocab_from_checkpoint(checkpoint_path)
    cfg = configs[mix]
    model = FemtoLlama(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        context_window_len=512,
    )
    sd = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    model.load_state_dict(sd)
    model.eval()
    return model


def compute_accuracy_and_loss(
    model: nn.Module, batch_size: int = 64
) -> Tuple[float, float]:
    """Compute next-token accuracy and average loss on the test set."""
    dataset = TestDataset(TEST_SEQS)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    total = 0
    correct = 0
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for input_ids, targets in loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            logits = model(input_ids)  # (B, seq_len, vocab)
            last_logits = logits[:, -1, :]  # (B, vocab)
            preds = last_logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.numel()

            loss = criterion(last_logits, targets)
            total_loss += loss.item()
            n_batches += 1

    acc = correct / total if total > 0 else 0.0
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    return acc, avg_loss


def measure_latency(
    model: nn.Module, seq_len: int = 512, runs: int = 200, warmup: int = 20
) -> float:
    """
    Measure average forward pass latency (ms) for a single sample (batch_size=1).

    Returns average milliseconds per forward.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # sample input (batch_size=1)
    dummy = torch.randint(0, 1000, (1, seq_len), dtype=torch.long, device=device)

    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    times = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    return float(np.mean(times))


def main():
    # Load configs and logs
    with open(MODEL_CONFIGS, "r") as f:
        configs = json.load(f)

    all_dfs: Dict[str, pd.DataFrame] = {}
    for mix in MIX_ORDER:
        try:
            df = load_logs(mix)
            all_dfs[mix] = df
        except FileNotFoundError:
            print(f"Log for {mix} not found, skipping plots for it.")

    # Plot combined losses/perplexity
    plot_losses(all_dfs, PLOTS_DIR)

    # For each model, load best checkpoint, compute accuracy, perplexity, latency
    summary = []
    for mix in MIX_ORDER:
        chk = os.path.join(ARTIFACTS_DIR, f"femtollama_{mix}_best.pth")
        if not os.path.exists(chk):
            print(f"Checkpoint for {mix} not found at {chk}, skipping.")
            continue

        print(f"\nEvaluating {mix}...")
        model = build_and_load_model(mix, chk, configs)
        acc, avg_loss = compute_accuracy_and_loss(model, batch_size=64)
        ppl = float(np.exp(avg_loss))
        latency_ms = measure_latency(model, seq_len=512, runs=100, warmup=10)
        summary.append((mix, acc, avg_loss, ppl, latency_ms))
        print(
            f"  Accuracy: {acc * 100:.2f}% | Val Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | Latency: {latency_ms:.2f} ms"
        )

    # Print summary table
    if summary:
        print("\nSummary:")
        print(
            f"{'Mix':<18}{'Accuracy':>10}{'ValLoss':>12}{'Perplexity':>12}{'Latency(ms)':>14}"
        )
        for mix, acc, loss, ppl, lat in summary:
            print(f"{mix:<18}{acc * 100:9.2f}%{loss:12.4f}{ppl:12.2f}{lat:14.2f}")

            # Save summary CSV
            os.makedirs(PLOTS_DIR, exist_ok=True)
            csv_path = os.path.join(PLOTS_DIR, "compare_summary.csv")
            try:
                import csv as _csv

                with open(csv_path, "w", newline="") as fh:
                    writer = _csv.writer(fh)
                    writer.writerow(
                        ["mix", "accuracy", "val_loss", "perplexity", "latency_ms"]
                    )
                    for mix, acc, loss, ppl, lat in summary:
                        writer.writerow(
                            [
                                mix,
                                f"{acc:.6f}",
                                f"{loss:.6f}",
                                f"{ppl:.6f}",
                                f"{lat:.6f}",
                            ]
                        )
                print(f"Saved summary CSV to {csv_path}")
            except Exception as exc:
                print(f"Failed to save summary CSV: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
