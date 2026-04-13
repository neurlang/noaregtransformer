"""
noareg_multi.py — trainer for the multi.tsv homograph disambiguation task.

Each sample is a sentence where homographs expand to N candidate slots.
The model learns to output 0xFFFFFFFF for the correct candidate and 0 for others.
Non-homograph slots are masked out of the loss (they're always 0xFFFFFFFF / trivial).

Usage:
    uv run --with torch noareg_multi.py
    uv run --with torch noareg_multi.py --lexicon real_dataset/lexicon.tsv \
        --multi real_dataset/multi.tsv --output weights_multi.bin
"""

import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from noareg_multidata import load_lexicon, load_multi
from noareg_traindata import int_to_bits_tensor, bits_to_int
from noareg_transformer import NoaregTransformer


def parse_args():
    p = argparse.ArgumentParser(description="Train homograph disambiguator on multi.tsv")
    p.add_argument("--lexicon",  default="toy_dataset/lexicon.tsv")
    p.add_argument("--multi",    default="toy_dataset/multi.tsv")
    p.add_argument("--output",   default="weights_multi.bin")
    p.add_argument("--seq-len",  type=int,   default=32)
    p.add_argument("--batch",    type=int,   default=512)
    p.add_argument("--lr",       type=float, default=0.01)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--patience", type=int,   default=120,
                   help="Stop if no improvement for this many seconds")
    return p.parse_args()


def active_mask(inputs):
    """
    Returns [B, L] bool mask: True for non-padding slots (input != 0).
    In the multi format, both 0x0 and 0xffffffff are valid targets,
    so we can't use mask_we_dont_care. Instead mask by input == 0 (padding).
    """
    # inputs: [B, L, 32] bits — padding slots have all-zero input
    return inputs.abs().sum(dim=-1) > 0  # [B, L]


# Precomputed bit pattern for 0xAAAAAAAA (alternating 10101010... LSB first)
_AAAA_BITS = torch.tensor([(0xAAAAAAAA >> i) & 1 for i in range(32)], dtype=torch.float32)


def irrelevant_mask(targets):
    """
    Returns [B, L] bool mask: True for slots whose target is 0xAAAAAAAA.
    These are non-homograph slots — irrelevant, should be excluded from loss.
    """
    global _AAAA_BITS
    if _AAAA_BITS.device != targets.device:
        _AAAA_BITS = _AAAA_BITS.to(targets.device)
    return (targets == _AAAA_BITS).all(dim=-1)  # [B, L]


def evaluate(model, inputs, outputs, device):
    """
    Returns accuracy: fraction of active (non-padding) slots predicted correctly.
    """
    model.eval()
    with torch.no_grad():
        preds = model(inputs.to(device))
    preds_bin = (preds > 0.5).float()
    targets = outputs.to(device)

    care = active_mask(inputs.to(device)) & ~irrelevant_mask(outputs.to(device))  # [B, L]
    if care.sum() == 0:
        return 0.0

    correct_bits = (preds_bin == targets).all(dim=-1)  # [B, L]
    acc = correct_bits[care].float().mean().item()
    model.train()
    return acc


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Load data ---
    lexicon = load_lexicon(args.lexicon)
    print(f"Lexicon: {len(lexicon)} words, "
          f"{sum(1 for v in lexicon.values() if len(v) > 1)} homographs")

    inputs_ints, outputs_ints = load_multi(args.multi, lexicon, seq_len=args.seq_len)
    print(f"Samples: {len(inputs_ints)}")

    if len(inputs_ints) == 0:
        print("No training data found — check your tsv files.")
        return

    train_inputs = int_to_bits_tensor(
        torch.tensor(inputs_ints, dtype=torch.uint32, device=device)
    )

    train_targets = int_to_bits_tensor(
        torch.tensor(outputs_ints, dtype=torch.uint32, device=device)
    )
    print(f"Input tensor:  {train_inputs.shape}")
    print(f"Target tensor: {train_targets.shape}")

    # --- Model ---
    model = NoaregTransformer().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    dataset = TensorDataset(train_inputs, train_targets)
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    best_loss    = float('inf')
    best_acc     = 0.0
    best_model   = None
    epoch_time   = time.time()
    step         = 0

    for epoch in range(100_000):
        if time.time() - epoch_time > args.patience:
            print("Patience reached — stopping.")
            break

        for batch_x, batch_y in loader:
            batch_x = batch_x.float()
            batch_y = batch_y.float()
            optimizer.zero_grad()

            # Only train on non-padding slots (input != 0)
            # and exclude 0xAAAAAAAA output slots (non-homograph, irrelevant)
            mask = active_mask(batch_x) & ~irrelevant_mask(batch_y)  # [B, L]
            mask_flat = mask.unsqueeze(-1).expand_as(batch_y)  # [B, L, 32]

            y_pred = model(batch_x)

            loss = criterion(y_pred[mask_flat], batch_y[mask_flat])
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            step += 1

            if loss_val < best_loss:
                best_loss  = loss_val
                best_model = model.copy().to('cpu')
                acc = evaluate(best_model, train_inputs.cpu(), train_targets.cpu(), 'cpu')
                print(f"Step {step:4d} | loss {best_loss:.6f} | acc {acc*100:.1f}%")

                if acc > best_acc:
                    best_acc   = acc
                    epoch_time = time.time()  # reset patience on improvement
                    best_model.eval()
                    best_model.export_to_binary(args.output)
                    print(f"  -> saved {args.output}  (acc {best_acc*100:.1f}%)")

        print(f"Epoch {epoch} | best_loss {best_loss:.6f} | best_acc {best_acc*100:.1f}%")

    print(f"\nDone. Best accuracy: {best_acc*100:.1f}%")


if __name__ == '__main__':
    main()
