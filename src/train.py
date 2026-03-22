import torch
from src.gpt_download3 import download_and_load_gpt2
from src.model import GPTModel, GPT2_SMALL_CONFIG, load_weights_into_gpt


from functools import partial
from src.dataset import custom_collate_fn, InstructionDataset

from torch.utils.data import DataLoader
from src.config import config

import torch
import torch.nn.functional as F

from src.model import GPTModel, GPT2_SMALL_CONFIG

def load_pretrained_gpt(device):
    # Download GPT-2 weights
    settings, params = download_and_load_gpt2(
        model_size="124M",
        models_dir="gpt2"
    )

    print("✅ Loaded GPT-2 pretrained weights")

    # Initialize model
    model = GPTModel(GPT2_SMALL_CONFIG)

    return model, params


def initialize_model(params, device):
    # Initialize model
    gpt = GPTModel(GPT2_SMALL_CONFIG)

    # Load pretrained weights
    load_weights_into_gpt(gpt, params)

    # Move to device
    gpt = gpt.to(device)

    print("✅ GPT-2 Small model ready!")

    return gpt


def calc_loss_batch(input_batch, target_batch, model):
    logits = model(input_batch)

    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_batch.view(-1),
        ignore_index=-100
    )

    return loss


def calc_loss_loader(data_loader, model, num_batches=None):
    total_loss = 0.0

    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break

        loss = calc_loss_batch(input_batch, target_batch, model)
        total_loss += loss.item()

    return total_loss / num_batches

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device



def get_collate_fn(device):
    return partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=1024
    )



def prepare_dataloaders(train_data, val_data, test_data, tokenizer, collate_fn):
    # Reproducibility
    torch.manual_seed(123)

    # Datasets
    train_dataset = InstructionDataset(train_data, tokenizer)
    val_dataset   = InstructionDataset(val_data, tokenizer)
    test_dataset  = InstructionDataset(test_data, tokenizer)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
    )

    return train_loader, val_loader, test_loader


def setup_finetuning(gpt, num_unfrozen_blocks=8):
    # Freeze all parameters
    for param in gpt.parameters():
        param.requires_grad = False

    # Unfreeze last N transformer blocks
    for block in gpt.trf_blocks[-num_unfrozen_blocks:]:
        for param in block.parameters():
            param.requires_grad = True

    # Unfreeze final normalization
    for param in gpt.final_norm.parameters():
        param.requires_grad = True

    # Unfreeze output head
    for param in gpt.out_head.parameters():
        param.requires_grad = True

    print(f"✅ Fine-tuning last {num_unfrozen_blocks} blocks + head")

    return gpt


import torch
import torch.nn.functional as F
from tqdm import tqdm
import math


def train_model(gpt, train_loader, val_loader, optimizer, device, config):
    EPOCHS = config.epochs
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):

        print("\n" + "=" * 70)
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print("=" * 70)

        # -------------------
        # TRAINING
        # -------------------
        gpt.train()
        train_loss = 0

        progress = tqdm(train_loader, desc="Training", leave=False)

        for inputs, targets in progress:
            inputs, targets = inputs.to(device), targets.to(device)

            logits = gpt(inputs)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)

            optimizer.step()

            train_loss += loss.item()

            progress.set_postfix(
                loss=loss.item(),
                grad_norm=float(grad_norm)
            )

        avg_train_loss = train_loss / len(train_loader)
        train_ppl = math.exp(avg_train_loss)

        print(f"\nAvg Train Loss: {avg_train_loss:.4f}")
        print(f"Train Perplexity: {train_ppl:.4f}")

        # -------------------
        # VALIDATION
        # -------------------
        gpt.eval()
        val_loss = 0

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validating", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)

                logits = gpt(inputs)

                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-100
                )

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_ppl = math.exp(avg_val_loss)

        print(f"✅ Avg Val Loss: {avg_val_loss:.4f}")
        print(f"Validation Perplexity: {val_ppl:.4f}")

        # -------------------
        # SAVE BEST MODEL
        # -------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            torch.save({
                "epoch": epoch + 1,
                "model_state": gpt.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": best_val_loss
            }, "checkpoints/best_model.pth")

            print("💾 Saved Best Model → checkpoints/best_model.pth")

    print("\n✅ Fine-tuning complete!")
    print("Best Validation Loss:", best_val_loss)

    return gpt