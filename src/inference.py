import os
import torch
import tiktoken

from src.model import GPTModel, GPT2_SMALL_CONFIG


def load_finetuned_model(checkpoint_path, device, strict=True):
    """
    Load fine-tuned GPT model from checkpoint.
    """

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Initialize model
    gpt = GPTModel(GPT2_SMALL_CONFIG)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "model_state" in checkpoint:
        gpt.load_state_dict(checkpoint["model_state"], strict=strict)

        print("Checkpoint Info:")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A')}")
    else:
        # Direct state_dict
        gpt.load_state_dict(checkpoint, strict=strict)
        print("Loaded raw state_dict")

    # Move to device
    gpt = gpt.to(device)
    gpt.eval()

    print("Fine-tuned model loaded successfully!")

    return gpt


def load_tokenizer():
    """
    Load GPT-2 tokenizer.
    """
    return tiktoken.get_encoding("gpt2")