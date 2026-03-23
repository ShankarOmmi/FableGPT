import json
import random
import torch
import tiktoken
from torch.utils.data import Dataset


def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_input(entry):
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input:\n{entry['input']}"
        if entry.get("input") else ""
    )

    return instruction_text + input_text


def format_training_text(entry):
    prompt = format_input(entry)
    response_text = f"\n\n### Response:\n{entry.get('output', '')}"
    return prompt + response_text


def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


def tokenize_entry(entry, tokenizer):
    full_text = format_training_text(entry)
    token_ids = tokenizer.encode(full_text)
    return token_ids


def split_dataset(data, train_split=0.8, test_split=0.1, seed=123):
    # Reproducibility
    random.seed(seed)
    data = data.copy()
    random.shuffle(data)

    train_portion = int(len(data) * train_split)
    test_portion  = int(len(data) * test_split)

    train_data = data[:train_portion]
    test_data  = data[train_portion:train_portion + test_portion]
    val_data   = data[train_portion + test_portion:]

    return train_data, val_data, test_data


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    """
    Collate function for InstructionDataset.
    Expects each item to have:
    - 'token_ids'
    - 'response_start'
    """

    batch_max_length = max(len(item['token_ids']) + 1 for item in batch)

    inputs_lst, targets_lst = [], []

    for item in batch:
        token_ids = item['token_ids']
        response_start = item['response_start']
        original_len = len(token_ids)

        # Add EOS token
        token_ids_with_eos = token_ids + [pad_token_id]

        # Pad
        padded = token_ids_with_eos + [pad_token_id] * (
            batch_max_length - len(token_ids_with_eos)
        )

        # Shift for next-token prediction
        inputs = torch.tensor(padded[:-1], dtype=torch.long).to(device)
        targets = torch.tensor(padded[1:], dtype=torch.long).to(device)

        # Mask instruction + input tokens (only train on response)
        if response_start > 0:
            mask_until = response_start - 1
            targets[:mask_until] = ignore_index

        # Mask padding tokens beyond sequence + EOS
        targets[original_len + 1:] = ignore_index

        # Truncate to max allowed length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    return (
        torch.stack(inputs_lst),
        torch.stack(targets_lst),
    )


class InstructionDataset(Dataset):
    """
    Dataset for instruction tuning.
    Stores tokenized text + response start index.
    """

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.encoded_texts = []

        response_marker = "\n\n### Response:\n"
        self.response_marker_ids = tokenizer.encode(response_marker)

        failed_count = 0

        for entry in data:
            # Build full text
            full_text = format_training_text(entry)

            # Tokenize
            token_ids = tokenizer.encode(full_text)

            # Find response start
            response_start = self._find_response_start(token_ids)

            if response_start is None:
                failed_count += 1
                response_start = 0  # fallback: no masking applied

            self.encoded_texts.append({
                "token_ids": token_ids,
                "response_start": response_start,
            })

        if failed_count > 0:
            print(
                f"Warning: Response marker not found in {failed_count}/{len(data)} examples"
            )

    def _find_response_start(self, token_ids):
        """Locate start index of response tokens."""
        marker = self.response_marker_ids
        marker_len = len(marker)

        for i in range(len(token_ids) - marker_len + 1):
            if token_ids[i:i + marker_len] == marker:
                return i + marker_len

        return None

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)