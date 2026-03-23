"""
examples/run_examples.py

Demonstrates inference with the fine-tuned GPT-2 model using several
example prompts. Run from the project root:

    python examples/run_examples.py
"""

import torch
import tiktoken
import os

from src.inference import load_finetuned_model
from src.utils import (
    generate_with_eos,
    text_to_token_ids,
    token_ids_to_text,
)
from src.model import GPT2_SMALL_CONFIG
from src.config import config


# ---------------------------------------------------------------------------
# EXACT PROMPT FORMAT (VERY IMPORTANT)
# ---------------------------------------------------------------------------
def create_instruction_prompt(instruction, input_text=""):
    prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    )

    prompt += f"\n\n### Instruction:\n{instruction}"

    if input_text:
        prompt += f"\n\n### Input:\n{input_text}"

    prompt += "\n\n### Response:\n"

    return prompt


# ---------------------------------------------------------------------------
# Example prompts
# ---------------------------------------------------------------------------
EXAMPLES = [
    {
        "instruction": "Try to write a fable/moral story with the characters and title given below.",
        "input": "Input 1 (Characters): Camel, Horse\nInput 2 (Title): The Long Journey",
    },
    {
        "instruction": "Try to write a fable/moral story with the characters and title given below.",
        "input": "Input 1 (Characters): Prince, Princess\nInput 2 (Title): Love and Care is all we need",
    },
    {
        "instruction": "Try to write a fable/moral story with the characters and title given below.",
        "input": "Input 1 (Characters): Witch, Brave boy\nInput 2 (Title): Fortune favours the brave",
    },
]


def run_inference(gpt, tokenizer, device, example, example_num):
    prompt = create_instruction_prompt(
        example["instruction"],
        example.get("input", ""),
    )

    input_ids = text_to_token_ids(prompt, tokenizer, device)


    generated_ids = generate_with_eos(
        model=gpt,
        idx=input_ids,
        max_new_tokens=config.max_new_tokens,
        context_size=GPT2_SMALL_CONFIG["Context_length"],
        top_k=config.top_k,
        temperature=config.temperature,
        eos_id=50256,
    )

    output = token_ids_to_text(generated_ids, tokenizer)

    header = f"\n{'=' * 80}\nEXAMPLE {example_num}\n{'=' * 80}\n"

    if "### Response:\n" in output:
        response = output.split("### Response:\n", 1)[1]
    else:
        response = output

    result_text = header + response.strip() + "\n"

    print(result_text)

    return result_text


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    checkpoint_path = "checkpoints/best_model.pth"

    gpt = load_finetuned_model(checkpoint_path, device)
    tokenizer = tiktoken.get_encoding("gpt2")

    results = []

    for i, example in enumerate(EXAMPLES, start=1):
        result_text = run_inference(gpt, tokenizer, device, example, i)
        results.append(result_text)

    # Save results
    os.makedirs("examples", exist_ok=True)
    output_file = "examples/sample_results.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results))

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()