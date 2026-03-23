import torch
import os

from src.inference import load_finetuned_model
from src.utils import (
    create_instruction_prompt,
    generate_with_eos,
    text_to_token_ids,
    token_ids_to_text
)
from src.model import GPT2_SMALL_CONFIG
from src.config import config


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    checkpoint_path = "checkpoints/best_model.pth"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Model checkpoint not found.")

    # Load model
    gpt = load_finetuned_model(checkpoint_path, device)

    # Tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    instruction = "Try to write a fable/moral story with the characters and title given below."
    input_text = "Input 1 (Characters): Camel, Horse\nInput 2 (Title): The Long Journey"

    prompt = create_instruction_prompt(instruction, input_text)

    input_ids = text_to_token_ids(prompt, tokenizer, device)

    torch.manual_seed(config.seed)

    generated_ids = generate_with_eos(
        model=gpt,
        idx=input_ids,
        max_new_tokens=config.max_new_tokens,
        context_size=GPT2_SMALL_CONFIG["Context_length"],
        top_k=config.top_k,
        temperature=config.temperature,
        eos_id=50256
    )

    output = token_ids_to_text(generated_ids, tokenizer)

    print("\n" + "="*80)
    print("📜 GENERATED STORY")
    print("="*80)

    if "### Response:\n" in output:
        print(output.split("### Response:\n", 1)[1])
    else:
        print(output)


if __name__ == "__main__":
    main()