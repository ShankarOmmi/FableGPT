import torch
import tiktoken

from src.inference import load_finetuned_model
from src.utils import (
    create_instruction_prompt,
    generate_with_eos,
    text_to_token_ids,
    token_ids_to_text
)
from src.model import GPT2_SMALL_CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load model
gpt = load_finetuned_model("checkpoints/best_model.pth", device)

# Tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

print("\n" + "="*80)
print("TEST GENERATION: Camel and Horse")
print("="*80)

instruction = "Try to write a fable/moral story with the characters and title given below."
input_text = "Input 1 (Characters): Camel, Horse\nInput 2 (Title): The Long Journey"

# Build prompt
prompt = create_instruction_prompt(instruction, input_text)

print("\nGenerated Prompt:\n", prompt)

# Tokenize
idx = text_to_token_ids(prompt, tokenizer, device)

# Generate
generated_ids = generate_with_eos(
    model=gpt,
    idx=idx,
    max_new_tokens=200,
    context_size=GPT2_SMALL_CONFIG["Context_length"],
    top_k=40,
    temperature=0.7,
    eos_id=50256
)

# Decode
output = token_ids_to_text(generated_ids, tokenizer)

print("\nGenerated Output:\n")

if "### Response:\n" in output:
    print(output.split("### Response:\n", 1)[1])
else:
    print(output)