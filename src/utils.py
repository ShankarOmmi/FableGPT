def show_samples(loader, tokenizer, num_samples=2):
    inputs, targets = next(iter(loader))

    print("\n📌 Sample Training Pairs Preview:\n")

    for i in range(num_samples):
        inp_ids = inputs[i].tolist()
        tgt_ids = targets[i].tolist()

        decoded_inp = tokenizer.decode(
            [x for x in inp_ids if x != 50256]
        )

        decoded_tgt = tokenizer.decode(
            [x for x in tgt_ids if x not in [-100, 50256]]
        )

        print("=" * 80)
        print(f"SAMPLE {i+1}")

        print("\nINPUT (first 100 tokens):")
        print(inp_ids[:100])

        print("\nFULL INPUT TEXT:")
        print(decoded_inp)

        print("\nTARGET (first 100 tokens):")
        print(tgt_ids[:100])

        print("\nFULL TARGET TEXT:")
        print(decoded_tgt)

        print("=" * 80)


def inspect_batch(loader, tokenizer):
    inputs, targets = next(iter(loader))

    print("Input shape:", inputs.shape)
    print("Target shape:", targets.shape)

    print("\nDecoded input preview:\n")
    print(tokenizer.decode(inputs[0].tolist()[:200]))

def inspect_single_sample(loader, tokenizer):
    inputs, targets = next(iter(loader))

    sample_input_ids = inputs[0].tolist()
    sample_target_ids = targets[0].tolist()

    print("Input Token IDs (first 50):")
    print(sample_input_ids[:50])

    print("\nTarget Token IDs (first 50):")
    print(sample_target_ids[:50])

    decoded_input = tokenizer.decode(
        [tid for tid in sample_input_ids if tid != 50256]
    )

    decoded_target = tokenizer.decode(
        [tid for tid in sample_target_ids if tid not in [-100, 50256]]
    )

    print("\n" + "="*80)
    print("DECODED INPUT TEXT:")
    print(decoded_input)

    print("\n" + "="*80)
    print("DECODED TARGET TEXT:")
    print(decoded_target)




def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        # Top-k filtering
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf"), device=logits.device),
                logits
            )

        # Temperature sampling
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Stop if EOS generated
        if eos_id is not None and (idx_next == eos_id).all():
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def text_to_token_ids(text, tokenizer, device):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())



def create_instruction_prompt(instruction, input_text=""):
    """
    Creates prompt in training format.
    """
    prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    )

    prompt += f"\n\n### Instruction:\n{instruction}"

    if input_text:
        prompt += f"\n\n### Input:\n{input_text}"

    prompt += "\n\n### Response:\n"

    return prompt


def generate_with_eos(
    model,
    idx,
    max_new_tokens,
    context_size,
    temperature=0.7,
    top_k=40,
    eos_id=50256
):
    model.eval()

    for step in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        # Top-k filtering
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf"), device=logits.device),
                logits
            )

        # Temperature sampling
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # EOS stopping
        if eos_id is not None and (idx_next == eos_id).any():
            print(f"[Stopped at EOS after {step+1} tokens]")
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx