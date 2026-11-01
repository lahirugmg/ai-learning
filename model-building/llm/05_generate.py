"""
Step 5 — Generate: Use the trained model to write text.

Generation works by repeatedly predicting the next word:
  1. Start with a few seed words (the prompt).
  2. Feed them to the model → get probabilities for the next word.
  3. Sample a word from those probabilities.
  4. Append it to the context and repeat.

The `temperature` parameter controls randomness:
  • temperature < 1 → more predictable (sharper distribution)
  • temperature = 1 → sample from the raw probabilities
  • temperature > 1 → more random (flatter distribution)

Run:
    uv run python 05_generate.py
"""

import json

import torch

import importlib

_model_mod = importlib.import_module("03_model")
TinyTransformer = _model_mod.TinyTransformer


def generate(
    model: TinyTransformer,
    word2idx: dict[str, int],
    idx2word: dict[int, str],
    prompt: str,
    max_words: int = 20,
    temperature: float = 0.8,
) -> str:
    """Generate text starting from a prompt.

    Args:
        model:       trained TinyTransformer
        word2idx:    word → integer mapping
        idx2word:    integer → word mapping
        prompt:      starting text (must be context_length words)
        max_words:   how many words to generate
        temperature: controls randomness (lower = more deterministic)
    """
    model.eval()
    tokens = [word2idx[w] for w in prompt.lower().split()]
    context_len = model.context_length

    if len(tokens) < context_len:
        raise ValueError(
            f"Prompt must be at least {context_len} words, got {len(tokens)}"
        )

    generated = list(tokens)

    with torch.no_grad():
        for _ in range(max_words):
            # Take the last context_length tokens
            context = torch.tensor(generated[-context_len:]).unsqueeze(0)
            logits = model(context)

            # Apply temperature
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)

            # Sample from the distribution
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)

    return " ".join(idx2word[t] for t in generated)


if __name__ == "__main__":
    with open("data/vocab.json") as f:
        word2idx = json.load(f)
    idx2word = {int(i): w for w, i in word2idx.items()}

    model = TinyTransformer(len(word2idx), embed_dim=64, context_length=4, num_heads=4)
    model.load_state_dict(torch.load("data/model.pt", weights_only=True))

    prompts = [
        "once upon a time",
        "the cat sat on",
        "the dog played in",
    ]

    print("Generated text (temperature=0.8):\n")
    for prompt in prompts:
        text = generate(model, word2idx, idx2word, prompt, max_words=15)
        print(f"  Prompt: '{prompt}'")
        print(f"  Output: {text}")
        print()

    # Show the effect of temperature
    print("Effect of temperature on 'once upon a time':\n")
    for temp in [0.3, 0.8, 1.5]:
        text = generate(model, word2idx, idx2word, "once upon a time", max_words=12, temperature=temp)
        print(f"  temp={temp}: {text}")
