"""
Part 1: Text Generation with Hugging Face
==========================================
Uses the open-source 'distilgpt2' model from Hugging Face.
Runs locally on CPU - no GPU needed.

Model: distilgpt2 (distilled version of GPT-2, ~350MB)
       https://huggingface.co/distilgpt2

Usage:
  python text_gen.py
  python text_gen.py --prompt "The future of AI is" --max-length 100
  python text_gen.py --model gpt2 --prompt "Once upon a time"
"""

import argparse
import sys
import time

from transformers import pipeline

# Available small text-generation models that run on CPU.
MODELS = {
    "distilgpt2": "distilgpt2",      # ~350MB, fastest
    "gpt2": "gpt2",                  # ~500MB, slightly better quality
    "gpt2-medium": "gpt2-medium",    # ~1.5GB, better but slower
}


def generate_text(prompt: str, model_name: str, max_length: int,
                  temperature: float) -> str:
    """Generate text using a Hugging Face model."""
    print(f"Loading model '{model_name}' (first run downloads it)...")
    start = time.time()

    gen = pipeline(
        "text-generation",
        model=model_name,
        dtype="auto",
    )

    load_time = time.time() - start
    print(f"Model loaded in {load_time:.1f}s\n")

    print(f"Prompt: {prompt}\n")
    print("Generating...")

    gen_start = time.time()
    output = gen(
        prompt,
        max_new_tokens=max_length,
        temperature=temperature,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.15,
        truncation=True,
    )
    gen_time = time.time() - gen_start

    generated = output[0]["generated_text"]
    print(f"--- Generated text ({gen_time:.1f}s) ---\n")
    print(generated)
    print(f"\n--- End (load: {load_time:.1f}s, generation: {gen_time:.1f}s) ---\n")
    return generated


def main():
    parser = argparse.ArgumentParser(description="Hugging Face Text Generation")
    parser.add_argument("--prompt", default="The future of artificial intelligence is",
                        help="Text prompt to continue")
    parser.add_argument("--model", default="distilgpt2",
                        choices=list(MODELS.keys()),
                        help="Model to use")
    parser.add_argument("--max-length", type=int, default=80,
                        help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Creativity (0.1=focused, 1.0=creative)")
    args = parser.parse_args()

    model_name = MODELS[args.model]
    generate_text(args.prompt, model_name, args.max_length, args.temperature)


if __name__ == "__main__":
    main()
