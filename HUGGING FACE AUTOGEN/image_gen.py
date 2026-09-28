"""
Part 2: Image Generation with Hugging Face
===========================================
Generates an image from a text prompt using HF Inference API.

Setup:
  1. Get a free token at https://huggingface.co/settings/tokens
    2. Paste it in .env:  HF_TOKEN=your-huggingface-token-here

Usage:
  python image_gen.py "a cat astronaut on Mars"
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import InferenceClient

MODEL = "black-forest-labs/FLUX.1-schnell"


def get_token():
    token = os.environ.get("HF_TOKEN", "").strip()
    if token and token != "your-hf-token-here":
        return token
    print("No HF_TOKEN found. Get one at https://huggingface.co/settings/tokens")
    print("Then paste it in the .env file.")
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print('Usage: python image_gen.py "your prompt here"')
        sys.exit(1)

    prompt = " ".join(sys.argv[1:])
    token = get_token()
    client = InferenceClient(token=token)

    print(f"Prompt: {prompt}")
    print("Generating...")

    start = time.time()
    try:
        image = client.text_to_image(prompt=prompt, model=MODEL)
    except Exception as e:
        print(f"Error: {e}")
        print("The model may be loading. Try again in a minute.")
        sys.exit(1)

    # Save to generated_images/ folder.
    out_dir = Path(__file__).parent / "generated_images"
    out_dir.mkdir(exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in prompt[:40]).strip("_")
    out_path = out_dir / f"{safe}.png"

    # Handle both PIL Image and raw bytes responses.
    if hasattr(image, "save"):
        image.save(out_path)
    else:
        out_path.write_bytes(image)

    print(f"\nSaved to: {out_path}")
    print(f"Time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
