#!/usr/bin/env python3
"""
Simple demo script for using a local LM Studio OpenAI-compatible endpoint
to generate a small Verilog module. It will:
  - Call the model served by LM Studio
  - Print the full response
  - Try to extract a Verilog code block and save it to demo_module.v

Requirements:
  pip install openai
  LM Studio server running at http://10.8.1.3:1234/v1 with model openai/gpt-oss-20b
"""

from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI


LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://10.8.1.3:1234/v1")
LMSTUDIO_API_KEY = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")  # dummy key is usually enough
LMSTUDIO_MODEL_ID = os.environ.get("LMSTUDIO_MODEL_ID", "openai/gpt-oss-20b")


SYSTEM_PROMPT = (
    "You are an expert digital design and Verilog RTL engineer. "
    "You write clean, synthesizable Verilog with clear module declarations."
)

USER_PROMPT = (
    "Write a simple synthesizable Verilog module called simple_counter that:\n"
    "- Has inputs: clk, rst_n, enable\n"
    "- Has an output: [3:0] count\n"
    "- On reset (rst_n == 0) sets count to 0\n"
    "- When enable is 1, increments count by 1 each clock cycle\n"
    "- When enable is 0, holds the current value.\n\n"
    "Return only Verilog code inside a ```verilog ... ``` code block."
)


def extract_verilog_from_markdown(text: str) -> Optional[str]:
    """Extract Verilog code from a markdown-style ```verilog code block."""
    if "```" not in text:
        return None

    parts = text.split("```")
    # Look for a block starting with 'verilog'
    for block in parts:
        lines = block.splitlines()
        if not lines:
            continue
        first_line = lines[0].strip().lower()
        if first_line == "verilog":
            code_lines = lines[1:]
            code = "\n".join(code_lines).strip()
            if code:
                return code

    # Fallback: if there is any non-empty block, return the first one
    for block in parts:
        candidate = block.strip()
        if candidate:
            return candidate

    return None


def main() -> None:
    client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)

    print(f"Using LM Studio endpoint: {LMSTUDIO_BASE_URL}")
    print(f"Model: {LMSTUDIO_MODEL_ID}")
    print("-" * 80)

    completion = client.chat.completions.create(
        model=LMSTUDIO_MODEL_ID,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        max_tokens=512,
    )

    content = completion.choices[0].message.content or ""

    print("=== Raw model response ===")
    print(content)
    print("=" * 80)

    verilog_code = extract_verilog_from_markdown(content)
    if verilog_code is None:
        print("Could not detect a Verilog code block in the response.")
        return

    output_path = "demo_module.v"
    with open(output_path, "w") as f:
        f.write(verilog_code + "\n")

    print(f"Extracted Verilog code saved to: {output_path}")


if __name__ == "__main__":
    main()

