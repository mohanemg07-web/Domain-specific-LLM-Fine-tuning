"""Hugging Face Space -- free CPU inference of the Q4_K_M GGUF.

Loads the quantized GGUF via llama-cpp-python (pure CPU, no GPU needed) and
serves a Gradio chat UI. Tokens/sec is MEASURED live on every response and
displayed -- we do NOT assume "8-12 tok/s"; we print what this hardware
actually delivers.

Config via env:
  GGUF_REPO_ID   HF repo holding the .gguf (e.g. "you/mistral7b-finance-gguf")
  GGUF_FILENAME  filename within that repo (e.g. "model-Q4_K_M.gguf")
  GGUF_PATH      OR a local path (used by the smoke test with a tiny GGUF)
"""

from __future__ import annotations

import os
import time

import gradio as gr

DOMAIN_DISCLAIMER = (
    "Educational demo, fine-tuned on public finance Q&A. "
    "Not financial advice."
)

_llm = None


def _resolve_gguf_path() -> str:
    local = os.environ.get("GGUF_PATH")
    if local and os.path.exists(local):
        return local
    repo_id = os.environ.get("GGUF_REPO_ID")
    filename = os.environ.get("GGUF_FILENAME", "model-Q4_K_M.gguf")
    if not repo_id:
        raise RuntimeError(
            "Set GGUF_PATH (local) or GGUF_REPO_ID + GGUF_FILENAME (hub)."
        )
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename)


def get_llm():
    global _llm
    if _llm is None:
        from llama_cpp import Llama

        _llm = Llama(
            model_path=_resolve_gguf_path(),
            n_ctx=2048,
            n_threads=os.cpu_count(),
            verbose=False,
        )
    return _llm


def chat_fn(message, history):
    llm = get_llm()
    messages = [{"role": "system", "content": DOMAIN_DISCLAIMER}]
    for user, assistant in (history or []):
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": message})

    t0 = time.time()
    out = llm.create_chat_completion(messages=messages, max_tokens=256, temperature=0.7)
    elapsed = time.time() - t0

    text = out["choices"][0]["message"]["content"]
    completion_tokens = out.get("usage", {}).get("completion_tokens", 0)
    tok_per_sec = (completion_tokens / elapsed) if elapsed > 0 else 0.0

    # MEASURED live -- not assumed.
    footer = (
        f"\n\n---\n_{completion_tokens} tokens in {elapsed:.2f}s "
        f"= **{tok_per_sec:.1f} tok/s** (CPU)_"
    )
    return text + footer


def build_demo():
    return gr.ChatInterface(
        fn=chat_fn,
        title="Finance QLoRA (Mistral-7B) -- free CPU demo",
        description=(
            "4-bit Q4_K_M GGUF served on free CPU via llama.cpp. "
            f"{DOMAIN_DISCLAIMER} Tokens/sec is measured live below each reply."
        ),
    )


if __name__ == "__main__":
    build_demo().launch()
