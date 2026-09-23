"""Text generation through OpenAI or a local Hugging Face model."""

import os
from functools import lru_cache


DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def provider_name():
    return os.environ.get("LLM_PROVIDER", "openai").strip().lower()


def model_name():
    if provider_name() == "huggingface":
        return os.environ.get("HF_MODEL", DEFAULT_HF_MODEL)
    return os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)


def configuration_error():
    """Return a user-facing configuration error, or None when ready."""
    provider = provider_name()
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        return "OPENAI_API_KEY is not set"
    if provider not in {"openai", "huggingface"}:
        return "LLM_PROVIDER must be 'openai' or 'huggingface'"
    return None


@lru_cache(maxsize=1)
def _load_huggingface_model():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face generation requires torch and transformers. "
            "Install them with: pip install torch 'transformers>=4.37'"
        ) from exc

    name = model_name()
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model.to(device)
    model.eval()
    return tokenizer, model, device


def _generate_huggingface(messages, temperature, max_new_tokens):
    import torch

    tokenizer, model, device = _load_huggingface_model()
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    generation_args = {"max_new_tokens": max_new_tokens}
    if temperature > 0:
        generation_args.update({"do_sample": True, "temperature": temperature})
    else:
        generation_args["do_sample"] = False

    with torch.inference_mode():
        output = model.generate(**inputs, **generation_args)

    prompt_tokens = inputs["input_ids"].shape[-1]
    return tokenizer.decode(output[0][prompt_tokens:], skip_special_tokens=True).strip()


def _generate_openai(messages, temperature, max_new_tokens):
    try:
        import openai
    except ImportError as exc:
        raise RuntimeError("OpenAI generation requires the openai package") from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name(),
        messages=messages,
        temperature=temperature,
        max_tokens=max_new_tokens,
    )
    return response.choices[0].message.content.strip()


def generate_chat(messages, temperature=0, max_new_tokens=512):
    """Generate a chat response with the provider selected in .env."""
    error = configuration_error()
    if error:
        raise RuntimeError(error)

    provider = provider_name()
    if provider == "openai":
        return _generate_openai(messages, temperature, max_new_tokens)
    if provider == "huggingface":
        return _generate_huggingface(messages, temperature, max_new_tokens)
    raise RuntimeError("LLM_PROVIDER must be 'openai' or 'huggingface'")
