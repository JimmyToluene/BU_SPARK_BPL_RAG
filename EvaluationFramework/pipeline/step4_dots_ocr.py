"""Step 4 — Run dots.ocr VLM on downloaded page images.

dots.ocr (rednote-hilab/dots.ocr) is a 3B-parameter Vision-Language Model
that replaces Tesseract + multi-tier cleaning with a single end-to-end model.
Handles multi-column newspaper layout and reading order natively.

Two modes:
  vllm         — calls a running vLLM server (recommended for batch)
  transformers — loads model in-process (simpler, slower)
"""

import base64
from pathlib import Path

from .shared import log, HAS_PIL, HAS_OPENAI, HAS_TRANSFORMERS

# Lazy imports — only used when the step actually runs
_Image = None
_OpenAI = None
_torch = None
_AutoModel = None
_AutoProcessor = None


def _ensure_imports(mode: str):
    global _Image, _OpenAI, _torch, _AutoModel, _AutoProcessor
    if HAS_PIL and _Image is None:
        from PIL import Image
        _Image = Image
    if mode == "vllm" and HAS_OPENAI and _OpenAI is None:
        from openai import OpenAI
        _OpenAI = OpenAI
    if mode == "transformers" and HAS_TRANSFORMERS and _torch is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        _torch = torch
        _AutoModel = AutoModelForCausalLM
        _AutoProcessor = AutoProcessor


def _image_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _ocr_vllm(image_path: Path, client, model: str, prompt: str,
              max_tokens: int, timeout: int) -> str:
    img_b64 = _image_to_base64(image_path)
    suffix = image_path.suffix.lstrip(".").lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}.get(suffix, "jpeg")

    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/{mime};base64,{img_b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        timeout=timeout,
    )
    return resp.choices[0].message.content.strip()


def _ocr_transformers(image_path: Path, model, processor, prompt: str,
                      max_tokens: int, device: str) -> str:
    img = _Image.open(image_path).convert("RGB")
    inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
    with _torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    input_len = inputs["input_ids"].shape[1]
    return processor.decode(out_ids[0][input_len:], skip_special_tokens=True).strip()


def _init_backend(cfg: dict):
    """Returns (mode, backend, processor, device) or (None, ...) on failure."""
    dots = cfg.get("dots_ocr", {})
    mode = dots.get("mode", "vllm")
    model_name = dots.get("model", "rednote-hilab/dots.ocr.1.5-3B")

    _ensure_imports(mode)

    if mode == "vllm":
        if not HAS_OPENAI:
            log.error("openai package required. Install: pip install openai")
            return None, None, None, None
        base_url = dots.get("vllm_base_url", "http://localhost:8000/v1")
        client = _OpenAI(base_url=base_url, api_key="dummy")
        try:
            client.models.list()
            log.info(f"  Connected to vLLM at {base_url}")
        except Exception as e:
            log.error(f"  Cannot reach vLLM at {base_url}: {e}")
            log.error(f"  Start: vllm serve {model_name} --dtype bfloat16 --max-model-len 8192")
            return None, None, None, None
        return "vllm", client, None, None

    if mode == "transformers":
        if not HAS_TRANSFORMERS:
            log.error("Need: pip install transformers torch accelerate")
            return None, None, None, None
        if not HAS_PIL:
            log.error("Need: pip install Pillow")
            return None, None, None, None
        device = "cuda" if _torch.cuda.is_available() else "cpu"
        if device == "cpu":
            log.warning("  No CUDA GPU -- dots.ocr will be very slow")
        log.info(f"  Loading {model_name} on {device} ...")
        model = _AutoModel.from_pretrained(
            model_name,
            torch_dtype=_torch.bfloat16 if device == "cuda" else _torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        processor = _AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        return "transformers", model, processor, device

    log.error(f"  Unknown mode: {mode}")
    return None, None, None, None


def run(cfg: dict, periods: list[str]) -> int:
    if not HAS_PIL:
        log.error("Pillow required. Install: pip install Pillow")
        return 0

    log.info("=" * 60)
    log.info("STEP 4: dots.ocr VLM OCR")
    log.info("=" * 60)

    dots = cfg.get("dots_ocr", {})
    model_name = dots.get("model", "rednote-hilab/dots.ocr.1.5-3B")
    prompt = dots.get("prompt", "<ocr>")
    max_tokens = dots.get("max_tokens", 8192)
    timeout = dots.get("timeout", 120)

    out = cfg["output"]
    data_dir = Path(out["data_dir"])
    pages_dir = data_dir / out["pages_subdir"]
    ocr_dir = data_dir / out["ocr_subdir"]
    ocr_dir.mkdir(parents=True, exist_ok=True)

    mode, backend, processor, device = _init_backend(cfg)
    if mode is None:
        return 0

    log.info(f"  Mode  : {mode}  |  Model : {model_name}  |  Prompt : {prompt}")

    issue_dirs = sorted([d for d in pages_dir.iterdir() if d.is_dir()])
    total = 0

    for issue_dir in issue_dirs:
        ark_id = issue_dir.name
        ocr_file = ocr_dir / f"{ark_id}.txt"
        if ocr_file.exists():
            continue

        page_images = sorted(issue_dir.glob("page_*.jpg"))
        if not page_images:
            continue

        parts = []
        for img in page_images:
            try:
                if mode == "vllm":
                    text = _ocr_vllm(img, backend, model_name, prompt, max_tokens, timeout)
                else:
                    text = _ocr_transformers(img, backend, processor, prompt, max_tokens, device)
                parts.append(text)
            except Exception as e:
                log.warning(f"    OCR failed {img.name}: {e}")
                parts.append("")

        ocr_file.write_text(
            "\n\n--- PAGE BREAK ---\n\n".join(parts), encoding="utf-8"
        )
        total += 1
        if total % 5 == 0:
            log.info(f"    ... {total} issues OCR'd")

    log.info(f"\n  Step 4 complete: {total} issues processed")
    return total
