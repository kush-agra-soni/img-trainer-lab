"""Text-to-image generation with SD-Turbo (adversarial diffusion distillation: 1-4 steps, CPU-friendly)."""
import re
import time
import threading
from PIL import Image
from .config import MODELS_DIR

MODELS = {"sd-turbo": {"repo": "stabilityai/sd-turbo", "title": "SD-Turbo (512 px, 1-4 steps)",
                       "license": "Stability AI Non-Commercial Research Community License"}}
BLOCKED = re.compile(r"\b(nude|naked|nsfw|porn|sexual|explicit|child abuse|gore|beheading)\b", re.I)
_pipes: dict[str, object] = {}
_lock = threading.Lock()


class PromptBlocked(ValueError):
    pass


def load(name: str = "sd-turbo", device=None):
    import torch
    from diffusers import AutoPipelineForText2Image

    from .config import pick_device
    device = device or pick_device()
    with _lock:
        key = f"{name}:{device}"
        if key not in _pipes:
            dtype = torch.float16 if device.type == "cuda" else torch.float32
            # fp16 weights are half the download; they are upcast to fp32 for CPU inference
            pipe = AutoPipelineForText2Image.from_pretrained(MODELS[name]["repo"], torch_dtype=dtype, variant="fp16",
                                                             cache_dir=str(MODELS_DIR / "hf"))
            if device.type == "cuda":
                pipe.enable_model_cpu_offload()  # fits small GPUs (e.g. 2 GB) by moving modules on demand
            else:
                pipe.to(device)
            pipe.set_progress_bar_config(disable=True)
            _pipes[key] = pipe
        return _pipes[key]


def generate(prompt: str, steps: int = 1, size: int = 512, n: int = 1, seed: int | None = None,
             model: str = "sd-turbo", device=None, pipe=None) -> dict:
    import torch
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt is empty")
    if BLOCKED.search(prompt):
        raise PromptBlocked("prompt contains blocked content")
    if not 1 <= steps <= 8 or size not in (256, 384, 512) or not 1 <= n <= 4:
        raise ValueError("steps 1-8, size 256/384/512 and 1-4 images are supported")
    pipe = pipe or load(model, device)
    seed = seed if seed is not None else int(time.time()) % 2_000_000_000
    images, times = [], []
    for i in range(n):
        g = torch.Generator().manual_seed(seed + i)
        t0 = time.perf_counter()
        img = pipe(prompt, num_inference_steps=steps, guidance_scale=0.0, height=size, width=size, generator=g).images[0]
        times.append(round(time.perf_counter() - t0, 2))
        images.append(img)
    return {"images": images, "seeds": [seed + i for i in range(n)], "seconds": times, "steps": steps, "size": size,
            "model": model}


def contact_sheet(images: list[Image.Image], cols: int = 2) -> Image.Image:
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (w * min(cols, len(images)), h * rows), "white")
    for i, im in enumerate(images):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    return sheet
