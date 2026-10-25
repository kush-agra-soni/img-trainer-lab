"""Image enhancement: Real-ESRGAN super-resolution (via spandrel) with tiling, plus classical baselines."""
import math
import time
import threading
import numpy as np
from PIL import Image
from pathlib import Path
from .config import MODELS_DIR

MODELS = {
    "realesrgan-general-x4": {
        "title": "Real-ESRGAN general v3 (fast, 4x)",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "scale": 4, "license": "BSD-3-Clause"},
    "realesrnet-x4plus": {
        "title": "Real-ESRNet x4plus (fidelity: best PSNR, smoother)",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
        "scale": 4, "license": "BSD-3-Clause"},
    "realesrgan-x4plus": {
        "title": "Real-ESRGAN x4plus (best quality, slower)",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "scale": 4, "license": "BSD-3-Clause"},
}
TARGETS = {"480p": 480, "720p": 720, "1080p": 1080, "2K": 1440, "4K": 2160}
CLASSIC = {"lanczos": Image.LANCZOS, "bicubic": Image.BICUBIC}
_loaded: dict[str, object] = {}
_lock = threading.Lock()


def download(url: str, dest: Path) -> Path:
    import httpx
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    tmp.replace(dest)
    return dest


def load_model(name: str, device):
    from spandrel import ImageModelDescriptor, ModelLoader
    with _lock:
        key = f"{name}:{device}"
        if key not in _loaded:
            spec = MODELS[name]
            path = download(spec["url"], MODELS_DIR / "sr" / Path(spec["url"]).name)
            m = ModelLoader().load_from_file(path)
            if not isinstance(m, ImageModelDescriptor):
                raise ValueError(f"{name} is not an image-to-image model")
            _loaded[key] = m.to(device).eval()
        return _loaded[key]


def _run_tiles(model, img: np.ndarray, device, tile: int = 256, pad: int = 12) -> np.ndarray:
    """Run a 4x model over overlapping tiles so memory stays bounded for large inputs."""
    import torch
    h, w, _ = img.shape
    s = model.scale
    out = np.zeros((h * s, w * s, 3), dtype=np.float32)
    x = torch.from_numpy(img).permute(2, 0, 1).float().div(255).unsqueeze(0)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            py0, px0, py1, px1 = max(y0 - pad, 0), max(x0 - pad, 0), min(y1 + pad, h), min(x1 + pad, w)
            with torch.inference_mode():
                o = model(x[:, :, py0:py1, px0:px1].to(device)).clamp(0, 1).cpu()[0].permute(1, 2, 0).numpy()
            oy, ox = (y0 - py0) * s, (x0 - px0) * s
            out[y0 * s:y1 * s, x0 * s:x1 * s] = o[oy:oy + (y1 - y0) * s, ox:ox + (x1 - x0) * s]
    return (out * 255).round().astype(np.uint8)


def target_size(w: int, h: int, target: str | None, factor: int | None) -> tuple[int, int]:
    if target:
        th = TARGETS[target]
        return max(1, round(w * th / h)), th
    f = factor or 4
    return w * f, h * f


def enhance(img: Image.Image, method: str = "realesrgan-general-x4", target: str | None = None,
            factor: int | None = 4, device=None, max_input_pixels: int = 1_200_000) -> dict:
    """Upscale to a named target height (480p..4K) or by a factor. Neural models run 4x passes, then resize."""
    from .config import pick_device
    device = device or pick_device()
    img = img.convert("RGB")
    w, h = img.size
    tw, th = target_size(w, h, target, factor)
    if tw * th > 9000 * 9000:
        raise ValueError("requested output is larger than 81 megapixels")
    if th <= h and tw <= w:
        raise ValueError("target is not larger than the input; choose a higher resolution")
    t0 = time.perf_counter()
    if method in CLASSIC:
        out = img.resize((tw, th), CLASSIC[method])
        passes = 0
    else:
        if w * h > max_input_pixels:
            raise ValueError(f"input is {w * h / 1e6:.1f} MP; neural upscaling accepts up to {max_input_pixels / 1e6:.1f} MP")
        model = load_model(method, device)
        s = model.scale
        passes = 1 if max(tw / w, th / h) <= s + 1e-9 else 2
        # pre-resize so the last neural pass lands exactly on the target instead of overshooting and shrinking
        pre = (max(1, round(tw / s ** passes)), max(1, round(th / s ** passes)))
        cur = np.asarray(img.resize(pre, Image.LANCZOS) if pre != (w, h) else img)
        for _ in range(passes):
            cur = _run_tiles(model, cur, device)
        out = Image.fromarray(cur)
        if out.size != (tw, th):
            out = out.resize((tw, th), Image.LANCZOS)
    return {"image": out, "input_size": [w, h], "output_size": [tw, th], "method": method, "neural_passes": passes,
            "seconds": round(time.perf_counter() - t0, 2), "device": str(device)}


def sharpness(img: Image.Image) -> float:
    """No-reference sharpness: variance of the Laplacian of the grey image (higher = more fine detail)."""
    import cv2
    g = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 1)


def quality_metrics(reference: Image.Image, candidate: Image.Image) -> dict:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    a = np.asarray(reference.convert("RGB"), dtype=np.float64)
    b = np.asarray(candidate.convert("RGB").resize(reference.size, Image.LANCZOS), dtype=np.float64)
    return {"psnr": round(float(peak_signal_noise_ratio(a, b, data_range=255)), 2),
            "ssim": round(float(structural_similarity(a, b, channel_axis=2, data_range=255)), 4)}


def benchmark(img: Image.Image, method: str, factor: int = 4) -> dict:
    """Downscale a high-resolution image, upscale it back, and compare with the original."""
    w, h = img.size
    small = img.convert("RGB").resize((max(1, w // factor), max(1, h // factor)), Image.BICUBIC)
    res = enhance(small, method, factor=factor)
    ref = img.convert("RGB").resize((small.width * factor, small.height * factor))
    return {**quality_metrics(ref, res["image"]), "sharpness": sharpness(res["image"]),
            "reference_sharpness": sharpness(ref), "seconds": res["seconds"], "method": method}
