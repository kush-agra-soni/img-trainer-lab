"""Object detection with torchvision COCO models, image/video annotation and a live webcam streamer."""
import io
import time
import threading
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MODELS = {
    "ssdlite": {"title": "SSDlite MobileNetV3 (fast)", "fn": "ssdlite320_mobilenet_v3_large",
                "weights": "SSDLite320_MobileNet_V3_Large_Weights"},
    "fasterrcnn_mobilenet": {"title": "Faster R-CNN MobileNetV3 FPN (accurate)", "fn": "fasterrcnn_mobilenet_v3_large_fpn",
                             "weights": "FasterRCNN_MobileNet_V3_Large_FPN_Weights"},
}
_loaded: dict[str, tuple] = {}
_lock = threading.Lock()
PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
           (240, 50, 230), (210, 245, 60), (0, 128, 128), (170, 110, 40)]


def load(name: str, device):
    import torchvision.models.detection as tvd
    with _lock:
        key = f"{name}:{device}"
        if key not in _loaded:
            spec = MODELS[name]
            weights = getattr(tvd, spec["weights"]).DEFAULT
            model = getattr(tvd, spec["fn"])(weights=weights).to(device).eval()
            _loaded[key] = (model, weights.meta["categories"], weights.transforms())
        return _loaded[key]


def detect(img: Image.Image, model_name: str = "ssdlite", threshold: float = 0.5, classes: list[str] | None = None,
           device=None) -> dict:
    import torch

    from .config import pick_device
    device = device or pick_device()
    model, cats, tf = load(model_name, device)
    rgb = img.convert("RGB")
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model([tf(rgb).to(device)])[0]
    ms = (time.perf_counter() - t0) * 1000
    dets = []
    wanted = {c.lower() for c in classes} if classes else None
    for box, label, score in zip(out["boxes"].cpu().numpy(), out["labels"].cpu().numpy(), out["scores"].cpu().numpy()):
        name = cats[int(label)]
        if score < threshold or (wanted and name.lower() not in wanted):
            continue
        x0, y0, x1, y1 = (float(v) for v in box)
        dets.append({"label": name, "score": round(float(score), 4), "box": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]})
    counts = {}
    for d in dets:
        counts[d["label"]] = counts.get(d["label"], 0) + 1
    return {"detections": dets, "counts": counts, "ms": round(ms, 1), "size": list(rgb.size), "model": model_name}


def annotate(img: Image.Image, dets: list[dict]) -> Image.Image:
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    lw = max(2, round(min(out.size) / 250))
    try:
        font = ImageFont.truetype("arial.ttf", max(12, round(min(out.size) / 40)))
    except OSError:
        font = ImageFont.load_default()
    for det in dets:
        color = PALETTE[hash(det["label"]) % len(PALETTE)]
        x0, y0, x1, y1 = det["box"]
        d.rectangle([x0, y0, x1, y1], outline=color, width=lw)
        text = f"{det['label']} {det['score']:.0%}"
        tb = d.textbbox((x0, y0), text, font=font)
        d.rectangle([tb[0], tb[1] - 2, tb[2] + 4, tb[3] + 2], fill=color)
        d.text((x0 + 2, y0), text, fill=(255, 255, 255), font=font)
    return out


def to_coco(dets: list[dict], size: list[int], file_name: str) -> dict:
    cats = sorted({d["label"] for d in dets})
    return {"images": [{"id": 1, "file_name": file_name, "width": size[0], "height": size[1]}],
            "categories": [{"id": i + 1, "name": c} for i, c in enumerate(cats)],
            "annotations": [{"id": i + 1, "image_id": 1, "category_id": cats.index(d["label"]) + 1, "score": d["score"],
                             "bbox": [d["box"][0], d["box"][1], d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]]}
                            for i, d in enumerate(dets)]}


def process_video(path: str, out_path: str, model_name: str, threshold: float, every: int = 2, max_frames: int = 600) -> dict:
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps / every, (w, h))
    n, totals, last = 0, {}, []
    t0 = time.perf_counter()
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if n % every == 0:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = detect(pil, model_name, threshold)
            last = res["detections"]
            for k, v in res["counts"].items():
                totals[k] = max(totals.get(k, 0), v)
            writer.write(cv2.cvtColor(np.asarray(annotate(pil, last)), cv2.COLOR_RGB2BGR))
        n += 1
    cap.release()
    writer.release()
    return {"frames_read": n, "frames_processed": (n + every - 1) // every, "max_simultaneous": totals,
            "seconds": round(time.perf_counter() - t0, 1)}


class Webcam:
    """Captures from a local camera on a background thread and keeps the latest annotated frame."""

    def __init__(self):
        self._thread = None
        self._stop = threading.Event()
        self.state = {"running": False, "fps": 0.0, "frames": 0, "error": None, "counts": {}, "detections": []}
        self._jpeg = None
        self._lock = threading.Lock()

    def start(self, index: int = 0, model_name: str = "ssdlite", threshold: float = 0.5, classes=None,
              capture_factory=None) -> None:
        self.stop()
        self._stop.clear()
        self.state.update(running=True, error=None, frames=0, fps=0.0)

        def loop():
            import cv2
            cap = capture_factory(index) if capture_factory else cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                self.state.update(running=False, error=f"Camera {index} could not be opened (in use or no permission)")
                return
            times = []
            try:
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        self.state.update(error="Camera stopped delivering frames")
                        break
                    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    res = detect(pil, model_name, threshold, classes)
                    buf = io.BytesIO()
                    annotate(pil, res["detections"]).save(buf, "JPEG", quality=80)
                    now = time.perf_counter()
                    times = [t for t in times + [now] if now - t < 2]
                    with self._lock:
                        self._jpeg = buf.getvalue()
                    self.state.update(frames=self.state["frames"] + 1, fps=round(len(times) / 2, 1),
                                      counts=res["counts"], detections=res["detections"], ms=res["ms"])
            finally:
                cap.release()
                self.state["running"] = False

        self._thread = threading.Thread(target=loop, daemon=True, name="webcam")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.state["running"] = False

    def frame(self) -> bytes | None:
        with self._lock:
            return self._jpeg
