"""Image AI Platform API: transfer learning, detection (images, video, webcam), generation, enhancement."""
import io
import re
import json
import time
import uuid
import shutil
import threading
import traceback
from pathlib import Path
from . import detect as det
from . import enhance as enh
from . import transfer as tl
from . import generate as gen
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
from fastapi.responses import FileResponse, Response
from .config import Settings, get_settings, pick_device
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
MAX_UPLOAD = 25 * 1024 * 1024


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    steps: int = Field(1, ge=1, le=8)
    size: int = 512
    n: int = Field(1, ge=1, le=4)
    seed: int | None = None


class PublishRequest(BaseModel):
    repo_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    private: bool = True


class WebcamRequest(BaseModel):
    index: int = Field(0, ge=0, le=9)
    model: str = "ssdlite"
    threshold: float = Field(0.5, ge=0.05, le=0.99)
    classes: list[str] | None = None


def create_app(settings: Settings | None = None, *, generate_pipe=None, hf_api=None, capture_factory=None) -> FastAPI:
    settings = settings or get_settings()
    runs = settings.runs_dir
    models_dir = settings.data_dir / "models"
    datasets_dir = settings.data_dir / "datasets"
    gallery = settings.data_dir / "gallery"
    for d in (runs, models_dir, datasets_dir, gallery):
        d.mkdir(parents=True, exist_ok=True)
    jobs: dict[str, dict] = {}
    stops: dict[str, threading.Event] = {}
    cam = det.Webcam()
    app = FastAPI(title="Image AI Platform", version="1.0.0")
    app.state.jobs, app.state.webcam = jobs, cam

    def start_job(kind: str, fn) -> str:
        jid = f"{kind[:4]}-{uuid.uuid4().hex[:8]}"
        job = {"id": jid, "kind": kind, "status": "running", "log": [], "progress": [], "result": None, "error": None,
               "created": time.time()}
        jobs[jid] = job
        stops[jid] = threading.Event()

        def progress(msg: str, data: dict | None = None):
            job["log"].append({"ts": time.time(), "msg": msg})
            if data:
                job["progress"].append(data)

        def run():
            try:
                job["result"] = fn(progress, stops[jid])
                job["status"] = "stopped" if stops[jid].is_set() else "done"
            except Exception as e:
                job["status"], job["error"] = "failed", f"{type(e).__name__}: {e}"
                progress(job["error"])
                progress(traceback.format_exc()[-800:])
            job["finished"] = time.time()

        threading.Thread(target=run, daemon=True, name=jid).start()
        return jid

    async def read_image(file: UploadFile) -> Image.Image:
        data = await file.read()
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "image larger than 25 MB")
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except (UnidentifiedImageError, OSError) as e:
            raise HTTPException(422, "not a readable image") from e
        return img

    def png(img: Image.Image, fmt: str = "PNG") -> bytes:
        buf = io.BytesIO()
        img.save(buf, fmt, **({"quality": 90} if fmt == "JPEG" else {}))
        return buf.getvalue()

    @app.get("/health")
    def health() -> dict:
        import torch
        dev = pick_device(settings.device)
        return {"status": "ok", "device": str(dev), "cuda": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "hf_token": bool(settings.hf_token), "models_trained": len(list(models_dir.glob("*/config.json")))}

    @app.get("/jobs/{jid}")
    def job(jid: str) -> dict:
        j = jobs.get(jid)
        if not j:
            raise HTTPException(404, "job not found")
        return {k: v for k, v in j.items() if k != "result" or not isinstance(v, (bytes, Image.Image))}

    @app.post("/jobs/{jid}/stop")
    def stop_job(jid: str) -> dict:
        if jid not in stops:
            raise HTTPException(404, "job not found")
        stops[jid].set()
        return {"stopping": jid}

    # ---------- transfer learning ----------
    @app.get("/transfer/options")
    def transfer_options() -> dict:
        return {"backbones": {k: v["title"] for k, v in tl.BACKBONES.items()},
                "modes": {"feature_extraction": "Feature extraction (frozen backbone, fastest)",
                          "fine_tune": "Fine-tune top blocks (slower, adapts features)"}}

    @app.post("/transfer/datasets/upload")
    async def upload_dataset(name: str = Form(...), file: UploadFile = File(...)) -> dict:
        if not NAME_RE.match(name):
            raise HTTPException(422, "name: lowercase letters, digits, - and _")
        data = await file.read()
        try:
            root = tl.extract_zip(data, datasets_dir / name)
            classes = tl.scan(root)
            tl.validate(classes)
        except tl.DatasetError as e:
            shutil.rmtree(datasets_dir / name, ignore_errors=True)
            raise HTTPException(422, str(e)) from e
        (datasets_dir / name / ".root").write_text(str(root))
        return {"name": name, "classes": {k: len(v) for k, v in classes.items()}}

    def dataset_root(name: str) -> Path:
        d = datasets_dir / name
        if name.startswith("imagenette_") and d.exists():
            return d
        marker = d / ".root"
        if not marker.exists():
            raise HTTPException(404, "dataset not found")
        return Path(marker.read_text())

    @app.post("/transfer/datasets/demo")
    def demo_dataset(classes: int = 5, per_class: int = 60) -> dict:
        if not 2 <= classes <= 10 or not 10 <= per_class <= 300:
            raise HTTPException(422, "2-10 classes, 10-300 images each")
        jid = start_job("demo-data", lambda progress, stop: (lambda root: {"name": root.name, "classes": {
            k: len(v) for k, v in tl.scan(root).items()}})(tl.imagenette_subset(datasets_dir, per_class, classes)))
        return {"job_id": jid}

    @app.get("/transfer/datasets")
    def list_datasets() -> list[dict]:
        out = []
        for d in sorted(datasets_dir.iterdir()):
            if not d.is_dir():
                continue
            try:
                root = dataset_root(d.name)
            except HTTPException:
                continue
            out.append({"name": d.name, "classes": {k: len(v) for k, v in tl.scan(root).items()}})
        return out

    @app.post("/transfer/train")
    def train(dataset: str = Form(...), name: str = Form(...), backbone: str = Form("mobilenet_v3_large"),
              mode: str = Form("feature_extraction"), epochs: int = Form(10), lr: float = Form(1e-3),
              batch_size: int = Form(32), unfreeze_blocks: int = Form(2), val_fraction: float = Form(0.2)) -> dict:
        if not NAME_RE.match(name):
            raise HTTPException(422, "model name: lowercase letters, digits, - and _")
        if backbone not in tl.BACKBONES or mode not in ("feature_extraction", "fine_tune"):
            raise HTTPException(422, "unknown backbone or mode")
        if not 1 <= epochs <= 100 or not 1e-6 <= lr <= 1 or not 1 <= batch_size <= 256 or not 0.05 <= val_fraction <= 0.5:
            raise HTTPException(422, "invalid hyperparameters")
        root = dataset_root(dataset)
        jid = start_job("train", lambda progress, stop: tl.train(
            root, models_dir / name, backbone, mode, epochs, lr, batch_size, unfreeze_blocks, val_fraction,
            device=pick_device(settings.device), progress=progress, stop=stop.is_set))
        return {"job_id": jid}

    @app.get("/transfer/models")
    def list_models() -> list[dict]:
        out = []
        for cfg in sorted(models_dir.glob("*/config.json")):
            c = json.loads(cfg.read_text())
            out.append({"name": cfg.parent.name, "backbone": c["backbone"], "mode": c["mode"], "labels": c["labels"],
                        "accuracy": c["metrics"]["accuracy"], "f1_macro": c["metrics"]["f1_macro"]})
        return out

    @app.get("/transfer/models/{name}")
    def get_model(name: str) -> dict:
        p = models_dir / name / "config.json"
        if not NAME_RE.match(name) or not p.exists():
            raise HTTPException(404, "model not found")
        return json.loads(p.read_text())

    @app.post("/transfer/models/{name}/classify")
    async def classify(name: str, file: UploadFile = File(...), top_k: int = Form(3)) -> dict:
        if not NAME_RE.match(name) or not (models_dir / name / "config.json").exists():
            raise HTTPException(404, "model not found")
        res = tl.classify(models_dir / name, await read_image(file), top_k)
        res["low_confidence"] = res["predictions"][0]["probability"] < 0.5
        return res

    @app.post("/transfer/models/{name}/publish")
    def publish(name: str, req: PublishRequest) -> dict:
        if not NAME_RE.match(name) or not (models_dir / name / "config.json").exists():
            raise HTTPException(404, "model not found")
        if not settings.hf_token and hf_api is None:
            raise HTTPException(400, "Set HF_TOKEN in .env to publish to the Hugging Face Hub")
        try:
            url = tl.publish(models_dir / name, req.repo_id, settings.hf_token or "", req.private, api=hf_api)
        except Exception as e:
            raise HTTPException(502, f"Hugging Face upload failed: {e}") from e
        return {"url": url, "private": req.private}

    # ---------- detection ----------
    @app.get("/detect/options")
    def detect_options() -> dict:
        import torchvision.models.detection as tvd
        cats = [c for c in tvd.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT.meta["categories"] if c not in ("__background__", "N/A")]
        return {"models": {k: v["title"] for k, v in det.MODELS.items()}, "classes": cats}

    @app.post("/detect/image")
    async def detect_image(file: UploadFile = File(...), model: str = Form("ssdlite"), threshold: float = Form(0.5),
                           classes: str = Form(""), output: str = Form("json")) -> Response:
        if model not in det.MODELS:
            raise HTTPException(422, "unknown model")
        img = await read_image(file)
        res = det.detect(img, model, threshold, [c.strip() for c in classes.split(",") if c.strip()] or None,
                         pick_device(settings.device))
        if output == "image":
            return Response(png(det.annotate(img, res["detections"]), "JPEG"), media_type="image/jpeg",
                            headers={"X-Detections": json.dumps(res["counts"]), "X-Latency-Ms": str(res["ms"])})
        if output == "coco":
            return Response(json.dumps(det.to_coco(res["detections"], res["size"], file.filename or "image")),
                            media_type="application/json")
        return Response(json.dumps(res), media_type="application/json")

    @app.post("/detect/video")
    async def detect_video(file: UploadFile = File(...), model: str = Form("ssdlite"), threshold: float = Form(0.5),
                           every: int = Form(2)) -> dict:
        data = await file.read()
        if len(data) > 200 * 1024 * 1024:
            raise HTTPException(413, "video larger than 200 MB")
        jid_dir = runs / f"video-{uuid.uuid4().hex[:8]}"
        jid_dir.mkdir(parents=True)
        src = jid_dir / ("input" + Path(file.filename or "v.mp4").suffix)
        src.write_bytes(data)
        out = jid_dir / "annotated.mp4"
        jid = start_job("video", lambda progress, stop: {**det.process_video(str(src), str(out), model, threshold, every),
                                                         "output": str(out.relative_to(settings.data_dir))})
        return {"job_id": jid}

    @app.get("/files/{path:path}")
    def files(path: str) -> FileResponse:
        p = (settings.data_dir / path).resolve()
        if settings.data_dir.resolve() not in p.parents or not p.is_file():
            raise HTTPException(404, "file not found")
        return FileResponse(p)

    @app.post("/detect/webcam/start")
    def webcam_start(req: WebcamRequest) -> dict:
        if req.model not in det.MODELS:
            raise HTTPException(422, "unknown model")
        cam.start(req.index, req.model, req.threshold, req.classes, capture_factory=capture_factory)
        return {"started": True}

    @app.post("/detect/webcam/stop")
    def webcam_stop() -> dict:
        cam.stop()
        return {"stopped": True}

    @app.get("/detect/webcam/status")
    def webcam_status() -> dict:
        return cam.state

    @app.get("/detect/webcam/frame")
    def webcam_frame() -> Response:
        f = cam.frame()
        if f is None:
            raise HTTPException(404, cam.state.get("error") or "no frame yet")
        return Response(f, media_type="image/jpeg")

    # ---------- generation ----------
    @app.post("/generate")
    def generate(req: GenerateRequest) -> dict:
        if req.size not in (256, 384, 512):
            raise HTTPException(422, "size must be 256, 384 or 512")
        if gen.BLOCKED.search(req.prompt):
            raise HTTPException(422, "prompt contains blocked content")

        def work(progress, stop):
            progress("Loading the image model (first time downloads about 2.6 GB)...")
            res = gen.generate(req.prompt, req.steps, req.size, req.n, req.seed, device=pick_device(settings.device),
                               pipe=generate_pipe)
            items = []
            for img, seed, secs in zip(res["images"], res["seeds"], res["seconds"]):
                fname = f"{int(time.time() * 1000)}_{seed}.png"
                img.save(gallery / fname)
                meta = {"file": f"gallery/{fname}", "prompt": req.prompt, "seed": seed, "steps": req.steps,
                        "size": req.size, "seconds": secs, "created": time.time()}
                (gallery / (fname + ".json")).write_text(json.dumps(meta))
                items.append(meta)
                progress(f"Generated image with seed {seed} in {secs}s")
            return {"images": items}
        return {"job_id": start_job("generate", work)}

    @app.get("/gallery")
    def list_gallery(limit: int = 60) -> list[dict]:
        metas = sorted(gallery.glob("*.png.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
        return [json.loads(m.read_text()) for m in metas]

    # ---------- enhancement ----------
    @app.get("/enhance/options")
    def enhance_options() -> dict:
        return {"methods": {**{k: v["title"] for k, v in enh.MODELS.items()}, "lanczos": "Lanczos (classical)",
                            "bicubic": "Bicubic (classical)"}, "targets": list(enh.TARGETS), "factors": [2, 4, 8]}

    @app.post("/enhance")
    async def enhance(file: UploadFile = File(...), method: str = Form("realesrgan-general-x4"), target: str = Form(""),
                      factor: int = Form(4)) -> Response:
        if method not in enh.MODELS and method not in enh.CLASSIC:
            raise HTTPException(422, "unknown method")
        if target and target not in enh.TARGETS:
            raise HTTPException(422, f"target must be one of {list(enh.TARGETS)}")
        if not target and factor not in (2, 4, 8):
            raise HTTPException(422, "factor must be 2, 4 or 8")
        img = await read_image(file)
        try:
            res = enh.enhance(img, method, target or None, None if target else factor, pick_device(settings.device))
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return Response(png(res["image"]), media_type="image/png",
                        headers={"X-Info": json.dumps({k: v for k, v in res.items() if k != "image"})})

    @app.post("/enhance/benchmark")
    async def enhance_benchmark(file: UploadFile = File(...)) -> list[dict]:
        img = await read_image(file)
        if min(img.size) < 128:
            raise HTTPException(422, "use an image at least 128 px on each side")
        img = img.convert("RGB")
        if img.width * img.height > 1_000_000:
            img.thumbnail((1000, 1000))
        return [enh.benchmark(img, m) for m in ["bicubic", "lanczos", *enh.MODELS]]

    return app
