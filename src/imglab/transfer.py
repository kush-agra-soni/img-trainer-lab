"""Transfer learning for image classification: frozen-feature linear probe or partial fine-tuning."""
import io
import json
import time
import random
import shutil
import tarfile
import zipfile
import numpy as np
from PIL import Image
from pathlib import Path
from .config import MODELS_DIR
from collections.abc import Callable

BACKBONES = {
    "mobilenet_v3_large": {"title": "MobileNetV3-Large (fast)", "weights": "MobileNet_V3_Large_Weights"},
    "efficientnet_b0": {"title": "EfficientNet-B0 (balanced)", "weights": "EfficientNet_B0_Weights"},
    "resnet18": {"title": "ResNet-18 (classic)", "weights": "ResNet18_Weights"},
}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
IMAGENETTE_CLASSES = {"n01440764": "tench", "n02102040": "english_springer", "n02979186": "cassette_player",
                      "n03000684": "chain_saw", "n03028079": "church", "n03394916": "french_horn",
                      "n03417042": "garbage_truck", "n03425413": "gas_pump", "n03445777": "golf_ball",
                      "n03888605": "parachute"}


class DatasetError(ValueError):
    pass


# ---------- datasets ----------
def scan(root: Path) -> dict[str, list[Path]]:
    classes = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "__"))):
        imgs = sorted(f for f in d.rglob("*") if f.suffix.lower() in IMAGE_EXT)
        if imgs:
            classes[d.name] = imgs
    return classes


def validate(classes: dict[str, list[Path]], min_per_class: int = 3) -> None:
    if len(classes) < 2:
        raise DatasetError("need at least 2 class folders with images")
    small = [c for c, v in classes.items() if len(v) < min_per_class]
    if small:
        raise DatasetError(f"classes with fewer than {min_per_class} images: {', '.join(small)}")


def extract_zip(data: bytes, dest: Path) -> Path:
    """Extract an uploaded zip safely (no absolute paths or ..), unwrapping a single top-level folder."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise DatasetError("not a valid zip file") from e
    total = 0
    for info in z.infolist():
        name = info.filename.replace("\\", "/")
        if name.startswith("/") or ".." in name.split("/"):
            raise DatasetError(f"unsafe path in zip: {name}")
        total += info.file_size
        if total > 2 * 1024 ** 3:
            raise DatasetError("zip expands to more than 2 GB")
    z.extractall(dest)
    entries = [p for p in dest.iterdir() if not p.name.startswith(("__MACOSX", "."))]
    if len(entries) == 1 and entries[0].is_dir() and len(scan(entries[0])) >= 2:
        return entries[0]
    return dest


def imagenette_subset(dest: Path, per_class: int = 60, classes: int = 5, seed: int = 0) -> Path:
    """Build a small demo dataset from Imagenette (ImageNet subset, 160px) on first use."""
    import httpx
    out = dest / f"imagenette_{classes}x{per_class}"
    if out.exists() and len(scan(out)) == classes:
        return out
    archive = MODELS_DIR / "imagenette2-160.tgz"
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", IMAGENETTE_URL, timeout=300, follow_redirects=True) as r:
            r.raise_for_status()
            with open(archive.with_suffix(".part"), "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
        archive.with_suffix(".part").replace(archive)
    wanted = list(IMAGENETTE_CLASSES)[:classes]
    rng = random.Random(seed)
    with tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".JPEG") and "/train/" in m.name]
        for wnid in wanted:
            pick = [m for m in members if f"/{wnid}/" in m.name]
            rng.shuffle(pick)
            target = out / IMAGENETTE_CLASSES[wnid]
            target.mkdir(parents=True, exist_ok=True)
            for m in pick[:per_class]:
                (target / Path(m.name).name).write_bytes(tar.extractfile(m).read())
    return out


def split(classes: dict[str, list[Path]], val_fraction: float = 0.2, seed: int = 0):
    rng = random.Random(seed)
    names = sorted(classes)
    train, val = [], []
    for i, c in enumerate(names):
        files = list(classes[c])
        rng.shuffle(files)
        k = max(1, round(len(files) * val_fraction))
        val += [(f, i) for f in files[:k]]
        train += [(f, i) for f in files[k:]]
    return names, train, val


# ---------- models ----------
def build(backbone: str, n_classes: int):
    import torch.nn as nn
    import torchvision.models as tvm
    weights = getattr(tvm, BACKBONES[backbone]["weights"]).DEFAULT
    m = getattr(tvm, backbone)(weights=weights)
    if backbone.startswith("resnet"):
        feat = m.fc.in_features
        m.fc = nn.Identity()
    else:
        feat = m.classifier[-1].in_features
        m.classifier[-1] = nn.Identity()
    head = nn.Linear(feat, n_classes)
    return m, head, weights.transforms()


def blocks(model) -> list:
    if hasattr(model, "features"):
        return list(model.features)
    return [model.layer1, model.layer2, model.layer3, model.layer4]


def _tensor(path: Path, tf):
    return tf(Image.open(path).convert("RGB"))


def _metrics(y_true, y_pred, names) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    rep = classification_report(y_true, y_pred, labels=list(range(len(names))), target_names=names, output_dict=True,
                                zero_division=0)
    return {"accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
            "per_class": {n: {"precision": round(rep[n]["precision"], 3), "recall": round(rep[n]["recall"], 3),
                              "support": int(rep[n]["support"])} for n in names},
            "confusion": confusion_matrix(y_true, y_pred, labels=list(range(len(names)))).tolist()}


def train(data_root: Path, out_dir: Path, backbone: str = "mobilenet_v3_large", mode: str = "feature_extraction",
          epochs: int = 10, lr: float = 1e-3, batch_size: int = 32, unfreeze_blocks: int = 2, val_fraction: float = 0.2,
          patience: int = 4, device=None, progress: Callable[[str, dict], None] = lambda m, d: None,
          stop: Callable[[], bool] = lambda: False) -> dict:
    import torch
    import torch.nn as nn
    import torchvision.transforms.v2 as T

    from .config import pick_device
    torch.manual_seed(0)
    device = device or pick_device()
    classes = scan(data_root)
    validate(classes)
    names, tr, va = split(classes, val_fraction)
    progress(f"{len(names)} classes, {len(tr)} training and {len(va)} validation images on {device}", {})
    model, head, tf = build(backbone, len(names))
    model.to(device)
    head.to(device)
    history, best = [], {"acc": -1.0, "loss": float("inf"), "epoch": 0}

    def improved(acc: float, loss: float) -> bool:  # accuracy first; validation loss breaks ties
        return acc > best["acc"] + 1e-9 or (abs(acc - best["acc"]) <= 1e-9 and loss < best["loss"] - 1e-4)
    t_start = time.time()

    if mode == "feature_extraction":
        model.eval()

        def featurize(items):
            feats, ys = [], []
            with torch.inference_mode():
                for i in range(0, len(items), batch_size):
                    chunk = items[i:i + batch_size]
                    x = torch.stack([_tensor(f, tf) for f, _ in chunk]).to(device)
                    feats.append(model(x).cpu())
                    ys += [y for _, y in chunk]
            return torch.cat(feats), torch.tensor(ys)
        Xtr, ytr = featurize(tr)
        Xva, yva = featurize(va)
        progress(f"Extracted {Xtr.shape[1]}-dimensional features in {time.time() - t_start:.1f}s", {})
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
        best_state = {k: v.clone() for k, v in head.state_dict().items()}  # fallback if stopped before epoch 1
        for ep in range(1, epochs + 1):
            if stop():
                break
            head.train()
            perm = torch.randperm(len(Xtr))
            tot = 0.0
            for i in range(0, len(perm), batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                loss = lossf(head(Xtr[idx].to(device)), ytr[idx].to(device))
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
            sched.step()
            head.eval()
            with torch.inference_mode():
                logits = head(Xva.to(device))
                vloss = float(lossf(logits, yva.to(device)))
                pred = logits.argmax(1).cpu()
            acc = float((pred == yva).float().mean())
            history.append({"epoch": ep, "train_loss": round(tot / len(Xtr), 4), "val_loss": round(vloss, 4),
                            "val_accuracy": round(acc, 4)})
            progress(f"epoch {ep}/{epochs}: loss {tot / len(Xtr):.4f}, val loss {vloss:.4f}, val accuracy {acc:.3f}",
                     history[-1])
            if improved(acc, vloss):
                best = {"acc": acc, "loss": vloss, "epoch": ep}
                best_state = {k: v.clone() for k, v in head.state_dict().items()}
            elif ep - best["epoch"] >= patience:
                progress(f"early stop: no improvement for {patience} epochs", {})
                break
        head.load_state_dict(best_state)
        with torch.inference_mode():
            y_pred = head(Xva.to(device)).argmax(1).cpu().numpy()
        y_true = yva.numpy()
    else:
        for p in model.parameters():
            p.requires_grad = False
        for b in blocks(model)[-unfreeze_blocks:]:
            for p in b.parameters():
                p.requires_grad = True
        aug = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0), antialias=True), T.RandomHorizontalFlip(),
                         T.ColorJitter(0.2, 0.2, 0.2)])
        net = nn.Sequential(model, head).to(device)
        params = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr / 10},
                  {"params": head.parameters(), "lr": lr}]
        opt = torch.optim.AdamW(params, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

        def evaluate():
            net.eval()
            logits = []
            with torch.inference_mode():
                for i in range(0, len(va), batch_size):
                    x = torch.stack([_tensor(f, tf) for f, _ in va[i:i + batch_size]]).to(device)
                    logits.append(net(x).cpu())
            lg = torch.cat(logits)
            return lg.argmax(1).numpy(), float(lossf(lg, torch.tensor([y for _, y in va])))
        y_true = np.array([y for _, y in va])
        for ep in range(1, epochs + 1):
            if stop():
                break
            net.train()
            random.Random(ep).shuffle(tr)
            tot = 0.0
            for i in range(0, len(tr), batch_size):
                chunk = tr[i:i + batch_size]
                x = torch.stack([aug(_tensor(f, tf)) for f, _ in chunk]).to(device)
                y = torch.tensor([c for _, c in chunk], device=device)
                opt.zero_grad()
                loss = lossf(net(x), y)
                loss.backward()
                opt.step()
                tot += loss.item() * len(chunk)
            sched.step()
            pred, vloss = evaluate()
            acc = float((pred == y_true).mean())
            history.append({"epoch": ep, "train_loss": round(tot / len(tr), 4), "val_loss": round(vloss, 4),
                            "val_accuracy": round(acc, 4)})
            progress(f"epoch {ep}/{epochs}: loss {tot / len(tr):.4f}, val loss {vloss:.4f}, val accuracy {acc:.3f}",
                     history[-1])
            if improved(acc, vloss):
                best = {"acc": acc, "loss": vloss, "epoch": ep}
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            elif ep - best["epoch"] >= patience:
                progress(f"early stop: no improvement for {patience} epochs", {})
                break
        net.load_state_dict(best_state)
        y_pred, _ = evaluate()

    metrics = _metrics(y_true, y_pred, names)
    out_dir.mkdir(parents=True, exist_ok=True)
    save(out_dir, model, head, names, {"backbone": backbone, "mode": mode, "epochs_run": len(history), "lr": lr,
                                       "batch_size": batch_size, "best_epoch": best["epoch"], "train_images": len(tr),
                                       "val_images": len(va), "seconds": round(time.time() - t_start, 1),
                                       "metrics": metrics, "history": history})
    progress(f"Best validation accuracy {best['acc']:.3f} (epoch {best['epoch']}); saved to {out_dir.name}", {})
    return json.loads((out_dir / "config.json").read_text())


def save(out_dir: Path, model, head, names, info: dict) -> None:
    from safetensors.torch import save_file
    state = {f"backbone.{k}": v.contiguous().cpu() for k, v in model.state_dict().items()}
    state.update({f"head.{k}": v.contiguous().cpu() for k, v in head.state_dict().items()})
    save_file(state, str(out_dir / "model.safetensors"))
    (out_dir / "config.json").write_text(json.dumps({"labels": names, **info}, indent=2), encoding="utf-8")


_cache: dict[str, tuple] = {}


def load_classifier(model_dir: Path, device=None):
    import torch
    from safetensors.torch import load_file

    from .config import pick_device
    device = device or pick_device()
    key = f"{model_dir}:{device}"
    if key not in _cache:
        cfg = json.loads((model_dir / "config.json").read_text())
        model, head, tf = build(cfg["backbone"], len(cfg["labels"]))
        state = load_file(str(model_dir / "model.safetensors"))
        model.load_state_dict({k[9:]: v for k, v in state.items() if k.startswith("backbone.")})
        head.load_state_dict({k[5:]: v for k, v in state.items() if k.startswith("head.")})
        net = torch.nn.Sequential(model, head).to(device).eval()
        _cache[key] = (net, tf, cfg)
    return _cache[key]


def classify(model_dir: Path, img: Image.Image, top_k: int = 3) -> dict:
    import torch
    net, tf, cfg = load_classifier(model_dir)
    dev = next(net.parameters()).device
    t0 = time.perf_counter()
    with torch.inference_mode():
        p = torch.softmax(net(tf(img.convert("RGB")).unsqueeze(0).to(dev)), 1)[0].cpu()
    top = torch.topk(p, min(top_k, len(cfg["labels"])))
    return {"predictions": [{"label": cfg["labels"][int(i)], "probability": round(float(v), 4)}
                            for v, i in zip(top.values, top.indices)], "ms": round((time.perf_counter() - t0) * 1000, 1)}


def model_card(cfg: dict, repo_id: str) -> str:
    m = cfg["metrics"]
    rows = "\n".join(f"| {c} | {v['precision']} | {v['recall']} | {v['support']} |" for c, v in m["per_class"].items())
    return f"""---
library_name: pytorch
tags: [image-classification, transfer-learning, torchvision]
---
# {repo_id}

Image classifier fine-tuned from torchvision **{cfg['backbone']}** (ImageNet weights) with
**{cfg['mode'].replace('_', ' ')}**. Classes: {', '.join(cfg['labels'])}.

## Results (validation split, {cfg['val_images']} images)

- Accuracy: **{m['accuracy']}**
- Macro F1: **{m['f1_macro']}**

| Class | Precision | Recall | Support |
|---|---|---|---|
{rows}

## Training

{cfg['train_images']} training images, {cfg['epochs_run']} epochs (best epoch {cfg['best_epoch']}), learning rate {cfg['lr']},
batch size {cfg['batch_size']}, {cfg['seconds']} s.

## Usage

Weights are in `model.safetensors` (keys `backbone.*` and `head.*`); labels and settings in `config.json`.
Rebuild the torchvision backbone with its classifier replaced by identity and a linear head of size {len(cfg['labels'])}.

## Limitations

Trained on a small dataset; validate on your own images before relying on it.
"""


def publish(model_dir: Path, repo_id: str, token: str, private: bool = True, api=None) -> str:
    """Upload the model folder plus a generated model card to the Hugging Face Hub."""
    from huggingface_hub import HfApi
    api = api or HfApi(token=token)
    cfg = json.loads((model_dir / "config.json").read_text())
    (model_dir / "README.md").write_text(model_card(cfg, repo_id), encoding="utf-8")
    api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=str(model_dir), repo_id=repo_id, repo_type="model",
                      commit_message=f"Upload {cfg['backbone']} classifier ({cfg['metrics']['accuracy']} val accuracy)")
    return f"https://huggingface.co/{repo_id}"
