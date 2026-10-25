"""Live evaluation of all four tools on this machine; writes reports/eval.json.

Usage: python scripts/run_eval.py [--skip-generation]"""
import sys
import json
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402
from skimage import data  # noqa: E402

from imglab import detect as det  # noqa: E402
from imglab import enhance as enh  # noqa: E402
from imglab import transfer as tl  # noqa: E402
from imglab.config import pick_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-generation", action="store_true")
    a = ap.parse_args()
    dev = pick_device()
    out = {"device": str(dev)}

    # transfer learning on a 10-class Imagenette subset (60 images per class)
    root = tl.imagenette_subset(ROOT / "data" / "datasets", per_class=60, classes=10)
    rows = []
    for bb in tl.BACKBONES:
        cfg = tl.train(root, ROOT / "data" / "models" / f"eval-{bb}", bb, "feature_extraction", epochs=20, device=dev)
        rows.append({"backbone": bb, "mode": "feature_extraction", "accuracy": cfg["metrics"]["accuracy"],
                     "f1_macro": cfg["metrics"]["f1_macro"], "seconds": cfg["seconds"]})
        print("transfer", rows[-1], flush=True)
    cfg = tl.train(root, ROOT / "data" / "models" / "eval-ft", "mobilenet_v3_large", "fine_tune", epochs=4,
                   unfreeze_blocks=2, device=dev)
    rows.append({"backbone": "mobilenet_v3_large", "mode": "fine_tune (2 blocks)", "accuracy": cfg["metrics"]["accuracy"],
                 "f1_macro": cfg["metrics"]["f1_macro"], "seconds": cfg["seconds"]})
    print("transfer", rows[-1], flush=True)
    out["transfer"] = {"dataset": "Imagenette 10 classes x 60 images (48 train / 12 val each)", "results": rows}

    # detection
    photos = {n: Image.fromarray(getattr(data, n)()) for n in ("astronaut", "chelsea", "coffee", "rocket")}
    det_rows = []
    for m in det.MODELS:
        det.detect(photos["coffee"], m, 0.5, device=dev)  # warm-up
        for n, im in photos.items():
            r = det.detect(im, m, 0.5, device=dev)
            det_rows.append({"model": m, "image": n, "ms": r["ms"], "found": r["counts"]})
    out["detection"] = det_rows
    print("detection", det_rows, flush=True)

    # enhancement benchmark (downscale x4 then restore)
    enh_rows = []
    for n in ("astronaut", "chelsea", "coffee"):
        for m in ["bicubic", "lanczos", *enh.MODELS]:
            enh_rows.append({"image": n, **enh.benchmark(photos[n], m)})
    out["enhancement"] = enh_rows
    t = time.time()
    r = enh.enhance(photos["chelsea"].resize((256, 144)), "realesrgan-general-x4", target="4K", device=dev)
    out["enhance_4k"] = {"input": r["input_size"], "output": r["output_size"], "seconds": r["seconds"]}
    print("enhancement done", out["enhance_4k"], flush=True)

    if not a.skip_generation:
        from imglab import generate as gen
        gen.load(device=dev)
        g = []
        for steps in (1, 2, 4):
            res = gen.generate("a red fox in a snowy forest, photograph", steps=steps, size=512, n=1, seed=42, device=dev)
            res["images"][0].save(ROOT / "reports" / f"gen_{steps}step.png")
            g.append({"steps": steps, "size": 512, "seconds": res["seconds"][0]})
            print("generation", g[-1], flush=True)
        out["generation"] = g

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved reports/eval.json")


if __name__ == "__main__":
    main()
