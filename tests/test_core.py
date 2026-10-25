import json
import pytest
import zipfile
import numpy as np
from PIL import Image
from pathlib import Path
from imglab import detect as det
from imglab import enhance as enh
from imglab import transfer as tl
from imglab import generate as gen
from imglab.config import pick_device
from conftest import FakeCapture, FakeHfApi, FakePipe, photo, shapes_dataset, zip_dir


# ---------- enhancement ----------
def test_classical_targets_and_factors():
    img = photo("chelsea").resize((256, 144))
    r = enh.enhance(img, "lanczos", target="1080p")
    assert r["output_size"] == [1920, 1080] and r["image"].size == (1920, 1080) and r["neural_passes"] == 0
    assert enh.enhance(img, "bicubic", factor=2)["output_size"] == [512, 288]
    with pytest.raises(ValueError, match="not larger"):
        enh.enhance(Image.new("RGB", (1000, 700)), "lanczos", target="480p")


def test_neural_upscale_lands_on_target_with_one_pass():
    img = photo("chelsea").resize((160, 90))
    r = enh.enhance(img, "realesrgan-general-x4", target="480p")
    assert r["image"].size == (853, 480) and r["neural_passes"] == 2  # 480/90 > 4 needs two passes
    r = enh.enhance(photo("chelsea").resize((256, 180)), "realesrgan-general-x4", target="720p")
    assert r["neural_passes"] == 1 and r["image"].size == (1024, 720)
    with pytest.raises(ValueError, match="MP"):
        enh.enhance(Image.new("RGB", (1300, 1000)), "realesrgan-general-x4", factor=2)


def test_tiling_matches_untiled():
    model = enh.load_model("realesrgan-general-x4", pick_device("cpu"))
    arr = np.asarray(photo("coffee").resize((120, 80)))
    a = enh._run_tiles(model, arr, pick_device("cpu"), tile=40, pad=12).astype(int)
    b = enh._run_tiles(model, arr, pick_device("cpu"), tile=512).astype(int)
    assert a.shape == b.shape == (320, 480, 3) and np.abs(a - b).mean() < 1.5


def test_benchmark_shows_fidelity_vs_detail_tradeoff():
    img = photo("astronaut")
    lanczos = enh.benchmark(img, "lanczos")
    gan = enh.benchmark(img, "realesrgan-general-x4")
    assert gan["sharpness"] > 5 * lanczos["sharpness"] and 20 < lanczos["psnr"] < 40 and 0 < gan["ssim"] <= 1


# ---------- detection ----------
def test_detects_people_and_cats():
    d = det.detect(photo("astronaut"), "ssdlite", 0.5)
    assert d["counts"].get("person") == 1 and d["ms"] > 0
    d = det.detect(photo("chelsea"), "ssdlite", 0.5)
    assert "cat" in d["counts"]
    assert det.detect(photo("chelsea"), "ssdlite", 0.5, classes=["person"])["detections"] == []


def test_annotate_and_coco_export():
    img = photo("coffee")
    d = det.detect(img, "ssdlite", 0.5)
    ann = det.annotate(img, d["detections"])
    assert ann.size == img.size and np.abs(np.asarray(ann, int) - np.asarray(img.convert("RGB"), int)).sum() > 0
    coco = det.to_coco(d["detections"], d["size"], "coffee.png")
    assert coco["images"][0]["width"] == img.width and len(coco["annotations"]) == len(d["detections"])
    x, y, w, h = coco["annotations"][0]["bbox"]
    assert w > 0 and h > 0


def test_webcam_streamer_with_fake_camera():
    cam = det.Webcam()
    cam.start(0, "ssdlite", 0.5, capture_factory=lambda i: FakeCapture([photo("astronaut"), photo("chelsea")]))
    for _ in range(200):
        if cam.state["frames"] >= 3:
            break
        import time
        time.sleep(0.05)
    cam.stop()
    assert cam.state["frames"] >= 3 and cam.frame()[:2] == b"\xff\xd8" and not cam.state["running"]
    bad = det.Webcam()
    closed = FakeCapture([photo("coffee")])
    closed.open = False
    bad.start(0, capture_factory=lambda i: closed)
    bad._thread.join(5)
    assert "could not be opened" in bad.state["error"]


def test_video_annotation(tmp_path):
    import cv2
    src = tmp_path / "in.mp4"
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10, (256, 256))
    for f in [photo("astronaut").resize((256, 256))] * 6:
        w.write(cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2BGR))
    w.release()
    res = det.process_video(str(src), str(tmp_path / "out.mp4"), "ssdlite", 0.4, every=2)
    assert res["frames_read"] == 6 and res["frames_processed"] == 3 and res["max_simultaneous"].get("person") == 1
    assert (tmp_path / "out.mp4").stat().st_size > 0


# ---------- transfer learning ----------
def test_dataset_validation_and_safe_zip(tmp_path):
    root = shapes_dataset(tmp_path / "shapes", per_class=4)
    tl.validate(tl.scan(root))
    with pytest.raises(tl.DatasetError, match="at least 2"):
        tl.validate({"one": [Path("a")] * 5})
    with pytest.raises(tl.DatasetError, match="fewer than 3"):
        tl.validate({"a": [Path("x")] * 5, "b": [Path("y")]})
    out = tl.extract_zip(zip_dir(root), tmp_path / "unzipped")
    assert out.name == "dataset" and sorted(tl.scan(out)) == ["circles", "squares"]
    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../evil.txt", "x")
    with pytest.raises(tl.DatasetError, match="unsafe"):
        tl.extract_zip(buf.getvalue(), tmp_path / "evil")
    with pytest.raises(tl.DatasetError, match="valid zip"):
        tl.extract_zip(b"not a zip", tmp_path / "bad")


def test_split_is_stratified():
    classes = {"a": [Path(f"a{i}") for i in range(10)], "b": [Path(f"b{i}") for i in range(20)]}
    names, tr, va = tl.split(classes, 0.2)
    assert names == ["a", "b"] and sum(y == 0 for _, y in va) == 2 and sum(y == 1 for _, y in va) == 4


def test_feature_extraction_learns_and_saves(tmp_path):
    root = shapes_dataset(tmp_path / "shapes", per_class=14)
    logs = []
    cfg = tl.train(root, tmp_path / "m", "mobilenet_v3_large", "feature_extraction", epochs=6, batch_size=16,
                   device=pick_device("cpu"), progress=lambda m, d: logs.append(m))
    assert cfg["metrics"]["accuracy"] >= 0.9 and cfg["labels"] == ["circles", "squares"]
    assert (tmp_path / "m" / "model.safetensors").exists() and any("epoch 1/6" in m for m in logs)
    tl._cache.clear()
    out = tl.classify(tmp_path / "m", Image.open(next((root / "squares").iterdir())))
    assert out["predictions"][0]["label"] == "squares"


def test_fine_tune_runs_and_can_stop(tmp_path):
    root = shapes_dataset(tmp_path / "shapes", per_class=8)
    cfg = tl.train(root, tmp_path / "ft", "mobilenet_v3_large", "fine_tune", epochs=1, batch_size=8,
                   unfreeze_blocks=1, device=pick_device("cpu"))
    assert cfg["mode"] == "fine_tune" and cfg["epochs_run"] == 1
    stopped = tl.train(root, tmp_path / "st", "mobilenet_v3_large", "feature_extraction", epochs=50,
                       device=pick_device("cpu"), stop=lambda: True)
    assert stopped["epochs_run"] == 0 and (tmp_path / "st" / "model.safetensors").exists()


def test_publish_writes_card_and_uploads(tmp_path):
    root = shapes_dataset(tmp_path / "shapes", per_class=6)
    tl.train(root, tmp_path / "m", "mobilenet_v3_large", "feature_extraction", epochs=2, device=pick_device("cpu"))
    api = FakeHfApi()
    url = tl.publish(tmp_path / "m", "user/shapes", "tok", private=True, api=api)
    assert url == "https://huggingface.co/user/shapes" and api.calls[0] == ("create_repo", "user/shapes", True)
    assert api.calls[1][2] == ["README.md", "config.json", "model.safetensors"]
    card = (tmp_path / "m" / "README.md").read_text()
    assert "image-classification" in card and "circles" in card


# ---------- generation ----------
def test_generation_with_fake_pipeline():
    pipe = FakePipe()
    r = gen.generate("a cat", steps=2, size=256, n=3, seed=10, pipe=pipe)
    assert r["seeds"] == [10, 11, 12] and len(r["images"]) == 3 and r["images"][0].size == (256, 256)
    assert pipe.calls[0] == ("a cat", 2, 256)
    sheet = gen.contact_sheet(r["images"])
    assert sheet.size == (512, 512)
    with pytest.raises(gen.PromptBlocked):
        gen.generate("a nude person", pipe=pipe)
    for bad in [dict(steps=9), dict(size=300), dict(n=5)]:
        with pytest.raises(ValueError):
            gen.generate("x", pipe=pipe, **bad)
