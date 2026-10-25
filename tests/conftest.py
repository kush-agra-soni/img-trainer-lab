import io
import sys
import time
import pytest
import zipfile
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "launcher")]


def shapes_dataset(root: Path, per_class: int = 14, seed: int = 0) -> Path:
    """Two easy classes (red circles vs blue squares) with random size/position/background noise."""
    rng = np.random.default_rng(seed)
    for cls in ("circles", "squares"):
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            bg = rng.integers(180, 255, (96, 96, 3), dtype=np.uint8)
            img = Image.fromarray(bg)
            dr = ImageDraw.Draw(img)
            r = int(rng.integers(18, 34))
            cx, cy = (int(v) for v in rng.integers(r, 96 - r, 2))
            if cls == "circles":
                dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 30, 30))
            else:
                dr.rectangle([cx - r, cy - r, cx + r, cy + r], fill=(30, 60, 220))
            img.save(d / f"{cls}_{i}.png")
    return root


def zip_dir(root: Path, wrap: str | None = "dataset") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in root.rglob("*.png"):
            arc = f.relative_to(root).as_posix()
            z.write(f, f"{wrap}/{arc}" if wrap else arc)
    return buf.getvalue()


def photo(name: str = "astronaut") -> Image.Image:
    from skimage import data
    return Image.fromarray(getattr(data, name)())


class FakePipe:
    """Stands in for the diffusion pipeline: returns a gradient image whose colour depends on the seed."""

    def __init__(self):
        self.calls = []

    def __call__(self, prompt, num_inference_steps, guidance_scale, height, width, generator):
        self.calls.append((prompt, num_inference_steps, height))
        seed = int(generator.initial_seed())
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        arr[..., 0] = seed % 255
        arr[..., 1] = np.linspace(0, 255, width, dtype=np.uint8)
        return type("Out", (), {"images": [Image.fromarray(arr)]})()


class FakeCapture:
    def __init__(self, frames):
        import cv2
        self.frames = [cv2.cvtColor(np.asarray(f.convert("RGB")), cv2.COLOR_RGB2BGR) for f in frames]
        self.i, self.open = 0, True

    def isOpened(self):
        return self.open

    def read(self):
        time.sleep(0.02)
        f = self.frames[self.i % len(self.frames)]
        self.i += 1
        return True, f

    def release(self):
        self.open = False


class FakeHfApi:
    def __init__(self):
        self.calls = []

    def create_repo(self, repo_id, private, exist_ok, repo_type):
        self.calls.append(("create_repo", repo_id, private))

    def upload_folder(self, folder_path, repo_id, repo_type, commit_message):
        self.calls.append(("upload_folder", repo_id, sorted(p.name for p in Path(folder_path).iterdir())))


def wait_job(c, jid, timeout=300):
    end = time.time() + timeout
    while time.time() < end:
        j = c.get(f"/jobs/{jid}").json()
        if j["status"] != "running":
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from imglab.api import create_app
    from imglab.config import Settings

    hf, pipe = FakeHfApi(), FakePipe()
    app = create_app(Settings(tmp_path, hf_token="hf_test", device="cpu"), generate_pipe=pipe, hf_api=hf,
                     capture_factory=lambda i: FakeCapture([photo("astronaut"), photo("chelsea")]))
    c = TestClient(app)
    c.hf, c.pipe = hf, pipe
    return c


def img_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, fmt)
    return buf.getvalue()
