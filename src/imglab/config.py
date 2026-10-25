import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TORCH_HOME", str(ROOT / "models" / "torch"))
MODELS_DIR = ROOT / "models"


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    hf_token: str | None = None
    device: str = "auto"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "imglab.db"


def get_settings() -> Settings:
    return Settings(data_dir=Path(os.getenv("IMGLAB_DATA", str(ROOT / "data"))),
                    hf_token=os.getenv("HF_TOKEN") or None, device=os.getenv("IMGLAB_DEVICE", "auto"))


def pick_device(pref: str = "auto"):
    import torch
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available() and pref in ("auto", "cuda"):
        return torch.device("cuda")
    return torch.device("cpu")
