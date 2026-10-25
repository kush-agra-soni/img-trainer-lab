"""Renders the UI headlessly against an in-process backend (fake generator and Hugging Face API)."""
import time
import httpx
import pytest
import socket
import uvicorn
import threading
from conftest import ROOT, FakeCapture, FakeHfApi, FakePipe, photo


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    from imglab.api import create_app
    from imglab.config import Settings

    app = create_app(Settings(tmp_path_factory.mktemp("ui"), hf_token=None, device="cpu"), generate_pipe=FakePipe(),
                     hf_api=FakeHfApi(), capture_factory=lambda i: FakeCapture([photo("astronaut")]))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            httpx.get(f"{url}/health")
            break
        except httpx.HTTPError:
            time.sleep(0.05)
    mp = pytest.MonkeyPatch()
    mp.setenv("BACKEND_URL", url)
    yield url
    mp.undo()
    server.should_exit = True


def test_ui_tabs_and_generation(backend):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "ui" / "app.py"), default_timeout=120).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Transfer Learning", "Object Detection", "Image Generation", "Image Enhancement"]
    next(b for b in at.button if b.label == "Generate").click().run()
    assert not at.exception
    assert len(at.get("imgs")) >= 1
