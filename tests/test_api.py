import json
import time
from conftest import img_bytes, photo, shapes_dataset, wait_job, zip_dir


def test_health_and_options(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["device"] == "cpu" and h["hf_token"]
    assert "mobilenet_v3_large" in client.get("/transfer/options").json()["backbones"]
    assert "person" in client.get("/detect/options").json()["classes"]
    assert "4K" in client.get("/enhance/options").json()["targets"]


def test_transfer_flow_upload_train_classify_publish(client, tmp_path):
    z = zip_dir(shapes_dataset(tmp_path / "s", per_class=10))
    r = client.post("/transfer/datasets/upload", data={"name": "shapes"}, files={"file": ("s.zip", z)})
    assert r.json()["classes"] == {"circles": 10, "squares": 10}
    assert client.get("/transfer/datasets").json()[0]["name"] == "shapes"
    jid = client.post("/transfer/train", data={"dataset": "shapes", "name": "shapes-m", "epochs": 5,
                                                "batch_size": 8}).json()["job_id"]
    j = wait_job(client, jid)
    assert j["status"] == "done" and j["result"]["metrics"]["accuracy"] >= 0.75 and j["progress"]
    assert client.get("/transfer/models").json()[0]["name"] == "shapes-m"
    sq = next((tmp_path / "s" / "squares").iterdir()).read_bytes()
    res = client.post("/transfer/models/shapes-m/classify", files={"file": ("x.png", sq)}).json()
    assert res["predictions"][0]["label"] == "squares" and "low_confidence" in res
    pub = client.post("/transfer/models/shapes-m/publish", json={"repo_id": "me/shapes", "private": True}).json()
    assert pub["url"].endswith("me/shapes") and client.hf.calls[0][2] is True


def test_transfer_validation(client):
    assert client.post("/transfer/datasets/upload", data={"name": "Bad Name"}, files={"file": ("x.zip", b"x")}).status_code == 422
    assert client.post("/transfer/datasets/upload", data={"name": "bad"}, files={"file": ("x.zip", b"not zip")}).status_code == 422
    assert client.post("/transfer/train", data={"dataset": "missing", "name": "m"}).status_code == 404
    assert client.post("/transfer/train", data={"dataset": "x", "name": "m", "backbone": "vgg"}).status_code == 422
    assert client.post("/transfer/models/nope/classify", files={"file": ("x.png", b"x")}).status_code == 404


def test_detect_image_json_image_coco(client):
    data = img_bytes(photo("astronaut"))
    j = client.post("/detect/image", data={"model": "ssdlite", "threshold": 0.5}, files={"file": ("a.png", data)}).json()
    assert j["counts"] == {"person": 1}
    r = client.post("/detect/image", data={"output": "image"}, files={"file": ("a.png", data)})
    assert r.headers["content-type"] == "image/jpeg" and json.loads(r.headers["x-detections"]) == {"person": 1}
    coco = client.post("/detect/image", data={"output": "coco"}, files={"file": ("a.png", data)}).json()
    assert coco["categories"][0]["name"] == "person"
    assert client.post("/detect/image", files={"file": ("a.png", b"garbage")}).status_code == 422
    assert client.post("/detect/image", data={"model": "yolo"}, files={"file": ("a.png", data)}).status_code == 422


def test_webcam_endpoints(client):
    assert client.post("/detect/webcam/start", json={"index": 0, "threshold": 0.5}).status_code == 200
    for _ in range(100):
        if client.get("/detect/webcam/status").json()["frames"] >= 2:
            break
        time.sleep(0.05)
    assert client.get("/detect/webcam/frame").headers["content-type"] == "image/jpeg"
    client.post("/detect/webcam/stop")
    assert client.get("/detect/webcam/status").json()["running"] is False


def test_generate_gallery_and_files(client):
    jid = client.post("/generate", json={"prompt": "a red fox", "n": 2, "size": 256, "seed": 5}).json()["job_id"]
    j = wait_job(client, jid)
    assert j["status"] == "done" and [i["seed"] for i in j["result"]["images"]] == [5, 6]
    gal = client.get("/gallery").json()
    assert len(gal) == 2 and gal[0]["prompt"] == "a red fox"
    assert client.get(f"/files/{gal[0]['file']}").headers["content-type"] == "image/png"
    assert client.get("/files/../../etc/passwd").status_code == 404
    assert client.post("/generate", json={"prompt": "nsfw thing"}).status_code == 422
    assert client.post("/generate", json={"prompt": "x", "size": 1024}).status_code == 422


def test_enhance_and_benchmark(client):
    small = img_bytes(photo("chelsea").resize((200, 120)))
    r = client.post("/enhance", data={"method": "realesrgan-general-x4", "target": "480p"}, files={"file": ("c.png", small)})
    info = json.loads(r.headers["x-info"])
    assert r.status_code == 200 and info["output_size"] == [800, 480]
    r = client.post("/enhance", data={"method": "lanczos", "target": "", "factor": 2}, files={"file": ("c.png", small)})
    assert json.loads(r.headers["x-info"])["output_size"] == [400, 240]
    assert client.post("/enhance", data={"method": "lanczos", "target": "8K"}, files={"file": ("c.png", small)}).status_code == 422
    big = img_bytes(photo("chelsea"))
    assert client.post("/enhance", data={"method": "lanczos", "target": "144p"}, files={"file": ("c.png", big)}).status_code == 422
    bench = client.post("/enhance/benchmark", files={"file": ("a.png", img_bytes(photo("coffee")))}).json()
    assert {b["method"] for b in bench} >= {"bicubic", "lanczos", "realesrgan-general-x4"}
