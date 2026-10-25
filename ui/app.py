import io
import os
import json
import time
import httpx
import pandas as pd
from PIL import Image
import streamlit as st

BACKEND = os.getenv("BACKEND_URL", f"http://127.0.0.1:{os.getenv('BACKEND_PORT', '8006')}")
st.set_page_config(page_title="Image AI Platform", layout="wide")
ss = st.session_state


def api(method, path, **kw):
    return httpx.request(method, f"{BACKEND}{path}", timeout=kw.pop("timeout", 900), **kw)


def ok(r):
    if r.status_code != 200:
        try:
            st.error(r.json().get("detail", r.text))
        except ValueError:
            st.error(r.text[:300])
        return None
    return r.json()


def wait_job(jid: str, log_box, chart_box=None) -> dict:
    while True:
        j = api("GET", f"/jobs/{jid}").json()
        log_box.code("\n".join(e["msg"] for e in j["log"][-12:]) or "starting...", language=None)
        if chart_box is not None and j["progress"]:
            hist = pd.DataFrame(j["progress"]).set_index("epoch")
            chart_box.line_chart(hist[[c for c in ("val_accuracy", "train_loss", "val_loss") if c in hist]])
        if j["status"] != "running":
            return j
        time.sleep(0.7)


try:
    health = api("GET", "/health").json()
except httpx.HTTPError:
    st.error(f"Backend not reachable at {BACKEND}")
    st.stop()

st.title("Image AI Platform")
st.caption(f"Compute: {health['device'].upper()}" + (f" ({health['gpu']})" if health.get("gpu") else "") +
           f"  |  Trained classifiers: {health['models_trained']}  |  Hugging Face token: "
           f"{'configured' if health['hf_token'] else 'not set'}")
tabs = st.tabs(["Transfer Learning", "Object Detection", "Image Generation", "Image Enhancement"])

# ======================= Transfer learning =======================
with tabs[0]:
    st.caption("Fine-tune an ImageNet model on your own image classes. Upload a zip with one folder per class "
               "(at least 3 images each), or build a demo dataset from Imagenette.")
    opts = api("GET", "/transfer/options").json()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Dataset**")
        z = st.file_uploader("Zip of class folders", type=["zip"])
        dname = st.text_input("Dataset name", "my-images")
        if z and st.button("Upload dataset"):
            r = ok(api("POST", "/transfer/datasets/upload", data={"name": dname}, files={"file": (z.name, z.getvalue())}))
            if r:
                st.success(f"{len(r['classes'])} classes: {r['classes']}")
        if st.button("Build demo dataset (Imagenette, 5 classes x 60 images)"):
            j = wait_job(ok(api("POST", "/transfer/datasets/demo"))["job_id"], st.empty())
            st.success(f"Ready: {j['result']['classes']}") if j["status"] == "done" else st.error(j["error"])
    datasets = api("GET", "/transfer/datasets").json()
    with c2:
        st.markdown("**Training**")
        if not datasets:
            st.info("Add a dataset first.")
        else:
            ds = st.selectbox("Dataset", [d["name"] for d in datasets])
            st.caption(", ".join(f"{k} ({v})" for k, v in next(d for d in datasets if d["name"] == ds)["classes"].items()))
            bb = st.selectbox("Backbone", list(opts["backbones"]), format_func=lambda k: opts["backbones"][k])
            mode = st.radio("Mode", list(opts["modes"]), format_func=lambda k: opts["modes"][k])
            h1, h2, h3, h4 = st.columns(4)
            epochs = h1.number_input("Epochs", 1, 100, 15 if mode == "feature_extraction" else 4)
            lr = h2.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=1e-3)
            bs = h3.select_slider("Batch size", [8, 16, 32, 64], value=32)
            unfreeze = h4.number_input("Blocks to unfreeze", 1, 6, 2, disabled=mode != "fine_tune")
            val = st.slider("Validation split", 0.1, 0.4, 0.2, 0.05)
            mname = st.text_input("Model name", f"{ds}-{bb.split('_')[0]}")
            if st.button("Train", type="primary"):
                jid = ok(api("POST", "/transfer/train", data={"dataset": ds, "name": mname, "backbone": bb, "mode": mode,
                                                              "epochs": epochs, "lr": lr, "batch_size": bs,
                                                              "unfreeze_blocks": unfreeze, "val_fraction": val}))
                if jid:
                    ss.train_job = jid["job_id"]
                    chart, log = st.empty(), st.empty()
                    j = wait_job(jid["job_id"], log, chart)
                    if j["status"] == "done":
                        ss.cls_model = mname
                        st.success(f"Validation accuracy {j['result']['metrics']['accuracy']:.1%}")
                    else:
                        st.error(j.get("error") or j["status"])
    models = api("GET", "/transfer/models").json()
    if models:
        st.divider()
        names = [m["name"] for m in models]
        sel = st.selectbox("Trained model", names, index=names.index(ss.cls_model) if ss.get("cls_model") in names else 0)
        cfg = api("GET", f"/transfer/models/{sel}").json()
        m = st.columns(4)
        m[0].metric("Validation accuracy", f"{cfg['metrics']['accuracy']:.1%}")
        m[1].metric("Macro F1", f"{cfg['metrics']['f1_macro']:.3f}")
        m[2].metric("Training time", f"{cfg['seconds']} s")
        m[3].metric("Best epoch", cfg["best_epoch"])
        g1, g2 = st.columns(2)
        g1.markdown("**Confusion matrix** (rows = actual)")
        g1.dataframe(pd.DataFrame(cfg["metrics"]["confusion"], index=cfg["labels"], columns=cfg["labels"]))
        if cfg.get("history"):
            g2.markdown("**Training curve**")
            g2.line_chart(pd.DataFrame(cfg["history"]).set_index("epoch")[["val_accuracy"]])
        t1, t2 = st.columns(2)
        test = t1.file_uploader("Classify an image", type=["jpg", "jpeg", "png", "webp", "bmp"], key="cls_img")
        if test:
            r = ok(api("POST", f"/transfer/models/{sel}/classify", files={"file": (test.name, test.getvalue())}))
            if r:
                t1.image(test, width=260)
                for p in r["predictions"]:
                    t1.progress(p["probability"], text=f"{p['label']}: {p['probability']:.1%}")
                if r["low_confidence"]:
                    t1.warning("Low confidence: this image may not belong to any trained class.")
        with t2:
            st.markdown("**Publish to the Hugging Face Hub**")
            repo = st.text_input("Repository (user/name)", f"your-username/{sel}")
            private = st.checkbox("Private repository", value=True)
            if st.button("Publish", disabled=not health["hf_token"],
                         help=None if health["hf_token"] else "Set HF_TOKEN in .env"):
                r = ok(api("POST", f"/transfer/models/{sel}/publish", json={"repo_id": repo, "private": private}))
                if r:
                    st.success(f"Published: {r['url']}")

# ======================= Detection =======================
with tabs[1]:
    dopts = api("GET", "/detect/options").json()
    c1, c2, c3 = st.columns([2, 2, 3])
    dmodel = c1.selectbox("Model", list(dopts["models"]), format_func=lambda k: dopts["models"][k])
    thr = c2.slider("Confidence threshold", 0.1, 0.95, 0.5, 0.05)
    only = c3.multiselect("Only these classes (optional)", dopts["classes"])
    src = st.radio("Source", ["Image", "Webcam (live)", "Video file"], horizontal=True)
    if src == "Image":
        f = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp", "bmp"], key="det_img")
        if f:
            form = {"model": dmodel, "threshold": thr, "classes": ",".join(only)}
            r = api("POST", "/detect/image", data={**form, "output": "image"}, files={"file": (f.name, f.getvalue())})
            if r.status_code == 200:
                i1, i2 = st.columns([3, 1])
                i1.image(r.content, use_container_width=True)
                i2.metric("Objects", sum(json.loads(r.headers["x-detections"]).values()))
                i2.metric("Inference", f"{r.headers['x-latency-ms']} ms")
                i2.json(json.loads(r.headers["x-detections"]))
                i2.download_button("Annotated image", r.content, file_name="detections.jpg")
                coco = api("POST", "/detect/image", data={**form, "output": "coco"}, files={"file": (f.name, f.getvalue())})
                i2.download_button("COCO JSON", coco.content, file_name="detections.json")
            else:
                ok(r)
    elif src == "Webcam (live)":
        st.caption("Uses the camera on this computer. Detection runs locally; frames are not stored.")
        w1, w2, w3 = st.columns([1, 1, 4])
        cam_index = w3.number_input("Camera index", 0, 9, 0)
        if w1.button("Start camera", type="primary"):
            ok(api("POST", "/detect/webcam/start", json={"index": int(cam_index), "model": dmodel, "threshold": thr,
                                                         "classes": only or None}))
        if w2.button("Stop camera"):
            api("POST", "/detect/webcam/stop")

        @st.fragment(run_every=0.3)
        def live():
            s = api("GET", "/detect/webcam/status").json()
            if s.get("error"):
                st.error(s["error"])
            if s.get("running"):
                fr = api("GET", "/detect/webcam/frame")
                if fr.status_code == 200:
                    st.image(fr.content, use_container_width=True)
                st.caption(f"{s['fps']} FPS | {s.get('ms', 0)} ms per frame | " +
                           (", ".join(f"{k}: {v}" for k, v in s.get("counts", {}).items()) or "no objects"))
            elif not s.get("error"):
                st.info("Camera is off.")
        live()
    else:
        v = st.file_uploader("Video (mp4, avi, mov)", type=["mp4", "avi", "mov", "mkv"])
        every = st.slider("Analyse every Nth frame", 1, 10, 3)
        if v and st.button("Annotate video", type="primary"):
            jid = ok(api("POST", "/detect/video", data={"model": dmodel, "threshold": thr, "every": every},
                         files={"file": (v.name, v.getvalue())}))
            if jid:
                j = wait_job(jid["job_id"], st.empty())
                if j["status"] == "done":
                    res = j["result"]
                    st.success(f"{res['frames_processed']} frames analysed in {res['seconds']} s; most objects at once: "
                               f"{res['max_simultaneous']}")
                    vid = api("GET", f"/files/{res['output']}")
                    st.download_button("Download annotated video", vid.content, file_name="annotated.mp4")
                else:
                    st.error(j["error"])

# ======================= Generation =======================
with tabs[2]:
    st.caption(f"Text-to-image with SD-Turbo (runs locally on {health['device'].upper()}; the first use downloads "
               "about 2.6 GB). One step gives a good image; more steps add detail.")
    with st.form("gen"):
        prompt = st.text_area("Prompt", "a lighthouse on a rocky coast at sunset, oil painting", height=80)
        g1, g2, g3, g4 = st.columns(4)
        steps = g1.slider("Steps", 1, 4, 1)
        size = g2.select_slider("Size", [256, 384, 512], value=512)
        n = g3.slider("Images", 1, 4, 1)
        seed = g4.number_input("Seed (0 = random)", 0, 2_000_000_000, 0)
        go = st.form_submit_button("Generate", type="primary")
    if go:
        jid = ok(api("POST", "/generate", json={"prompt": prompt, "steps": steps, "size": size, "n": n,
                                                "seed": int(seed) or None}))
        if jid:
            j = wait_job(jid["job_id"], st.empty())
            if j["status"] == "done":
                cols = st.columns(min(n, 4))
                for col, im in zip(cols, j["result"]["images"]):
                    data = api("GET", f"/files/{im['file']}").content
                    col.image(data, caption=f"seed {im['seed']} | {im['seconds']} s", use_container_width=True)
                    col.download_button("Download PNG", data, file_name=f"generated_{im['seed']}.png", key=f"dl{im['seed']}")
            else:
                st.error(j["error"])
    gal = api("GET", "/gallery", params={"limit": 12}).json()
    if gal:
        st.markdown("**Gallery**")
        cols = st.columns(6)
        for i, im in enumerate(gal):
            cols[i % 6].image(api("GET", f"/files/{im['file']}").content, caption=im["prompt"][:50], use_container_width=True)

# ======================= Enhancement =======================
with tabs[3]:
    eo = api("GET", "/enhance/options").json()
    up = st.file_uploader("Image to enhance", type=["jpg", "jpeg", "png", "webp", "bmp"], key="enh_img")
    e1, e2, e3 = st.columns(3)
    method = e1.selectbox("Method", list(eo["methods"]), format_func=lambda k: eo["methods"][k])
    how = e2.radio("Upscale to", ["Target resolution", "Factor"], horizontal=True)
    target = e3.selectbox("Target", eo["targets"], index=2) if how == "Target resolution" else None
    factor = e3.selectbox("Factor", eo["factors"], index=1) if how == "Factor" else None
    if up:
        src_img = Image.open(io.BytesIO(up.getvalue()))
        st.caption(f"Input: {src_img.width} x {src_img.height}")
        if st.button("Enhance", type="primary"):
            with st.spinner("Upscaling..."):
                r = api("POST", "/enhance", data={"method": method, "target": target or "", "factor": factor or 4},
                        files={"file": (up.name, up.getvalue())})
            if r.status_code == 200:
                info = json.loads(r.headers["x-info"])
                out = Image.open(io.BytesIO(r.content))
                st.success(f"{info['input_size'][0]}x{info['input_size'][1]} -> {info['output_size'][0]}x{info['output_size'][1]} "
                           f"in {info['seconds']} s on {info['device']}")
                b1, b2 = st.columns(2)
                crop_w, crop_h = max(1, src_img.width // 3), max(1, src_img.height // 3)
                sx, sy = out.width / src_img.width, out.height / src_img.height
                b1.markdown("**Before** (centre crop, enlarged with nearest neighbour)")
                box = (src_img.width // 3, src_img.height // 3, src_img.width // 3 + crop_w, src_img.height // 3 + crop_h)
                b1.image(src_img.convert("RGB").crop(box).resize((round(crop_w * sx), round(crop_h * sy)), Image.NEAREST),
                         use_container_width=True)
                b2.markdown("**After** (same crop)")
                b2.image(out.crop((round(box[0] * sx), round(box[1] * sy), round(box[2] * sx), round(box[3] * sy))),
                         use_container_width=True)
                st.download_button("Download enhanced PNG", r.content, file_name="enhanced.png")
            else:
                ok(r)
        if st.button("Benchmark all methods on this image"):
            with st.spinner("Downscaling 4x and restoring with every method..."):
                res = ok(api("POST", "/enhance/benchmark", files={"file": (up.name, up.getvalue())}))
            if res:
                st.dataframe(pd.DataFrame(res), hide_index=True, use_container_width=True)
                st.caption("PSNR/SSIM measure fidelity to the original; sharpness measures restored fine detail. "
                           "GAN models trade some fidelity for much more detail.")
