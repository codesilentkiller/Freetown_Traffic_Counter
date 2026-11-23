
"""Streamlit front-end for the Freetown traffic counter.

Run from the project root:   streamlit run app/app.py

The page is a thin wrapper over scripts/count_traffic.py (process_video). It
streams the video frame by frame and updates a progress bar as it goes, so long
videos never freeze the page. The upload is written to disk in chunks (Streamlit
itself holds the uploaded file in RAM, so very large files need matching memory).
"""
import os
import sys
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)                                   # config.yaml, models/, outputs/ are relative to the root
sys.path.insert(0, str(ROOT / "scripts"))
from count_traffic import build_line, load_config, process_video  # noqa: E402

st.set_page_config(page_title="Freetown Traffic Counter", page_icon="🚦", layout="wide")

CLASS_INFO = {
    "car": "Private cars, saloons, SUVs and similar light vehicles.",
    "truck": "Trucks and lorries.",
    "poda-poda": "Locally operated minibus, a common form of public transport in Freetown (not a generic bus).",
    "keke": "Three-wheeled tricycle taxi.",
    "okada": "Commercial motorcycle taxi (not every motorcycle).",
    "pedestrian": "People on foot.",
}


def save_upload(uploaded) -> Path:
    """Write the uploaded file to disk in 8 MB chunks (avoids a second full in-memory copy)."""
    run_dir = ROOT / "outputs" / "app_runs" / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    dst = run_dir / f"input{Path(uploaded.name).suffix.lower()}"
    uploaded.seek(0)
    with open(dst, "wb") as f:
        while chunk := uploaded.read(8 * 1024 * 1024):
            f.write(chunk)
    return dst


def first_frame_with_line(video: Path, orientation: str, position: float, flip: bool):
    """First video frame with the counting line drawn on it, as an RGB array."""
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = frame.shape[:2]
    (x1, y1), (x2, y2) = build_line(orientation, position, w, h, flip)
    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), max(2, w // 300))
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


cfg_default = load_config("config.yaml")

# ------------------------------------------------------------------ sidebar
st.sidebar.title("Settings")
models = sorted((ROOT / "models").glob("*.pt"))
if not models:
    st.sidebar.warning("No .pt files in `models/`. Train a model (see README) and copy `best.pt` there.")
    model_path = None
else:
    names = [m.name for m in models]
    default_idx = names.index("best.pt") if "best.pt" in names else 0
    model_path = models[st.sidebar.selectbox("Model", range(len(names)), index=default_idx,
                                             format_func=lambda i: names[i])]

st.sidebar.subheader("Counting line")
orientation = st.sidebar.radio("Orientation", ["horizontal", "vertical"],
                               index=["horizontal", "vertical"].index(cfg_default["line"]["orientation"]),
                               help="Horizontal: counts traffic moving up/down the frame. Vertical: left/right.")
position = st.sidebar.slider("Position", 0.0, 1.0, float(cfg_default["line"]["position"]), 0.01,
                             help="Fraction of the frame height (horizontal) or width (vertical).")
flip = st.sidebar.checkbox("Swap in / out", value=bool(cfg_default["line"]["flip"]))

st.sidebar.subheader("Detection")
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, float(cfg_default["inference"]["conf"]), 0.05,
                         help="Higher = fewer false alarms but more missed vehicles.")
stride = st.sidebar.select_slider("Process every Nth frame", options=[1, 2, 3, 4, 5],
                                  value=int(cfg_default["inference"]["vid_stride"]),
                                  help="Higher = much faster on a CPU, but tracking is coarser.")
max_frames = st.sidebar.number_input("Quick test: stop after N frames (0 = whole video)", 0, 1_000_000, 0, 100)

# --------------------------------------------------------------------- main
st.title("🚦 Freetown Traffic Counter")
tab_run, tab_about = st.tabs(["Count traffic", "About"])

with tab_run:
    uploaded = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"])
    if uploaded is not None:
        key = f"{uploaded.name}-{uploaded.size}"
        if st.session_state.get("upload_key") != key:         # new file: save it, forget old results
            st.session_state["upload_key"] = key
            st.session_state["video_path"] = save_upload(uploaded)
            st.session_state.pop("result", None)
        video_path = st.session_state["video_path"]

        st.subheader("1. Place the counting line")
        st.caption("Vehicles are counted when their centre crosses the red line. Put it where traffic is "
                   "spread out and moving, away from junctions where vehicles stop.")
        preview = first_frame_with_line(video_path, orientation, position, flip)
        if preview is None:
            st.error("Could not read this video. Try re-exporting it as MP4.")
        else:
            st.image(preview, use_container_width=True)

        st.subheader("2. Process")
        if st.button("Process Video", type="primary", disabled=model_path is None or preview is None):
            cfg = load_config("config.yaml", {
                "line": {"orientation": orientation, "position": position, "flip": flip},
                "inference": {"conf": conf, "vid_stride": stride, "max_frames": int(max_frames) or None},
            })
            bar = st.progress(0.0, text="Loading model ...")
            t_start = time.time()

            def on_progress(done, total):
                frac = min(done / total, 1.0) if total else 0.0
                elapsed = time.time() - t_start
                eta = f", about {elapsed / done * (total - done):.0f}s left" if done and total and done < total else ""
                bar.progress(frac, text=f"Processed {done} of {total or '?'} frames{eta}")

            try:
                result = process_video(video_path, model_path, cfg, video_path.parent, on_progress)
                bar.progress(1.0, text=f"Done in {result['seconds']:.0f}s ({result['processing_fps']:.1f} frames/s)")
                st.session_state["result"] = {"res": result, "model": model_path.name}
            except Exception as e:                                # show the problem instead of a stack trace
                bar.empty()
                st.error(f"Processing failed: {e}")

    # Results live in session_state so clicking a download button does not re-run the video.
    stored = st.session_state.get("result")
    if stored:
        res = stored["res"]
        st.divider()
        st.header("Results")
        if res["warning"]:
            st.warning(res["warning"])
        if res["video"] and Path(res["video"]).exists():
            st.video(str(res["video"]))

        # Explicit columns: the list can be empty (nothing crossed the line), and an empty
        # DataFrame would otherwise have no "total" column.
        df = pd.DataFrame(res["counts"], columns=["class", "in", "out", "total"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Total crossings", int(df["total"].sum()))
        c2.metric("In", int(df["in"].sum()))
        c3.metric("Out", int(df["out"].sum()))

        if df.empty:
            st.info("Nothing crossed the counting line. Check that the line sits across where vehicles actually "
                    "travel, try a lower confidence threshold, and make sure the clip is long enough (not just "
                    "a few frames). Watch the annotated video above to see what the model detected.")
        else:
            left, right = st.columns(2)
            with left:
                st.dataframe(df, hide_index=True, use_container_width=True)
            with right:
                st.bar_chart(df.set_index("class")[["in", "out"]])

        d1, d2, d3 = st.columns(3)
        if res["csv"]:
            d1.download_button("Download counts (CSV)", Path(res["csv"]).read_bytes(),
                               file_name=Path(res["csv"]).name, mime="text/csv")
        if res["json"]:
            d2.download_button("Download detections (JSON)", Path(res["json"]).read_bytes(),
                               file_name=Path(res["json"]).name, mime="application/json")
        if res["video"] and Path(res["video"]).stat().st_size < 300 * 1024 * 1024:
            d3.download_button("Download annotated video", Path(res["video"]).read_bytes(),
                               file_name=Path(res["video"]).name, mime="video/mp4")
        st.caption(f"Model: {stored['model']}. Files are saved in `{Path(res['out_dir']).relative_to(ROOT)}`.")

with tab_about:
    st.header("About")
    st.write("This tool detects, tracks and counts road users in video using a YOLO object detector trained on "
             "Freetown traffic, with ByteTrack following each object between frames. It runs on a normal laptop CPU.")
    st.subheader("The six classes")
    st.table(pd.DataFrame({"Class": list(CLASS_INFO), "What it is": list(CLASS_INFO.values())}))
    st.subheader("Limitations")
    st.markdown(
        "- **Daytime only.** Not designed for night footage or heavy rain; accuracy will drop sharply.\n"
        "- **No license plate reading** and no identification of individual people or vehicles.\n"
        "- **Counts are estimates.** Crowded scenes, occlusion and a camera that is too low or too far can cause "
        "missed vehicles or an object being counted twice when its track ID switches.\n"
        "- **Line placement matters.** Put the line where traffic flows freely, not where vehicles queue.\n"
        "- **Look-alike classes.** Poda-poda vs. other minibuses/trucks, and okada vs. private motorcycles, are the "
        "most likely confusions. Check the confusion matrix in the evaluation report.\n"
        "- **Speed.** A CPU processes only a few frames per second. For long videos raise *Process every Nth frame*.")
