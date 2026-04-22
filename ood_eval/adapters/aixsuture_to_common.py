"""
Adapter: AIxSuture (Zenodo record 7940583, Package 11) -> common format.

Pipeline (idempotent):
 1. Download Package 11.zip (~4.4 GB) with wget -c.
 2. Inspect zip contents (listing).
 3. Extract ONE video only (smallest .mp4).
 4. Sample ~30 frames at 1 Hz via OpenCV, write PNGs.
 5. Write samples.pkl with no GT, placeholder center-crop bbox.
 6. Delete zip + extracted video to reclaim space.

Output:
  common_format/aixsuture/
    images/<video_name>_frame_XXXX.png   (~30 files)
    samples.pkl

The samples.pkl has NO ground truth. WiLoR will run GT-bbox mode against
the placeholder ~60% center-crop bbox as a weak hand-location prior; no
metrics are computed, only qualitative visualization.
"""

import os
import pickle
import shutil
import subprocess
import sys
import time
import zipfile

import cv2
import numpy as np

# --------- paths ---------
ROOT = "/mnt/ssd/yuchang/SurgicalVLA/AIM_2_Project/ood_eval"
RAW_DIR = os.path.join(ROOT, "aixsuture_raw")
ZIP_PATH = os.path.join(RAW_DIR, "Package_11.zip")
DST = os.path.join(ROOT, "common_format", "aixsuture")
IMG_DIR = os.path.join(DST, "images")

ZIP_URL = "https://zenodo.org/api/records/7940583/files/Package%2011.zip/content"
EXPECTED_ZIP_BYTES = 4_428_559_519

TARGET_FRAMES = 30
SAMPLE_HZ = 1.0


def log(msg: str):
    print(f"[aixsuture] {msg}", flush=True)


def download_zip() -> float:
    """Download Package_11.zip with wget -c (resume). Returns elapsed sec."""
    os.makedirs(RAW_DIR, exist_ok=True)

    if os.path.exists(ZIP_PATH) and os.path.getsize(ZIP_PATH) == EXPECTED_ZIP_BYTES:
        log(f"zip already present and complete ({EXPECTED_ZIP_BYTES} bytes) - skipping download")
        return 0.0

    log(f"downloading Package_11.zip (~4.4 GB) -> {ZIP_PATH}")
    t0 = time.time()
    cmd = [
        "wget", "-c",
        "--no-verbose",
        "--show-progress",
        "--progress=dot:giga",
        "-O", ZIP_PATH,
        ZIP_URL,
    ]
    proc = subprocess.run(cmd)
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"wget failed with exit code {proc.returncode}")

    actual = os.path.getsize(ZIP_PATH)
    if actual != EXPECTED_ZIP_BYTES:
        log(f"WARNING: expected {EXPECTED_ZIP_BYTES} bytes, got {actual}")
    log(f"download complete in {dt:.1f} s ({actual} bytes)")
    return dt


def list_zip_contents() -> list:
    """Return list of ZipInfo entries; log first 40 entries."""
    log("inspecting zip contents")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        entries = zf.infolist()

    log(f"total entries in zip: {len(entries)}")
    for e in entries[:40]:
        log(f"  {e.file_size:>12} bytes  {e.filename}")
    return entries


def pick_video(entries) -> zipfile.ZipInfo:
    """Pick the smallest .mp4 entry."""
    videos = [e for e in entries if e.filename.lower().endswith(".mp4") and not e.is_dir()]
    if not videos:
        # fall back to any video extension
        exts = (".mp4", ".mov", ".avi", ".mkv", ".mpg", ".mpeg", ".wmv")
        videos = [e for e in entries if e.filename.lower().endswith(exts) and not e.is_dir()]
    if not videos:
        raise RuntimeError("no video files found in zip")

    log(f"found {len(videos)} video files")
    videos.sort(key=lambda e: e.file_size)
    chosen = videos[0]
    log(f"chose smallest video: {chosen.filename} ({chosen.file_size} bytes)")
    return chosen


def extract_video(entry: zipfile.ZipInfo) -> str:
    """Extract single video entry to RAW_DIR. Returns absolute path."""
    target = os.path.join(RAW_DIR, entry.filename)
    if os.path.exists(target) and os.path.getsize(target) == entry.file_size:
        log(f"video already extracted at {target}")
        return target

    log(f"extracting {entry.filename}")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extract(entry, RAW_DIR)
    if not os.path.exists(target):
        raise RuntimeError(f"extraction failed, {target} not found")
    log(f"extracted to {target} ({os.path.getsize(target)} bytes)")
    return target


def sanitize_video_name(path_in_zip: str) -> str:
    """Turn zip-internal path into a filesystem-friendly stem."""
    base = os.path.basename(path_in_zip)
    stem, _ = os.path.splitext(base)
    return stem.replace(" ", "_").replace("/", "_")


def sample_frames(video_path: str, video_name: str) -> tuple:
    """Sample ~TARGET_FRAMES frames at SAMPLE_HZ. Returns (samples_list, meta)."""
    log(f"opening video with OpenCV: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture failed to open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = total / fps if fps > 0 else 0.0
    log(f"video meta: fps={fps:.3f} total_frames={total} duration={duration:.1f}s res={width}x{height}")

    if fps <= 0 or total <= 0:
        raise RuntimeError("video has invalid fps or frame count")

    # Pick frame indices: try 1 Hz, cap at TARGET_FRAMES, uniformly spread.
    step = int(round(fps / SAMPLE_HZ))
    if step < 1:
        step = 1
    indices_1hz = list(range(0, total, step))
    if len(indices_1hz) >= TARGET_FRAMES:
        # subsample uniformly down to TARGET_FRAMES
        sel = np.linspace(0, len(indices_1hz) - 1, TARGET_FRAMES).round().astype(int)
        indices = [indices_1hz[i] for i in sel]
    else:
        indices = indices_1hz
    log(f"sampling {len(indices)} frames (step={step}, target={TARGET_FRAMES})")

    os.makedirs(IMG_DIR, exist_ok=True)

    samples = []
    saved = 0
    for n, fidx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok or frame is None:
            log(f"  WARN: failed to read frame {fidx}, skipping")
            continue
        H, W = frame.shape[:2]
        img_rel = f"images/{video_name}_frame_{n:04d}.png"
        img_abs = os.path.join(DST, img_rel)
        cv2.imwrite(img_abs, frame)
        saved += 1

        K = np.array(
            [[W, 0, W / 2], [0, W, H / 2], [0, 0, 1]],
            dtype=np.float32,
        )
        bbox = np.array(
            [W * 0.2, H * 0.2, W * 0.8, H * 0.8],
            dtype=np.float32,
        )
        samples.append({
            "frame_id": f"aixsuture/{video_name}/frame_{n:04d}",
            "image_path": img_rel,
            "K": K,
            "joints_3d": None,
            "joints_3d_frame_trustworthy": False,
            "vertices_3d": None,
            "joints_2d": None,
            "bbox": bbox,
            "is_right": True,
            "hand_side": "right",
        })

    cap.release()

    meta = {
        "fps": fps,
        "total_frames": total,
        "duration_s": duration,
        "width": W if samples else width,
        "height": H if samples else height,
        "saved_frames": saved,
    }
    log(f"saved {saved} frames to {IMG_DIR}")
    return samples, meta


def write_samples(samples):
    out = os.path.join(DST, "samples.pkl")
    with open(out, "wb") as f:
        pickle.dump(samples, f)
    log(f"wrote {len(samples)} samples -> {out}")


def cleanup(video_abs_path: str):
    """Remove the downloaded zip and extracted video to reclaim space."""
    if os.path.exists(ZIP_PATH):
        sz = os.path.getsize(ZIP_PATH)
        os.remove(ZIP_PATH)
        log(f"removed zip ({sz} bytes)")
    if os.path.exists(video_abs_path):
        sz = os.path.getsize(video_abs_path)
        os.remove(video_abs_path)
        log(f"removed extracted video ({sz} bytes)")

    # tidy empty dirs inside RAW_DIR (but keep RAW_DIR itself)
    for dirpath, dirnames, filenames in os.walk(RAW_DIR, topdown=False):
        if dirpath == RAW_DIR:
            continue
        if not os.listdir(dirpath):
            try:
                os.rmdir(dirpath)
            except OSError:
                pass


def main():
    os.makedirs(DST, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    dl_sec = download_zip()
    entries = list_zip_contents()
    chosen = pick_video(entries)
    video_path = extract_video(chosen)
    video_name = sanitize_video_name(chosen.filename)

    samples, meta = sample_frames(video_path, video_name)
    write_samples(samples)

    cleanup(video_path)

    log("=" * 60)
    log(f"chosen video     : {chosen.filename}")
    log(f"resolution       : {meta['width']}x{meta['height']}")
    log(f"fps / duration   : {meta['fps']:.3f} / {meta['duration_s']:.1f}s")
    log(f"frames saved     : {meta['saved_frames']}")
    log(f"download seconds : {dl_sec:.1f}")
    log(f"output dir       : {DST}")


if __name__ == "__main__":
    main()
