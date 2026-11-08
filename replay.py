"""Replay a KITTI camera log with exact timing, run a detector, and track:
  1. replay determinism  (stream hash + detection hash identical every run)
  2. timing jitter       (actual release time - intended release time)
  3. end-to-end latency  (detection done - intended release time, per frame)

Usage: python replay.py [--runs 3] [--speed 1.0] [--device cpu]
"""
import argparse, glob, hashlib, json, os, queue, sys, threading, time
from datetime import datetime

import cv2
import numpy as np
import torch
import torchvision

sys.setswitchinterval(0.0002)  # keep the release thread from waiting on the detector thread's GIL
HERE = os.path.dirname(os.path.abspath(__file__))
DRIVE = glob.glob(f"{HERE}/data/kitti/*/*_sync/image_02")[0]
REPORT = f"{HERE}/report.json"


def load_log():
    """Frames (preloaded, so disk I/O never perturbs timing) + relative ts in ns."""
    stamps = []
    for line in open(f"{DRIVE}/timestamps.txt"):
        dt = datetime.strptime(line.strip()[:26], "%Y-%m-%d %H:%M:%S.%f")  # us precision
        stamps.append(int(dt.timestamp() * 1e6) * 1000)
    files = sorted(glob.glob(f"{DRIVE}/data/*.png"))
    assert len(files) == len(stamps), (len(files), len(stamps))
    frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files]
    rel = [s - stamps[0] for s in stamps]
    return frames, rel


def stream_hash(frames, rel):
    h = hashlib.sha256()
    for i, (f, t) in enumerate(zip(frames, rel)):
        h.update(f"{i}:{t}:".encode()); h.update(f.tobytes())
    return h.hexdigest()


def make_detector(device):
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    w = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    m = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=w).eval().to(device)
    def detect(img):  # img: HxWx3 uint8 RGB
        x = torch.from_numpy(img).permute(2, 0, 1).float().div(255).to(device)
        with torch.inference_mode():
            return m([x])[0]
    return detect


def det_digest(d):
    """Quantised so only real changes (not float noise below 0.1px / 0.001) matter."""
    keep = d["scores"] > 0.4
    b = d["boxes"][keep].cpu().numpy().round(1)
    return b.tolist(), d["labels"][keep].tolist(), d["scores"][keep].cpu().numpy().round(3).tolist()


def spin_until(t):
    while True:
        rem = t - time.perf_counter()
        if rem <= 0: return
        if rem > 0.02: time.sleep(rem - 0.015)  # coarse sleep far out, busy-wait the last 15 ms


def run_once(frames, rel, detect, speed):
    q = queue.Queue()
    release = [None] * len(frames); done = [None] * len(frames); dets = [None] * len(frames)

    def worker():
        for _ in frames:
            i = q.get()
            dets[i] = det_digest(detect(frames[i]))
            done[i] = time.perf_counter()

    th = threading.Thread(target=worker); th.start()
    t0 = time.perf_counter() + 0.2
    intended = [t0 + r / 1e9 / speed for r in rel]
    for i in range(len(frames)):
        spin_until(intended[i])
        release[i] = time.perf_counter()
        q.put(i)
    th.join()

    jitter = np.array([(release[i] - intended[i]) * 1e3 for i in range(len(frames))])
    latency = np.array([(done[i] - intended[i]) * 1e3 for i in range(len(frames))])
    dh = hashlib.sha256(json.dumps(dets).encode()).hexdigest()
    return jitter, latency, dh


def pct(a, p): return float(np.percentile(a, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = real time")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    frames, rel = load_log()
    sh = stream_hash(frames, rel)
    detect = make_detector(a.device)
    detect(frames[0])  # warm-up, excluded from stats
    print(f"{len(frames)} frames, {rel[-1]/1e9:.2f}s log, speed x{a.speed}, device {a.device}")
    print(f"stream hash {sh[:16]}")

    runs = []
    for r in range(a.runs):
        j, l, dh = run_once(frames, rel, detect, a.speed)
        runs.append(dict(det_hash=dh, jitter_ms=j.tolist(), latency_ms=l.tolist()))
        print(f"run {r}: det {dh[:12]} | jitter mean|.| {np.abs(j).mean():.3f} p95 {pct(np.abs(j),95):.3f} "
              f"max {np.abs(j).max():.3f} ms | latency p50 {pct(l,50):.1f} p95 {pct(l,95):.1f} max {l.max():.1f} ms")

    in_proc = len({r["det_hash"] for r in runs}) == 1
    prev = json.load(open(REPORT)) if os.path.exists(REPORT) else None
    cross = None if prev is None else (prev["stream_hash"] == sh and prev["det_hash"] == runs[0]["det_hash"])
    print(f"\ndeterminism: runs identical in this process = {in_proc}"
          f" | identical to previous invocation = {cross if cross is not None else 'n/a (first run)'}")
    json.dump(dict(stream_hash=sh, det_hash=runs[0]["det_hash"], speed=a.speed, device=a.device,
                   deterministic=in_proc, runs=runs), open(REPORT, "w"))


if __name__ == "__main__":
    main()
