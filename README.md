# sensor-replay

Replays a public driving log (KITTI raw `2011_09_26_drive_0001`, camera `image_02`, 108 frames, ~10 Hz)
at its original timestamps and runs a detector (torchvision SSDLite-MobileNetV3, COCO) on every frame.

    python replay.py                      # 3 runs, real time, CPU
    python replay.py --speed 2 --runs 5   # faster than real time

Tracked per run (printed, full per-frame arrays saved to `report.json`):

| metric | definition |
|---|---|
| determinism | SHA-256 of the frame stream (bytes + timestamps) and of quantised detections; compared across runs and against the previous invocation |
| timing jitter | actual frame release time minus intended time (`t0 + log timestamp`), ms |
| end-to-end latency | detection finished minus intended release time, per frame, ms (includes queueing) |

Data lives in `data/kitti/` (images + `timestamps.txt` only; frames are preloaded so disk I/O never affects timing).
Re-download: `https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip`
