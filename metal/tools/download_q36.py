#!/usr/bin/env python3
import os
import sys
import time
import urllib.request

URL = "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf"
DEST = "/Users/yebyen/w/Strata-data/models/qwen36/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf"
PART = DEST + ".part"

os.makedirs(os.path.dirname(DEST), exist_ok=True)

if os.path.exists(DEST):
    print(f"{DEST} already exists, skipping download.")
    sys.exit(0)

existing_size = os.path.getsize(PART) if os.path.exists(PART) else 0

req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
if existing_size > 0:
    req.add_header("Range", f"bytes={existing_size}-")

print(f"Connecting to {URL} (resuming from {existing_size / (1024**3):.2f} GB)...")
response = urllib.request.urlopen(req)

total_size = existing_size + int(response.headers.get("Content-Length", 0))
print(f"Total size: {total_size / (1024**3):.2f} GB")

chunk_size = 4 * 1024 * 1024  # 4 MB
downloaded = existing_size
t0 = time.time()
last_report = t0
last_bytes = downloaded

with open(PART, "ab" if existing_size > 0 else "wb") as f:
    while True:
        chunk = response.read(chunk_size)
        if not chunk:
            break
        f.write(chunk)
        downloaded += len(chunk)
        now = time.time()
        if now - last_report >= 5.0:
            speed = (downloaded - last_bytes) / (now - last_report) / (1024 * 1024)
            pct = downloaded / total_size * 100 if total_size else 0
            eta_s = (total_size - downloaded) / (speed * 1024 * 1024) if speed > 0 and total_size else 0
            print(f"[{pct:5.1f}%] {downloaded / (1024**3):.2f} / {total_size / (1024**3):.2f} GB "
                  f"({speed:.1f} MB/s, ETA: {eta_s / 60:.1f} min)", flush=True)
            last_report = now
            last_bytes = downloaded

os.rename(PART, DEST)
print(f"Downloaded successfully: {DEST} ({os.path.getsize(DEST) / (1024**3):.2f} GB in {(time.time() - t0)/60:.1f} min)")
