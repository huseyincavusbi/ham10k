#!/usr/bin/env python
"""Lightweight smoke test for the Skin Cancer Detection API.

Checks:
 1. / root
 2. /health
 3. /predict (tta disabled) with generated dummy image
"""
from __future__ import annotations
import io, os, sys, json, time, random, argparse
from typing import Any, Dict
import requests
from PIL import Image

def create_dummy_image(size: int = 224) -> bytes:
    img = Image.new("RGB", (size, size), (
        random.randint(0,255), random.randint(0,255), random.randint(0,255)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()

def check(url: str) -> Dict[str, Any]:
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return {"raw": r.text}

def predict(api: str, img_bytes: bytes) -> Dict[str, Any]:
    files = {"file": ("dummy.jpg", img_bytes, "image/jpeg")}
    data = {"use_tta": "false"}
    r = requests.post(f"{api}/predict", files=files, data=data, timeout=60)
    r.raise_for_status()
    return r.json()

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--api-base', default=os.getenv('API_BASE','http://localhost:8002'))
    p.add_argument('--wait', type=int, default=120)
    args = p.parse_args()
    api = args.api_base.rstrip('/')
    print(f"[smoke] API base: {api}")

    deadline = time.time() + args.wait
    last_err = None
    while time.time() < deadline:
        try:
            h = check(f"{api}/health")
            if h.get('status') in {'healthy','degraded'}:
                print(f"[smoke] Health OK: {h.get('status')}")
                break
        except Exception as e:
            last_err = e
        time.sleep(2)
    else:
        print(f"[smoke] FAIL: health not ready: {last_err}")
        return 2

    try:
        root = check(f"{api}/")
        print("[smoke] Root OK")
    except Exception as e:
        print(f"[smoke] FAIL: root: {e}")
        return 3

    try:
        pred = predict(api, create_dummy_image())
        if not ({'ensemble_prediction','final_prediction'} & set(pred.keys())):
            print(f"[smoke] FAIL: prediction missing expected keys: {pred.keys()}")
            return 4
        print("[smoke] Predict OK")
    except Exception as e:
        print(f"[smoke] FAIL: predict: {e}")
        return 5

    print("[smoke] SUCCESS")
    return 0

if __name__ == '__main__':
    sys.exit(main())
