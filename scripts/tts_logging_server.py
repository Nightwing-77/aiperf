# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simple TTS logging server for deployment testing.

Logs every incoming request to stdout and a JSONL file. Returns a minimal
audio response so AIPerf doesn't error on missing response fields.

Usage:
    uv run python scripts/tts_logging_server.py --port 8000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="TTS Logging Server")

_log_file: Path | None = None
_request_count = 0
_start_time = 0.0


@app.middleware("http")
async def log_request(request: Request, call_next):
    global _request_count
    body = await request.body()
    payload = {}
    if body:
        try:
            payload = json.loads(body)
        except Exception:
            payload = {"raw": body.decode("utf-8", errors="replace")[:500]}

    elapsed = time.time() - _start_time
    _request_count += 1
    entry = {
        "seq": _request_count,
        "elapsed_sec": round(elapsed, 3),
        "method": request.method,
        "path": request.url.path,
        "input_len": len(payload.get("input", "")),
        "max_tokens": payload.get("max_tokens"),
        "voice": payload.get("voice"),
        "timestamp": time.time(),
    }
    print(
        f"[#{entry['seq']:04d}] t={entry['elapsed_sec']:7.1f}s "
        f"input={entry['input_len']:6d}c  max_tokens={entry['max_tokens']}  "
        f"voice={entry['voice']}"
    )
    if _log_file is not None:
        with open(_log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    response = await call_next(request)
    return response


@app.post("/v1/audio/speech")
async def tts_speech(request: Request):
    return JSONResponse(
        content={
            "audio": "base64placeholder",
            "duration_ms": 1000,
            "audio_format": "wav",
        }
    )


@app.get("/health")
async def health():
    return {"status": "ok", "requests_received": _request_count}


@app.on_event("startup")
async def startup():
    global _start_time
    _start_time = time.time()
    print(f"\n{'='*60}")
    print("TTS Logging Server started")
    print(f"Logging to: {_log_file}")
    print(f"{'='*60}\n")


def main():
    global _log_file
    parser = argparse.ArgumentParser(description="TTS Logging Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--log-file",
        default="tts_requests.jsonl",
        help="JSONL file to log requests to",
    )
    args = parser.parse_args()

    _log_file = Path(args.log_file)
    _log_file.parent.mkdir(parents=True, exist_ok=True)
    _log_file.unlink(missing_ok=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
