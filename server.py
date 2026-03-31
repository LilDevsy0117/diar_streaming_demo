#!/usr/bin/env python3
"""Real-time speaker diarization web demo — microphone + WebSocket + FastAPI."""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

# NeMo restore_from / HF cache unpack may need a large TMPDIR.
def _bootstrap_tmpdir_for_nemo_unpack() -> None:
    if os.environ.get("TMPDIR"):
        return
    data_root = Path("/mnt/data")
    if not data_root.is_dir():
        return
    tdir = data_root / "tmp" / "diar_streaming_demo"
    try:
        tdir.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(tdir)
    except OSError:
        pass


_bootstrap_tmpdir_for_nemo_unpack()

_DEMO = Path(__file__).resolve().parent
sys.path.insert(0, str(_DEMO))
_ROOT = _DEMO.parent
if (_ROOT / "NeMo").is_dir():
    sys.path.insert(0, str(_ROOT / "NeMo"))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from demo_service import StreamingDiarDemoConfig, Ultra8StreamingDiar

app = FastAPI(title="Streaming Sortformer Diarization Demo (mic)")

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

diar: Ultra8StreamingDiar | None = None
diar_lock = asyncio.Lock()
current_preset_id: str = "ultra_8spk"

STREAM_PRESET_4SPK_VOICE_AGENT: dict = {
    "chunk_len": 6,
    "chunk_left_context": 1,
    "chunk_right_context": 7,
    "fifo_len": 188,
    "spkcache_update_period": 144,
}

MODEL_REGISTRY: dict[str, dict] = {
    "ultra_8spk": {
        "label": "Ultra 8-speaker v1",
        "path": "devsy0117/ultra_diar_streaming_sortformer_8spk_v1",
        "spk_max": 8,
        **STREAM_PRESET_4SPK_VOICE_AGENT,
    },
    "nvidia_4spk_v21": {
        "label": "NVIDIA Sortformer 4spk v2.1",
        "path": "nvidia/diar_streaming_sortformer_4spk-v2.1",
        "spk_max": 4,
        **STREAM_PRESET_4SPK_VOICE_AGENT,
    },
}


def make_config(
    preset_id: str,
    device: str,
    path_override: str | None = None,
    spkcache_override: int | None = None,
    *,
    return_aux: bool = True,
    aux_pre_encode: bool = False,
) -> StreamingDiarDemoConfig:
    if preset_id not in MODEL_REGISTRY:
        raise ValueError(f"unknown preset: {preset_id}")
    r = MODEL_REGISTRY[preset_id]
    path = str(Path(path_override)) if path_override else str(r["path"])
    spkcache_len = r.get("spkcache_len")
    if spkcache_override is not None:
        spkcache_len = spkcache_override
    return StreamingDiarDemoConfig(
        model_path=str(path),
        device=device,
        chunk_len=r["chunk_len"],
        chunk_left_context=r["chunk_left_context"],
        chunk_right_context=r["chunk_right_context"],
        fifo_len=r["fifo_len"],
        spkcache_update_period=r["spkcache_update_period"],
        spkcache_len=spkcache_len,
        return_aux=return_aux,
        aux_pre_encode=aux_pre_encode,
    )


async def load_diar(preset_id: str, device: str, path_override: str | None = None) -> Ultra8StreamingDiar:
    global diar, current_preset_id
    args = get_cli_args()
    cfg = make_config(
        preset_id,
        device,
        path_override=path_override,
        spkcache_override=args.spkcache_len,
        return_aux=not getattr(args, "no_aux", False),
        aux_pre_encode=getattr(args, "aux_pre_encode", False),
    )
    new_diar = Ultra8StreamingDiar(cfg, sample_rate=get_cli_args().sample_rate)
    async with diar_lock:
        old = diar
        diar = new_diar
        current_preset_id = preset_id
        del old
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return diar


def get_diar() -> Ultra8StreamingDiar:
    if diar is None:
        raise RuntimeError("Model not loaded")
    return diar


def make_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--preset",
        default="ultra_8spk",
        choices=list(MODEL_REGISTRY.keys()),
        help="Initial model preset to load",
    )
    p.add_argument(
        "--nemo",
        default=None,
        help="Override with local .nemo path or HF org/model id",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--device", default="cuda")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument(
        "--spkcache-len",
        type=int,
        default=None,
        dest="spkcache_len",
        help="Force sortformer_modules.spkcache_len",
    )
    p.add_argument(
        "--no-aux",
        action="store_true",
        help="Omit mel/FIFO etc. aux fields in WebSocket responses",
    )
    p.add_argument(
        "--aux-pre-encode",
        action="store_true",
        help="(compat flag, ignored)",
    )
    return p


def default_cli_args() -> argparse.Namespace:
    return make_cli_parser().parse_args([])


def get_cli_args():
    args = getattr(app.state, "cli_args", None)
    if args is None:
        args = default_cli_args()
        app.state.cli_args = args
    return args


@app.on_event("startup")
async def _startup():
    global diar, current_preset_id
    if getattr(app.state, "cli_args", None) is None:
        app.state.cli_args = default_cli_args()
    args = app.state.cli_args
    cfg = make_config(
        args.preset,
        args.device,
        path_override=args.nemo,
        spkcache_override=getattr(args, "spkcache_len", None),
        return_aux=not getattr(args, "no_aux", False),
        aux_pre_encode=getattr(args, "aux_pre_encode", False),
    )
    diar = Ultra8StreamingDiar(cfg, sample_rate=args.sample_rate)
    current_preset_id = args.preset


@app.get("/")
async def index():
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/api/models")
async def api_models():
    out = []
    for mid, r in MODEL_REGISTRY.items():
        out.append(
            {
                "id": mid,
                "label": r["label"],
                "spk_max": r["spk_max"],
                "default_path": str(r["path"]),
            }
        )
    d = diar
    n_spk = int(d.max_num_speakers) if d is not None else None
    heatmap_params = d.get_heatmap_params() if d is not None else None
    sil_threshold = float(heatmap_params["sil_threshold"]) if heatmap_params else 0.2
    vad_onset = float(heatmap_params["vad_onset"]) if heatmap_params else 0.5
    vad_offset = float(heatmap_params["vad_offset"]) if heatmap_params else 0.5
    return {
        "models": out,
        "current_id": current_preset_id,
        "current_n_spk": n_spk,
        "sil_threshold": sil_threshold,
        "vad_onset": vad_onset,
        "vad_offset": vad_offset,
        "heatmap_params": heatmap_params,
    }


@app.websocket("/ws")
async def ws_audio(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("text"):
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue

                if data.get("cmd") == "reset":
                    async with diar_lock:
                        get_diar().reset_state()
                    await ws.send_json({"type": "reset", "ok": True})
                    continue

                if data.get("cmd") == "set_model" and data.get("id") in MODEL_REGISTRY:
                    pid = data["id"]
                    args = get_cli_args()
                    try:
                        new_d = await load_diar(pid, args.device, path_override=None)
                        hp = new_d.get_heatmap_params()
                        await ws.send_json(
                            {
                                "type": "model",
                                "ok": True,
                                "id": pid,
                                "label": MODEL_REGISTRY[pid]["label"],
                                "n_spk": int(new_d.max_num_speakers),
                                "spk_max": MODEL_REGISTRY[pid]["spk_max"],
                                "sil_threshold": float(hp["sil_threshold"]),
                                "vad_onset": float(hp["vad_onset"]),
                                "vad_offset": float(hp["vad_offset"]),
                                "heatmap_params": hp,
                            }
                        )
                    except Exception as e:
                        await ws.send_json({"type": "model", "ok": False, "error": str(e)})
                    continue
                continue

            raw = msg.get("bytes")
            if raw is None or len(raw) < 2:
                continue
            audio_duration_sec = (len(raw) / 2) / 16000.0
            async with diar_lock:
                d = get_diar()
                probs, aux = d.diarize(raw)
            aux = dict(aux)
            postproc_log = aux.pop("postproc_log", None)
            n_spk = int(probs.shape[1]) if probs.size else int(d.max_num_speakers)
            out: dict[str, Any] = {
                "type": "probs",
                "frame_len_sec": 0.08,
                "audio_duration_sec": audio_duration_sec,
                "n_spk": n_spk,
                "shape": [int(probs.shape[0]), n_spk],
                "data": probs.tolist(),
            }
            if postproc_log is not None:
                out["postproc_log"] = postproc_log
            if aux:
                out["aux"] = aux
            await ws.send_json(out)
    except WebSocketDisconnect:
        pass


def main():
    p = make_cli_parser()
    p.add_argument("-h", "--help", action="help", help="Show this help message and exit")
    args = p.parse_args()
    app.state.cli_args = args

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
