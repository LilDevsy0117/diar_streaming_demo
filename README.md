# Live speaker diarization demo (microphone only)

Browser microphone → WebSocket → NeMo Sortformer streaming diarization. **There is no YouTube or sample-WAV playback path.**

## Screenshot

![Demo UI: live / running heatmaps, post-processed segment log, model and mic controls](assets/streaming_diar_demo.png)

The **segment log** (right) lists finalized speech intervals per speaker as `start : end` (seconds), using the same per-channel `ts_vad_post_processing` + merge step as NeMo Sortformer RTTM-style output (see `demo_service.py` comments). Heatmaps show raw streaming probabilities.

## Quick start

1. **Python** — use a version supported by your `nemo_toolkit` wheel (often **3.10–3.12**).

2. **PyTorch** — install **before** NeMo, matching CPU or CUDA: [pytorch.org](https://pytorch.org/get-started/locally/).

   ```bash
   # Example: CPU-only
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

3. **This repo + NeMo ASR stack**

   ```bash
   git clone https://github.com/LilDevsy0117/diar_streaming_demo.git
   cd diar_streaming_demo
   pip install -r requirements.txt
   ```

4. **Hugging Face** — default checkpoints load from the Hub.

   ```bash
   huggingface-cli login
   # or: export HF_TOKEN=...
   ```

5. **Run** — use CPU if you have no GPU or a driver/PyTorch CUDA mismatch:

   ```bash
   python server.py --host 0.0.0.0 --port 8765 --device cpu --preset nvidia_4spk_v21
   ```

   Open `http://localhost:8765/`.

## Repository layout

- **No sibling NeMo source tree is required.** Post-processing defaults ship in `conf/post_processing/sortformer_diar_4spk-v1_dihard3-dev.yaml`. If you also keep a `NeMo` folder next to this repo, that file is used only when the bundled copy is missing.
- `server.py` still prepends a sibling `../NeMo` to `sys.path` when present (optional, for developers working from source).

## Hugging Face models

| preset            | Hugging Face ID |
|-------------------|-----------------|
| `ultra_8spk`      | `devsy0117/ultra_diar_streaming_sortformer_8spk_v1` |
| `nvidia_4spk_v21` | `nvidia/diar_streaming_sortformer_4spk-v2.1` |

The **NVIDIA** preset is usually the easiest for a fresh clone (public checkpoint). The **ultra_8spk** preset must be **public** or your HF account must have access; otherwise use `--nemo` with a local `.nemo` file or another Hub id.

## Server options

```bash
python server.py --host 0.0.0.0 --port 8765 --device cuda --preset ultra_8spk
```

- **`--preset`**: `ultra_8spk`, `nvidia_4spk_v21`.
- **`--nemo`**: local `.nemo` or `org/model` for the **initial** load only; the UI preset switch uses registry IDs again.
- **`--device`**: `cuda` or `cpu`. `CUDA_VISIBLE_DEVICES=""` alone does not switch this app off CUDA—you still need `--device cpu` if CUDA init fails.
- **`--sample-rate`**: default `16000`.
- **`--spkcache-len`**, **`--no-aux`**: see `python server.py -h`.

## Temporary directory (`TMPDIR`)

Unpacking Hub checkpoints can need a large temp directory. On this demo image, if `/mnt/data` exists, the server sets `TMPDIR=/mnt/data/tmp/diar_streaming_demo` automatically. Elsewhere:

```bash
export TMPDIR=/path/to/large/tmp
```

## Browser (microphone)

- **Chrome / Edge** recommended; allow the microphone.
- Raw **HTTP + non-localhost IP** often blocks `getUserMedia` — use HTTPS, `localhost`, or an SSH tunnel.

## API summary

- `GET /` — UI (`static/index.html`)
- `GET /api/models` — models, channel count, heatmap parameters
- `WebSocket /ws` — binary PCM; JSON `{"cmd":"reset"}`, `{"cmd":"set_model","id":"<preset>"}`

## Troubleshooting

- **`pip install nemo_toolkit` fails or imports break** — install the matching **torch** build first; check Python version against [NeMo docs](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/).
- **Checkpoint download errors** — `hf login`, `HF_TOKEN`, network, and free disk (cache + `TMPDIR`).
- **CUDA errors / OOM** — `python server.py --device cpu` or a smaller preset.
- **403 / model not found** — use `nvidia_4spk_v21`, or your own `--nemo` checkpoint with Hub access fixed.
