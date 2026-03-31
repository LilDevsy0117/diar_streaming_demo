# Live speaker diarization demo (microphone only)

Browser microphone → WebSocket → NeMo Sortformer streaming diarization. **There is no YouTube or sample-WAV playback path.**

## Prerequisites

1. **Repository layout**  
   This folder (`diar_streaming_demo`) should sit next to a sibling **NeMo** source tree. `demo_service.py` reads post-processing YAML from `../NeMo/examples/speaker_tasks/diarization/conf/post_processing/`.

2. **Python environment**  
   Install NeMo and hardware-appropriate dependencies, then add the demo packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Hugging Face**  
   Default checkpoints are loaded from the Hub via `from_pretrained`.

   - Interactive: `huggingface-cli login` or `hf auth login`
   - Servers / CI: set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`
   - For gated models, accept the terms on the Hub with the same account you use for login.

4. **Temporary directory (`TMPDIR`)**  
   Unpacking `.nemo` archives may need a large temp area.  
   If you do not set `TMPDIR` and **`/mnt/data` exists**, the server sets `TMPDIR=/mnt/data/tmp/diar_streaming_demo` automatically. On other machines, set it yourself if needed:

   ```bash
   export TMPDIR=/path/to/large/tmp
   ```

## Default models (registry)

| preset            | Hugging Face ID |
|-------------------|-----------------|
| `ultra_8spk`      | `devsy0117/ultra_diar_streaming_sortformer_8spk_v1` |
| `nvidia_4spk_v21` | `nvidia/diar_streaming_sortformer_4spk-v2.1` |

## Run the server

```bash
cd diar_streaming_demo
python server.py --host 0.0.0.0 --port 8765 --device cuda --preset ultra_8spk
```

- **`--preset`**: initial model (`ultra_8spk`, `nvidia_4spk_v21`).
- **`--nemo`**: override with a local `.nemo` path or HF `org/model` string **for the initial load only**; switching preset in the UI uses the registry path again.
- **`--device`**: `cuda` or `cpu`.
- **`--sample-rate`**: default `16000` (client also sends 16 kHz mono).
- **`--spkcache-len`**, **`--no-aux`**: advanced flags (`server.py -h`).

Open `http://<host>:8765/` in a browser.

## Browser (microphone)

- **Chrome / Edge** recommended; grant microphone permission.
- **Non-HTTPS** access from a raw IP (not `localhost`) may block `getUserMedia`. Use TLS behind a reverse proxy or an SSH tunnel to localhost.
- **Start microphone** streams PCM over the WebSocket and updates the live / running heatmaps and segment log. **Reset state** clears server-side streaming state only.

## API summary

- `GET /` — demo UI (`static/index.html`)
- `GET /api/models` — model list, current channel count, heatmap parameters
- `WebSocket /ws` — binary: 16-bit PCM mono 16 kHz chunks; text JSON: `{"cmd":"reset"}`, `{"cmd":"set_model","id":"<preset>"}`

## Troubleshooting

- **Download failures**: check HF login/token, network, and disk (cache + `TMPDIR`).
- **CUDA OOM**: try `--device cpu` or a smaller preset.
- **NeMo import errors**: verify install path and `PYTHONPATH` for your environment.
