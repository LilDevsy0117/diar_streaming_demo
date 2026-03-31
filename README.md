# diar_streaming_demo

Mic → WebSocket → NeMo Sortformer streaming diarization (browser UI).

![Screenshot](assets/streaming_diar_demo.png)

## Setup

1. Install [PyTorch](https://pytorch.org/get-started/locally/) (CPU or CUDA) **before** NeMo.
2. `pip install -r requirements.txt`
3. `hf auth login` (or set `HF_TOKEN`) — checkpoints download from the Hub.

## Run

`--preset` picks the **startup model** from the server registry (`ultra_8spk` = 8-ch HF model, `nvidia_4spk_v21` = NVIDIA 4-ch). You can switch in the browser later.

```bash
python server.py --device cpu --preset nvidia_4spk_v21
```

Open `http://localhost:8765/`. GPU / CUDA issues → `--device cpu`. More flags → `python server.py -h`.
