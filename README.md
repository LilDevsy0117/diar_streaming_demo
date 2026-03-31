# diar_streaming_demo

Mic → WebSocket → NeMo Sortformer streaming diarization (browser UI).

![Screenshot](assets/streaming_diar_demo.png)

## Setup

1. Install [PyTorch](https://pytorch.org/get-started/locally/) (CPU or CUDA) **before** NeMo.
2. `pip install -r requirements.txt`
3. `hf auth login` (or set `HF_TOKEN`) — checkpoints download from the Hub.

## Run

```bash
python server.py --device cpu
```
