"""
Ultra Diar Streaming Sortformer 8spk — real-time streaming speaker diarization (NeMo streaming pattern, no pipecat).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor

from cache_feature_bufferer import CacheFeatureBufferer
from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.modules.sortformer_modules import StreamingSortformerState
from nemo.collections.asr.parts.utils.speaker_utils import merge_float_intervals
from nemo.collections.asr.parts.utils.vad_utils import PostProcessingParams, ts_vad_post_processing

# RTTM / pred_rttm parity reference (offline Sortformer):
#   nemo/collections/asr/models/sortformer_diar_models.py::_diarize_output_processing
#   — per spk: ts_vad_post_processing(..., unit_10ms_frame_count=subsampling_factor)
#   — then generate_diarization_output_lines → merge_float_intervals per speaker (not across speakers).

# Post-processing YAML: bundled in this repo first; else sibling NeMo checkout (dev).
_DEMO_ROOT = Path(__file__).resolve().parent
_BUNDLED_POSTPROCESSING_YAML = (
    _DEMO_ROOT / "conf" / "post_processing" / "sortformer_diar_4spk-v1_dihard3-dev.yaml"
)
_NEMO_SIBLING_POSTPROCESSING_YAML = (
    _DEMO_ROOT.parent
    / "NeMo"
    / "examples"
    / "speaker_tasks"
    / "diarization"
    / "conf"
    / "post_processing"
    / "sortformer_diar_4spk-v1_dihard3-dev.yaml"
)


def default_postprocessing_yaml_path() -> Path:
    if _BUNDLED_POSTPROCESSING_YAML.is_file():
        return _BUNDLED_POSTPROCESSING_YAML
    return _NEMO_SIBLING_POSTPROCESSING_YAML


def _load_nemo_postprocessing_yaml(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        import yaml  # type: ignore[import-untyped]

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    params = data.get("parameters")
    if not isinstance(params, dict):
        return None
    return params


@dataclass
class StreamingDiarDemoConfig:
    model_path: str
    device: str = "cuda"
    frame_len_in_secs: float = 0.08
    chunk_len: int = 340
    chunk_left_context: int = 1
    chunk_right_context: int = 40
    fifo_len: int = 40
    spkcache_update_period: int = 300
    #: If None, keep sortformer_modules.spkcache_len from the checkpoint
    spkcache_len: Optional[int] = None
    log: bool = False
    #: Send intermediate tensor summaries over WebSocket (collected on the same path as the streaming step)
    return_aux: bool = True
    #: Compatibility flag (ignored). Previously duplicated pre_encode only.
    aux_pre_encode: bool = False


Ultra8DiarConfig = StreamingDiarDemoConfig


def _downsample_2d_np(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim != 2:
        raise ValueError("expected 2D array")
    h, w = arr.shape
    if h == 0 or w == 0:
        return np.zeros((out_h, out_w), dtype=np.float32)
    yi = np.linspace(0, h - 1, out_h).astype(np.int64).clip(0, h - 1)
    xi = np.linspace(0, w - 1, out_w).astype(np.int64).clip(0, w - 1)
    return arr[np.ix_(yi, xi)]


def _pack_2d_for_json(
    tensor_2d: Tensor, out_h: int, out_w: int, *, use_abs: bool = True
) -> Dict[str, Any]:
    a = tensor_2d.detach().float().cpu().numpy()
    if use_abs:
        a = np.abs(a)
    a = _downsample_2d_np(a, out_h, out_w)
    vmin, vmax = float(a.min()), float(a.max())
    if vmax <= vmin:
        norm = np.zeros_like(a)
    else:
        norm = (a - vmin) / (vmax - vmin + 1e-8)
    return {
        "h": out_h,
        "w": out_w,
        "data": norm.astype(np.float32).flatten().tolist(),
        "raw_min": vmin,
        "raw_max": vmax,
    }


def _pack_preds_heatmap(pr: Tensor) -> Dict[str, Any]:
    """(T, N_spk) sigmoid — time × speaker heatmap."""
    if pr.dim() != 2:
        pr = pr.reshape(pr.shape[0], -1)
    oh = min(28, max(4, int(pr.shape[0])))
    ow = min(16, max(4, int(pr.shape[1])))
    return _pack_2d_for_json(pr, oh, ow, use_abs=False)


class Ultra8StreamingDiar:
    """Sortformer streaming diarization demo (overrides chunk, fifo, spkcache_len, etc. from config)."""

    def __init__(
        self,
        cfg: StreamingDiarDemoConfig,
        sample_rate: int = 16000,
        left_offset: int = 8,
        right_offset: int = 8,
    ):
        self._ultra_cfg = cfg
        self.cfg = cfg
        self.device = cfg.device
        self.frame_len_in_secs = cfg.frame_len_in_secs
        self.left_offset = left_offset
        self.right_offset = right_offset
        self.chunk_size = cfg.chunk_len
        self.buffer_size_in_secs = (
            cfg.chunk_len * self.frame_len_in_secs + (self.left_offset + self.right_offset) * 0.01
        )

        self.diarizer = self.build_diarizer()
        self.max_num_speakers = int(self.diarizer.sortformer_modules.n_spk)
        self.use_amp = False
        self.compute_dtype = torch.float32

        self.feature_bufferer = CacheFeatureBufferer(
            sample_rate=sample_rate,
            buffer_size_in_secs=self.buffer_size_in_secs,
            chunk_size_in_secs=cfg.chunk_len * self.frame_len_in_secs,
            preprocessor_cfg=self.diarizer.cfg.preprocessor,
            device=self.device,
        )
        self.streaming_state = self.init_streaming_state(batch_size=1)
        self.total_preds = torch.zeros((1, 0, self.max_num_speakers), device=self.diarizer.device)

        hp = self.get_heatmap_params()
        self._pp_omega = OmegaConf.structured(
            PostProcessingParams(
                onset=float(hp["vad_onset"]),
                offset=float(hp["vad_offset"]),
                pad_onset=float(hp["pad_onset"]),
                pad_offset=float(hp["pad_offset"]),
                min_duration_on=float(hp["min_duration_on"]),
                min_duration_off=float(hp["min_duration_off"]),
            )
        )
        self._pp_subsampling = int(getattr(self.diarizer.encoder, "subsampling_factor", 8))
        #: Wall-clock seconds of PCM fed into diarize (16 kHz mono int16 chunks).
        self._cumulative_audio_sec = 0.0
        #: Finalized post-processed segments already logged (spk, neural_start, neural_end) rounded.
        self._pp_logged_segment_keys: set[tuple[int, float, float]] = set()

    def get_heatmap_params(self) -> Dict[str, Any]:
        """
        NeMo `vad_utils.binarization` hysteresis: speech on when `> onset`, off when `< offset`.
        Defaults load from `sortformer_diar_4spk-v1_dihard3-dev.yaml` (DIHARD3 dev tuning).
        If the checkpoint cfg has postprocessing, that is used when YAML is missing.
        """
        sm = self.diarizer.sortformer_modules
        vad_onset = 0.5
        vad_offset = 0.5
        pad_onset = 0.0
        pad_offset = 0.0
        min_duration_on = 0.0
        min_duration_off = 0.0
        pp_yaml = _load_nemo_postprocessing_yaml(default_postprocessing_yaml_path())
        if pp_yaml is not None:
            vad_onset = float(pp_yaml.get("onset", vad_onset))
            vad_offset = float(pp_yaml.get("offset", vad_offset))
            pad_onset = float(pp_yaml.get("pad_onset", pad_onset))
            pad_offset = float(pp_yaml.get("pad_offset", pad_offset))
            min_duration_on = float(pp_yaml.get("min_duration_on", min_duration_on))
            min_duration_off = float(pp_yaml.get("min_duration_off", min_duration_off))
        else:
            cfg = getattr(self.diarizer, "cfg", None) or getattr(self.diarizer, "_cfg", None)
            if cfg is not None:
                try:
                    pp = None
                    if isinstance(cfg, dict):
                        d = cfg.get("diarization") or cfg.get("diarize") or {}
                        pp = d.get("postprocessing_params") if isinstance(d, dict) else None
                    else:
                        for key in ("diarization", "diarize"):
                            sub = getattr(cfg, key, None)
                            if sub is not None:
                                pp = getattr(sub, "postprocessing_params", None) or (
                                    sub.get("postprocessing_params") if isinstance(sub, dict) else None
                                )
                                if pp is not None:
                                    break
                    if pp is not None:
                        if isinstance(pp, dict):
                            vad_onset = float(pp.get("onset", vad_onset))
                            vad_offset = float(pp.get("offset", vad_offset))
                            pad_onset = float(pp.get("pad_onset", pad_onset))
                            pad_offset = float(pp.get("pad_offset", pad_offset))
                            min_duration_on = float(pp.get("min_duration_on", min_duration_on))
                            min_duration_off = float(pp.get("min_duration_off", min_duration_off))
                        else:
                            vad_onset = float(getattr(pp, "onset", vad_onset))
                            vad_offset = float(getattr(pp, "offset", vad_offset))
                except (TypeError, ValueError, AttributeError):
                    pass

        pp_path = default_postprocessing_yaml_path()
        out: Dict[str, Any] = {
            "sil_threshold": float(getattr(sm, "sil_threshold", 0.2)),
            "vad_onset": vad_onset,
            "vad_offset": vad_offset,
            "pad_onset": pad_onset,
            "pad_offset": pad_offset,
            "min_duration_on": min_duration_on,
            "min_duration_off": min_duration_off,
            "postprocessing_yaml": str(pp_path) if pp_path.is_file() else None,
            "viz_rule": "per_channel_hysteresis_onset_offset",
        }
        if hasattr(sm, "n_base_spks"):
            out["n_base_spks"] = int(sm.n_base_spks)
        if hasattr(sm, "base_speech_prob_threshold"):
            out["base_speech_prob_threshold"] = float(sm.base_speech_prob_threshold)
        if hasattr(sm, "new_speech_prob_threshold"):
            out["new_speech_prob_threshold"] = float(sm.new_speech_prob_threshold)
        return out

    @staticmethod
    def _amp_device_type(device) -> str:
        s = str(device).lower()
        return "cuda" if "cuda" in s else "cpu"

    def init_streaming_state(self, batch_size: int = 1) -> StreamingSortformerState:
        return self.diarizer.sortformer_modules.init_streaming_state(
            batch_size=batch_size, async_streaming=self.diarizer.async_streaming, device=self.device
        )

    def _stream_step_with_aux(
        self,
        processed_signal: Tensor,
        processed_signal_length: Tensor,
        streaming_state: StreamingSortformerState,
        total_preds: Tensor,
        left_offset: int = 0,
        right_offset: int = 0,
        drop_extra_pre_encoded: int = 0,
    ) -> Tuple[StreamingSortformerState, Tensor, Dict[str, Any]]:
        """One forward pass matching `SortformerEncLabelModel.forward_streaming_step` plus aux tensor capture."""
        m = self.diarizer
        sm = m.sortformer_modules
        aux: Dict[str, Any] = {}

        if processed_signal.device != self.device:
            processed_signal = processed_signal.to(self.device)
        if processed_signal_length.device != self.device:
            processed_signal_length = processed_signal_length.to(self.device)
        if total_preds is not None and total_preds.device != self.device:
            total_preds = total_preds.to(self.device)

        amp_dev = self._amp_device_type(self.device)
        with (
            torch.amp.autocast(device_type=amp_dev, dtype=self.compute_dtype, enabled=self.use_amp),
            torch.inference_mode(),
            torch.no_grad(),
        ):
            if self.cfg.return_aux:
                if streaming_state.spkcache is not None and streaming_state.spkcache.numel() > 0:
                    aux["spkcache_before"] = _pack_2d_for_json(streaming_state.spkcache[0], 14, 22, use_abs=True)
                if streaming_state.fifo is not None and streaming_state.fifo.numel() > 0:
                    aux["fifo_before"] = _pack_2d_for_json(streaming_state.fifo[0], 14, 22, use_abs=True)

            chunk_pre_encode_embs, chunk_pre_encode_lengths = m.encoder.pre_encode(
                x=processed_signal, lengths=processed_signal_length
            )
            if drop_extra_pre_encoded > 0:
                chunk_pre_encode_embs = chunk_pre_encode_embs[:, drop_extra_pre_encoded:, :]
                chunk_pre_encode_lengths = chunk_pre_encode_lengths - drop_extra_pre_encoded

            if self.cfg.return_aux:
                aux["chunk_pre_encode"] = _pack_2d_for_json(chunk_pre_encode_embs[0], 16, 26, use_abs=True)

            if m.async_streaming:
                spkcache_fifo_chunk_pre_encode_embs, spkcache_fifo_chunk_pre_encode_lengths = (
                    sm.concat_and_pad(
                        [streaming_state.spkcache, streaming_state.fifo, chunk_pre_encode_embs],
                        [streaming_state.spkcache_lengths, streaming_state.fifo_lengths, chunk_pre_encode_lengths],
                    )
                )
            else:
                spkcache_fifo_chunk_pre_encode_embs = sm.concat_embs(
                    [streaming_state.spkcache, streaming_state.fifo, chunk_pre_encode_embs],
                    dim=1,
                    device=m.device,
                )
                spkcache_fifo_chunk_pre_encode_lengths = (
                    streaming_state.spkcache.shape[1]
                    + streaming_state.fifo.shape[1]
                    + chunk_pre_encode_lengths
                )

            if self.cfg.return_aux:
                aux["concat_pre_encode"] = _pack_2d_for_json(
                    spkcache_fifo_chunk_pre_encode_embs[0], 18, 28, use_abs=True
                )

            spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths = m.frontend_encoder(
                processed_signal=spkcache_fifo_chunk_pre_encode_embs,
                processed_signal_length=spkcache_fifo_chunk_pre_encode_lengths,
                bypass_pre_encode=True,
            )

            if self.cfg.return_aux:
                aux["encoder_seq"] = _pack_2d_for_json(
                    spkcache_fifo_chunk_fc_encoder_embs[0], 18, 28, use_abs=True
                )

            encoder_mask = sm.length_to_mask(
                spkcache_fifo_chunk_fc_encoder_lengths, spkcache_fifo_chunk_fc_encoder_embs.shape[1]
            )
            trans_emb_seq = m.transformer_encoder(
                encoder_states=spkcache_fifo_chunk_fc_encoder_embs, encoder_mask=encoder_mask
            )
            _preds = sm.forward_speaker_sigmoids(trans_emb_seq)
            preds = _preds * encoder_mask.unsqueeze(-1)
            spkcache_fifo_chunk_preds = sm.apply_mask_to_preds(preds, spkcache_fifo_chunk_fc_encoder_lengths)

            if self.cfg.return_aux:
                aux["transformer_seq"] = _pack_2d_for_json(trans_emb_seq[0], 18, 28, use_abs=True)
                aux["diar_logits"] = _pack_preds_heatmap(spkcache_fifo_chunk_preds[0])

            lc = round(left_offset / m.encoder.subsampling_factor)
            rc = math.ceil(right_offset / m.encoder.subsampling_factor)
            if m.async_streaming:
                streaming_state, chunk_preds = sm.streaming_update_async(
                    streaming_state=streaming_state,
                    chunk=chunk_pre_encode_embs,
                    chunk_lengths=chunk_pre_encode_lengths,
                    preds=spkcache_fifo_chunk_preds,
                    lc=lc,
                    rc=rc,
                )
            else:
                streaming_state, chunk_preds = sm.streaming_update(
                    streaming_state=streaming_state,
                    chunk=chunk_pre_encode_embs,
                    preds=spkcache_fifo_chunk_preds,
                    lc=lc,
                    rc=rc,
                )

            total_preds = torch.cat([total_preds, chunk_preds], dim=1)

            if self.cfg.return_aux:
                if streaming_state.spkcache is not None and streaming_state.spkcache.numel() > 0:
                    aux["spkcache_after"] = _pack_2d_for_json(streaming_state.spkcache[0], 14, 24, use_abs=True)
                if streaming_state.fifo is not None and streaming_state.fifo.numel() > 0:
                    aux["fifo_after"] = _pack_2d_for_json(streaming_state.fifo[0], 14, 22, use_abs=True)

        return streaming_state, total_preds, aux

    def build_diarizer(self):
        mp = str(self._ultra_cfg.model_path)
        p = Path(mp)
        if p.is_file():
            diar_model = SortformerEncLabelModel.restore_from(mp, map_location=self._ultra_cfg.device)
        else:
            diar_model = SortformerEncLabelModel.from_pretrained(mp, map_location=self._ultra_cfg.device)
        sm = diar_model.sortformer_modules
        sm.chunk_len = self._ultra_cfg.chunk_len
        sm.chunk_left_context = self._ultra_cfg.chunk_left_context
        sm.chunk_right_context = self._ultra_cfg.chunk_right_context
        sm.fifo_len = self._ultra_cfg.fifo_len
        sm.spkcache_update_period = self._ultra_cfg.spkcache_update_period
        if self._ultra_cfg.spkcache_len is not None:
            sm.spkcache_len = int(self._ultra_cfg.spkcache_len)
        sm.log = self._ultra_cfg.log
        sm._check_streaming_parameters()
        diar_model.eval()
        return diar_model

    def reset_state(self, stream_id: str = "default"):
        self.feature_bufferer.reset()
        self.streaming_state = self.init_streaming_state(batch_size=1)
        self.total_preds = torch.zeros((1, 0, self.max_num_speakers), device=self.diarizer.device)
        self._cumulative_audio_sec = 0.0
        self._pp_logged_segment_keys.clear()

    def _neural_time_to_wall_sec(self, t_neural: float) -> float:
        """Map NeMo postproc timeline (seconds) to wall time using PCM vs model-frame span."""
        tlen = int(self.total_preds.shape[1])
        fl = float(self.frame_len_in_secs)
        neural_span = max(tlen * fl, 1e-9)
        audio_sec = float(self._cumulative_audio_sec)
        w = float(t_neural) * (audio_sec / neural_span)
        return max(0.0, min(w, audio_sec))

    def pop_postproc_segment_log_events(self) -> List[Dict[str, Any]]:
        """
        Same core post-processing as NeMo pred_rttm for Sortformer (per-speaker ts_vad_post_processing
        + per-speaker merge_float_intervals). See module docstring for file/line reference.

        Differences vs offline pred_rttm: (1) preds are streaming-accumulated total_preds, not a single
        full-file forward — values can differ from offline diarize() on the same wav. (2) Wall-time
        scaling maps neural seconds to cumulative PCM length. (3) Trailing margin defers segments
        still inside the lookahead window. (4) Output is JSON log lines, not RTTM strings.
        """
        tp = self.total_preds[0]
        tlen = int(tp.shape[0])
        out: List[Dict[str, Any]] = []
        if tlen == 0:
            return out
        fl = float(self.frame_len_in_secs)
        neural_span = tlen * fl
        margin = max(3.0 * fl, 0.24)
        cutoff = neural_span - margin

        for j in range(int(tp.shape[1])):
            col = tp[:, j].detach().float().cpu()
            segs = ts_vad_post_processing(
                col,
                self._pp_omega,
                unit_10ms_frame_count=self._pp_subsampling,
                bypass_postprocessing=False,
            )
            if segs is None or segs.numel() == 0:
                continue
            arr = segs.detach().cpu().numpy()
            raw: List[List[float]] = []
            for k in range(arr.shape[0]):
                s0_n = float(arr[k, 0])
                s1_n = float(arr[k, 1])
                if s1_n > cutoff:
                    continue
                raw.append([round(s0_n, 2), round(s1_n, 2)])
            if not raw:
                continue
            merged = merge_float_intervals(raw)
            for s0_n, s1_n in merged:
                if s1_n > cutoff:
                    continue
                key = (j, round(float(s0_n), 2), round(float(s1_n), 2))
                if key in self._pp_logged_segment_keys:
                    continue
                self._pp_logged_segment_keys.add(key)
                w0 = self._neural_time_to_wall_sec(s0_n)
                w1 = self._neural_time_to_wall_sec(s1_n)
                if w1 < w0:
                    w0, w1 = w1, w0
                out.append({"start": w0, "end": w1, "spk": j})

        out.sort(key=lambda e: (e["start"], e["spk"]))
        return out

    def _aux_mel(self, features: Tensor) -> Dict[str, Any]:
        """features: (F, T) log-mel, etc."""
        a = features.detach().float().cpu().numpy()
        a = _downsample_2d_np(a, 32, 48)
        vmin, vmax = float(a.min()), float(a.max())
        if vmax <= vmin:
            norm = np.zeros_like(a)
        else:
            norm = (a - vmin) / (vmax - vmin + 1e-8)
        return {
            "h": 32,
            "w": 48,
            "data": norm.astype(np.float32).flatten().tolist(),
            "raw_min": vmin,
            "raw_max": vmax,
        }

    def diarize(self, audio: bytes, stream_id: str = "default") -> Tuple[np.ndarray, Dict[str, Any]]:
        self._cumulative_audio_sec += (len(audio) / 2) / 16000.0
        audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        self.feature_bufferer.update(audio_array)

        features = self.feature_bufferer.get_feature_buffer()
        feature_buffers = features.unsqueeze(0)
        feature_buffers = feature_buffers.transpose(1, 2)
        feature_buffer_lens = torch.tensor([feature_buffers.shape[1]], device=self.device)
        prev_len = int(self.total_preds.shape[1])

        aux: Dict[str, Any] = {}
        if self.cfg.return_aux:
            aux["mel"] = self._aux_mel(features)

        self.streaming_state, self.total_preds, aux_step = self._stream_step_with_aux(
            processed_signal=feature_buffers,
            processed_signal_length=feature_buffer_lens,
            streaming_state=self.streaming_state,
            total_preds=self.total_preds,
            left_offset=self.left_offset,
            right_offset=self.right_offset,
        )
        if self.cfg.return_aux:
            aux.update(aux_step)

        new_len = int(self.total_preds.shape[1])
        if new_len > prev_len:
            log_events = self.pop_postproc_segment_log_events()
            if log_events:
                aux["postproc_log"] = log_events
        # If we only slice the last chunk_size every call, responses overlap heavily with the previous
        # step and the heatmap looks broken or repeats. Return only newly produced frames.
        if new_len <= prev_len:
            return np.zeros((0, self.max_num_speakers), dtype=np.float32), aux
        delta = self.total_preds[:, prev_len:new_len, :].detach().cpu().numpy().astype(np.float32)
        return delta[0], aux
