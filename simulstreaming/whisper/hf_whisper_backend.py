import os
import logging
from typing import Optional, Tuple, List

import numpy as np
import copy


logger = logging.getLogger(__name__)


def is_hf_whisper_dir(path: str) -> bool:
    if not path:
        return False
    if not os.path.isdir(path):
        return False
    has_config = os.path.isfile(os.path.join(path, "config.json"))
    has_weights = (
        os.path.isfile(os.path.join(path, "pytorch_model.bin"))
        or os.path.isfile(os.path.join(path, "model.safetensors"))
    )
    return has_config and has_weights


class HFWhisperASR:
    """Minimal HuggingFace Whisper ASR wrapper for SimulStreaming.

    This backend is intentionally simple: it transcribes each audio segment
    independently (optionally with a fixed prompt), so we can keep the
    SimulStreaming/VAC chunking while avoiding OpenAI .pt conversion.
    """

    def __init__(
        self,
        *,
        language: str,
        model_path: str,
        beams: int,
        task: str,
        audio_min_len: float,
        init_prompt: Optional[str] = None,
        static_init_prompt: Optional[str] = None,
    ) -> None:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.language = language
        self.model_path = model_path
        self.beams = max(1, int(beams))
        self.task = task
        self.audio_min_len = float(audio_min_len)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch

        logger.info(f"Loading HuggingFace Whisper from: {model_path}")
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Prefer forced decoder ids (works even when generation_config is missing lang_to_id)
        self.forced_decoder_ids = None
        try:
            if language and language != "auto":
                self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language=language, task=task
                )
        except Exception as e:
            logger.warning(f"Could not build forced_decoder_ids: {e}")
            self.forced_decoder_ids = None

        prompt_text = " ".join([p for p in [static_init_prompt, init_prompt] if p])
        self.prompt_ids = None
        if prompt_text:
            get_prompt_ids = getattr(self.processor, "get_prompt_ids", None)
            if callable(get_prompt_ids):
                try:
                    self.prompt_ids = get_prompt_ids(prompt_text)
                except Exception as e:
                    logger.warning(f"Could not build prompt_ids: {e}")
                    self.prompt_ids = None

    def set_translate_task(self):
        self.task = "translate"
        try:
            if self.language and self.language != "auto":
                self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language=self.language, task=self.task
                )
        except Exception:
            self.forced_decoder_ids = None

    def warmup(self, audio: np.ndarray):
        _ = self.transcribe_segment(audio)

    def _generate_max_new_tokens(self, seconds: float) -> int:
        # Conservative heuristic to prevent runaway generations.
        # Whisper typically emits a few tokens per second of speech.
        base = int(seconds * 6) + 16
        return int(max(32, min(448, base)))

    def transcribe_segment(self, audio: np.ndarray) -> Tuple[str, List[int]]:
        if audio is None:
            return "", []
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return "", []

        seconds = audio.size / 16000.0
        if seconds < self.audio_min_len:
            return "", []

        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)

        gen_kwargs = {
            "num_beams": self.beams,
            "do_sample": False,
            "max_new_tokens": self._generate_max_new_tokens(seconds),
        }
        generation_config = None
        if self.forced_decoder_ids is not None:
            # Newer transformers validate generate() kwargs strictly; for Whisper,
            # forced_decoder_ids should be provided via GenerationConfig.
            try:
                base_cfg = getattr(self.model, "generation_config", None)
                if base_cfg is not None:
                    generation_config = base_cfg.clone() if hasattr(base_cfg, "clone") else copy.deepcopy(base_cfg)
                    setattr(generation_config, "forced_decoder_ids", self.forced_decoder_ids)
            except Exception as e:
                logger.warning(f"Could not set forced_decoder_ids in generation_config: {e}")
                generation_config = None
        if self.prompt_ids is not None:
            # transformers Whisper generation expects prompt_ids to be a torch.Tensor
            # (and will torch.cat() it with decoder_input_ids). Some tokenizers return
            # numpy arrays here, so normalize to a 1D LongTensor on the right device.
            # (WhisperGenerationMixin will batch it internally via prompt_ids[None].repeat(B, 1)).
            prompt_ids = self.torch.as_tensor(self.prompt_ids, dtype=self.torch.long, device=self.device).reshape(-1)
            gen_kwargs["prompt_ids"] = prompt_ids

        with self.torch.no_grad():
            if generation_config is not None:
                predicted_ids = self.model.generate(input_features, generation_config=generation_config, **gen_kwargs)
            else:
                predicted_ids = self.model.generate(input_features, **gen_kwargs)

        token_ids = predicted_ids[0].tolist()
        text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return (text or "").strip(), token_ids


class HFWhisperOnline:
    SAMPLING_RATE = 16000

    def __init__(self, asr: HFWhisperASR):
        self.asr = asr
        self.init()

    def init(self, offset: Optional[float] = None):
        self.audio_chunks: List[np.ndarray] = []
        self.offset = float(offset) if offset is not None else 0.0
        self.beg = self.offset
        self.end = self.offset
        self.is_last = False

    def insert_audio_chunk(self, audio):
        if audio is None:
            return
        arr = np.asarray(audio, dtype=np.float32)
        if arr.size == 0:
            return
        self.audio_chunks.append(arr)
        self.end += arr.size / float(self.SAMPLING_RATE)

    def _consume_audio(self) -> Optional[np.ndarray]:
        if not self.audio_chunks:
            return None
        audio = np.concatenate(self.audio_chunks) if len(self.audio_chunks) > 1 else self.audio_chunks[0]
        self.audio_chunks = []
        return audio

    def process_iter(self):
        audio = self._consume_audio()
        if audio is None:
            return {}

        text, token_ids = self.asr.transcribe_segment(audio)
        if not text:
            self.beg = self.end
            return {}

        out = {
            "start": self.beg,
            "end": self.end,
            "text": text,
            "tokens": token_ids,
        }
        self.beg = self.end
        return out

    def finish(self):
        self.is_last = True
        out = self.process_iter()
        self.is_last = False
        # Prepare for next utterance
        self.init(offset=self.end)
        return out
