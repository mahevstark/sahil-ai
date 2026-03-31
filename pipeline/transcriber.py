from pathlib import Path

from faster_whisper import WhisperModel


def load_model(model_size: str = "base") -> WhisperModel:
    import ctranslate2
    try:
        cuda_types = ctranslate2.get_supported_compute_types("cuda")
        device = "cuda" if cuda_types else "cpu"
    except Exception:
        device = "cpu"
    compute_type = "int8_float16" if device == "cuda" else "int8"
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe(model: WhisperModel, wav_path: Path) -> list:
    segments, _info = model.transcribe(str(wav_path), beam_size=5)
    return [
        {"text": seg.text, "start": seg.start, "end": seg.end}
        for seg in segments
    ]


def chunk_segments(segments: list, max_chars: int = 1500) -> list:
    chunks = []
    current_texts = []
    current_start = None
    current_end = None
    accumulated = 0
    chunk_index = 0

    for seg in segments:
        text = seg["text"]
        start = seg["start"]
        end = seg["end"]

        if current_start is None:
            current_start = start

        if accumulated + len(text) > max_chars and current_texts:
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "text": "".join(current_texts).strip(),
                    "start_sec": current_start,
                    "end_sec": current_end,
                }
            )
            chunk_index += 1
            current_texts = [text]
            current_start = start
            current_end = end
            accumulated = len(text)
        else:
            current_texts.append(text)
            current_end = end
            accumulated += len(text)

    if current_texts:
        chunks.append(
            {
                "chunk_index": chunk_index,
                "text": "".join(current_texts).strip(),
                "start_sec": current_start,
                "end_sec": current_end,
            }
        )

    return chunks
