"""Local embedding backend for Vexor."""

from __future__ import annotations

import os
import platform
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from ..config import local_model_dir
from ..text import Messages

# Lazy-loaded mlx_embedding_models module (None if not installed)
try:
    import mlx_embedding_models
    import mlx_embedding_models.embedding as mlx_embedding
except ImportError:
    mlx_embedding_models = None  # type: ignore[assignment]
    mlx_embedding = None  # type: ignore[assignment]


@contextmanager
def _suppress_mlx_stderr():
    """Suppress C++-level stderr output from MLX libraries."""
    old_fd = os.dup(2)
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        _ = os.dup2(devnull, 2)
        os.close(devnull)
        yield
    finally:
        _ = os.dup2(old_fd, 2)
        os.close(old_fd)


def _load_fastembed():
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise RuntimeError(Messages.ERROR_LOCAL_DEP_MISSING) from exc
    return TextEmbedding


def resolve_fastembed_cache_dir(*, create: bool = True) -> Path:
    """Return the fixed cache directory used for local models."""
    cache_dir = local_model_dir()
    if create:
        cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


_CUSTOM_TEXT_MODELS: dict[str, dict[str, object]] = {
    "intfloat/multilingual-e5-small": {
        "model": "intfloat/multilingual-e5-small",
        "pooling": "MEAN",
        "normalization": True,
        "hf": "intfloat/multilingual-e5-small",
        "dim": 384,
        "model_file": "onnx/model.onnx",
        "description": "Multilingual E5 model for cross-lingual retrieval",
        "license": "MIT",
        "size_in_gb": 0.12,
    },
}


def _is_unsupported_model_error(exc: Exception) -> bool:
    return isinstance(exc, ValueError) and "not supported in TextEmbedding" in str(exc)


def _register_custom_model(text_embedding_cls, model_name: str) -> bool:
    spec = _CUSTOM_TEXT_MODELS.get(model_name.strip().lower())
    if not spec:
        return False
    try:
        from fastembed.common.model_description import ModelSource, PoolingType
    except Exception as exc:
        raise RuntimeError(
            Messages.ERROR_LOCAL_MODEL_LOAD.format(model=model_name, reason=str(exc))
        ) from exc
    try:
        text_embedding_cls.add_custom_model(
            model=spec["model"],
            pooling=getattr(PoolingType, str(spec["pooling"])),
            normalization=bool(spec["normalization"]),
            sources=ModelSource(hf=str(spec["hf"])),
            dim=int(spec["dim"]),
            model_file=str(spec["model_file"]),
            description=str(spec["description"]),
            license=str(spec["license"]),
            size_in_gb=float(spec["size_in_gb"]),
        )
    except ValueError as exc:
        if "already registered" not in str(exc).lower():
            raise
    return True


def is_mlx_available() -> bool:
    """Check if MLX is available (requires macOS on Apple Silicon)."""
    if mlx_embedding_models is None:
        return False
    try:
        return platform.system() == "Darwin" and platform.machine() == "arm64"
    except Exception:
        return False


class LocalEmbeddingBackend:
    """Embedding backend that runs a lightweight local model via fastembed."""

    def __init__(
        self,
        *,
        model_name: str,
        chunk_size: int | None = None,
        concurrency: int = 1,
        cuda: bool = False,
    ) -> None:
        self.model_name = model_name
        self.chunk_size = chunk_size if chunk_size and chunk_size > 0 else None
        self.concurrency = max(int(concurrency or 1), 1)
        self.cuda = bool(cuda)
        TextEmbedding = _load_fastembed()
        cache_dir = resolve_fastembed_cache_dir()
        try:
            self._model = TextEmbedding(
                model_name=model_name,
                cache_dir=str(cache_dir),
                cuda=self.cuda,
            )
        except Exception as exc:
            if _is_unsupported_model_error(exc) and _register_custom_model(
                TextEmbedding, model_name
            ):
                try:
                    self._model = TextEmbedding(
                        model_name=model_name,
                        cache_dir=str(cache_dir),
                        cuda=self.cuda,
                    )
                except Exception as retry_exc:
                    raise RuntimeError(
                        Messages.ERROR_LOCAL_MODEL_LOAD.format(
                            model=model_name, reason=str(retry_exc)
                        )
                    ) from retry_exc
            else:
                raise RuntimeError(
                    Messages.ERROR_LOCAL_MODEL_LOAD.format(
                        model=model_name, reason=str(exc)
                    )
                ) from exc

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors: list[np.ndarray] = []
        for chunk in _chunk(texts, self.chunk_size):
            try:
                for embedding in self._model.embed(list(chunk)):
                    vectors.append(np.asarray(embedding, dtype=np.float32))
            except Exception as exc:
                raise RuntimeError(
                    Messages.ERROR_LOCAL_MODEL_EMBED.format(reason=str(exc))
                ) from exc
        if not vectors:
            raise RuntimeError(Messages.ERROR_NO_EMBEDDINGS)
        return np.vstack(vectors)


class MLXEmbeddingBackend:
    """Embedding backend using MLX for native Apple Silicon GPU acceleration."""

    def __init__(
        self,
        *,
        model_name: str,
        chunk_size: int | None = None,
        concurrency: int = 1,
    ) -> None:
        if mlx_embedding_models is None:
            raise RuntimeError(
                "mlx-embedding-models is required for MLX backend. "
                "Install with: uv pip install 'vexor[local-mlx]'"
            )

        self.model_name = model_name
        self.chunk_size = chunk_size if chunk_size and chunk_size > 0 else None
        self.concurrency = max(int(concurrency or 1), 1)

        # Try from_registry first (strips org prefix if present)
        # Example: "intfloat/multilingual-e5-small" → "multilingual-e5-small"
        short_name = model_name.split("/")[-1] if "/" in model_name else model_name
        with _suppress_mlx_stderr():
            try:
                self._model = mlx_embedding_models.EmbeddingModel.from_registry(
                    short_name
                )
            except Exception:
                # Fall back to from_pretrained for arbitrary HuggingFace models
                try:
                    self._model = mlx_embedding_models.EmbeddingModel.from_pretrained(
                        model_name
                    )
                except Exception as exc:
                    raise RuntimeError(
                        Messages.ERROR_LOCAL_MODEL_LOAD.format(
                            model=model_name, reason=str(exc)
                        )
                    ) from exc

        # Extend SEQ_LENS for long-context models — mlx-embedding-models' internal
        # SEQ_LENS bucketing only covers up to 512 by default. Long-context models
        # (bge-m3=8192, nomic-v1.5=2048) crash with IndexError without extension.
        # See: https://github.com/taylorai/mlx_embedding_models/issues/7
        if mlx_embedding is not None and self._model.max_length > 512:
            # Defensive check: ensure SEQ_LENS is mutable (not stripped by -O)
            if not isinstance(mlx_embedding.SEQ_LENS, list):
                raise TypeError(
                    "mlx_embedding_models.embedding.SEQ_LENS must be a list for mutation"
                )

            current_max = max(mlx_embedding.SEQ_LENS)
            model_max = self._model.max_length

            # Only extend if needed (idempotent)
            # The -1 accounts for the need for a sentinel value above max_length
            if model_max > current_max - 1:
                # Extension strategy:
                # - 32-token steps from 544 to 1024 (matches 128-512 pattern)
                # - 512-token steps from 1024 to max_length
                # - Sentinel at max_length + 32 (for _construct_batch strict > comparison)
                new_buckets = []

                # Add 32-token steps in 512-1024 transition zone
                for bucket in range(544, 1024 + 1, 32):
                    if bucket > current_max:
                        new_buckets.append(bucket)

                # Add 512-token steps from 1024 to max_length
                for bucket in range(1024, model_max + 1, 512):
                    if bucket > current_max and bucket not in new_buckets:
                        new_buckets.append(bucket)

                # Add sentinel value above max_length
                sentinel = model_max + 32
                if sentinel > current_max:
                    new_buckets.append(sentinel)

                # Extend SEQ_LENS with new buckets
                if new_buckets:
                    mlx_embedding.SEQ_LENS.extend(sorted(new_buckets))

        # CRITICAL: Apply monkey-patch for transformers>=5.0 compatibility
        # transformers>=5.0 removed batch_encode_plus from TokenizersBackend,
        # but mlx-embedding-models==0.0.11 still calls it
        if not hasattr(self._model.tokenizer, "batch_encode_plus"):
            self._model.tokenizer.batch_encode_plus = self._model.tokenizer.__call__

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        # Guard empty input - mlx-embedding-models crashes with IndexError
        if not texts:
            # Return (0, dim) array - we don't know dim yet, so use 384 as default
            # (multilingual-e5-small dimension)
            return np.empty((0, 384), dtype=np.float32)

        try:
            # mlx-embedding-models handles batching, sorting by length, and L2 normalization internally
            with _suppress_mlx_stderr():
                result = self._model.encode(
                    list(texts),
                    batch_size=self.chunk_size or 64,
                    show_progress=False,
                )
            return result  # Already numpy float32, L2-normalized
        except Exception as exc:
            raise RuntimeError(
                Messages.ERROR_LOCAL_MODEL_EMBED.format(reason=str(exc))
            ) from exc


def _chunk(items: Sequence[str], size: int | None) -> Iterator[Sequence[str]]:
    if size is None or size <= 0:
        yield items
        return
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]
