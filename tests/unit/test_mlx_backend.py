"""Tests for MLX embedding backend."""

import os
import sys
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from vexor.providers.local import is_mlx_available, _suppress_mlx_stderr


class TestSuppressMLXStderr:
    def test_suppresses_stderr_output(self):
        """_suppress_mlx_stderr suppresses writes to stderr fd 2."""
        # Write to stderr before suppression - should be visible
        original_stderr_fd = sys.stderr.fileno()

        with _suppress_mlx_stderr():
            # During suppression, writes to fd 2 should be suppressed
            # We can't easily verify suppression without actually writing to fd 2,
            # but we can verify the fd was redirected
            assert (
                os.fstat(2).st_dev != os.fstat(original_stderr_fd).st_dev
                or os.fstat(2).st_ino != os.fstat(original_stderr_fd).st_ino
            )

        # After context exit, stderr should be restored
        assert os.fstat(2).st_dev == os.fstat(original_stderr_fd).st_dev
        assert os.fstat(2).st_ino == os.fstat(original_stderr_fd).st_ino

    def test_restores_stderr_after_exception(self):
        """_suppress_mlx_stderr restores stderr even if exception occurs."""
        original_stat = os.fstat(2)

        with pytest.raises(ValueError):
            with _suppress_mlx_stderr():
                raise ValueError("test exception")

        # After exception, stderr should still be restored
        restored_stat = os.fstat(2)
        assert restored_stat.st_dev == original_stat.st_dev
        assert restored_stat.st_ino == original_stat.st_ino


class TestIsMlxAvailable:
    def test_returns_true_when_on_apple_silicon(self):
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
            patch("vexor.providers.local.mlx_embedding_models") as mock_mlx_models,
        ):
            mock_mlx_models.__version__ = "0.0.11"
            assert is_mlx_available() is True

    def test_returns_false_when_not_darwin(self):
        with (
            patch("platform.system", return_value="Linux"),
            patch("platform.machine", return_value="x86_64"),
        ):
            assert is_mlx_available() is False

    def test_returns_false_when_not_arm64(self):
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="x86_64"),
        ):
            assert is_mlx_available() is False

    def test_returns_false_when_mlx_models_not_installed(self):
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
            patch("vexor.providers.local.mlx_embedding_models", None),
        ):
            assert is_mlx_available() is False


class TestMLXEmbeddingBackend:
    @pytest.fixture(autouse=True)
    def _preserve_seq_lens(self):
        """Save and restore SEQ_LENS to prevent pollution across tests."""
        import mlx_embedding_models.embedding as embedding_module

        original = embedding_module.SEQ_LENS.copy()
        yield embedding_module
        embedding_module.SEQ_LENS[:] = original

    def _make_backend(self, **kwargs):
        from vexor.providers.local import MLXEmbeddingBackend

        defaults = {
            "model_name": "multilingual-e5-small",
            "chunk_size": None,
            "concurrency": 1,
        }
        defaults.update(kwargs)
        return MLXEmbeddingBackend(**defaults)

    @patch("vexor.providers.local.mlx_embedding_models")
    def test_constructor_loads_from_registry_short_name(self, mock_mlx_models):
        """MLXEmbeddingBackend tries from_registry with short names first."""
        mock_model = MagicMock()
        mock_model.tokenizer = MagicMock()
        mock_model.max_length = 512
        mock_mlx_models.EmbeddingModel.from_registry.return_value = mock_model

        self._make_backend(model_name="multilingual-e5-small")

        # Verify from_registry was called with the short name
        mock_mlx_models.EmbeddingModel.from_registry.assert_called_once_with(
            "multilingual-e5-small"
        )
        # Verify monkey-patch was applied
        assert hasattr(mock_model.tokenizer, "batch_encode_plus")

    @patch("vexor.providers.local.mlx_embedding_models")
    def test_constructor_strips_org_prefix_for_registry(self, mock_mlx_models):
        """MLXEmbeddingBackend strips org prefix before trying from_registry."""
        mock_model = MagicMock()
        mock_model.tokenizer = MagicMock()
        mock_model.max_length = 512
        mock_mlx_models.EmbeddingModel.from_registry.return_value = mock_model

        self._make_backend(model_name="intfloat/multilingual-e5-small")

        # Verify from_registry was called with org prefix stripped
        mock_mlx_models.EmbeddingModel.from_registry.assert_called_once_with(
            "multilingual-e5-small"
        )

    @patch("vexor.providers.local.mlx_embedding_models")
    def test_constructor_falls_back_to_from_pretrained(self, mock_mlx_models):
        """MLXEmbeddingBackend falls back to from_pretrained if registry fails."""
        mock_model = MagicMock()
        mock_model.tokenizer = MagicMock()
        mock_model.max_length = 512
        # from_registry raises KeyError (model not in registry)
        mock_mlx_models.EmbeddingModel.from_registry.side_effect = KeyError(
            "not in registry"
        )
        mock_mlx_models.EmbeddingModel.from_pretrained.return_value = mock_model

        self._make_backend(model_name="some/custom-model")

        # Verify fallback to from_pretrained with full name
        mock_mlx_models.EmbeddingModel.from_pretrained.assert_called_once_with(
            "some/custom-model"
        )

    @patch("vexor.providers.local.mlx_embedding_models")
    def test_embed_returns_correct_shape(self, mock_mlx_models):
        """MLXEmbeddingBackend.embed returns numpy array with correct shape."""
        mock_model = MagicMock()
        mock_model.tokenizer = MagicMock()
        mock_model.max_length = 512
        # Simulate encode() returning (N, 384) float32 array
        mock_embeddings = np.random.randn(3, 384).astype(np.float32)
        mock_model.encode.return_value = mock_embeddings
        mock_mlx_models.EmbeddingModel.from_registry.return_value = mock_model

        backend = self._make_backend()
        result = backend.embed(["hello", "world", "test"])

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 384)
        assert result.dtype == np.float32

    @patch("vexor.providers.local.mlx_embedding_models")
    def test_embed_empty_returns_empty_array(self, mock_mlx_models):
        """MLXEmbeddingBackend.embed guards empty input and returns empty array."""
        mock_model = MagicMock()
        mock_model.tokenizer = MagicMock()
        mock_model.max_length = 512
        mock_mlx_models.EmbeddingModel.from_registry.return_value = mock_model

        backend = self._make_backend()
        result = backend.embed([])

        # Should return (0, dim) array, not crash
        assert result.shape[0] == 0
        assert result.dtype == np.float32
        # encode() should NOT have been called (guarded before call)
        mock_model.encode.assert_not_called()

    @patch("vexor.providers.local.mlx_embedding_models")
    def test_embed_passes_batch_size_to_encode(self, mock_mlx_models):
        """MLXEmbeddingBackend.embed passes chunk_size as batch_size."""
        mock_model = MagicMock()
        mock_model.tokenizer = MagicMock()
        mock_model.max_length = 512
        mock_embeddings = np.random.randn(2, 384).astype(np.float32)
        mock_model.encode.return_value = mock_embeddings
        mock_mlx_models.EmbeddingModel.from_registry.return_value = mock_model

        backend = self._make_backend(chunk_size=128)
        backend.embed(["test1", "test2"])

        # Verify batch_size=chunk_size was passed
        mock_model.encode.assert_called_once()
        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs["batch_size"] == 128
        assert call_kwargs["show_progress"] is False

    @patch("vexor.providers.local.mlx_embedding_models")
    def test_embed_uses_default_batch_size_when_chunk_size_none(self, mock_mlx_models):
        """MLXEmbeddingBackend.embed uses default batch_size=64 when chunk_size is None."""
        mock_model = MagicMock()
        mock_model.tokenizer = MagicMock()
        mock_model.max_length = 512
        mock_embeddings = np.random.randn(1, 384).astype(np.float32)
        mock_model.encode.return_value = mock_embeddings
        mock_mlx_models.EmbeddingModel.from_registry.return_value = mock_model

        backend = self._make_backend(chunk_size=None)
        backend.embed(["test"])

        # Verify default batch_size=64
        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs["batch_size"] == 64

    def test_raises_when_mlx_models_not_installed(self):
        """MLXEmbeddingBackend raises RuntimeError when mlx_embedding_models not installed."""
        with patch("vexor.providers.local.mlx_embedding_models", None):
            with pytest.raises(RuntimeError, match="mlx-embedding-models"):
                self._make_backend()

    @patch("vexor.providers.local._suppress_mlx_stderr")
    @patch("vexor.providers.local.mlx_embedding_models")
    def test_init_suppresses_mlx_stderr_during_model_load(
        self, mock_mlx_models, mock_suppress
    ):
        """MLXEmbeddingBackend.__init__ uses _suppress_mlx_stderr during model loading."""
        mock_model = MagicMock()
        mock_model.tokenizer = MagicMock()
        mock_model.max_length = 512
        mock_mlx_models.EmbeddingModel.from_registry.return_value = mock_model
        # Make the context manager work as a no-op for the test
        mock_suppress.return_value.__enter__ = MagicMock()
        mock_suppress.return_value.__exit__ = MagicMock()

        self._make_backend()

        # Verify _suppress_mlx_stderr was called during init
        mock_suppress.assert_called_once()

    @patch("vexor.providers.local._suppress_mlx_stderr")
    @patch("vexor.providers.local.mlx_embedding_models")
    def test_embed_suppresses_mlx_stderr_during_encode(
        self, mock_mlx_models, mock_suppress
    ):
        """MLXEmbeddingBackend.embed uses _suppress_mlx_stderr during encode."""
        mock_model = MagicMock()
        mock_model.tokenizer = MagicMock()
        mock_model.max_length = 512
        mock_embeddings = np.random.randn(2, 384).astype(np.float32)
        mock_model.encode.return_value = mock_embeddings
        mock_mlx_models.EmbeddingModel.from_registry.return_value = mock_model
        # Make the context manager work as a no-op for the test
        mock_suppress.return_value.__enter__ = MagicMock()
        mock_suppress.return_value.__exit__ = MagicMock()

        backend = self._make_backend()
        # Reset the mock after __init__ call
        mock_suppress.reset_mock()

        backend.embed(["test1", "test2"])

        # Verify _suppress_mlx_stderr was called during embed
        mock_suppress.assert_called_once()

    @patch("vexor.providers.local.mlx_embedding_models")
    def test_constructor_propagates_error_through_stderr_suppression(
        self, mock_mlx_models
    ):
        """Errors during model loading propagate even with stderr suppression active."""
        mock_mlx_models.EmbeddingModel.from_registry.side_effect = KeyError("not found")
        mock_mlx_models.EmbeddingModel.from_pretrained.side_effect = Exception(
            "load failed"
        )

        with pytest.raises(RuntimeError, match="load failed"):
            self._make_backend(model_name="broken-model")

    @patch("vexor.providers.local.mlx_embedding_models")
    def test_constructor_extends_seq_lens_for_long_context_model(self, mock_mlx_models):
        """MLXEmbeddingBackend extends SEQ_LENS for models with max_length > 512."""
        import mlx_embedding_models.embedding as embedding_module

        mock_model = MagicMock()
        mock_model.tokenizer = MagicMock()
        mock_model.max_length = 8192  # bge-m3 default
        mock_mlx_models.EmbeddingModel.from_registry.return_value = mock_model

        self._make_backend()

        assert max(embedding_module.SEQ_LENS) > 8192, (
            "SEQ_LENS should include sentinel above max_length"
        )
        assert 8192 in embedding_module.SEQ_LENS, (
            "SEQ_LENS should include max_length value"
        )
        assert max(embedding_module.SEQ_LENS) >= 8192 + 32, (
            "Sentinel should be at least max_length + 32"
        )

    @patch("vexor.providers.local.mlx_embedding_models")
    def test_constructor_preserves_model_max_length(self, mock_mlx_models):
        """MLXEmbeddingBackend preserves model max_length (does not cap it)."""
        mock_model = MagicMock()
        mock_model.tokenizer = MagicMock()
        mock_model.max_length = 8192  # bge-m3 default
        mock_mlx_models.EmbeddingModel.from_registry.return_value = mock_model

        self._make_backend()

        assert mock_model.max_length == 8192, (
            "Model max_length should be preserved, not capped"
        )

    @patch("vexor.providers.local.mlx_embedding_models")
    def test_constructor_does_not_modify_seq_lens_for_short_context_model(
        self, mock_mlx_models
    ):
        """MLXEmbeddingBackend does not modify SEQ_LENS for models with max_length <= 512."""
        import mlx_embedding_models.embedding as embedding_module

        mock_model = MagicMock()
        mock_model.tokenizer = MagicMock()
        mock_model.max_length = 512  # Standard model
        mock_mlx_models.EmbeddingModel.from_registry.return_value = mock_model

        original_len = len(embedding_module.SEQ_LENS)
        self._make_backend()

        assert len(embedding_module.SEQ_LENS) == original_len, (
            "SEQ_LENS should not be modified for models with max_length <= 512"
        )

    @patch("vexor.providers.local.mlx_embedding_models")
    def test_seq_lens_sentinel_covers_construct_batch_boundary(self, mock_mlx_models):
        """SEQ_LENS sentinel value is strictly greater than max_length for _construct_batch safety."""
        import mlx_embedding_models.embedding as embedding_module

        mock_model = MagicMock()
        mock_model.tokenizer = MagicMock()
        mock_model.max_length = 2048  # nomic-text-v1.5 default
        mock_mlx_models.EmbeddingModel.from_registry.return_value = mock_model

        self._make_backend()

        max_seq_len = max(embedding_module.SEQ_LENS)
        assert max_seq_len > 2048, (
            f"Sentinel {max_seq_len} must be > max_length {2048} for _construct_batch safety"
        )
        assert max_seq_len >= 2048 + 32, (
            f"Sentinel {max_seq_len} should be at least max_length + 32 = {2048 + 32}"
        )

    @patch("vexor.providers.local.mlx_embedding_models")
    def test_seq_lens_extension_is_idempotent(self, mock_mlx_models):
        """SEQ_LENS extension is idempotent — creating two backends doesn't double-extend."""
        import mlx_embedding_models.embedding as embedding_module

        mock_model = MagicMock()
        mock_model.tokenizer = MagicMock()
        mock_model.max_length = 8192
        mock_mlx_models.EmbeddingModel.from_registry.return_value = mock_model

        self._make_backend()
        len_after_first = len(embedding_module.SEQ_LENS)

        self._make_backend()
        len_after_second = len(embedding_module.SEQ_LENS)

        assert len_after_second == len_after_first, (
            f"SEQ_LENS grew from {len_after_first} to {len_after_second} on second instantiation"
        )
