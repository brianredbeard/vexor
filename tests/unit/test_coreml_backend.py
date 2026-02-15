"""Tests for CoreML embedding backend."""

from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import pytest

from vexor.providers.local import is_coreml_available


class TestIsCoremlAvailable:
    def test_returns_true_when_coreml_ep_present(self):
        with patch("vexor.providers.local.ort") as mock_ort:
            mock_ort.get_available_providers.return_value = [
                "CoreMLExecutionProvider",
                "CPUExecutionProvider",
            ]
            assert is_coreml_available() is True

    def test_returns_false_when_coreml_ep_absent(self):
        with patch("vexor.providers.local.ort") as mock_ort:
            mock_ort.get_available_providers.return_value = [
                "CPUExecutionProvider",
            ]
            assert is_coreml_available() is False

    def test_returns_false_when_ort_not_installed(self):
        with patch("vexor.providers.local.ort", None):
            assert is_coreml_available() is False


class TestCoreMLEmbeddingBackend:
    def _make_backend(self, **kwargs):
        from vexor.providers.local import CoreMLEmbeddingBackend

        defaults = {
            "model_name": "intfloat/multilingual-e5-small",
            "chunk_size": None,
            "concurrency": 1,
            "compute_units": "ALL",
        }
        defaults.update(kwargs)
        return CoreMLEmbeddingBackend(**defaults)

    @patch("vexor.providers.local._resolve_onnx_model_path")
    @patch("vexor.providers.local._resolve_tokenizer_path")
    @patch("vexor.providers.local.ort")
    def test_constructor_creates_session_with_coreml_provider(
        self, mock_ort, mock_tok_path, mock_model_path
    ):
        mock_model_path.return_value = "/fake/model.onnx"
        mock_tok_path.return_value = "/fake/tokenizer.json"

        mock_session = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.get_available_providers.return_value = [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]

        with patch("vexor.providers.local.Tokenizer") as mock_tokenizer_cls:
            mock_tokenizer_cls.from_file.return_value = MagicMock()
            backend = self._make_backend()

        # Verify CoreMLExecutionProvider was requested
        call_args = mock_ort.InferenceSession.call_args
        providers = call_args[1].get("providers") or call_args[0][1]
        assert "CoreMLExecutionProvider" in str(providers)

    @patch("vexor.providers.local._resolve_onnx_model_path")
    @patch("vexor.providers.local._resolve_tokenizer_path")
    @patch("vexor.providers.local.ort")
    def test_embed_returns_correct_shape(
        self, mock_ort, mock_tok_path, mock_model_path
    ):
        mock_model_path.return_value = "/fake/model.onnx"
        mock_tok_path.return_value = "/fake/tokenizer.json"

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        encoding = MagicMock()
        encoding.ids = list(range(10))
        encoding.attention_mask = [1] * 10
        mock_tokenizer.encode_batch.return_value = [encoding, encoding]

        # Mock ORT session: return (batch, seq_len, dim) output
        mock_session = MagicMock()
        # Simulate last_hidden_state output: shape (2, 10, 384)
        mock_output = np.random.randn(2, 10, 384).astype(np.float32)
        mock_session.run.return_value = [mock_output]
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.get_available_providers.return_value = [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]

        with patch("vexor.providers.local.Tokenizer") as mock_tokenizer_cls:
            mock_tokenizer_cls.from_file.return_value = mock_tokenizer
            backend = self._make_backend()

        result = backend.embed(["hello", "world"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 384)

    @patch("vexor.providers.local._resolve_onnx_model_path")
    @patch("vexor.providers.local._resolve_tokenizer_path")
    @patch("vexor.providers.local.ort")
    def test_embed_empty_returns_empty(
        self, mock_ort, mock_tok_path, mock_model_path
    ):
        mock_model_path.return_value = "/fake/model.onnx"
        mock_tok_path.return_value = "/fake/tokenizer.json"

        mock_ort.InferenceSession.return_value = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]

        with patch("vexor.providers.local.Tokenizer") as mock_tokenizer_cls:
            mock_tokenizer_cls.from_file.return_value = MagicMock()
            backend = self._make_backend()

        result = backend.embed([])
        assert result.shape == (0, 0)

    @patch("vexor.providers.local._resolve_onnx_model_path")
    @patch("vexor.providers.local._resolve_tokenizer_path")
    @patch("vexor.providers.local.ort")
    def test_embed_normalizes_vectors(
        self, mock_ort, mock_tok_path, mock_model_path
    ):
        mock_model_path.return_value = "/fake/model.onnx"
        mock_tok_path.return_value = "/fake/tokenizer.json"

        mock_tokenizer = MagicMock()
        encoding = MagicMock()
        encoding.ids = list(range(5))
        encoding.attention_mask = [1] * 5
        mock_tokenizer.encode_batch.return_value = [encoding]

        mock_session = MagicMock()
        mock_output = np.array([[[3.0, 4.0, 0.0] * 128]], dtype=np.float32)  # (1, 5, 384)
        mock_output = np.broadcast_to(
            np.array([[[3.0, 4.0, 0.0]]], dtype=np.float32),
            (1, 5, 3),
        ).copy()
        # Make it (1, 5, 384) by repeating
        mock_output = np.tile(mock_output, (1, 1, 128))
        mock_session.run.return_value = [mock_output]
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.get_available_providers.return_value = [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]

        with patch("vexor.providers.local.Tokenizer") as mock_tokenizer_cls:
            mock_tokenizer_cls.from_file.return_value = mock_tokenizer
            backend = self._make_backend()

        result = backend.embed(["test"])
        # Vectors should be L2-normalized (unit length)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_raises_when_ort_not_installed(self):
        with patch("vexor.providers.local.ort", None):
            with pytest.raises(RuntimeError, match="onnxruntime"):
                self._make_backend()
