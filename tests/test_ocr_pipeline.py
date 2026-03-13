import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


def _make_mock_vlm():
    model = MagicMock()
    tokenizer = MagicMock()
    model.chat = MagicMock(return_value="Extracted text from page")
    return model, tokenizer


def test_extract_text_returns_list_of_strings(tmp_path):
    fake_pdf = tmp_path / "test.pdf"
    fake_pdf.write_bytes(b"%PDF fake")

    fake_image = MagicMock()
    mock_model, mock_tokenizer = _make_mock_vlm()

    with patch("services.ocr_pipeline.convert_from_path", return_value=[fake_image, fake_image]), \
         patch("services.ocr_pipeline._load_model", return_value=(mock_model, mock_tokenizer)), \
         patch("services.ocr_pipeline._run_inference", return_value="Page text"):
        from services import ocr_pipeline
        ocr_pipeline._model = None
        ocr_pipeline._processor = None
        result = ocr_pipeline.extract_text(str(fake_pdf))
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(t, str) for t in result)


def test_extract_text_single_image(tmp_path):
    fake_img = tmp_path / "scan.png"
    fake_img.write_bytes(b"PNG fake")

    fake_pil = MagicMock()
    mock_model, mock_tokenizer = _make_mock_vlm()

    with patch("services.ocr_pipeline.Image.open", return_value=fake_pil), \
         patch("services.ocr_pipeline._load_model", return_value=(mock_model, mock_tokenizer)), \
         patch("services.ocr_pipeline._run_inference", return_value="Image text"):
        from services import ocr_pipeline
        ocr_pipeline._model = None
        ocr_pipeline._processor = None
        result = ocr_pipeline.extract_text(str(fake_img))
        assert len(result) == 1
        assert result[0] == "Image text"


def test_extract_text_page_failure_skips_page(tmp_path):
    fake_pdf = tmp_path / "test.pdf"
    fake_pdf.write_bytes(b"%PDF fake")

    fake_images = [MagicMock(), MagicMock()]
    mock_model, mock_tokenizer = _make_mock_vlm()

    call_count = {"n": 0}
    def side_effect(image, model, tokenizer):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("OCR failed on page 1")
        return "Page 2 text"

    with patch("services.ocr_pipeline.convert_from_path", return_value=fake_images), \
         patch("services.ocr_pipeline._load_model", return_value=(mock_model, mock_tokenizer)), \
         patch("services.ocr_pipeline._run_inference", side_effect=side_effect):
        from services import ocr_pipeline
        ocr_pipeline._model = None
        ocr_pipeline._processor = None
        result = ocr_pipeline.extract_text(str(fake_pdf))
        assert len(result) == 1
        assert result[0] == "Page 2 text"


def test_cuda_fallback_to_cpu_when_unavailable():
    import os
    with patch.dict(os.environ, {"OCR_DEVICE": "cuda"}), \
         patch("services.ocr_pipeline.torch.cuda.is_available", return_value=False):
        from services import ocr_pipeline
        device = ocr_pipeline._resolve_device()
        assert device == "cpu"


def test_cuda_used_when_available():
    import os
    with patch.dict(os.environ, {"OCR_DEVICE": "cuda"}), \
         patch("services.ocr_pipeline.torch.cuda.is_available", return_value=True):
        from services import ocr_pipeline
        device = ocr_pipeline._resolve_device()
        assert device == "cuda"