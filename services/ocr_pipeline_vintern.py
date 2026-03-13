import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)

# Lazy imports for model
_model = None
_processor = None

OCR_PROMPT = "Trích xuất toàn bộ văn bản trong ảnh và xuất ra định dạng markdown"


def _resolve_device() -> str:
    """Resolve the device to use for OCR (cpu or cuda)."""
    requested = os.getenv("OCR_DEVICE", "cpu").lower()
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("OCR_DEVICE=cuda but CUDA unavailable — falling back to cpu")
        return "cpu"
    return requested


def _load_model():
    """Load the Vintern 1B V3.5 model."""
    global _model, _processor

    if _model is None:
        model_name = os.getenv("OCR_MODEL_VINTERN", "VietnamOpenAI/Vintern-1B-v3.5")
        device = _resolve_device()
        logger.info("Loading Vintern 1B v3.5 model %s on %s", model_name, device)

        # Import here to avoid loading if not needed
        from transformers import AutoModelForVision2Seq, AutoProcessor

        _processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Load model - use bfloat16 for better performance
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

        _model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )

        _model = _model.eval()
        logger.info("Vintern 1B v3.5 model loaded.")

    return _model, _processor


def _run_inference(image: Image.Image, model, processor) -> str:
    """Run Vintern OCR inference on a single PIL image."""
    try:
        # Save image temporarily since model requires file path
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            image_path = tmp.name

        try:
            # Prepare inputs with prompt
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": OCR_PROMPT},
                    ]
                }
            ]

            # Apply chat template
            text = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )

            # Process inputs
            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
            )

            # Move inputs to device
            device = model.device
            inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}

            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=8192,
                    do_sample=False,
                )

            # Decode output - skip the prompt tokens
            generated_ids = [
                output_ids[len(inputs.get("input_ids", [])): ]
                for output_ids in generated_ids
            ]

            output_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            return output_text[0].strip() if output_text else ""

        finally:
            os.unlink(image_path)

    except Exception as e:
        logger.error("Vintern inference failed: %s", e)
        raise


def extract_text(file_path: str) -> list[str]:
    """Extract text from a PDF or image file using Vintern 1B v3.5.

    - PDF: each page is converted to an image and passed through the VLM.
    - PNG/JPG: treated as a single-page document.
    - Failed pages are logged and skipped; remaining pages are returned.

    Returns:
        List of per-page text strings (empty pages omitted).
    """
    path = Path(file_path)
    model, processor = _load_model()

    if path.suffix.lower() == ".pdf":
        try:
            images = convert_from_path(str(path))
        except Exception as exc:
            logger.error("pdf2image failed for %s: %s", file_path, exc)
            return []
    elif path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        try:
            images = [Image.open(str(path)).convert("RGB")]
        except Exception as exc:
            logger.error("Image open failed for %s: %s", file_path, exc)
            return []
    else:
        logger.warning("Unsupported file type for OCR: %s", path.suffix)
        return []

    results = []
    for page_num, image in enumerate(images, start=1):
        try:
            logger.info("Processing page %d of %s (image size: %s)", page_num, file_path, image.size)
            text = _run_inference(image, model, processor)
            if text:
                results.append(text)
            else:
                logger.warning("Empty text result for page %d of %s", page_num, file_path)
        except Exception as exc:
            logger.warning("OCR failed on page %d of %s: %s", page_num, file_path, exc, exc_info=True)
            continue

    return results


def reset_model():
    """Reset the loaded model (useful for switching models)."""
    global _model, _processor
    _model = None
    _processor = None
    logger.info("Vintern OCR model reset")
