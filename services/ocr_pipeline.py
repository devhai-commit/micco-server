import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from pdf2image import convert_from_path
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Model selection: "firered" or "vintern"
OCR_ENGINE = os.getenv("OCR_ENGINE", "firered").lower()

# Lazy imports
_model = None
_tokenizer = None

# For Vintern image preprocessing
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

OCR_PROMPT = "Trích xuất toàn bộ văn bản trong ảnh và xuất ra định dạng markdown"


def _resolve_device() -> str:
    """Resolve the device to use for OCR (cpu or cuda)."""
    requested = os.getenv("OCR_DEVICE", "cpu").lower()
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("OCR_DEVICE=cuda but CUDA unavailable — falling back to cpu")
        return "cpu"
    return requested


# === Vintern Image Preprocessing Functions ===
def build_transform(input_size=448):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []

    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def load_image_for_vintern(image: Image.Image, input_size=448, max_num=6):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# === Model Loading ===
def _load_model():
    """Load the selected OCR model based on OCR_ENGINE setting."""
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    device = _resolve_device()

    if OCR_ENGINE == "vintern":
        model_name = os.getenv("OCR_MODEL_VINTERN", "5CD-AI/Vintern-1B-v3_5")
        logger.info("Loading Vintern model %s on %s", model_name, device)

        from transformers import AutoModel, AutoConfig

        _tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False
        )

        # Load config and fix meta tensor issue before loading model
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # Fix vision config - convert any tensors to plain values
        if hasattr(config, 'vision_config'):
            vision_cfg = config.vision_config
            for key in ['drop_path_rate', 'num_hidden_layers']:
                if hasattr(vision_cfg, key):
                    val = getattr(vision_cfg, key)
                    if hasattr(val, 'item'):
                        setattr(vision_cfg, key, val.item())
                    elif hasattr(val, 'cpu'):
                        # It's a tensor, need to convert
                        setattr(vision_cfg, key, val.cpu().tolist() if hasattr(val, 'tolist') else float(val))

        # Load model without low_cpu_mem_usage to avoid meta tensor issue
        # Use float32 to avoid bfloat16 issues
        _model = AutoModel.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            use_flash_attn=False,
        )

        if device == "cuda":
            _model = _model.to(torch.bfloat16).eval().cuda()
        else:
            _model = _model.eval()

        logger.info("Vintern model loaded.")

    else:  # Default to firered
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model_name = os.getenv("OCR_MODEL", "FireRedTeam/FireRed-OCR")
        logger.info("Loading FireRed-OCR model %s on %s", model_name, device)

        _tokenizer = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

        _model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )

        _model = _model.eval()
        logger.info("FireRed-OCR model loaded.")

    return _model, _tokenizer


def _run_inference(image: Image.Image, model, processor) -> str:
    """Run OCR inference on a single PIL image."""
    try:
        if OCR_ENGINE == "vintern":
            # Vintern uses custom preprocessing and chat method
            pixel_values = load_image_for_vintern(image, max_num=6)

            device = next(model.parameters()).device
            pixel_values = pixel_values.to(device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)

            question = f'<image>\n{OCR_PROMPT}'

            generation_config = dict(
                max_new_tokens=8192,
                do_sample=False,
                num_beams=3,
                repetition_penalty=2.5
            )

            result = model.chat(
                processor,
                pixel_values,
                question,
                generation_config,
                history=None,
                return_history=False
            )

            # Handle different return formats
            if isinstance(result, tuple):
                response = result[0]
            else:
                response = result

            return response.strip() if response else ""

        else:  # FireRed-OCR (Qwen-based)
            device = model.device

            prompt_text = f"<|image_pad|>{OCR_PROMPT}"

            inputs = processor(
                text=prompt_text,
                images=image,
                return_tensors="pt",
                padding=True,
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            if "pixel_values" not in inputs or inputs["pixel_values"] is None:
                raise ValueError("Processor did not return pixel_values - image may be invalid")

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=8192,
                    do_sample=False,
                )

            output_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            return output_text[0].strip() if output_text else ""

    except Exception as e:
        logger.error("Inference failed: %s", e)
        raise


def extract_text(file_path: str) -> Tuple[list[str], str]:
    """Extract text from a PDF or image file using the selected OCR engine.

    - PDF: each page is converted to an image and passed through the VLM.
    - PNG/JPG: treated as a single-page document.
    - Failed pages are logged and skipped; remaining pages are returned.
    - Results are saved as a .md file in the same directory.

    Returns:
        Tuple of:
            - List of per-page text strings (empty pages omitted)
            - Path to the saved .md file
    """
    path = Path(file_path)
    model, processor = _load_model()

    engine_name = "Vintern" if OCR_ENGINE == "vintern" else "FireRed-OCR"

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

    logger.info("Using OCR engine: %s", engine_name)

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

    # Save results as .md file
    md_path = _save_as_markdown(results, file_path)

    return results, md_path


def _save_as_markdown(pages: list[str], original_path: str) -> str:
    """Save OCR results as a markdown file.

    Args:
        pages: List of text from each page
        original_path: Path to the original PDF/image file

    Returns:
        Path to the saved .md file
    """
    if not pages:
        return ""

    # Create markdown content
    md_content = f"# {Path(original_path).stem}\n\n"
    for i, page_text in enumerate(pages, 1):
        md_content += f"## Page {i}\n\n{page_text}\n\n"

    # Determine output path (same directory as original, with .md extension)
    original = Path(original_path)
    md_path = original.with_suffix(".md")

    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info("Saved OCR results to: %s", md_path)
    except Exception as exc:
        logger.error("Failed to save markdown file: %s", exc)
        return ""

    return str(md_path)


def reset_model():
    """Reset the loaded model (useful for switching models)."""
    global _model, _tokenizer
    _model = None
    _tokenizer = None
    logger.info("OCR model reset")


def get_engine() -> str:
    """Get the current OCR engine name."""
    return OCR_ENGINE
