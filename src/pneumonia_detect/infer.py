import logging
from pathlib import Path

import torch
from omegaconf import DictConfig
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

log = logging.getLogger(__name__)


def inference(cfg: DictConfig, image_path: Path):
    trained_model_path = Path(cfg.training.output_dir) / "final_model"

    if trained_model_path.exists():
        model_path = trained_model_path
        log.info(f"Загрузка дообученной модели из {model_path}")
    else:
        model_path = cfg.model.name
        log.warning("Дообученная модель не найдена. Используем базовую (необученную)!")

    model = ViTForImageClassification.from_pretrained(model_path)
    processor = ViTImageProcessor.from_pretrained(model_path)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)

        pred_idx = logits.argmax(-1).item()
        confidence = probs[0][pred_idx].item()

        label = model.config.id2label[pred_idx]

    result = {
        "filename": image_path.name,
        "prediction": label,
        "confidence": f"{confidence:.4f}",
    }

    print("\n---------------- RESULT ----------------")
    print(result)
    print("----------------------------------------\n")

    return result
