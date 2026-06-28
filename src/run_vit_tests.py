from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor

from utils import IMAGE_URLS, RESULTS_CSV, RESULTS_JSON, download_image, ensure_directories, get_device, save_json, setup_logger


MODELS = {
    "ViT Base": "google/vit-base-patch16-224",
    "ViT Large": "google/vit-large-patch16-224",
}


def serialize_top5(top5: List[Dict[str, Any]]) -> str:
    return json.dumps(top5, ensure_ascii=False)


def load_model(model_label: str, model_name: str, device: torch.device, logger) -> Dict[str, Any]:
    logger.info("Carregando %s (%s)...", model_label, model_name)
    start = time.time()
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    end = time.time()

    load_time = end - start
    logger.info(
        "Modelo %s carregado em %.4f s no dispositivo %s",
        model_name,
        load_time,
        device,
    )
    return {
        "label": model_label,
        "name": model_name,
        "processor": processor,
        "model": model,
        "load_time_seconds": load_time,
        "status": "ok",
        "error": "",
    }


def infer_image(model_entry: Dict[str, Any], image, device: torch.device) -> Dict[str, Any]:
    processor = model_entry["processor"]
    model = model_entry["model"]

    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    end = time.time()

    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]
    confidence = probabilities[0][predicted_class_idx].item() * 100.0

    top5: List[Dict[str, Any]] = []
    for prob, catid in zip(top5_prob[0], top5_catid[0]):
        top5.append(
            {
                "classe": model.config.id2label[catid.item()],
                "confianca_percentual": prob.item() * 100.0,
            }
        )

    return {
        "predicted_class": predicted_class,
        "top1_confidence_percent": confidence,
        "inference_time_seconds": end - start,
        "top5": top5,
    }


def main() -> int:
    ensure_directories()
    logger = setup_logger("run_vit_tests")
    device = get_device()
    logger.info("Dispositivo detectado: %s", device)

    model_entries: Dict[str, Dict[str, Any]] = {}
    results: List[Dict[str, Any]] = []

    for model_label, model_name in MODELS.items():
        try:
            model_entries[model_label] = load_model(model_label, model_name, device, logger)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Falha ao carregar %s: %s", model_name, exc)
            model_entries[model_label] = {
                "label": model_label,
                "name": model_name,
                "processor": None,
                "model": None,
                "load_time_seconds": None,
                "status": "error",
                "error": str(exc),
            }

    for image_name, url in tqdm(IMAGE_URLS.items(), desc="Processando imagens"):
        logger.info("Baixando imagem %s (%s)", image_name, url)
        downloaded_path: str | None = None
        image = None
        image_error = ""
        try:
            image, path = download_image(url, image_name, logger)
            downloaded_path = str(path)
        except Exception as exc:  # noqa: BLE001
            image_error = str(exc)
            logger.exception("Falha ao baixar/processar %s: %s", image_name, exc)

        for model_label, model_name in MODELS.items():
            model_entry = model_entries[model_label]
            row: Dict[str, Any] = {
                "image_name": image_name,
                "url": url,
                "downloaded_image_path": downloaded_path or "",
                "model_label": model_label,
                "model_name": model_name,
                "device": str(device),
                "status": "ok",
                "error": "",
                "predicted_class": "",
                "top1_confidence_percent": None,
                "model_load_time_seconds": model_entry["load_time_seconds"],
                "inference_time_seconds": None,
                "top5_predictions": [],
                "top5_predictions_json": "[]",
            }

            if image is None:
                row["status"] = "error"
                row["error"] = f"Falha no download/processamento da imagem: {image_error}"
                results.append(row)
                continue

            if model_entry["status"] != "ok":
                row["status"] = "error"
                row["error"] = f"Falha ao carregar modelo: {model_entry['error']}"
                results.append(row)
                continue

            try:
                inference = infer_image(model_entry, image, device)
                row["predicted_class"] = inference["predicted_class"]
                row["top1_confidence_percent"] = inference["top1_confidence_percent"]
                row["inference_time_seconds"] = inference["inference_time_seconds"]
                row["top5_predictions"] = inference["top5"]
                row["top5_predictions_json"] = serialize_top5(inference["top5"])
                logger.info(
                    "Imagem=%s | Modelo=%s | Classe=%s | Confianca=%.2f%% | Inferencia=%.4f s",
                    image_name,
                    model_label,
                    row["predicted_class"],
                    row["top1_confidence_percent"],
                    row["inference_time_seconds"],
                )
            except Exception as exc:  # noqa: BLE001
                row["status"] = "error"
                row["error"] = str(exc)
                logger.exception(
                    "Falha na inferencia da imagem %s com modelo %s: %s",
                    image_name,
                    model_label,
                    exc,
                )

            results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False, encoding="utf-8")
    save_json(results, RESULTS_JSON)

    success_count = int((df["status"] == "ok").sum())
    error_count = int((df["status"] == "error").sum())
    logger.info("Resultados salvos em %s e %s", RESULTS_CSV, RESULTS_JSON)
    logger.info("Execucao finalizada com %s sucessos e %s falhas", success_count, error_count)

    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
