from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import requests
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
IMAGES_DIR = OUTPUT_DIR / "imagens"
RESULTS_DIR = OUTPUT_DIR / "resultados"
LOGS_DIR = OUTPUT_DIR / "logs"

RESULTS_CSV = RESULTS_DIR / "resultados_vit.csv"
RESULTS_JSON = RESULTS_DIR / "resultados_vit.json"
LOG_FILE = LOGS_DIR / "execucao.log"
REPORT_FILE = OUTPUT_DIR / "relatorio_vit_abnt.docx"
TEXT_DOCX = INPUT_DIR / "texto.docx"
CODE_DOCX = INPUT_DIR / "Documento 1.docx"

IMAGE_URLS: Dict[str, str] = {
    "Carro": "https://images.unsplash.com/photo-1492144534655-ae79c964c9d7",
    "Bicicleta": "https://images.unsplash.com/photo-1541625602330-2277a4c46182",
    "Cachorro": "https://images.unsplash.com/photo-1517849845537-4d257902454a",
    "Praia": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e",
    "Floresta": "https://images.unsplash.com/photo-1441974231531-c6227db76b6e",
    "Rua urbana": "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df",
    "Laboratorio": "https://images.unsplash.com/photo-1532187643603-ba119ca4109e",
}

EXPECTED_KEYWORDS: Dict[str, List[str]] = {
    "Carro": ["car", "sports car", "convertible", "racer", "grille", "wheel"],
    "Bicicleta": ["bicycle", "bike", "mountain bike", "velodrome"],
    "Cachorro": ["dog", "retriever", "terrier", "poodle", "spaniel", "canis", "griffon", "pug"],
    "Praia": ["seashore", "coast", "shore", "beach", "ocean", "sandbar"],
    "Floresta": ["forest", "woodland", "rainforest", "grove", "park", "jungle"],
    "Rua urbana": ["street", "crosswalk", "downtown", "building", "alley", "traffic"],
    "Laboratorio": ["laboratory", "lab", "microscope", "clean room", "research", "beaker"],
}


def ensure_directories() -> None:
    for directory in (INPUT_DIR, OUTPUT_DIR, IMAGES_DIR, RESULTS_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str = "vit_relatorio") -> logging.Logger:
    ensure_directories()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sanitize_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip().lower())
    return safe.strip("_") or "imagem"


def download_image(url: str, image_name: str, logger: logging.Logger) -> tuple[Image.Image, Path]:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    image = Image.open(response.raw).convert("RGB")
    extension = image.format.lower() if image.format else "jpg"
    output_path = IMAGES_DIR / f"{sanitize_filename(image_name)}.{extension}"
    image.save(output_path)
    logger.info("Imagem salva em %s", output_path)
    return image, output_path


def save_json(data: object, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def read_docx_text(path: Path) -> str:
    if not path.exists():
        return ""

    with ZipFile(path) as archive:
        xml_data = archive.read("word/document.xml")

    root = ET.fromstring(xml_data)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: List[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        texts = [node.text for node in paragraph.findall(".//w:t", namespace) if node.text]
        if texts:
            paragraphs.append("".join(texts).strip())
    return "\n".join(paragraphs)


def format_seconds(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}".replace(".", ",")


def format_percentage(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}%".replace(".", ",")


def is_prediction_coherent(image_name: str, predicted_class: str) -> bool | None:
    if not predicted_class:
        return None
    terms = EXPECTED_KEYWORDS.get(image_name, [])
    lowered = predicted_class.lower()
    return any(term in lowered for term in terms)
