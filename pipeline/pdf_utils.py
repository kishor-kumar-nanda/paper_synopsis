import fitz
import base64
from typing import List


def pdf_to_base64_images(pdf_bytes: bytes, dpi: int = 200) -> List[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []

    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")
        images.append(base64.b64encode(img_bytes).decode())

    return images


def extract_page_texts(pdf_bytes: bytes) -> List[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return [page.get_text() for page in doc]


def build_context(page_text: str, max_chars: int = 1500) -> str:
    lines = page_text.split("\n")

    captions = [
        l for l in lines
        if l.lower().startswith("figure")
    ]

    if captions:
        return "\n".join(captions[:3])[:max_chars]

    return page_text[:max_chars]