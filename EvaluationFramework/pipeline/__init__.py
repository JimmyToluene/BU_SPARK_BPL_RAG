"""War-period pseudo GT pipeline — step registry."""

from .step1_fetch_ids import run as step1_fetch_ids
from .step2_fetch_metadata import run as step2_fetch_metadata
from .step3_download_pages import run as step3_download_pages
from .step4_dots_ocr import run as step4_dots_ocr
from .step5_extract_articles import run as step5_extract_articles
from .step6_generate_qa import run as step6_generate_qa
from .step7_export import run as step7_export

STEPS = {
    1: ("Fetch Record IDs", step1_fetch_ids),
    2: ("Fetch Metadata + OCR Text", step2_fetch_metadata),
    3: ("Download IIIF Page Images", step3_download_pages),
    4: ("dots.ocr VLM OCR", step4_dots_ocr),
    5: ("Extract War Articles", step5_extract_articles),
    6: ("Generate Q&A Pairs", step6_generate_qa),
    7: ("Export Final JSONL", step7_export),
}
