import os

import bs4
import loguru
import requests
from bs4 import BeautifulSoup

from michael_mauboussin_twin.feature.extract import constants

logger = loguru.logger

URL = "https://www.michaelmauboussin.com/writing"
CHUNK_SIZE = 8192


def get_data(url: str = URL) -> bs4.element.Tag:
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        logger.error(f"Failed to download HTML from {url}")
        raise ValueError(f"Failed to download HTML from {url}")

    soup = BeautifulSoup(response.text, "html.parser")

    research_data = soup.find_all(class_="sqs-html-content")[-2]
    links = research_data.find_all("a")
    consilient_observer_link = process_links(links)
    return consilient_observer_link


def process_links(links: list[bs4.element.Tag]) -> bs4.element.Tag:
    for link in links:
        if link.text.startswith("Research"):
            pdf_data(link["href"], link.text)
        elif link.text.startswith("The Consilient Observer"):
            return link


def pdf_data(link: str, filename: str) -> None:
    safe_filename = (
        "".join(c if c.isalnum() or c in "-_" else "_" for c in filename) + ".pdf"
    )
    response = requests.get(link, timeout=100, stream=True)
    if response.status_code != 200:
        logger.error(f"Failed to download PDF from {link}")
        raise ValueError(f"Failed to download PDF from {link}")
    os.makedirs(constants.DATA_DIR, exist_ok=True)
    filepath = os.path.join(constants.DATA_DIR, safe_filename)

    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)
    logger.info(f"Downloaded PDF {filename} to {filepath}")


def restore_original_filename(safe_filename: str) -> str:
    """
    Convert a safe filename back to its original form.
    Example: "Research_Articles_and_Interviews_1995-2004.pdf" ->
             "Research, Articles and Interviews (1995-2004)"
    """
    original_name = safe_filename.replace(".pdf", "")
    original_name = original_name.replace("_", " ")
    if "-" in original_name:
        parts = original_name.split(" ")
        year_range = parts[-1]
        original_name = " ".join(parts[:-1]) + f" ({year_range})"
    original_name = original_name.replace("Research ", "Research, ")

    return original_name


if __name__ == "__main__":
    get_data()
