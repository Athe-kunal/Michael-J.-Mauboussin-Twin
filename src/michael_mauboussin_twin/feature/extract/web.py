import os
from typing import Annotated

import bs4
import loguru
import requests
import zenml
from bs4 import BeautifulSoup

from michael_mauboussin_twin.feature.extract import constants, datamodels

logger = loguru.logger


@zenml.step(
    name="download previous pdf and get consilient observer link",
    enable_cache=True,
)
def get_consilient_observer_link_after_saving_previous_data(
    url: str = constants.URL,
) -> tuple[list[datamodels.ExtractData], bs4.element.Tag]:
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        logger.error(f"Failed to download HTML from {url}")
        raise ValueError(f"Failed to download HTML from {url}")

    soup = BeautifulSoup(response.text, "html.parser")

    research_data = soup.find_all(class_="sqs-html-content")[-2]
    links = research_data.find_all("a")
    previous_pdf_data_list, consilient_observer_link = process_links(links)
    return previous_pdf_data_list, consilient_observer_link


@zenml.step(
    name="process links",
    enable_cache=True,
)
def process_links(
    links: list[bs4.element.Tag],
) -> Annotated[
    tuple[list[datamodels.ExtractData], bs4.element.Tag | None],
    zenml.ArtifactConfig(name="previous_pdf_data", version="2025"),
]:
    pdf_data_list: list[datamodels.ExtractData] = []
    consilient_observer_link: bs4.element.Tag | None = None
    for link in links:
        if link.text.startswith("Research"):
            filepath = pdf_data_save(link["href"], link.text)
            pdf_data_list.append(
                datamodels.ExtractData(
                    url=link["href"],
                    title=link.text,
                    pdf_path=filepath,
                    date=link.text[link.text.find("(") + 1 : link.text.rfind(")")],
                )
            )
        elif link.text.startswith("The Consilient Observer"):
            consilient_observer_link = link
    return pdf_data_list, consilient_observer_link


def pdf_data_save(link: str, filename: str) -> str:
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
        for chunk in response.iter_content(chunk_size=constants.CHUNK_SIZE):
            f.write(chunk)
    logger.info(f"Downloaded PDF {filename} to {filepath}")
    return filepath


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
    get_consilient_observer_link_after_saving_previous_data(constants.URL)
