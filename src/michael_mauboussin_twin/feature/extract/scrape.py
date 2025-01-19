import os
import zenml
import loguru
import requests
import selenium.common.exceptions
from bs4 import BeautifulSoup
import bs4
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from typing import Annotated
from michael_mauboussin_twin.feature.extract import constants, datamodels

logger = loguru.logger

DATE_CLASS = "insightHeaderTextRegular text-uppercase customColor insightDateColor"
TITLE_CLASS = "heroProductName equalSpace noMargin customColor"
TEXT_CLASS = "blockText"
PDF_CLASS = "buttoncomponent left custom-btn btn btn-default btn-lg"


def _setup_driver(url: str) -> webdriver.Chrome:
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get(url)
    driver.maximize_window()
    return driver


def _open_links(driver: webdriver.Chrome) -> None:
    articles = driver.find_elements(By.CSS_SELECTOR, ".series-detail-articles a")

    seen_urls = set()

    for article in articles:
        url = article.get_attribute("href")
        if url not in seen_urls:
            driver.execute_script("window.open(arguments[0], '_blank');", url)
            seen_urls.add(url)


def _close_tab(driver: webdriver.Chrome) -> None:
    driver.switch_to.window(driver.window_handles[0])
    driver.close()


def _extract_data(driver: webdriver.Chrome) -> tuple[str, str, str]:
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    date = soup.find(class_=DATE_CLASS).text.strip()
    title = soup.find(class_=TITLE_CLASS).text.strip()
    text = soup.find(class_=TEXT_CLASS).text.strip()
    return date, title, text


def _extract_pdf(driver: webdriver.Chrome, sanitized_title: str) -> str:
    os.makedirs(constants.DATA_DIR, exist_ok=True)
    pdf_link = driver.find_element(
        By.CLASS_NAME, PDF_CLASS.replace(" ", ".")
    ).get_attribute("href")
    pdf_path = os.path.join(constants.DATA_DIR, f"{sanitized_title}.pdf")
    response = requests.get(pdf_link, timeout=100, stream=True)
    if response.status_code != 200:
        logger.error(f"Failed to download PDF from {pdf_link}")
        raise ValueError(f"Failed to download PDF from {pdf_link}")
    with open(pdf_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info(f"Downloaded PDF {sanitized_title} to {pdf_path}")
    return pdf_path


def _sanitize_title_name(title: str) -> str:
    return title.replace(" ", "_")


@zenml.step(
    name="scrape consilient observer data",
    enable_cache=True,
)
def scrape_data(
    url_tag: bs4.element.Tag,
) -> Annotated[
    list[datamodels.ExtractData],
    zenml.ArtifactConfig(name="consilient_observer_data", version="2025"),
]:
    url = url_tag.get("href")
    consilient_data: list[datamodels.ExtractData] = []
    driver = _setup_driver(url)
    _open_links(driver)
    _close_tab(driver)
    while True:
        driver.switch_to.window(driver.window_handles[0])
        try:
            date, title, text = _extract_data(driver)
        except selenium.common.exceptions.NoSuchElementException as e:
            logger.error(f"Failed to extract data from {driver.current_url}: {e}")
            driver.close()
            continue
        sanitized_title = _sanitize_title_name(title)
        try:
            pdf_path = _extract_pdf(driver, sanitized_title)
        except selenium.common.exceptions.WebDriverException as e:
            logger.error(f"Failed to extract PDF from {driver.current_url}: {e}")
            driver.close()
            continue
        consilient_data.append(
            datamodels.ExtractData(
                date=date,
                title=title,
                text=text,
                pdf_path=pdf_path,
                url=driver.current_url,
            )
        )
        driver.close()
        try:
            hasattr(driver, "window_handles")
        except selenium.common.exceptions.InvalidSessionIdException:
            break
    return consilient_data


if __name__ == "__main__":
    URL = "https://www.morganstanley.com/im/en-us/financial-advisor/insights/series/consilient-observer.html"
    data = scrape_data(URL)
    print(data)
