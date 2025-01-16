import json
import os

from michael_mauboussin_twin.feature.extract import constants, datamodels, scrape, web


def extract_and_save_data() -> list[datamodels.ExtractData]:
    pdf_data_list, url = web.get_consilient_observer_link_after_saving_previous_data(
        constants.URL
    )
    consilient_observer_data = scrape.scrape_data(url["href"])
    combined_data = pdf_data_list + consilient_observer_data
    return combined_data


def save_data(pdf_data_list: list[datamodels.ExtractData]) -> None:
    output_path = os.path.join(constants.DATA_DIR, constants.EXTRACTION_METADATA_FILE)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([data.model_dump() for data in pdf_data_list], f, indent=2)


if __name__ == "__main__":
    extraction_metadata = extract_and_save_data()
    save_data(extraction_metadata)
