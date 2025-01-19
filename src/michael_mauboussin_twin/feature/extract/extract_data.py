import json
import os
import zenml
import loguru
import sys
from typing import Annotated, Tuple
from michael_mauboussin_twin.feature.extract import constants, datamodels, web, scrape


sys.setrecursionlimit(3000)

logger = loguru.logger


@zenml.step(enable_cache=False)
def extraction_data() -> Tuple[
    Annotated[list[datamodels.ExtractData], "previous_pdf_data"],
    Annotated[list[datamodels.ExtractData], "consilient_observer_data"],
]:
    # client = zenml_client.Client()
    pdf_data_list, url = web.get_consilient_observer_link_after_saving_previous_data()
    # pdf_data_list, url = client.get_artifact_version(
    #     name_id_or_prefix="previous_pdf_data", version="2025"
    # )
    consilient_observer_data = scrape.scrape_data(url)
    # consilient_observer_data = client.get_artifact_version(
    #     name_id_or_prefix="consilient_observer_data", version="2025"
    # )
    # save_metadata(pdf_data_list, consilient_observer_data)
    save_metadata(pdf_data_list)
    save_metadata(consilient_observer_data)
    return (pdf_data_list, consilient_observer_data)


# @zenml.step(name="save metadata", enable_cache=True)
def save_metadata(
    extract_data: list[datamodels.ExtractData],
    # consilient_observer_data: list[datamodels.ExtractData],
) -> None:
    # combined_data = pdf_data_list + consilient_observer_data
    output_path = os.path.join(constants.DATA_DIR, constants.EXTRACTION_METADATA_FILE)
    with open(output_path, "r+", encoding="utf-8") as f:
        try:
            existing_data = json.load(f)
            f.seek(0)
            f.truncate()
            json.dump(
                existing_data.extend([data.model_dump() for data in extract_data]),
                f,
                indent=2,
            )
        except json.JSONDecodeError:
            json.dump([data.model_dump() for data in extract_data], f, indent=2)
        logger.info(f"Saved metadata to {output_path}")


if __name__ == "__main__":
    extraction_data()
