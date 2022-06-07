from pathlib import Path

import pandas as pd

ROOT_DIRECTORY = Path("/code_execution")
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
OUTPUT_FILE = ROOT_DIRECTORY / "submission" / "submission.csv"


def predict(query_image_id, database_image_ids):
    raise NotImplementedError(
        "This script is just a template. You should adapt it with your own code."
    )
    result_images = ...
    scores = ...
    return result_images, scores


def main():
    scenarios_df = pd.read_csv(DATA_DIRECTORY / "query_scenarios.csv")

    predictions = []

    for scenario_row in scenarios_df.itertuples():

        queries_df = pd.read_csv(DATA_DIRECTORY / scenario_row.queries_path)
        database_df = pd.read_csv(DATA_DIRECTORY / scenario_row.database_path)

        for query_row in queries_df.itertuples():
            query_id = query_row.query_id
            query_image_id = query_row.query_image_id
            database_image_ids = database_df["database_image_id"].values

            ### Prediction happens here ######
            result_images, scores = predict(query_image_id, database_image_ids)
            ##################################

            predictions.extend({"query_id": query_id, "database_image_id": pred_image_id, "score": score,} for pred_image_id, score in zip(result_images, scores))

    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()
