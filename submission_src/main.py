from pathlib import Path

import pandas as pd
import typer

ROOT_DIRECTORY = Path("/code_execution")
PREDICTION_FILE = ROOT_DIRECTORY / "submission" / "submission.csv"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"


def main():
    query_ids = ['Q00', 'Q00', 'Q00', 'Q01', 'Q01', 'Q02', 'Q02']
    reference_ids = ['R00A', 'R00B', 'R00C', 'R01A', 'R01B', 'R02A', 'R02B']
    scores = [1., .9, .8, 1., .9, 1., .9]
    submission = pd.DataFrame({"query_id": query_ids, "reference_id": reference_ids, "score": scores})
    submission.to_csv(PREDICTION_FILE, index=False)


if __name__ == "__main__":
    typer.run(main)
