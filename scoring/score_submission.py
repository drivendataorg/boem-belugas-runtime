import json
from multiprocessing.sharedctypes import Value
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score
import typer


PREDICTION_LIMIT = 20
QUERY_ID_COL = "query_id"
DATABASE_ID_COL = "database_image_id"
SCORE_COL = "score"


class MeanAveragePrecision:
    @classmethod
    def score(cls, predicted: pd.DataFrame, actual: pd.DataFrame, prediction_limit: int):
        """Calculates mean average precision for a ranking task.

        :param predicted: The predicted values as a dataframe with specified column names
        :param actual: The ground truth values as a dataframe with specified column names
        """
        if not predicted[SCORE_COL].between(0.0, 1.0).all():
            raise ValueError("Scores must be in range [0, 1].")
        if predicted.index.name != QUERY_ID_COL:
            raise ValueError(
                f"First column of submission must be named '{QUERY_ID_COL}', "
                f"got {predicted.index.name}."
            )
        if predicted.columns.to_list() != [DATABASE_ID_COL, SCORE_COL]:
            raise ValueError(
                f"Columns of submission must be named '{[DATABASE_ID_COL, SCORE_COL]}', "
                f"got {predicted.columns.to_list()}."
            )

        unadjusted_aps, predicted_n_pos, actual_n_pos = cls._score_per_query(
            predicted, actual, prediction_limit
        )
        adjusted_aps = unadjusted_aps.multiply(predicted_n_pos).divide(actual_n_pos)
        return adjusted_aps.mean()

    @classmethod
    def _score_per_query(
        cls, predicted: pd.DataFrame, actual: pd.DataFrame, prediction_limit: int
    ):
        """Calculates per-query mean average precision for a ranking task."""
        merged = predicted.merge(
            right=actual.assign(actual=1.0),
            how="left",
            on=[QUERY_ID_COL, DATABASE_ID_COL],
        ).fillna({"actual": 0.0})
        # Per-query raw average precisions based on predictions
        unadjusted_aps = merged.groupby(QUERY_ID_COL).apply(
            lambda df: average_precision_score(df["actual"].values, df[SCORE_COL].values)
            if df["actual"].sum()
            else 0.0
        )
        # Total ground truth positive counts for rescaling
        predicted_n_pos = merged["actual"].groupby(QUERY_ID_COL).sum().astype("int64").rename()
        actual_n_pos = actual.groupby(QUERY_ID_COL).size().clip(upper=prediction_limit)
        return unadjusted_aps, predicted_n_pos, actual_n_pos


def main(
    submission_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to submission file (submission.csv)",
    ),
    ground_truth_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to ground truth CSV file.",
    ),
):
    """Evaluate a submission for the "Where's Whale-do?" beluga whale photo-identification
    challenge."""

    submission_df = pd.read_csv(submission_path, index_col=QUERY_ID_COL)
    gt_df = pd.read_csv(ground_truth_path, index_col=QUERY_ID_COL)

    mean_avg_prec = MeanAveragePrecision.score(
        predicted=submission_df, actual=gt_df, prediction_limit=PREDICTION_LIMIT
    )

    typer.echo(
        json.dumps(
            {
                "mean_average_precision": mean_avg_prec,
            }
        )
    )


if __name__ == "__main__":
    typer.run(main)
