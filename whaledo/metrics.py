from typing import ClassVar, Tuple, cast

import pandas as pd  # type: ignore
from sklearn.metrics import average_precision_score  # type: ignore

__all__ = ["MeanAveragePrecision"]


class MeanAveragePrecision:
    PREDICTION_LIMIT: ClassVar[int] = 20
    QUERY_ID_COL: ClassVar[str] = "query_id"
    DATABASE_ID_COL: ClassVar[str] = "database_image_id"
    SCORE_COL: ClassVar[str] = "score"

    @classmethod
    def score(cls, predicted: pd.DataFrame, *, actual: pd.DataFrame):
        """Calculates mean average precision for a ranking task.

        :param predicted: The predicted values as a dataframe with specified column names
        :param actual: The ground truth values as a dataframe with specified column names
        """
        if not predicted[cls.SCORE_COL].between(0.0, 1.0).all():
            raise ValueError("Scores must be in range [0, 1].")
        if predicted.index.name != cls.QUERY_ID_COL:
            raise ValueError(
                f"First column of submission must be named '{cls.QUERY_ID_COL}', "
                f"got {predicted.index.name}."
            )
        if predicted.columns.to_list() != [cls.DATABASE_ID_COL, cls.SCORE_COL]:
            raise ValueError(
                f"Columns of submission must be named '{[cls.DATABASE_ID_COL, cls.SCORE_COL]}', "
                f"got {predicted.columns.to_list()}."
            )

        unadjusted_aps, predicted_n_pos, actual_n_pos = cls._score_per_query(
            predicted, actual=actual
        )
        adjusted_aps = unadjusted_aps.multiply(predicted_n_pos).divide(actual_n_pos)
        return adjusted_aps.mean()

    @classmethod
    def _score_per_query(
        cls, predicted: pd.DataFrame, *, actual: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculates per-query mean average precision for a ranking task."""
        merged = predicted.merge(
            right=actual.assign(actual=1.0),
            how="left",
            on=[cls.QUERY_ID_COL, cls.DATABASE_ID_COL],
        ).fillna({"actual": 0.0})
        # Per-query raw average precisions based on predictions
        unadjusted_aps = cast(
            pd.Series,
            merged.groupby(cls.QUERY_ID_COL).apply(
                lambda df: average_precision_score(df["actual"].values, df[cls.SCORE_COL].values)
                if df["actual"].sum()
                else 0.0
            ),
        )
        # Total ground truth positive counts for rescaling
        predicted_n_pos = cast(
            pd.Series, merged["actual"].groupby(cls.QUERY_ID_COL).sum().astype("int64").rename()
        )
        actual_n_pos = cast(
            pd.Series, actual.groupby(cls.QUERY_ID_COL).size().clip(upper=cls.PREDICTION_LIMIT)
        )
        return unadjusted_aps, predicted_n_pos, actual_n_pos
