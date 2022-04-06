import pandas as pd
from typing import Tuple


def predict_dst(
    solar_wind_7d: pd.DataFrame,
    satellite_positions_7d: pd.DataFrame,
    latest_sunspot_number: float,
) -> Tuple[float, float]:
    """
    Take all of the data up until time t-1, and then make predictions for
    times t and t+1.

    Parameters
    ----------
    solar_wind_7d: pd.DataFrame
        The last 7 days of satellite data up until (t - 1) minutes [exclusive of t], with `timedelta` as index
    satellite_positions_7d: pd.DataFrame
        The last 7 days of satellite position data up until the present time [inclusive of t], with `timedelta` as index
    latest_sunspot_number: float
        The latest monthly sunspot number (SSN) to be available

    Returns
    -------
    predictions : Tuple[float, float]
        A tuple of two predictions, for (t and t + 1 hour) respectively; these should
        be between -2,000 and 500.
    """

    ########################################################################
    #                         YOUR CODE HERE!                              #
    ########################################################################

    # this is a naive baseline where we just guess the training data mean every time
    prediction_at_t0 = -12
    prediction_at_t1 = -12

    return prediction_at_t0, prediction_at_t1
