import pandas as pd

def create_time_features(data: pd.DataFrame, special_holidays=None) -> pd.DataFrame:

    if special_holidays is not None:
        if not isinstance(special_holidays[0], pd.Timestamp):
            special_holidays = [pd.to_datetime(date) for date in special_holidays]

    """
    Extract tim related features from the dataframe:

    - hour
    - dayofweek
    - dayofmonth
    - dayofyear
    - weekofyear
    - month
    - quarter
    - year
    - weekend
    - distance to weekend (numeric feature indicating the distance to the next weekend)
    - special_holiday (binary feature indicating special holidays)

    This function can be used to extract time related features from
    train and test sets.

    ------------------------------------------------------------

    Parameters:
    df : pandas.DataFrame
        The dataframe to extract time related features from
    special_holidays : list
        The list of special holidays to extract time related features from
    Returns:
    df : pandas.DataFrame
        The dataframe with time related features
    """
    df = data.copy()

    # Basic time features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['distance_to_weekend'] = df['dayofweek'].apply(lambda x: (5 - x) if x < 5 else 0)
    df['dayofmonth'] = df.index.day
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year

    # Binary feature indicating weekend
    df['weekend'] = (df['dayofweek'] >= 5).astype(int)

    # special holiday feature
    if special_holidays is not None:
        df['date'] = df.index.date
        df['special_holiday'] = df['date'].isin(special_holidays).astype(int)

    return df

def update_workingday_and_holiday(data: pd.DataFrame) -> pd.DataFrame:
    """
    Update workingday and holiday columns based on special holidays

    This function is used to update the workingday and holiday columns
    based on the special holidays. Which means that if the date is a special holiday,
    the workingday and holiday columns will be set to 0 and 1 respectively. Otherwise,
    they will remain with the original values.

    ------------------------------------------------------------
    Parameters:
    df : pandas.DataFrame
        The dataframe to update workingday and holiday columns
    Returns:
    df : pandas.DataFrame
        The dataframe with updated workingday and holiday columns
    """

    df = data.copy()

    cols_to_update = ['workingday', 'holiday']

    if 'special_holiday' in df.columns:
        df.loc[df['special_holiday'] == 1, cols_to_update] = [0, 1]
    else:
        raise ValueError("special_holiday column not found in dataframe, please run create_time_features first")

    return df

def create_peak_feature(data: pd.DataFrame, peak_hours=None) -> pd.DataFrame:
    """
    Create peak feature based on hour and workingday

    This function is used to create a peak feature based on the hour 
    and workingday columns. The peak feature is a binary feature indicating
    if the hour is a previously identified peak hour.

    Peak hours are typically identified using external knowledge. They
    can be modified using the config.config.PEAK_HOURS.

    ------------------------------------------------------------

    Parameters:
    df : pandas.DataFrame
        The dataframe to create peak feature from
    peak_hours : list
        The list of peak hours to create peak feature from
    Returns:
    df : pandas.DataFrame
        The dataframe with peak feature
    """
    df = data.copy()

    df['peak'] = (
        df[['hour', 'workingday']]
        .apply(
            lambda row: 1 if row['hour'] in peak_hours and row['workingday'] == 1 else 0,
            axis=1
        )
        .astype(int)
    )

    return df

    return df

def create_time_slots(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create time slots based on hour

    This function is used to create time slots based on the hour column.
    The time slots are created by dividing the hour into 3 equal parts.
    This is a common approach to create time-based features.

    ------------------------------------------------------------

    Parameters:
    df : pandas.DataFrame
        The dataframe to create time slots from
    Returns:
    df : pandas.DataFrame
        The dataframe with time slots
    """
    df = data.copy()

    df['time_slot'] = df['hour'].apply(lambda x: (x // 3) + 1)

    return df