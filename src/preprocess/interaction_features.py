import pandas as pd


def create_interaction_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    This function is used to create interaction features between temperature,
    humidity and windspeed. The interaction features are created by multiplying
    the temperature, humidity and windspeed columns.

    ------------------------------------------------------------

    Parameters:
    data : pandas.DataFrame
        The dataframe to create interaction features from
    Returns:
    data : pandas.DataFrame
        The dataframe with interaction features
    """
    df = data.copy()

    df['temp_vs_humidity'] = df['temp'] * df['humidity']
    df['temp_vs_windspeed'] = df['temp'] * df['windspeed']
    df['humidity_vs_windspeed'] = df['humidity'] * df['windspeed']

    return df