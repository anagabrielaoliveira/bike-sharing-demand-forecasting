from dataclasses import dataclass
from dataclasses import field
from typing import Tuple
from typing import List

SPECIAL_HOLIDAYS = [

    # Christmas
    '2011-12-24', '2011-12-25', '2011-12-26',  # 26th - observance
    '2012-12-24', '2012-12-25',

    # New Year's Eve & New Year
    '2011-01-01', '2011-12-31', '2012-01-01',
    '2012-12-31',

    # Hurricane Sandy
    '2012-10-29', '2012-10-30'
]

PEAK_HOURS = [7, 8, 17, 18] # Peak hours on working days

QUANTITATIVE_COLS = ['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']
Y_COLS = ['casual', 'registered', 'count' ]
RANDOM_STATE = 42