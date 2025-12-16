import datetime
import numpy as np
import talib.abstract as ta
from pandas import DataFrame
from datetime import datetime
from technical import qtpylib
from freqtrade.strategy import IStrategy
from scipy.signal import argrelextrema


class SlopeV8_fixed(IStrategy):
    INTERFACE_VERSION = 3
    
    timeframe = '15m'
    can_short = True

    minimal_roi = { '0': 1 }

    stoploss = -0.999

    trailing_stop = False

    max_open_trades = -1

    @property
    def plot_config(self):
        plot_config = {
            'main_plot' : {},
            'subplots' : {},
        }
        return plot_config
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag:str, side: str, **kwargs) -> float:
        return 10.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        order = 5 

        # 2. Find the indices of local peaks and valleys
        # Note: This uses future data! We must handle this below.
        maxima_idx = argrelextrema(dataframe['close'].values, np.greater, order=order)[0]
        minima_idx = argrelextrema(dataframe['close'].values, np.less, order=order)[0]

        # 3. Initialize columns
        dataframe['maxima'] = np.nan
        dataframe['minima'] = np.nan

        # 4. Mark the peaks/valleys
        dataframe.loc[maxima_idx, 'maxima'] = dataframe.loc[maxima_idx, 'close']
        dataframe.loc[minima_idx, 'minima'] = dataframe.loc[minima_idx, 'close']

        # -------------------------------------------------------------------
        # CRITICAL FIX: Shift the signals forward by 'order' candles.
        # This simulates the delay required to confirm the peak/valley in real-time.
        # Without this, the strategy is repainting (cheating).
        # -------------------------------------------------------------------
        dataframe['maxima'] = dataframe['maxima'].shift(order)
        dataframe['minima'] = dataframe['minima'].shift(order)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['minima'].notnull())
            ),
        'enter_long'] = 1

        dataframe.loc[
            (
                (dataframe['maxima'].notnull())
            ),
        'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['maxima'].notnull())
            ),
        'exit_long'] = 1

        dataframe.loc[
            (
                (dataframe['minima'].notnull())
            ),
        'exit_short'] = 1

        return dataframe
