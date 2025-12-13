# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple, List

from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib
from technical.indicators import ichimoku

# ==========================================
# Ultimate Ichimoku Cloud Strategy [2944% Profit Backtest]
# YouTube Link: https://youtu.be/EumlRRIx0WA
# ==========================================


# ================================
# Download Historical Data
# ================================

"""
freqtrade download-data \
    -c user_data/binance_futures_Ichimoku.json \
    --timerange 20220101- \
    -t 5m 15m 30m 1h 4h 1d

"""

# ================================
# Lookahead Analysis 
# ================================

"""
freqtrade lookahead-analysis \
    --strategy Ichimoku \
    --timeframe 4h \
    --timerange 20230101-20240501 \
    -c user_data/binance_futures_Ichimoku.json \
    --max-open-trades 1 \
    -p AVAX/USDT:USDT
"""

# ================================
# Hyperopt Optimization
# ================================
"""
freqtrade hyperopt \
    --strategy Ichimoku \
    --config user_data/binance_futures_Ichimoku.json \
    --timeframe 4h \
    --timerange 20230101-20240501 \
    --hyperopt-loss MultiMetricHyperOptLoss \
    --spaces buy\
    -e 50 \
    --j -2 \
    --random-state 9319 \
    --min-trades 20 \
    -p AVAX/USDT:USDT \
    --max-open-trades 1 \
    --analyze-per-epoch
"""

# ================================
# Backtesting
# ================================

"""
freqtrade backtesting \
    --strategy Ichimoku \
    --timeframe 4h \
    --timerange 20230101-20250101 \
    --breakdown month \
    -c user_data/binance_futures_Ichimoku.json \
    --max-open-trades 1 \
    -p AVAX/USDT:USDT \
    --cache none \
    --timeframe-detail 30m
"""

# ================================
# Start FreqUI Web Interface
# ================================

"""
freqtrade webserver \
    --config user_data/config_binance_futures.json
"""


class Ichimoku(IStrategy):
            
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "4h"

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {}
    
    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.25
    
    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured
    
    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    conversion_line_periods = IntParameter(5, 15, default=9, space="buy")    
    base_line_periods = IntParameter(20, 35, default=26, space="buy")  
    atr_mult = CategoricalParameter([1.5, 2, 2.5, 3], default=2.5, space="buy")
    converstion_cross_rolling_window = CategoricalParameter([3, 6, 9, 12, 15], default=6, space="buy")

    leverage_level = IntParameter(1, 10, default=1, space='buy', optimize=False, load=False)

    @property
    def plot_config(self):

        plot_config = {

            'main_plot': {
                f'conversion_line_{self.conversion_line_periods.value}_{self.base_line_periods.value}': {'color': 'rgb(41, 98, 255)', 'style': 'line', 'width': 2},
                f'base_line_{self.conversion_line_periods.value}_{self.base_line_periods.value}': {'color': 'rgb(183, 28, 28)', 'style': 'line', 'width': 2},
                f'upper_{self.conversion_line_periods.value}_{self.base_line_periods.value}': {
                    'color': 'rgb(165, 214, 167)',
                    'fill_to': f'lower_{self.conversion_line_periods.value}_{self.base_line_periods.value}',
                    'fill_label': 'Ichimoku Cloud',
                    'fill_color': 'rgba(67, 160, 71, 0.1)',
                },
                f'lower_{self.conversion_line_periods.value}_{self.base_line_periods.value}': {
                    'color': 'rgb(239, 154, 154)',
                },
            }
        }
        
        return plot_config
    
    
    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        
        # get access to all pairs available in whitelist.
        # pairs = self.dp.current_whitelist()

        # # Assign tf to each pair so they can be downloaded and cached for strategy.
        # informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        
        return []
    
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        for conversion_line_val in self.conversion_line_periods.range:
            for base_line_val in self.base_line_periods.range:
                
                ichi = ichimoku(dataframe,
                                conversion_line_period=conversion_line_val, 
                                base_line_periods=base_line_val)
                
                dataframe[f'conversion_line_{conversion_line_val}_{base_line_val}'] = ichi['tenkan_sen']
                dataframe[f'base_line_{conversion_line_val}_{base_line_val}'] = ichi['kijun_sen']
                dataframe[f'upper_{conversion_line_val}_{base_line_val}'] = np.maximum(ichi['senkou_span_a'], ichi['senkou_span_b'])
                dataframe[f'lower_{conversion_line_val}_{base_line_val}'] = np.minimum(ichi['senkou_span_a'], ichi['senkou_span_b'])

        # ATR
        dataframe["atr"] = ta.ATR(dataframe)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:        
        
        conversion_line_val = self.conversion_line_periods.value
        base_line_val = self.base_line_periods.value

        dataframe.loc[
            (

                (dataframe[f'conversion_line_{conversion_line_val}_{base_line_val}'] > dataframe[f'base_line_{conversion_line_val}_{base_line_val}']) &
                (dataframe['close'] > dataframe[f'upper_{conversion_line_val}_{base_line_val}']) &
                (dataframe[f'conversion_line_{conversion_line_val}_{base_line_val}'] >= dataframe[f'conversion_line_{conversion_line_val}_{base_line_val}'].shift(1)) &
                (dataframe[f'base_line_{conversion_line_val}_{base_line_val}'] >= dataframe[f'base_line_{conversion_line_val}_{base_line_val}'].shift(1)) &

                # Checks if conversion line crossed above the base line within the specified rolling window
                (dataframe[f'conversion_line_{conversion_line_val}_{base_line_val}'].rolling(window=self.converstion_cross_rolling_window.value).apply(
                    lambda x: any(qtpylib.crossed_above(x, dataframe[f'base_line_{conversion_line_val}_{base_line_val}'].iloc[x.index[0]:x.index[-1]+1])))
                ) &
                
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1
        
        dataframe.loc[
            (

                (dataframe[f'conversion_line_{conversion_line_val}_{base_line_val}'] < dataframe[f'base_line_{conversion_line_val}_{base_line_val}']) &
                (dataframe['close'] < dataframe[f'lower_{conversion_line_val}_{base_line_val}']) &
                (dataframe[f'conversion_line_{conversion_line_val}_{base_line_val}'] <= dataframe[f'conversion_line_{conversion_line_val}_{base_line_val}'].shift(1)) &
                (dataframe[f'base_line_{conversion_line_val}_{base_line_val}'] <= dataframe[f'base_line_{conversion_line_val}_{base_line_val}'].shift(1)) &

                # Checks if conversion line crossed below the base line within the specified rolling window
                (dataframe[f'conversion_line_{conversion_line_val}_{base_line_val}'].rolling(window=self.converstion_cross_rolling_window.value).apply(
                    lambda x: any(qtpylib.crossed_below(x, dataframe[f'base_line_{conversion_line_val}_{base_line_val}'].iloc[x.index[0]:x.index[-1]+1])))
                ) &
                (dataframe['volume'] > 0)
            ),
            'enter_short'] = 1

        return dataframe
    

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        conversion_line_val = self.conversion_line_periods.value
        base_line_val = self.base_line_periods.value
        exit_threshold = dataframe['atr'] * self.atr_mult.value
        
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['close'], (dataframe[f'base_line_{conversion_line_val}_{base_line_val}'] - exit_threshold))) &
                (dataframe['volume'] > 0)
            ),
        'exit_long'] = 1
        
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['close'], (dataframe[f'base_line_{conversion_line_val}_{base_line_val}'] + exit_threshold))) &
                (dataframe['volume'] > 0)
            ),
        'exit_short'] = 1

        return dataframe

    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        return self.leverage_level.value