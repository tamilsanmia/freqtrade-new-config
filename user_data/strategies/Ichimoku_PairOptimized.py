# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple, List
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

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
    -c user_data/binance_futures_Ichimoku_PairOptimized.json \
    --timerange 20220101- \
    -t 5m 15m 30m 1h 4h 1d
"""

# ================================
# Backtesting
# ================================

"""
freqtrade backtesting \
    --strategy Ichimoku_PairOptimized \
    --timeframe 4h \
    --timerange 20230101-20250101 \
    --breakdown month \
    -c user_data/binance_futures_Ichimoku_PairOptimized.json \
    --max-open-trades 4 \
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


class Ichimoku_PairOptimized(IStrategy):
    
    def __init__(self, config):
        
        # Initialize the strategy with the given configuration and load pair-specific settings.   
        super().__init__(config)
        self.load_pair_settings()

    def load_pair_settings(self) -> None:
        
        # Get the class name dynamically to locate the appropriate settings file
        class_name = self.__class__.__name__
        settings_filename = Path(__file__).parent / f'{class_name}_Settings.json'
        
        try:
            # Attempt to open and load the JSON settings file
            with open(settings_filename, "r") as f:
                self.custom_info = json.load(f)
                logger.info(f"Settings successfully loaded from {settings_filename}.")
                logger.info(f"Settings: {self.custom_info}")

        except FileNotFoundError:
            # Raise an error if the settings file is missing
            raise SystemExit(f"Settings file not found at {settings_filename}. Program will exit.")
        
        except json.JSONDecodeError as e:
            # Raise an error if the JSON file contains invalid data
            raise SystemExit(f"Error decoding JSON from settings file: {e}. Program will exit.")
    
    
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

    @property
    def plot_config(self):

        plot_config = {
            'main_plot': {
                'conversion_line': {'color': 'rgb(41, 98, 255)', 'style': 'line', 'width': 2},
                'base_line': {'color': 'rgb(183, 28, 28)', 'style': 'line', 'width': 2},
                'upper': {
                    'color': 'rgb(165, 214, 167)', 
                    'fill_to': 'lower',
                    'fill_label': 'Ichimoku Cloud', 
                    'fill_color': 'rgba(67, 160, 71, 0.1)',
                },
                'lower': {
                    'color': 'rgb(239, 154, 154)',
                }
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
        
        pair = metadata['pair']
        
        if pair in self.custom_info:
            
            pair_settings = self.custom_info[pair]

            ichi = ichimoku(dataframe,
                            conversion_line_period=pair_settings['conversion_line_periods'], 
                            base_line_periods=pair_settings['base_line_periods'])
            
            dataframe['conversion_line'] = ichi['tenkan_sen']
            dataframe['base_line'] = ichi['kijun_sen']
            dataframe['upper'] = np.maximum(ichi['senkou_span_a'], ichi['senkou_span_b'])
            dataframe['lower'] = np.minimum(ichi['senkou_span_a'], ichi['senkou_span_b'])

            # ATR
            dataframe["atr"] = ta.ATR(dataframe)
            
            return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:        
        
        pair = metadata['pair']
        
        if pair in self.custom_info:
            
            pair_settings = self.custom_info[pair]

            dataframe.loc[
                (

                    (dataframe['conversion_line'] > dataframe[f'base_line']) &
                    (dataframe['close'] > dataframe[f'upper']) &
                    (dataframe['conversion_line'] >= dataframe[f'conversion_line'].shift(1)) &
                    (dataframe['base_line'] >= dataframe[f'base_line'].shift(1)) &

                    # Checks if conversion line crossed above the base line within the specified rolling window
                    (dataframe['conversion_line'].rolling(window=pair_settings['converstion_cross_rolling_window']).apply(
                        lambda x: any(qtpylib.crossed_above(x, dataframe['base_line'].iloc[x.index[0]:x.index[-1]+1])))
                    ) &
                    
                    (dataframe['volume'] > 0)
                ),

                'enter_long'] = 1
            
            dataframe.loc[
                (

                    (dataframe['conversion_line'] < dataframe['base_line']) &
                    (dataframe['close'] < dataframe['lower']) &
                    (dataframe['conversion_line'] <= dataframe['conversion_line'].shift(1)) &
                    (dataframe['base_line'] <= dataframe['base_line'].shift(1)) &

                    # Checks if conversion line crossed below the base line within the specified rolling window
                    (dataframe['conversion_line'].rolling(window=pair_settings['converstion_cross_rolling_window']).apply(
                        lambda x: any(qtpylib.crossed_below(x, dataframe['base_line'].iloc[x.index[0]:x.index[-1]+1])))
                    ) &
                    (dataframe['volume'] > 0)
                ),

                'enter_short'] = 1

        
            return dataframe
    

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        pair = metadata['pair']
        
        if pair in self.custom_info:
            
            pair_settings = self.custom_info[pair]
            
            exit_threshold = dataframe['atr'] * pair_settings['atr_mult']
            
            dataframe.loc[
                (
                    (qtpylib.crossed_below(dataframe['close'], (dataframe['base_line'] - exit_threshold))) &
                    (dataframe['volume'] > 0)
                ),
            'exit_long'] = 1
            
            dataframe.loc[
                (
                    (qtpylib.crossed_above(dataframe['close'], (dataframe['base_line'] + exit_threshold))) &
                    (dataframe['volume'] > 0)
                ),
            'exit_short'] = 1

            return dataframe

    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:
        
        if pair in self.custom_info:
            
            pair_settings = self.custom_info[pair]
            
            return pair_settings['leverage_level']