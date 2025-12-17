import logging
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
import talib.abstract as ta
import pandas_ta as pta

import numpy as np
from technical import qtpylib
from datetime import datetime
from pandas import DataFrame, Series
from typing import Optional
from freqtrade.persistence import Trade
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
from scipy.signal import argrelextrema
import pandas as pd
from technical import qtpylib
from technical.indicators import ichimoku

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class FreqAiProV4_5m_15m(IStrategy):
    exit_profit_only = False
    trailing_stop = False
    position_adjustment_enable = False
    ignore_roi_if_entry_signal = True
    max_entry_position_adjustment = 0
    max_dca_multiplier = 1
    process_only_new_candles = True
    can_short = True
    use_exit_signal = True
    startup_candle_count: int = 200
    stoploss = -0.296
    timeframe = "5m"
    informative_timeframe = '15m'

    # Strategy Parameters
    rsi_period = 14
    bb_period = 20
    bb_stddev = 2
    dispersion = 0.1

    # DCA
    position_adjustment_enable = True
    initial_safety_order_trigger = DecimalParameter(
        low=-0.02, high=-0.01, default=-0.018, decimals=3, space="entry", optimize=True, load=True
    )
    max_safety_orders = IntParameter(1, 6, default=2, space="entry", optimize=True)
    safety_order_step_scale = DecimalParameter(
        low=1.05, high=1.5, default=1.25, decimals=2, space="entry", optimize=True, load=True
    )
    safety_order_volume_scale = DecimalParameter(
        low=1.1, high=2, default=1.4, decimals=1, space="entry", optimize=True, load=True
    )

    # Custom Functions
    increment = DecimalParameter(
        low=1.0005, high=1.002, default=1.001, decimals=4, space="entry", optimize=True, load=True
    )
    last_entry_price = None

    # Protections
    cooldown_lookback = IntParameter(2, 48, default=1, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=4, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    minimal_roi = {
        "0": 0.5,
        "60": 0.45,
        "120": 0.4,
        "240": 0.3,
        "360": 0.25,
        "720": 0.2,
        "1440": 0.15,
        "2880": 0.1,
        "3600": 0.05,
        "7200": 0.02,
    }

    plot_config = {
        "main_plot": {
            'tenkan': {'color': 'green'},
            'kijun': {'color': 'red'},
        },
        "subplots": {
            "5m RSI BB Dispersion": {
                "rsi": {"color": "blue", "type": "line"},
                "rsi_ema": {"color": "red", "type": "line"},
                "rsi_upper": {"color": "cyan", "type": "line"},
                "rsi_lower": {"color": "cyan", "type": "line"},
                "rsi_disp_up": {"color": "white", "type": "line"},
                "rsi_disp_down": {"color": "white", "type": "line"},
            },
            "15m RSI BB Dispersion": {
                "rsi_15m": {"color": "blue", "type": "line"},
                "rsi_ema_15m": {"color": "red", "type": "line"},
                "rsi_upper_15m": {"color": "cyan", "type": "line"},
                "rsi_lower_15m": {"color": "cyan", "type": "line"},
                "rsi_disp_up_15m": {"color": "white", "type": "line"},
                "rsi_disp_down_15m": {"color": "white", "type": "line"},
            },
            "extrema": {
                "&s-extrema": {"color": "#f53580", "type": "line"},
                "&s-minima_sort_threshold": {"color": "#4ae747", "type": "line"},
                "&s-maxima_sort_threshold": {"color": "#5b5e4b", "type": "line"},
            },
            "min_max": {
                "maxima": {"color": "#a29db9", "type": "line"},
                "minima": {"color": "#ac7fc", "type": "line"},
                "maxima_check": {"color": "#a29db9", "type": "line"},
                "minima_check": {"color": "#ac7fc", "type": "line"},
            },
        },
    }

    @property
    def protections(self):
        prot = []
        prot.append(
            {"method": "CooldownPeriod", "stop_duration_candles": self.cooldown_lookback.value}
        )
        if self.use_stop_protection.value:
            prot.append(
                {
                    "method": "StoplossGuard",
                    "lookback_period_candles": 24 * 3,
                    "trade_limit": 2,
                    "stop_duration_candles": self.stop_duration.value,
                    "only_per_pair": False,
                }
            )
        return prot

    def custom_stake_amount(
            self,
            pair: str,
            current_time: datetime,
            current_rate: float,
            proposed_stake: float,
            min_stake: Optional[float],
            max_stake: float,
            leverage: float,
            entry_tag: Optional[str],
            side: str,
            **kwargs,
    ) -> float:
        return proposed_stake / self.max_dca_multiplier

    def custom_entry_price(
            self,
            pair: str,
            trade: Optional["Trade"],
            current_time: datetime,
            proposed_rate: float,
            entry_tag: Optional[str],
            side: str,
            **kwargs,
    ) -> float:
        dataframe, last_updated = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.timeframe
        )
        entry_price = (dataframe["close"].iloc[-1] + dataframe["open"].iloc[-1] + proposed_rate) / 3
        if proposed_rate < entry_price:
            entry_price = proposed_rate

        logger.info(
            f"{pair} Using Entry Price: {entry_price} | close: {dataframe['close'].iloc[-1]} open: {dataframe['open'].iloc[-1]} proposed_rate: {proposed_rate}"
        )

        if self.last_entry_price is not None and abs(entry_price - self.last_entry_price) < 0.0005:
            entry_price *= self.increment.value
            logger.info(
                f"{pair} Incremented entry price: {entry_price} based on previous entry price : {self.last_entry_price}."
            )

        self.last_entry_price = entry_price

        return entry_price

    def confirm_trade_exit(
            self,
            pair: str,
            trade: Trade,
            order_type: str,
            amount: float,
            rate: float,
            time_in_force: str,
            exit_reason: str,
            current_time: datetime,
            **kwargs,
    ) -> bool:
        if exit_reason == "partial_exit" and trade.calc_profit_ratio(rate) < 0:
            logger.info(f"{trade.pair} partial exit is below 0")
            self.dp.send_msg(f"{trade.pair} partial exit is below 0")
            return False
        if exit_reason == "trailing_stop_loss" and trade.calc_profit_ratio(rate) < 0:
            logger.info(f"{trade.pair} trailing stop price is below 0")
            self.dp.send_msg(f"{trade.pair} trailing stop price is below 0")
            return False
        return True

    def adjust_trade_position(
            self,
            trade: Trade,
            current_time: datetime,
            current_rate: float,
            current_profit: float,
            min_stake: Optional[float],
            max_stake: float,
            current_entry_rate: float,
            current_exit_rate: float,
            current_entry_profit: float,
            current_exit_profit: float,
            **kwargs,
    ) -> Optional[float]:
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries
        trade_duration = (current_time - trade.open_date_utc).seconds / 60

        if current_profit > 0.25 and trade.nr_of_successful_exits == 0:
            return -(trade.stake_amount / 4)
        if current_profit > 0.40 and trade.nr_of_successful_exits == 1:
            return -(trade.stake_amount / 3)

        if current_profit > -0.15 and trade.nr_of_successful_entries == 1:
            return None
        if current_profit > -0.3 and trade.nr_of_successful_entries == 2:
            return None
        if current_profit > -0.6 and trade.nr_of_successful_entries == 3:
            return None

        try:
            stake_amount = filled_entries[0].cost
            if count_of_entries == 1:
                stake_amount = stake_amount * 1
            elif count_of_entries == 2:
                stake_amount = stake_amount * 1
            elif count_of_entries == 3:
                stake_amount = stake_amount * 1
            else:
                stake_amount = stake_amount
            return stake_amount
        except Exception as exception:
            return None
        return None

    def leverage(
            self,
            pair: str,
            current_time: "datetime",
            current_rate: float,
            proposed_leverage: float,
            max_leverage: float,
            side: str,
            **kwargs,
    ) -> float:
        window_size = 50
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        historical_close_prices = dataframe["close"].tail(window_size)
        historical_high_prices = dataframe["high"].tail(window_size)
        historical_low_prices = dataframe["low"].tail(window_size)
        base_leverage = 10

        rsi_values = ta.RSI(historical_close_prices, timeperiod=14)
        atr_values = ta.ATR(
            historical_high_prices, historical_low_prices, historical_close_prices, timeperiod=14
        )
        macd_line, signal_line, _ = ta.MACD(
            historical_close_prices, fastperiod=12, slowperiod=26, signalperiod=9
        )
        sma_values = ta.SMA(historical_close_prices, timeperiod=20)
        current_rsi = rsi_values[-1] if len(rsi_values) > 0 else 50.0
        current_atr = atr_values[-1] if len(atr_values) > 0 else 0.0
        current_macd = (
            macd_line[-1] - signal_line[-1] if len(macd_line) > 0 and len(signal_line) > 0 else 0.0
        )
        current_sma = sma_values[-1] if len(sma_values) > 0 else 0.0

        dynamic_rsi_low = (
            np.nanmin(rsi_values)
            if len(rsi_values) > 0 and not np.isnan(np.nanmin(rsi_values))
            else 30.0
        )
        dynamic_rsi_high = (
            np.nanmax(rsi_values)
            if len(rsi_values) > 0 and not np.isnan(np.nanmax(rsi_values))
            else 70.0
        )
        dynamic_atr_low = (
            np.nanmin(atr_values)
            if len(atr_values) > 0 and not np.isnan(np.nanmin(atr_values))
            else 0.002
        )
        dynamic_atr_high = (
            np.nanmax(atr_values)
            if len(atr_values) > 0 and not np.isnan(np.nanmax(atr_values))
            else 0.005
        )

        long_increase_factor = 1.5
        long_decrease_factor = 0.5
        short_increase_factor = 1.5
        short_decrease_factor = 0.5
        volatility_decrease_factor = 0.8

        if side == "long":
            if current_rsi < dynamic_rsi_low:
                base_leverage *= long_increase_factor
            elif current_rsi > dynamic_rsi_high:
                base_leverage *= long_decrease_factor

            if current_atr > (current_rate * 0.03):
                base_leverage *= volatility_decrease_factor

            if current_macd > 0:
                base_leverage *= long_increase_factor
            if current_rate < current_sma:
                base_leverage *= long_decrease_factor

        elif side == "short":
            if current_rsi > dynamic_rsi_high:
                base_leverage *= short_increase_factor
            elif current_rsi < dynamic_rsi_low:
                base_leverage *= short_decrease_factor

            if current_atr > (current_rate * 0.03):
                base_leverage *= volatility_decrease_factor

            if current_macd < 0:
                base_leverage *= short_increase_factor
            if current_rate > current_sma:
                base_leverage *= short_decrease_factor

        adjusted_leverage = max(min(base_leverage, max_leverage), 1.0)

        return adjusted_leverage

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    @informative('15m')
    def populate_indicators_15m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe)
        dataframe["DI_values"] = ta.PLUS_DI(dataframe) - ta.MINUS_DI(dataframe)
        dataframe["DI_cutoff"] = 0

        maxima = np.zeros(len(dataframe))
        minima = np.zeros(len(dataframe))

        maxima[argrelextrema(dataframe["close"].values, np.greater, order=5)] = 1
        minima[argrelextrema(dataframe["close"].values, np.less, order=5)] = 1

        dataframe["maxima"] = maxima
        dataframe["minima"] = minima

        dataframe["&s-extrema"] = 0
        min_peaks = argrelextrema(dataframe["close"].values, np.less, order=5)[0]
        max_peaks = argrelextrema(dataframe["close"].values, np.greater, order=5)[0]
        dataframe.loc[min_peaks, "&s-extrema"] = -1
        dataframe.loc[max_peaks, "&s-extrema"] = 1

        dataframe["DI_catch"] = np.where(dataframe["DI_values"] > dataframe["DI_cutoff"], 0, 1)

        dataframe["minima_sort_threshold"] = dataframe["close"].rolling(window=10).min()
        dataframe["maxima_sort_threshold"] = dataframe["close"].rolling(window=10).max()

        dataframe["min_threshold_mean"] = dataframe["minima_sort_threshold"].expanding().mean()
        dataframe["max_threshold_mean"] = dataframe["maxima_sort_threshold"].expanding().mean()

        dataframe["maxima_check"] = (
            dataframe["maxima"].rolling(4).apply(lambda x: int((x != 1).all()), raw=True).fillna(0)
        )
        dataframe["minima_check"] = (
            dataframe["minima"].rolling(4).apply(lambda x: int((x != 1).all()), raw=True).fillna(0)
        )

        # Ichimoku componentsAdd commentMore actions
        ichi = ichimoku(dataframe, conversion_line_period=9, base_line_periods=26, laggin_span=52, displacement=26)

        dataframe['tenkan'] = ichi['tenkan_sen']
        dataframe['kijun'] = ichi['kijun_sen']
        dataframe['cloud_green'] = ichi['cloud_green']
        dataframe['cloud_red'] = ichi['cloud_red']

        # 15m RSI and Bollinger Bands on RSI
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=self.rsi_period)
        dataframe['rsi_ema'] = ta.EMA(dataframe['rsi'], timeperiod=self.bb_period)
        dataframe['rsi_std'] = dataframe['rsi'].rolling(window=self.bb_period).std()
        dataframe['rsi_upper'] = dataframe['rsi_ema'] + self.bb_stddev * dataframe['rsi_std']
        dataframe['rsi_lower'] = dataframe['rsi_ema'] - self.bb_stddev * dataframe['rsi_std']

        # 15m Dispersion range
        dispersion_range = (dataframe['rsi_upper'] - dataframe['rsi_lower']) * self.dispersion
        dataframe['rsi_disp_up'] = dataframe['rsi_ema'] + dispersion_range
        dataframe['rsi_disp_down'] = dataframe['rsi_ema'] - dispersion_range


        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe)
        dataframe["DI_values"] = ta.PLUS_DI(dataframe) - ta.MINUS_DI(dataframe)
        dataframe["DI_cutoff"] = 0

        maxima = np.zeros(len(dataframe))
        minima = np.zeros(len(dataframe))

        maxima[argrelextrema(dataframe["close"].values, np.greater, order=5)] = 1
        minima[argrelextrema(dataframe["close"].values, np.less, order=5)] = 1

        dataframe["maxima"] = maxima
        dataframe["minima"] = minima

        dataframe["&s-extrema"] = 0
        min_peaks = argrelextrema(dataframe["close"].values, np.less, order=5)[0]
        max_peaks = argrelextrema(dataframe["close"].values, np.greater, order=5)[0]
        dataframe.loc[min_peaks, "&s-extrema"] = -1
        dataframe.loc[max_peaks, "&s-extrema"] = 1

        dataframe["DI_catch"] = np.where(dataframe["DI_values"] > dataframe["DI_cutoff"], 0, 1)

        dataframe["minima_sort_threshold"] = dataframe["close"].rolling(window=10).min()
        dataframe["maxima_sort_threshold"] = dataframe["close"].rolling(window=10).max()

        dataframe["min_threshold_mean"] = dataframe["minima_sort_threshold"].expanding().mean()
        dataframe["max_threshold_mean"] = dataframe["maxima_sort_threshold"].expanding().mean()

        dataframe["maxima_check"] = (
            dataframe["maxima"].rolling(4).apply(lambda x: int((x != 1).all()), raw=True).fillna(0)
        )
        dataframe["minima_check"] = (
            dataframe["minima"].rolling(4).apply(lambda x: int((x != 1).all()), raw=True).fillna(0)
        )

        # Ichimoku componentsAdd commentMore actions
        ichi = ichimoku(dataframe, conversion_line_period=9, base_line_periods=26, laggin_span=52, displacement=26)

        dataframe['tenkan'] = ichi['tenkan_sen']
        dataframe['kijun'] = ichi['kijun_sen']
        dataframe['cloud_green'] = ichi['cloud_green']
        dataframe['cloud_red'] = ichi['cloud_red']

        # RSI and Bollinger Bands on RSI
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=self.rsi_period)
        dataframe['rsi_ema'] = ta.EMA(dataframe['rsi'], timeperiod=self.bb_period)
        dataframe['rsi_std'] = dataframe['rsi'].rolling(window=self.bb_period).std()
        dataframe['rsi_upper'] = dataframe['rsi_ema'] + self.bb_stddev * dataframe['rsi_std']
        dataframe['rsi_lower'] = dataframe['rsi_ema'] - self.bb_stddev * dataframe['rsi_std']

        # Dispersion range
        dispersion_range = (dataframe['rsi_upper'] - dataframe['rsi_lower']) * self.dispersion
        dataframe['rsi_disp_up'] = dataframe['rsi_ema'] + dispersion_range
        dataframe['rsi_disp_down'] = dataframe['rsi_ema'] - dispersion_range

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Only use 15m indicators for entries
        # Long entries
        df.loc[
            (
                qtpylib.crossed_above(df['rsi_15m'], df['rsi_disp_down_15m'])
                & (df['tenkan_15m'] > df['kijun_15m'])
                & (df['tenkan'] > df['kijun'])
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "Long - RSI + BB + Dispersion")

        # Short entries
        df.loc[
            (
                qtpylib.crossed_below(df['rsi_15m'], df['rsi_disp_up_15m'])
                & (df['tenkan_15m'] < df['kijun_15m'])
                & (df['tenkan'] < df['kijun'])
            ),
            ["enter_short", "enter_tag"],
        ] = (1, "Short - RSI + BB + Dispersion")
        
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Only use 5m indicators for exits
        df.loc[((df["maxima_check"] == 0) & (df["volume"] > 0)), ["exit_long", "exit_tag"]] = (
            1,
            "Maxima Check",
        )
        df.loc[
            (
                    (df["DI_catch"] == 1)
                    & (df["&s-extrema"] > 0)
                    & (df["maxima"].shift(1) == 1)
                    & (df["volume"] > 0)
            ),
            ["exit_long", "exit_tag"],
        ] = (1, "Maxima")
        df.loc[((df["maxima_check"] == 0) & (df["volume"] > 0)), ["exit_long", "exit_tag"]] = (
            1,
            "Maxima Full Send",
        )

        df.loc[((df["minima_check"] == 0) & (df["volume"] > 0)), ["exit_short", "exit_tag"]] = (
            1,
            "Minima Check",
        )
        df.loc[
            (
                    (df["DI_catch"] == 1)
                    & (df["&s-extrema"] < 0)
                    & (df["minima"].shift(1) == 1)
                    & (df["volume"] > 0)
            ),
            ["exit_short", "exit_tag"],
        ] = (1, "Minima")
        df.loc[((df["minima_check"] == 0) & (df["volume"] > 0)), ["exit_short", "exit_tag"]] = (
            1,
            "Minima Full Send",
        )

        return df