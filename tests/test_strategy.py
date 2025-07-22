import types
import sys

import pandas as pd
import numpy as np
import pytest

# Provide a lightweight stub for pandas_ta to avoid dependency on the real package
class _TA:
    @staticmethod
    def atr(high, low, close, length=14):
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(length).mean()

    @staticmethod
    def adx(high, low, close, length=14):
        adx_series = pd.Series(np.ones(len(close)) * 50, index=close.index)
        return pd.DataFrame({
            f'ADX_{length}': adx_series,
            f'DMP_{length}': adx_series,
            f'DMN_{length}': adx_series,
        })

ta_module = types.ModuleType('pandas_ta')
ta_module.atr = _TA.atr
ta_module.adx = _TA.adx
sys.modules['pandas_ta'] = ta_module

from indicators import OptimizedIndicatorCalculator
from signals import SignalGenerator
from trade_recorder import OptimizedTradeRecorder


def create_sample_data():
    return pd.DataFrame({
        'open': [1, 2, 3, 4],
        'high': [2, 3, 4, 5],
        'low': [0.5, 1.5, 2.5, 3.5],
        'close': [1.5, 2.5, 3.5, 4.5],
        'volume': [10, 10, 10, 10],
    })


def test_calculate_all_indicators_optimized_columns():
    config = {'indicators': {
        'dema144_len': 2,
        'dema169_len': 3,
        'atr_period': 2,
        'atr_multiplier': 1.0,
        'adx_period': 2,
    }}
    calc = OptimizedIndicatorCalculator(config)
    df = create_sample_data()
    result = calc.calculate_all_indicators_optimized(df)
    expected = {
        'dema144', 'dema169', 'atr', 'adx', 'plus_di', 'minus_di',
        'supertrend_upper', 'supertrend_lower',
        'supertrend_direction', 'supertrend_buy', 'supertrend_sell'
    }
    assert expected.issubset(result.columns)


def test_signal_generator_basic_signals():
    config = {'signals': {'strategy': 'Supertrend和DEMA策略'}}
    generator = SignalGenerator(config)
    df = pd.DataFrame({
        'close': [100, 110, 120, 115],
        'dema144': [90, 100, 110, 120],
        'dema169': [85, 95, 105, 115],
        'supertrend_buy': [False, True, False, False],
        'supertrend_sell': [False, False, True, False],
        'adx': [30, 30, 30, 30],
    })
    signals = generator.generate_signals(df)
    assert signals['buy_signal'].iloc[1]
    assert signals['sell_signal'].iloc[2]


def test_process_signals_vectorized_pnl():
    config = {'backtest': {
        'initial_capital': 10000.0,
        'leverage': 1.0,
        'risk_per_trade': 0.02,
    }}
    recorder = OptimizedTradeRecorder(config)
    df = pd.DataFrame({
        'open': [100, 110, 120, 115],
        'high': [101, 111, 121, 116],
        'low': [99, 109, 119, 114],
        'close': [100, 110, 120, 115],
        'volume': [10, 10, 10, 10],
        'buy_signal': [False, True, False, False],
        'sell_signal': [False, False, False, True],
        'dema144': [90, 100, 110, 120],
        'dema169': [85, 95, 105, 115],
    })
    result = recorder.process_signals_vectorized(df)
    entry_price = 110
    exit_price = 115
    stop_loss = 95
    risk_amount = 10000.0 * 0.02
    risk_per_unit = max(entry_price - stop_loss, entry_price * 0.01)
    position_size = risk_amount / risk_per_unit
    expected_pnl = (exit_price - entry_price) * position_size
    pnl = result.loc[result.index[3], 'trade_pnl']
    assert pytest.approx(pnl, rel=1e-6) == expected_pnl
    assert recorder.trades[0]['盈亏'] == pytest.approx(expected_pnl, rel=1e-6)

