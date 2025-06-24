# 阿翔趋势交易系统2.1 (Axiang Trend Trading System 2.1)

基于多均线和ATR的趋势交易系统，用Python实现。

## 功能特点

- 使用DEMA144、DEMA169、EMA120和EMA200多均线系统
- 基于ATR的趋势识别
- 买入条件：趋势转为正向且价格突破DEMA144和DEMA169
- 卖出条件：趋势转为负向且价格跌破DEMA144和DEMA169
- 买入止损设置在DEMA169
- 卖出止损设置在DEMA144
- 风险收益比设为3:1
- 使用CCXT库获取币安交易所数据
- 支持时间转换为北京时间

## 安装依赖

```
pip install ccxt pandas numpy matplotlib pytz
```

## 使用方法

```python
from axiang_trend_system import AxiangTrendSystem

# 创建交易系统实例
trading_system = AxiangTrendSystem(
    symbol='BTC/USDT',  # 交易对
    timeframe='1h',     # 时间周期
    dema144_len=144,    # DEMA144均线长度
    dema169_len=169,    # DEMA169均线长度
    ema120_len=120,     # EMA120均线长度
    ema200_len=200,     # EMA200均线长度
    atr_period=34,      # ATR周期
    atr_multiplier=3.0  # ATR倍数
)

# 获取数据并计算指标
df = trading_system.run_strategy()

# 分析交易信号
trading_system.analyze_signals(df)

# 绘制策略图表
trading_system.plot_strategy(df)
```

## 参数说明

- `symbol`: 交易对，如 'BTC/USDT'
- `timeframe`: 时间周期，如 '1h', '4h', '1d'
- `dema144_len`: DEMA144均线周期
- `dema169_len`: DEMA169均线周期
- `ema120_len`: EMA120均线周期
- `ema200_len`: EMA200均线周期
- `atr_period`: ATR计算周期
- `atr_multiplier`: ATR倍数

## 策略逻辑

1. 使用ATR指标识别价格趋势
2. 当趋势由负转正且价格突破DEMA144和DEMA169时产生买入信号
3. 当趋势由正转负且价格跌破DEMA144和DEMA169时产生卖出信号
4. 买入止损设置在DEMA169，止盈为止损的3倍
5. 卖出止损设置在DEMA144，止盈为止损的3倍 