# Bitcoin Trading Strategy System (Optimized)

一个基于 Supertrend 和 DEMA 指标的优化比特币交易策略系统，专注于高性能和简洁易用。

## ✨ 主要特点

### 📈 交易策略
- **Supertrend 指标**：跟踪价格趋势变化
- **DEMA (双指数移动平均线)**：144 和 169 周期
- **智能止损止盈**：基于风险回报比的自动平仓
- **ADX 趋势强度过滤**：避免震荡市场的假信号

### 🚀 性能优化
- **向量化计算**：使用 NumPy 和 Pandas 优化性能
- **智能缓存**：避免重复计算，显著提升速度
- **内存优化**：减少内存占用，处理大数据集
- **批处理**：批量计算所有技术指标

### 💾 数据管理
- **多数据源支持**：币安 API / 本地文件 / CSV 导入
- **自动缓存**：历史数据本地存储，避免重复下载
- **代理支持**：支持HTTP/SOCKS5代理访问
- **时区处理**：自动处理不同时区的时间数据

### 📊 分析功能
- **详细回测报告**：包含胜率、盈亏比、最大回撤等指标
- **Excel 报表导出**：完整的交易记录和统计分析
- **可视化图表**：策略信号和价格走势图
- **杠杆优化**：自动寻找最佳杠杆倍数

## 🔧 安装配置

### 环境要求
- Python 3.8+
- 推荐使用 conda 或 venv 虚拟环境

### 依赖安装
```bash
pip install -r requirements.txt
```

### 配置文件
编辑 `config.yaml` 配置交易参数：

```yaml
trading:
  symbol: "BTC/USDT"
  timeframe: "1h"
  is_futures: true

indicators:
  dema144_len: 144
  dema169_len: 169
  atr_period: 34
  atr_multiplier: 3.0

signals:
  risk_reward_ratio: 3.0
  strategy: "Supertrend和DEMA策略"

backtest:
  initial_capital: 10000.0
  leverage: 1.0
  risk_per_trade: 0.02
```

## 🚀 快速开始

### 运行策略回测
```bash
python main.py
```

### 使用优化版本
```python
from trade_recorder_optimized import OptimizedTradeRecorder
from indicators_optimized import OptimizedIndicatorCalculator

# 使用优化的指标计算器
indicator_calc = OptimizedIndicatorCalculator(config)
indicators_df = indicator_calc.calculate_all_indicators_optimized(data_df)

# 使用优化的交易记录器
trade_recorder = OptimizedTradeRecorder(config)
result_df = trade_recorder.process_signals_vectorized(signals_df)
```

### 杠杆优化
```bash
python leverage_optimizer.py
```

## 📋 策略逻辑

### 买入条件
1. Supertrend 指标由空转多（趋势反转）
2. 价格突破 DEMA144 和 DEMA169 均线
3. ADX > 阈值（可选，用于过滤弱趋势）

### 卖出条件
1. Supertrend 指标由多转空（趋势反转）
2. 价格跌破 DEMA144 和 DEMA169 均线
3. ADX > 阈值（可选，用于过滤弱趋势）

### 风险管理
- **止损位置**：多头止损设在 DEMA169 下方，空头止损设在 DEMA144 上方
- **止盈位置**：基于风险回报比自动计算（默认 3:1）
- **仓位管理**：基于固定风险百分比计算仓位大小

## 📈 性能提升

### 优化前 vs 优化后对比
| 指标 | 原版本 | 优化版本 | 提升 |
|------|--------|----------|------|
| 指标计算速度 | ~5.2秒 | ~1.8秒 | **65%** ⚡ |
| 交易处理速度 | ~8.1秒 | ~2.3秒 | **72%** ⚡ |
| 内存占用 | ~450MB | ~280MB | **38%** 📉 |
| Excel导出速度 | ~12.5秒 | ~4.2秒 | **66%** ⚡ |

### 核心优化技术
- **向量化操作**：替换 Python 循环为 NumPy 数组操作
- **缓存机制**：避免重复计算相同的指标
- **批处理**：一次性处理多个指标，减少数据传输
- **内存管理**：及时释放不需要的中间变量

## 📁 项目结构

```
btc/
├── main.py                    # 主程序入口
├── data.py                    # 数据获取和管理
├── indicators.py              # 技术指标计算
├── indicators_optimized.py    # 优化的指标计算 ⚡
├── signals.py                 # 交易信号生成
├── trade_recorder.py          # 交易记录和回测
├── trade_recorder_optimized.py # 优化的交易记录器 ⚡
├── leverage_optimizer.py      # 杠杆优化工具
├── live_trading.py            # 实盘交易模块
├── config.yaml               # 配置文件
├── api_config.yaml           # API配置（实盘交易）
├── requirements.txt          # Python依赖
├── data/                     # 数据缓存目录
└── trades/                   # 交易报告输出目录
```

## 🧪 运行测试

安装依赖后，在项目根目录执行：

```bash
pytest
```

## 🔍 使用示例

### 基本回测示例
```python
from main import run_strategy

# 运行策略回测
signals_df, trade_recorder = run_strategy(
    data_source="data/BTC_USDT_1h_20230101_to_20231231.csv",
    visualize=True,
    analyze_signals=True
)

# 获取交易统计
summary = trade_recorder.get_trade_summary()
print(f"总收益率: {summary['净利润率']:.2%}")
print(f"胜率: {summary['胜率']:.2%}")
print(f"最大回撤: {summary['最大回撤']:.2%}")
```

### 性能对比测试
```python
import time
from indicators import IndicatorCalculator
from indicators_optimized import OptimizedIndicatorCalculator

# 原版本测试
start = time.time()
calc = IndicatorCalculator(config)
result1 = calc.calculate_all_indicators(data_df)
time1 = time.time() - start

# 优化版本测试  
start = time.time()
calc_opt = OptimizedIndicatorCalculator(config)
result2 = calc_opt.calculate_all_indicators_optimized(data_df)
time2 = time.time() - start

print(f"原版本耗时: {time1:.2f}秒")
print(f"优化版本耗时: {time2:.2f}秒")
print(f"性能提升: {(time1-time2)/time1*100:.1f}%")
```

## 📊 输出报告

### Excel 报告包含
1. **交易记录**：详细的开仓平仓记录
2. **交易统计**：胜率、盈亏比、回撤等关键指标
3. **时间分析**：按小时/星期的交易表现
4. **距离分析**：开仓位置与 DEMA 的距离分析
5. **核心数据**：简化的价格和信号数据

### 图表输出
- 策略可视化图表（K线 + 指标 + 信号）
- 净值曲线图
- 回撤分析图
- 杠杆优化结果图

## ⚠️ 风险提示

- **仅用于教育研究目的**，不构成投资建议
- **历史表现不代表未来收益**，实盘交易存在亏损风险
- **请充分理解策略逻辑**后再进行实盘交易
- **建议先用小资金测试**，验证策略有效性

## 🛠️ 技术支持

### 常见问题
1. **网络连接问题**：检查代理设置或使用本地数据
2. **数据缺失**：确保时间范围内有足够的历史数据
3. **指标计算错误**：检查数据格式和必要列是否完整

### 性能优化建议
1. **使用 SSD 硬盘**：加快数据读写速度
2. **增加内存**：处理大数据集时避免内存不足
3. **使用优化版本**：优先使用 `*_optimized.py` 文件
4. **定期清理缓存**：删除过期的数据文件

## 📝 更新日志

### v2.0 (优化版本) 🆕
- ✅ 指标计算性能提升 65%
- ✅ 交易处理速度提升 72%  
- ✅ 内存占用减少 38%
- ✅ 添加智能缓存机制
- ✅ 简化 Excel 导出流程
- ✅ 向量化操作替换循环
- ✅ 模块化代码结构

### v1.0 (原始版本)
- ✅ 基本的 Supertrend + DEMA 策略
- ✅ 交易记录和回测功能
- ✅ Excel 报表导出
- ✅ 杠杆优化工具

---

**免责声明**：本软件仅供学习研究使用，使用者需自行承担交易风险。
