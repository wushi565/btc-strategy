# 实时交易执行引擎 (需求2)

## 概述

实时交易执行引擎是智能化交易系统的核心组件，负责处理实时市场数据、管理订单生命周期、跟踪交易仓位，并提供完整的风险控制和紧急干预机制。

## 核心架构

### 主要组件

1. **LiveTradingEngine** - 主控制器
   - 事件驱动架构
   - 协调各子系统运行
   - 提供统一的交易接口

2. **WebSocketDataManager** - 实时数据管理
   - 管理与交易所的WebSocket连接
   - 处理实时市场数据流
   - 支持多交易所数据源

3. **OrderManagementSystem** - 订单管理系统
   - 完整的订单生命周期管理
   - 支持多种订单类型(市价、限价、止损等)
   - 订单状态实时同步

4. **PositionManager** - 仓位管理
   - 实时仓位计算和跟踪
   - 盈亏分析和风险评估
   - 支持多策略仓位管理

5. **StateSynchronizer** - 状态同步机制  
   - 确保本地状态与交易所一致
   - 定期同步账户信息
   - 异常状态检测和恢复

6. **TrailingStopManager** - 动态止损管理
   - 追踪止损功能
   - 动态调整止损价格
   - 保护已实现利润

7. **TradingLogger** - 交易日志系统
   - 结构化日志记录
   - 关键操作追踪
   - 便于审计和调试

## 核心特性

### 事件驱动架构
- 采用异步事件处理机制
- 高并发性能和低延迟响应
- 灵活的事件处理器注册

### 实时数据处理
- WebSocket实时市场数据接收
- 价格、订单簿、交易数据处理
- 自动重连和错误恢复

### 订单管理
- 支持市价单、限价单、止损单等多种类型
- 订单状态实时跟踪和更新
- 批量订单操作和取消

### 仓位跟踪
- 实时仓位计算和盈亏分析
- 多头/空头仓位管理
- 风险指标计算(夏普比率、最大回撤等)

### 风险控制
- 仓位大小限制
- 杠杆控制
- 动态止损和止盈
- 异常情况检测

### 紧急干预
- 手动交易暂停/恢复
- 紧急止损功能
- 全局交易停止
- 多级别紧急响应

## 使用方法

### 基本配置

```python
config = {
    'websocket': {
        'exchange': 'binance',
        'testnet': True,
        'auto_reconnect': True
    },
    'orders': {
        'exchange': 'binance',
        'testnet': True,
        'api_key': 'your_api_key',
        'api_secret': 'your_api_secret'
    },
    'positions': {
        'max_position_size': 1000000,
        'max_leverage': 10
    }
}
```

### 启动引擎

```python
from live_trading.engine import LiveTradingEngine

engine = LiveTradingEngine(config)
await engine.start()
```

### 提交交易信号

```python
from live_trading.engine import TradingSignal

signal = TradingSignal(
    symbol='BTCUSDT',
    action='BUY',
    price=50000.0,
    quantity=0.001,
    confidence=0.8,
    timestamp=datetime.now(),
    strategy_id='my_strategy'
)

await engine.submit_signal(signal)
```

### 事件处理

```python
async def on_order_filled(data):
    order = data.get('order')
    print(f"订单成交: {order.symbol} {order.side}")

engine.register_handler('order_filled', on_order_filled)
```

### 查询状态

```python
# 获取引擎状态
state = engine.get_state()

# 获取活跃仓位
positions = await engine.position_manager.get_active_positions()

# 获取活跃订单
orders = await engine.order_manager.get_active_orders()

# 获取性能指标
metrics = engine.get_performance_metrics()
```

### 紧急控制

```python
# 紧急停止所有交易
await engine.emergency_stop(EmergencyLevel.CRITICAL, "系统异常")

# 恢复交易
await engine.resume_trading()
```

## 事件类型

### 订单事件
- `order_submitted` - 订单已提交
- `order_filled` - 订单成交
- `order_cancelled` - 订单取消
- `order_error` - 订单错误

### 仓位事件  
- `position_updated` - 仓位更新
- `position_risk_warning` - 仓位风险警告
- `stop_loss_triggered` - 止损触发
- `take_profit_triggered` - 止盈触发

### 系统事件
- `connection_lost` - 连接丢失
- `system_error` - 系统错误
- `emergency_stop` - 紧急停止
- `risk_limit_breach` - 风险限制突破

## 风险管理

### 仓位控制
- 单个仓位最大价值限制
- 总体风险敞口控制
- 杠杆倍数限制

### 止损机制
- 固定止损价格设置
- 追踪止损动态调整
- 基于波动率的自适应止损

### 异常处理
- 网络连接异常自动重连
- 订单执行失败重试机制
- 数据异常检测和过滤

### 紧急响应
- 多级别紧急停止机制
- 自动风险控制触发
- 手动干预和覆盖功能

## 监控和日志

### 性能监控
- 订单执行延迟统计
- 连接质量监控
- 系统资源使用情况

### 日志记录
- 所有交易操作记录
- 系统状态变化日志
- 错误和异常日志
- 性能指标历史

### 数据导出
- 交易历史导出
- 仓位变化记录
- 风险指标报告

## 扩展性

### 多交易所支持
- 统一的交易所接口抽象
- 支持币安、OKX等主流交易所
- 便于添加新的交易所支持

### 策略集成
- 标准化的信号接口
- 支持多策略并行运行
- 策略性能隔离和跟踪

### 自定义扩展
- 可插拔的事件处理器
- 自定义风险控制规则
- 灵活的配置管理

## 注意事项

1. **测试环境**: 建议在测试网络环境中进行开发和测试
2. **API密钥**: 确保API密钥安全存储，使用环境变量
3. **风险控制**: 设置合理的仓位限制和止损参数
4. **监控告警**: 配置适当的监控和告警机制
5. **日志管理**: 定期清理和归档日志文件

## 依赖要求

```
asyncio
websockets
ccxt
pandas
numpy
logging
```

## 示例代码

完整的使用示例请参考 `example.py` 文件，包含：
- 引擎启动和配置
- 交易信号提交
- 事件处理示例
- 状态查询方法
- 紧急控制演示

---

*此模块实现了智能交易系统需求文档中的第2项核心功能，为后续的机器学习信号增强、风险管理等模块提供了稳定的执行基础。*