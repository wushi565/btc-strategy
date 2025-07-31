"""
实时交易执行引擎模块

这个模块提供了完整的实时交易执行引擎，包括：
- LiveTradingEngine: 主交易引擎
- WebSocketDataManager: WebSocket数据管理
- OrderManagementSystem: 订单管理系统  
- PositionManager: 仓位管理器
- StateSynchronizer: 状态同步器
- TrailingStopManager: 追踪止损管理
- TradingLogger: 交易日志记录

使用示例:
    from live_trading import LiveTradingEngine, TradingSignal
    
    config = {
        'websocket': {...},
        'orders': {...},
        'positions': {...}
    }
    
    engine = LiveTradingEngine(config)
    await engine.start()
"""

# 导入核心类和枚举
from .engine import (
    LiveTradingEngine, 
    TradingSignal, 
    EngineState, 
    EmergencyLevel
)

from .order_management import (
    OrderManagementSystem,
    Order,
    OrderStatus,
    OrderType,
    OrderSide,
    TimeInForce
)

from .position_manager import (
    PositionManager,
    Position,
    PositionSide
)

from .websocket_manager import WebSocketDataManager
from .state_synchronizer import StateSynchronizer  
from .trailing_stop_manager import TrailingStopManager
from .trading_logger import TradingLogger

# 版本信息
__version__ = "1.0.0"
__author__ = "BTC Trading System"

# 导出的公共接口
__all__ = [
    # 核心引擎
    'LiveTradingEngine',
    'TradingSignal',
    'EngineState', 
    'EmergencyLevel',
    
    # 订单管理
    'OrderManagementSystem',
    'Order',
    'OrderStatus',
    'OrderType', 
    'OrderSide',
    'TimeInForce',
    
    # 仓位管理
    'PositionManager',
    'Position',
    'PositionSide',
    
    # 数据和同步
    'WebSocketDataManager',
    'StateSynchronizer',
    'TrailingStopManager',
    'TradingLogger'
]