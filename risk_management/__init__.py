"""
高级风险管理系统模块

该模块实现了智能化交易系统的高级风险管理功能，包括：
- RiskManager: 风险管理主控制器
- PositionSizer: 动态头寸规模计算器
- LeverageAdjuster: 杠杆调整器（基于波动率）
- RiskMetrics: 风险指标计算器
- AdaptiveStopLoss: 自适应止损系统
- DrawdownController: 回撤控制系统

使用示例:
    from risk_management import RiskManager, PositionSizer
    
    config = {
        'max_daily_loss': 1000,
        'max_drawdown': 0.1,
        'position_size_factor': 0.02
    }
    
    risk_manager = RiskManager(config)
    position_sizer = PositionSizer(config)
"""

# 导入核心类
from .risk_manager import RiskManager, RiskEvent, RiskLevel
from .position_sizer import PositionSizer, PositionSizeCalculator, KellyCalculator
from .leverage_adjuster import LeverageAdjuster, VolatilityEstimator
from .risk_metrics import RiskMetrics, VaRCalculator, DrawdownCalculator
from .adaptive_stop_loss import AdaptiveStopLoss, StopLossStrategy
from .drawdown_controller import DrawdownController, DrawdownAlert

# 版本信息
__version__ = "1.0.0"
__author__ = "BTC Trading System"

# 导出的公共接口
__all__ = [
    # 核心组件
    'RiskManager',
    'PositionSizer', 
    'LeverageAdjuster',
    'RiskMetrics',
    'AdaptiveStopLoss',
    'DrawdownController',
    
    # 枚举和事件
    'RiskEvent',
    'RiskLevel',
    'StopLossStrategy',
    'DrawdownAlert',
    
    # 计算器类
    'PositionSizeCalculator',
    'KellyCalculator',
    'VolatilityEstimator',
    'VaRCalculator',
    'DrawdownCalculator'
]