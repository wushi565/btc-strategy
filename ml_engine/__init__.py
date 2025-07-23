"""
机器学习驱动的交易信号增强模块

该模块实现了需求1：机器学习驱动的交易信号增强
- 特征工程：价格模式、技术指标、交易量分析、波动率特征
- 模型集成：随机森林、XGBoost、LSTM网络、Transformer等
- 信号生成：方向概率和强度评分
- 在线学习：模型持续更新和性能监控
"""

from .ml_signal_enhancer import MLSignalEnhancer
from .features.feature_engineering import FeatureEngineering
from .training.model_trainer import ModelTrainer
from .models.signal_predictor import SignalPredictor
from .models.model_ensemble import ModelEnsemble
from .utils.model_version_manager import ModelVersionManager

__version__ = "1.0.0"
__all__ = [
    'MLSignalEnhancer',
    'FeatureEngineering', 
    'ModelTrainer',
    'SignalPredictor',
    'ModelEnsemble',
    'ModelVersionManager'
]