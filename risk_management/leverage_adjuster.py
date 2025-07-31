"""
杠杆调整器
基于市场波动率自动调整杠杆倍数
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
from collections import deque


class VolatilityModel(Enum):
    """波动率模型类型"""
    SIMPLE = "simple"
    EWMA = "ewma"  # 指数加权移动平均
    GARCH = "garch"
    REALIZED = "realized"


@dataclass
class LeverageAdjustment:
    """杠杆调整结果"""
    current_leverage: float
    recommended_leverage: float
    adjustment_ratio: float
    volatility_estimate: float
    risk_level: str
    confidence: float
    metadata: Dict[str, Any]


class VolatilityEstimator:
    """波动率估计器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('VolatilityEstimator')
        
        # 波动率参数
        self.lookback_period = config.get('volatility_lookback', 20)
        self.ewma_alpha = config.get('ewma_alpha', 0.94)
        self.min_observations = config.get('min_observations', 10)
        
        # 历史数据存储
        self.price_history = {}
        self.volatility_history = {}
    
    def estimate_volatility(self, 
                          symbol: str, 
                          prices: Optional[List[float]] = None,
                          model: VolatilityModel = VolatilityModel.EWMA) -> float:
        """
        估计波动率
        
        Args:
            symbol: 交易品种
            prices: 价格序列
            model: 波动率模型
            
        Returns:
            float: 波动率估计值
        """
        try:
            if prices:
                self._update_price_history(symbol, prices)
            
            price_data = self.price_history.get(symbol, [])
            if len(price_data) < self.min_observations:
                return self.config.get('default_volatility', 0.02)
            
            if model == VolatilityModel.SIMPLE:
                return self._simple_volatility(price_data)
            elif model == VolatilityModel.EWMA:
                return self._ewma_volatility(price_data)
            elif model == VolatilityModel.REALIZED:
                return self._realized_volatility(price_data)
            else:
                return self._ewma_volatility(price_data)
                
        except Exception as e:
            self.logger.error(f"波动率估计失败 {symbol}: {e}")
            return self.config.get('default_volatility', 0.02)
    
    def _simple_volatility(self, prices: List[float]) -> float:
        """简单波动率计算"""
        if len(prices) < 2:
            return 0.02
        
        returns = np.diff(np.log(prices))
        return np.std(returns) * np.sqrt(252)  # 年化波动率
    
    def _ewma_volatility(self, prices: List[float]) -> float:
        """指数加权移动平均波动率"""
        if len(prices) < 2:
            return 0.02
        
        returns = np.diff(np.log(prices))
        
        # EWMA计算
        variance = returns[0] ** 2
        for ret in returns[1:]:
            variance = self.ewma_alpha * variance + (1 - self.ewma_alpha) * ret ** 2
        
        return np.sqrt(variance * 252)  # 年化波动率
    
    def _realized_volatility(self, prices: List[float]) -> float:
        """已实现波动率"""
        if len(prices) < 2:
            return 0.02
        
        returns = np.diff(np.log(prices))
        realized_var = np.sum(returns ** 2)
        
        return np.sqrt(realized_var * 252 / len(returns))
    
    def _update_price_history(self, symbol: str, prices: List[float]):
        """更新价格历史"""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.lookback_period * 2)
        
        for price in prices:
            self.price_history[symbol].append(price)
    
    def get_volatility_percentile(self, symbol: str, current_vol: float) -> float:
        """获取当前波动率的历史分位数"""
        vol_history = self.volatility_history.get(symbol, [])
        if not vol_history:
            return 0.5  # 默认50%分位数
        
        return np.percentile(vol_history, current_vol * 100)


class LeverageAdjuster:
    """
    杠杆调整器
    
    核心功能：
    - 基于波动率的动态杠杆调整
    - 市场状态识别
    - 风险等级评估
    - 杠杆限制管理
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化杠杆调整器
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger('LeverageAdjuster')
        
        # 初始化波动率估计器
        self.volatility_estimator = VolatilityEstimator(config)
        
        # 杠杆参数
        self.base_leverage = config.get('base_leverage', 3.0)
        self.max_leverage = config.get('max_leverage', 10.0)
        self.min_leverage = config.get('min_leverage', 1.0)
        
        # 波动率阈值
        self.low_vol_threshold = config.get('low_vol_threshold', 0.01)
        self.medium_vol_threshold = config.get('medium_vol_threshold', 0.03)
        self.high_vol_threshold = config.get('high_vol_threshold', 0.05)
        
        # 调整参数
        self.adjustment_speed = config.get('adjustment_speed', 0.5)
        self.volatility_target = config.get('volatility_target', 0.02)
        self.max_adjustment_ratio = config.get('max_adjustment_ratio', 2.0)
        
        # 状态跟踪
        self.current_leverages = {}
        self.volatility_history = {}
        self.adjustment_history = {}
        
        self.logger.info("杠杆调整器初始化完成")
    
    def adjust_leverage(self,
                       symbol: str,
                       current_leverage: float,
                       market_data: Optional[Dict[str, Any]] = None,
                       position_data: Optional[Dict[str, Any]] = None) -> LeverageAdjustment:
        """
        调整杠杆倍数
        
        Args:
            symbol: 交易品种
            current_leverage: 当前杠杆
            market_data: 市场数据
            position_data: 仓位数据
            
        Returns:
            LeverageAdjustment: 杠杆调整结果
        """
        try:
            # 估计当前波动率
            volatility = self._get_current_volatility(symbol, market_data)
            
            # 评估风险等级
            risk_level = self._assess_risk_level(volatility, market_data)
            
            # 计算推荐杠杆
            recommended_leverage = self._calculate_recommended_leverage(
                volatility, risk_level, current_leverage
            )
            
            # 应用调整限制
            recommended_leverage = self._apply_adjustment_limits(
                current_leverage, recommended_leverage
            )
            
            # 计算调整比例
            adjustment_ratio = recommended_leverage / current_leverage if current_leverage > 0 else 1.0
            
            # 评估调整置信度
            confidence = self._calculate_adjustment_confidence(
                volatility, risk_level, market_data
            )
            
            # 更新历史记录
            self._update_adjustment_history(symbol, current_leverage, recommended_leverage, volatility)
            
            result = LeverageAdjustment(
                current_leverage=current_leverage,
                recommended_leverage=recommended_leverage,
                adjustment_ratio=adjustment_ratio,
                volatility_estimate=volatility,
                risk_level=risk_level,
                confidence=confidence,
                metadata={
                    'volatility_percentile': self.volatility_estimator.get_volatility_percentile(symbol, volatility),
                    'market_conditions': market_data.get('market_state', 'unknown') if market_data else 'unknown',
                    'adjustment_reason': self._get_adjustment_reason(volatility, risk_level)
                }
            )
            
            self.logger.debug(f"杠杆调整完成: {symbol} {current_leverage:.1f}x -> {recommended_leverage:.1f}x "
                            f"波动率={volatility:.2%} 风险={risk_level}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"杠杆调整失败 {symbol}: {e}")
            return self._create_error_result(current_leverage, f"调整错误: {e}")
    
    def _get_current_volatility(self, symbol: str, market_data: Optional[Dict[str, Any]]) -> float:
        """获取当前波动率"""
        if market_data and 'volatility' in market_data:
            return market_data['volatility']
        
        if market_data and 'prices' in market_data:
            return self.volatility_estimator.estimate_volatility(symbol, market_data['prices'])
        
        # 使用历史估计或默认值
        return self.volatility_estimator.estimate_volatility(symbol)
    
    def _assess_risk_level(self, volatility: float, market_data: Optional[Dict[str, Any]]) -> str:
        """评估风险等级"""
        # 基于波动率的基本评估
        if volatility <= self.low_vol_threshold:
            base_risk = "低风险"
        elif volatility <= self.medium_vol_threshold:
            base_risk = "中风险"
        elif volatility <= self.high_vol_threshold:
            base_risk = "高风险"
        else:
            base_risk = "极高风险"
        
        # 考虑市场状态调整
        if market_data:
            market_state = market_data.get('market_state', 'normal')
            
            if market_state in ['crash', 'panic']:
                return "极高风险"
            elif market_state == 'volatile':
                return "高风险" if base_risk in ["低风险", "中风险"] else base_risk
            elif market_state == 'trending':
                return base_risk
        
        return base_risk
    
    def _calculate_recommended_leverage(self, 
                                      volatility: float, 
                                      risk_level: str, 
                                      current_leverage: float) -> float:
        """计算推荐杠杆"""
        # 基于波动率的目标杠杆
        if self.volatility_target > 0:
            volatility_based_leverage = self.base_leverage * (self.volatility_target / volatility)
        else:
            volatility_based_leverage = self.base_leverage
        
        # 基于风险等级的杠杆调整
        risk_multipliers = {
            "低风险": 1.2,
            "中风险": 1.0,
            "高风险": 0.7,
            "极高风险": 0.4
        }
        
        risk_multiplier = risk_multipliers.get(risk_level, 1.0)
        risk_adjusted_leverage = volatility_based_leverage * risk_multiplier
        
        # 应用杠杆限制
        recommended_leverage = max(self.min_leverage, 
                                 min(self.max_leverage, risk_adjusted_leverage))
        
        # 平滑调整（避免剧烈变化）
        if current_leverage > 0:
            adjustment = (recommended_leverage - current_leverage) * self.adjustment_speed
            recommended_leverage = current_leverage + adjustment
        
        return recommended_leverage
    
    def _apply_adjustment_limits(self, current_leverage: float, recommended_leverage: float) -> float:
        """应用调整限制"""
        if current_leverage <= 0:
            return recommended_leverage
        
        # 单次调整比例限制
        max_ratio = self.max_adjustment_ratio
        min_ratio = 1.0 / max_ratio
        
        adjustment_ratio = recommended_leverage / current_leverage
        
        if adjustment_ratio > max_ratio:
            recommended_leverage = current_leverage * max_ratio
        elif adjustment_ratio < min_ratio:
            recommended_leverage = current_leverage * min_ratio
        
        # 确保在绝对限制范围内
        return max(self.min_leverage, min(self.max_leverage, recommended_leverage))
    
    def _calculate_adjustment_confidence(self, 
                                       volatility: float, 
                                       risk_level: str, 
                                       market_data: Optional[Dict[str, Any]]) -> float:
        """计算调整置信度"""
        confidence = 0.8  # 基础置信度
        
        # 基于数据质量调整
        if market_data:
            if 'data_quality' in market_data:
                confidence *= market_data['data_quality']
            
            if 'market_state_confidence' in market_data:
                confidence *= market_data['market_state_confidence']
        
        # 基于波动率稳定性调整
        if volatility > self.high_vol_threshold:
            confidence *= 0.8  # 高波动率降低置信度
        
        # 基于风险等级调整
        if risk_level == "极高风险":
            confidence *= 0.9  # 极端情况下稍微降低置信度
        
        return max(0.1, min(1.0, confidence))
    
    def _get_adjustment_reason(self, volatility: float, risk_level: str) -> str:
        """获取调整原因"""
        if volatility > self.high_vol_threshold:
            return "高波动率环境"
        elif volatility < self.low_vol_threshold:
            return "低波动率环境"
        elif risk_level == "极高风险":
            return "极高风险市场状态"
        elif risk_level == "高风险":
            return "高风险市场条件"
        else:
            return "正常市场调整"
    
    def _update_adjustment_history(self, 
                                 symbol: str, 
                                 current_leverage: float, 
                                 recommended_leverage: float, 
                                 volatility: float):
        """更新调整历史"""
        if symbol not in self.adjustment_history:
            self.adjustment_history[symbol] = deque(maxlen=100)
        
        self.adjustment_history[symbol].append({
            'timestamp': datetime.now(),
            'current_leverage': current_leverage,
            'recommended_leverage': recommended_leverage,
            'volatility': volatility
        })
        
        self.current_leverages[symbol] = recommended_leverage
    
    def _create_error_result(self, current_leverage: float, error_message: str) -> LeverageAdjustment:
        """创建错误结果"""
        return LeverageAdjustment(
            current_leverage=current_leverage,
            recommended_leverage=current_leverage,
            adjustment_ratio=1.0,
            volatility_estimate=0.02,
            risk_level="未知",
            confidence=0.0,
            metadata={'error': error_message}
        )
    
    def get_leverage_statistics(self, symbol: str) -> Dict[str, Any]:
        """获取杠杆统计信息"""
        history = self.adjustment_history.get(symbol, [])
        
        if not history:
            return {'error': '无历史数据'}
        
        leverages = [h['recommended_leverage'] for h in history]
        volatilities = [h['volatility'] for h in history]
        
        return {
            'current_leverage': self.current_leverages.get(symbol, 0),
            'average_leverage': np.mean(leverages),
            'leverage_std': np.std(leverages),
            'min_leverage': np.min(leverages),
            'max_leverage': np.max(leverages),
            'average_volatility': np.mean(volatilities),
            'volatility_std': np.std(volatilities),
            'adjustment_count': len(history),
            'last_adjustment': history[-1]['timestamp'] if history else None
        }
    
    def update_market_data(self, symbol: str, prices: List[float], market_state: Optional[str] = None):
        """更新市场数据"""
        # 更新波动率估计器的数据
        self.volatility_estimator._update_price_history(symbol, prices)
        
        # 估计并存储新的波动率
        volatility = self.volatility_estimator.estimate_volatility(symbol)
        
        if symbol not in self.volatility_history:
            self.volatility_history[symbol] = deque(maxlen=100)
        
        self.volatility_history[symbol].append({
            'timestamp': datetime.now(),
            'volatility': volatility,
            'market_state': market_state
        })
    
    def should_adjust_leverage(self, 
                             symbol: str, 
                             current_leverage: float, 
                             threshold: float = 0.1) -> bool:
        """判断是否需要调整杠杆"""
        adjustment = self.adjust_leverage(symbol, current_leverage)
        
        if adjustment.confidence < 0.5:
            return False
        
        adjustment_magnitude = abs(adjustment.adjustment_ratio - 1.0)
        return adjustment_magnitude > threshold