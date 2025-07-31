"""
动态头寸规模计算器
基于Kelly公式变体和风险管理原则动态计算最优仓位大小
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta


class PositionSizeMethod(Enum):
    """仓位计算方法"""
    FIXED_PERCENTAGE = "fixed_percentage"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    VAR_BASED = "var_based"
    ADAPTIVE = "adaptive"


@dataclass
class PositionSizeResult:
    """仓位大小计算结果"""
    position_size: float
    risk_amount: float
    confidence: float
    method_used: PositionSizeMethod
    metadata: Dict[str, Any]


class KellyCalculator:
    """Kelly准则计算器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('KellyCalculator')
        
        # Kelly公式参数
        self.lookback_period = config.get('kelly_lookback_period', 30)
        self.max_kelly_fraction = config.get('max_kelly_fraction', 0.25)
        self.min_kelly_fraction = config.get('min_kelly_fraction', 0.01)
    
    def calculate_kelly_fraction(self, historical_returns: List[float]) -> float:
        """
        计算Kelly分数
        
        Kelly = (bp - q) / b
        其中:
        b = 赔率 (平均盈利/平均亏损)
        p = 胜率
        q = 败率 (1-p)
        """
        if not historical_returns or len(historical_returns) < 10:
            return self.min_kelly_fraction
        
        try:
            returns = np.array(historical_returns)
            
            # 计算胜率
            wins = returns[returns > 0]
            losses = returns[returns < 0]
            
            if len(wins) == 0 or len(losses) == 0:
                return self.min_kelly_fraction
            
            win_rate = len(wins) / len(returns)
            loss_rate = 1 - win_rate
            
            # 计算平均盈利和亏损
            avg_win = np.mean(wins)
            avg_loss = abs(np.mean(losses))
            
            if avg_loss == 0:
                return self.min_kelly_fraction
            
            # 计算赔率
            odds = avg_win / avg_loss
            
            # Kelly公式
            kelly_fraction = (odds * win_rate - loss_rate) / odds
            
            # 限制Kelly分数在合理范围内
            kelly_fraction = max(self.min_kelly_fraction, 
                               min(self.max_kelly_fraction, kelly_fraction))
            
            self.logger.debug(f"Kelly计算: 胜率={win_rate:.2%}, 赔率={odds:.2f}, Kelly={kelly_fraction:.2%}")
            
            return kelly_fraction
            
        except Exception as e:
            self.logger.error(f"Kelly分数计算失败: {e}")
            return self.min_kelly_fraction


class PositionSizeCalculator:
    """仓位大小计算器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('PositionSizeCalculator')
        
        # 基本参数
        self.account_value = config.get('account_value', 10000)
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.02)
        self.min_position_size = config.get('min_position_size', 0.001)
        self.max_position_size = config.get('max_position_size', 0.2)
    
    def calculate_fixed_percentage(self, risk_amount: float) -> float:
        """固定百分比仓位计算"""
        risk_percentage = self.config.get('fixed_risk_percentage', 0.02)
        return self.account_value * risk_percentage / risk_amount if risk_amount > 0 else 0
    
    def calculate_volatility_adjusted(self, volatility: float, base_size: float) -> float:
        """波动率调整仓位计算"""
        volatility_target = self.config.get('volatility_target', 0.02)
        adjustment_factor = volatility_target / volatility if volatility > 0 else 1.0
        
        # 限制调整范围
        adjustment_factor = max(0.5, min(2.0, adjustment_factor))
        
        return base_size * adjustment_factor
    
    def calculate_var_based(self, var_estimate: float, confidence_level: float = 0.95) -> float:
        """基于VaR的仓位计算"""
        var_target = self.account_value * self.max_risk_per_trade
        return var_target / var_estimate if var_estimate > 0 else 0


class PositionSizer:
    """
    动态头寸规模计算器
    
    核心功能：
    - 基于Kelly公式的最优仓位计算
    - 波动率调整
    - 风险预算管理
    - 多种计算方法集成
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化仓位计算器
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger('PositionSizer')
        
        # 初始化子组件
        self.kelly_calculator = KellyCalculator(config)
        self.base_calculator = PositionSizeCalculator(config)
        
        # 仓位计算参数
        self.default_method = PositionSizeMethod(
            config.get('default_position_method', 'adaptive')
        )
        self.account_value = config.get('account_value', 10000)
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.02)
        
        # 自适应参数
        self.volatility_lookback = config.get('volatility_lookback', 20)
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        
        # 历史数据存储
        self.historical_returns = {}
        self.volatility_estimates = {}
        
        self.logger.info("动态头寸规模计算器初始化完成")
    
    def calculate_position_size(self, 
                              symbol: str,
                              entry_price: float,
                              stop_loss: float,
                              signal_confidence: float = 0.8,
                              method: Optional[PositionSizeMethod] = None,
                              market_data: Optional[Dict[str, Any]] = None) -> PositionSizeResult:
        """
        计算动态仓位大小
        
        Args:
            symbol: 交易品种
            entry_price: 入场价格
            stop_loss: 止损价格
            signal_confidence: 信号置信度
            method: 计算方法
            market_data: 市场数据
            
        Returns:
            PositionSizeResult: 仓位计算结果
        """
        try:
            method = method or self.default_method
            
            # 计算风险金额
            risk_per_unit = abs(entry_price - stop_loss)
            if risk_per_unit <= 0:
                return self._create_error_result("止损价格无效")
            
            # 根据方法计算仓位
            if method == PositionSizeMethod.FIXED_PERCENTAGE:
                result = self._calculate_fixed_percentage_position(
                    symbol, risk_per_unit, signal_confidence
                )
            elif method == PositionSizeMethod.KELLY_CRITERION:
                result = self._calculate_kelly_position(
                    symbol, risk_per_unit, signal_confidence
                )
            elif method == PositionSizeMethod.VOLATILITY_ADJUSTED:
                result = self._calculate_volatility_adjusted_position(
                    symbol, risk_per_unit, signal_confidence, market_data
                )
            elif method == PositionSizeMethod.VAR_BASED:
                result = self._calculate_var_based_position(
                    symbol, risk_per_unit, signal_confidence, market_data
                )
            elif method == PositionSizeMethod.ADAPTIVE:
                result = self._calculate_adaptive_position(
                    symbol, risk_per_unit, signal_confidence, market_data
                )
            else:
                result = self._calculate_fixed_percentage_position(
                    symbol, risk_per_unit, signal_confidence
                )
            
            # 应用安全限制
            result = self._apply_safety_limits(result, signal_confidence)
            
            self.logger.debug(f"仓位计算完成: {symbol} 大小={result.position_size:.4f} "
                            f"风险={result.risk_amount:.2f} 方法={result.method_used.value}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"仓位计算失败 {symbol}: {e}")
            return self._create_error_result(f"计算错误: {e}")
    
    def _calculate_fixed_percentage_position(self, 
                                           symbol: str, 
                                           risk_per_unit: float, 
                                           confidence: float) -> PositionSizeResult:
        """固定百分比仓位计算"""
        risk_percentage = self.max_risk_per_trade * confidence
        risk_amount = self.account_value * risk_percentage
        position_size = risk_amount / risk_per_unit
        
        return PositionSizeResult(
            position_size=position_size,
            risk_amount=risk_amount,
            confidence=confidence,
            method_used=PositionSizeMethod.FIXED_PERCENTAGE,
            metadata={'risk_percentage': risk_percentage}
        )
    
    def _calculate_kelly_position(self, 
                                symbol: str, 
                                risk_per_unit: float, 
                                confidence: float) -> PositionSizeResult:
        """Kelly准则仓位计算"""
        # 获取历史收益率
        historical_returns = self.historical_returns.get(symbol, [])
        
        # 计算Kelly分数
        kelly_fraction = self.kelly_calculator.calculate_kelly_fraction(historical_returns)
        
        # 根据信号置信度调整Kelly分数
        adjusted_kelly = kelly_fraction * confidence
        
        # 计算仓位
        risk_amount = self.account_value * adjusted_kelly
        position_size = risk_amount / risk_per_unit
        
        return PositionSizeResult(
            position_size=position_size,
            risk_amount=risk_amount,
            confidence=confidence,
            method_used=PositionSizeMethod.KELLY_CRITERION,
            metadata={
                'kelly_fraction': kelly_fraction,
                'adjusted_kelly': adjusted_kelly,
                'historical_trades': len(historical_returns)
            }
        )
    
    def _calculate_volatility_adjusted_position(self, 
                                              symbol: str, 
                                              risk_per_unit: float, 
                                              confidence: float,
                                              market_data: Optional[Dict[str, Any]]) -> PositionSizeResult:
        """波动率调整仓位计算"""
        # 基础仓位（固定百分比）
        base_result = self._calculate_fixed_percentage_position(symbol, risk_per_unit, confidence)
        
        # 获取波动率
        if market_data and 'volatility' in market_data:
            current_volatility = market_data['volatility']
        else:
            current_volatility = self._estimate_volatility(symbol)
        
        # 波动率调整
        adjusted_size = self.base_calculator.calculate_volatility_adjusted(
            current_volatility, base_result.position_size
        )
        
        return PositionSizeResult(
            position_size=adjusted_size,
            risk_amount=base_result.risk_amount,
            confidence=confidence,
            method_used=PositionSizeMethod.VOLATILITY_ADJUSTED,
            metadata={
                'base_position_size': base_result.position_size,
                'volatility': current_volatility,
                'volatility_adjustment': adjusted_size / base_result.position_size if base_result.position_size > 0 else 1
            }
        )
    
    def _calculate_var_based_position(self, 
                                    symbol: str, 
                                    risk_per_unit: float, 
                                    confidence: float,
                                    market_data: Optional[Dict[str, Any]]) -> PositionSizeResult:
        """基于VaR的仓位计算"""
        # 估计VaR
        if market_data and 'var_estimate' in market_data:
            var_estimate = market_data['var_estimate']
        else:
            var_estimate = self._estimate_var(symbol, risk_per_unit)
        
        # 基于VaR计算仓位
        position_size = self.base_calculator.calculate_var_based(var_estimate)
        
        # 计算风险金额
        risk_amount = position_size * risk_per_unit
        
        return PositionSizeResult(
            position_size=position_size,
            risk_amount=risk_amount,
            confidence=confidence,
            method_used=PositionSizeMethod.VAR_BASED,
            metadata={
                'var_estimate': var_estimate,
                'var_confidence': 0.95
            }
        )
    
    def _calculate_adaptive_position(self, 
                                   symbol: str, 
                                   risk_per_unit: float, 
                                   confidence: float,
                                   market_data: Optional[Dict[str, Any]]) -> PositionSizeResult:
        """自适应仓位计算"""
        # 根据可用数据和信号质量选择最佳方法
        historical_returns = self.historical_returns.get(symbol, [])
        
        if len(historical_returns) >= 20 and confidence >= self.confidence_threshold:
            # 有足够历史数据且信号置信度高，使用Kelly准则
            return self._calculate_kelly_position(symbol, risk_per_unit, confidence)
        elif market_data and 'volatility' in market_data:
            # 有波动率数据，使用波动率调整
            return self._calculate_volatility_adjusted_position(symbol, risk_per_unit, confidence, market_data)
        else:
            # 默认使用固定百分比
            return self._calculate_fixed_percentage_position(symbol, risk_per_unit, confidence)
    
    def _apply_safety_limits(self, result: PositionSizeResult, confidence: float) -> PositionSizeResult:
        """应用安全限制"""
        # 最小/最大仓位限制
        min_size = self.config.get('min_position_size', 0.001)
        max_size = self.config.get('max_position_size', 0.2) * self.account_value
        
        # 根据置信度调整最大仓位
        confidence_adjusted_max = max_size * confidence
        
        original_size = result.position_size
        result.position_size = max(min_size, min(confidence_adjusted_max, result.position_size))
        
        # 如果仓位被调整，更新风险金额
        if result.position_size != original_size:
            adjustment_ratio = result.position_size / original_size if original_size > 0 else 0
            result.risk_amount *= adjustment_ratio
            result.metadata['size_adjusted'] = True
            result.metadata['adjustment_ratio'] = adjustment_ratio
        
        return result
    
    def _estimate_volatility(self, symbol: str) -> float:
        """估计波动率"""
        if symbol in self.volatility_estimates:
            return self.volatility_estimates[symbol]
        
        # 默认波动率估计
        default_volatility = self.config.get('default_volatility', 0.02)
        return default_volatility
    
    def _estimate_var(self, symbol: str, risk_per_unit: float) -> float:
        """估计VaR"""
        # 简单的VaR估计，基于波动率
        volatility = self._estimate_volatility(symbol)
        confidence_level = 0.95
        z_score = 1.645  # 95%置信度的Z分数
        
        return volatility * z_score * risk_per_unit
    
    def _create_error_result(self, error_message: str) -> PositionSizeResult:
        """创建错误结果"""
        return PositionSizeResult(
            position_size=0,
            risk_amount=0,
            confidence=0,
            method_used=PositionSizeMethod.FIXED_PERCENTAGE,
            metadata={'error': error_message}
        )
    
    def update_historical_data(self, symbol: str, returns: List[float], volatility: Optional[float] = None):
        """更新历史数据"""
        self.historical_returns[symbol] = returns[-self.kelly_calculator.lookback_period:]
        
        if volatility is not None:
            self.volatility_estimates[symbol] = volatility
    
    def get_position_stats(self, symbol: str) -> Dict[str, Any]:
        """获取仓位统计信息"""
        historical_returns = self.historical_returns.get(symbol, [])
        
        if not historical_returns:
            return {'error': '无历史数据'}
        
        returns = np.array(historical_returns)
        
        return {
            'trade_count': len(returns),
            'win_rate': len(returns[returns > 0]) / len(returns),
            'avg_return': np.mean(returns),
            'return_std': np.std(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            'max_return': np.max(returns),
            'min_return': np.min(returns),
            'kelly_fraction': self.kelly_calculator.calculate_kelly_fraction(historical_returns.tolist()),
            'estimated_volatility': self.volatility_estimates.get(symbol, 'N/A')
        }