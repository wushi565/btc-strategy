"""
自适应止损系统
基于波动率和市场状态的动态止损调整
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Callable, Any
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
import asyncio


class StopLossType(Enum):
    """止损类型"""
    FIXED = "fixed"  # 固定止损
    TRAILING = "trailing"  # 追踪止损
    ATR_BASED = "atr_based"  # ATR基础止损
    VOLATILITY_BASED = "volatility_based"  # 波动率基础止损
    ADAPTIVE = "adaptive"  # 自适应止损


class MarketRegime(Enum):
    """市场状态"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class PositionSide(Enum):
    """仓位方向"""
    LONG = "long"
    SHORT = "short"


@dataclass
class StopLossLevel:
    """止损水平数据结构"""
    symbol: str
    position_side: PositionSide
    stop_price: float
    stop_type: StopLossType
    initial_price: float
    current_price: float
    risk_amount: float
    confidence: float  # 止损有效性信心度
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_distance_pct(self) -> float:
        """获取止损距离百分比"""
        if self.position_side == PositionSide.LONG:
            return (self.current_price - self.stop_price) / self.current_price
        else:
            return (self.stop_price - self.current_price) / self.current_price
    
    def get_risk_reward_ratio(self, target_price: float = None) -> float:
        """计算风险收益比"""
        if target_price is None:
            return 0.0
        
        if self.position_side == PositionSide.LONG:
            risk = self.current_price - self.stop_price
            reward = target_price - self.current_price
        else:
            risk = self.stop_price - self.current_price
            reward = self.current_price - target_price
        
        return reward / risk if risk > 0 else 0.0


@dataclass
class StopLossConfig:
    """止损配置"""
    symbol: str
    stop_type: StopLossType = StopLossType.ADAPTIVE
    min_stop_distance: float = 0.005  # 最小止损距离 0.5%
    max_stop_distance: float = 0.10   # 最大止损距离 10%
    atr_multiplier: float = 2.0       # ATR倍数
    volatility_lookback: int = 20     # 波动率回看期
    trailing_activation: float = 0.02 # 追踪止损激活阈值 2%
    trailing_distance: float = 0.01   # 追踪止损距离 1%
    reentry_threshold: float = 0.005  # 重新进入阈值 0.5%
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'stop_type': self.stop_type.value,
            'min_stop_distance': self.min_stop_distance,
            'max_stop_distance': self.max_stop_distance,
            'atr_multiplier': self.atr_multiplier,
            'volatility_lookback': self.volatility_lookback,
            'trailing_activation': self.trailing_activation,
            'trailing_distance': self.trailing_distance,
            'reentry_threshold': self.reentry_threshold
        }


class AdaptiveStopLoss:
    """
    自适应止损系统
    
    核心功能：
    - 基于波动率的动态止损调整
    - 市场状态识别和止损策略选择
    - 追踪止损和固定止损的智能切换
    - 止损有效性评估和历史分析
    - 风险管理集成
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化自适应止损系统
        
        Args:
            config: 系统配置
        """
        self.config = config
        self.logger = logging.getLogger('AdaptiveStopLoss')
        
        # 系统参数
        self.default_risk_per_trade = config.get('default_risk_per_trade', 0.02)
        self.volatility_window = config.get('volatility_window', 20)
        self.atr_window = config.get('atr_window', 14)
        self.market_regime_window = config.get('market_regime_window', 50)
        self.max_history_size = config.get('max_history_size', 10000)
        
        # 数据存储
        self.price_history: Dict[str, deque] = {}
        self.volume_history: Dict[str, deque] = {}
        self.stop_levels: Dict[str, StopLossLevel] = {}
        self.stop_configs: Dict[str, StopLossConfig] = {}
        
        # 技术指标缓存
        self.atr_cache: Dict[str, deque] = {}
        self.volatility_cache: Dict[str, deque] = {}
        self.market_regime_cache: Dict[str, MarketRegime] = {}
        
        # 历史记录
        self.stop_history: deque = deque(maxlen=1000)
        self.performance_stats: Dict[str, Any] = {}
        
        # 事件回调
        self.stop_triggered_callbacks: List[Callable] = []
        self.stop_updated_callbacks: List[Callable] = []
        
        self.logger.info("自适应止损系统初始化完成")
    
    def register_stop_triggered_callback(self, callback: Callable):
        """注册止损触发回调"""
        self.stop_triggered_callbacks.append(callback)
    
    def register_stop_updated_callback(self, callback: Callable):
        """注册止损更新回调"""
        self.stop_updated_callbacks.append(callback)
    
    def update_market_data(self, symbol: str, price: float, volume: float = 0, 
                          timestamp: datetime = None):
        """
        更新市场数据
        
        Args:
            symbol: 交易对符号
            price: 当前价格
            volume: 成交量
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # 初始化数据存储
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.max_history_size)
            self.volume_history[symbol] = deque(maxlen=self.max_history_size)
            self.atr_cache[symbol] = deque(maxlen=100)
            self.volatility_cache[symbol] = deque(maxlen=100)
        
        # 添加数据
        self.price_history[symbol].append({
            'price': price,
            'volume': volume,
            'timestamp': timestamp
        })
        
        # 更新技术指标
        self._update_technical_indicators(symbol)
        
        # 更新市场状态
        self._update_market_regime(symbol)
        
        # 更新止损水平
        if symbol in self.stop_levels:
            asyncio.create_task(self._update_stop_level(symbol, price))
    
    async def set_stop_loss(self, symbol: str, position_side: PositionSide, 
                           entry_price: float, position_size: float,
                           risk_amount: float = None, stop_config: StopLossConfig = None) -> StopLossLevel:
        """
        设置止损
        
        Args:
            symbol: 交易对符号
            position_side: 仓位方向
            entry_price: 入场价格
            position_size: 仓位大小
            risk_amount: 风险金额
            stop_config: 止损配置
            
        Returns:
            止损水平
        """
        try:
            # 使用默认配置或提供的配置
            if stop_config is None:
                stop_config = self.stop_configs.get(symbol, StopLossConfig(symbol))
            
            # 计算风险金额
            if risk_amount is None:
                risk_amount = position_size * self.default_risk_per_trade
            
            # 根据止损类型计算止损价格
            if stop_config.stop_type == StopLossType.ADAPTIVE:
                stop_price = await self._calculate_adaptive_stop(
                    symbol, position_side, entry_price, risk_amount, stop_config
                )
            elif stop_config.stop_type == StopLossType.ATR_BASED:
                stop_price = self._calculate_atr_stop(
                    symbol, position_side, entry_price, stop_config
                )
            elif stop_config.stop_type == StopLossType.VOLATILITY_BASED:
                stop_price = self._calculate_volatility_stop(
                    symbol, position_side, entry_price, stop_config
                )
            elif stop_config.stop_type == StopLossType.TRAILING:
                stop_price = self._calculate_trailing_stop(
                    symbol, position_side, entry_price, stop_config
                )
            else:  # FIXED
                stop_price = self._calculate_fixed_stop(
                    symbol, position_side, entry_price, risk_amount, position_size
                )
            
            # 创建止损水平
            stop_level = StopLossLevel(
                symbol=symbol,
                position_side=position_side,
                stop_price=stop_price,
                stop_type=stop_config.stop_type,
                initial_price=entry_price,
                current_price=entry_price,
                risk_amount=risk_amount,
                confidence=self._calculate_stop_confidence(symbol, stop_price, entry_price)
            )
            
            # 保存止损水平
            self.stop_levels[symbol] = stop_level
            self.stop_configs[symbol] = stop_config
            
            self.logger.info(f"设置止损: {symbol} {position_side.value} "
                           f"入场:{entry_price:.4f} 止损:{stop_price:.4f}")
            
            # 触发回调
            for callback in self.stop_updated_callbacks:
                try:
                    await callback(stop_level, 'created')
                except Exception as e:
                    self.logger.error(f"止损更新回调执行失败: {e}")
            
            return stop_level
            
        except Exception as e:
            self.logger.error(f"设置止损失败 {symbol}: {e}")
            raise
    
    async def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """
        检查是否触发止损
        
        Args:
            symbol: 交易对符号
            current_price: 当前价格
            
        Returns:
            是否触发止损
        """
        if symbol not in self.stop_levels:
            return False
        
        stop_level = self.stop_levels[symbol]
        triggered = False
        
        try:
            # 检查止损触发条件
            if stop_level.position_side == PositionSide.LONG:
                triggered = current_price <= stop_level.stop_price
            else:
                triggered = current_price >= stop_level.stop_price
            
            if triggered:
                await self._handle_stop_triggered(symbol, current_price)
                return True
            
            # 更新当前价格
            stop_level.current_price = current_price
            
            return False
            
        except Exception as e:
            self.logger.error(f"检查止损失败 {symbol}: {e}")
            return False
    
    async def _calculate_adaptive_stop(self, symbol: str, position_side: PositionSide,
                                     entry_price: float, risk_amount: float,
                                     config: StopLossConfig) -> float:
        """计算自适应止损价格"""
        # 获取市场状态
        market_regime = self.market_regime_cache.get(symbol, MarketRegime.SIDEWAYS)
        
        # 获取当前波动率
        current_volatility = self._get_current_volatility(symbol)
        
        # 获取ATR
        current_atr = self._get_current_atr(symbol)
        
        # 根据市场状态调整策略
        if market_regime == MarketRegime.HIGH_VOLATILITY:
            # 高波动率市场：使用更宽的止损
            multiplier = config.atr_multiplier * 1.5
        elif market_regime == MarketRegime.LOW_VOLATILITY:
            # 低波动率市场：使用更紧的止损
            multiplier = config.atr_multiplier * 0.7
        elif market_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            # 趋势市场：使用追踪止损逻辑
            multiplier = config.atr_multiplier * 1.2
        else:
            # 横盘市场：使用标准ATR
            multiplier = config.atr_multiplier
        
        # 计算基础止损距离
        if current_atr > 0:
            base_distance = current_atr * multiplier / entry_price
        else:
            base_distance = current_volatility * 2.0
        
        # 应用最小和最大距离限制
        distance = max(config.min_stop_distance, 
                      min(config.max_stop_distance, base_distance))
        
        # 计算止损价格
        if position_side == PositionSide.LONG:
            stop_price = entry_price * (1 - distance)
        else:
            stop_price = entry_price * (1 + distance)
        
        return stop_price
    
    def _calculate_atr_stop(self, symbol: str, position_side: PositionSide,
                           entry_price: float, config: StopLossConfig) -> float:
        """计算ATR基础止损价格"""
        atr = self._get_current_atr(symbol)
        
        if atr <= 0:
            # 如果没有ATR数据，使用固定百分比
            distance = config.min_stop_distance
        else:
            distance = (atr * config.atr_multiplier) / entry_price
            distance = max(config.min_stop_distance, 
                          min(config.max_stop_distance, distance))
        
        if position_side == PositionSide.LONG:
            return entry_price * (1 - distance)
        else:
            return entry_price * (1 + distance)
    
    def _calculate_volatility_stop(self, symbol: str, position_side: PositionSide,
                                  entry_price: float, config: StopLossConfig) -> float:
        """计算波动率基础止损价格"""
        volatility = self._get_current_volatility(symbol)
        
        if volatility <= 0:
            distance = config.min_stop_distance
        else:
            # 使用2倍波动率作为止损距离
            distance = volatility * 2.0
            distance = max(config.min_stop_distance, 
                          min(config.max_stop_distance, distance))
        
        if position_side == PositionSide.LONG:
            return entry_price * (1 - distance)
        else:
            return entry_price * (1 + distance)
    
    def _calculate_trailing_stop(self, symbol: str, position_side: PositionSide,
                               entry_price: float, config: StopLossConfig) -> float:
        """计算追踪止损价格"""
        # 追踪止损初始设置与固定止损类似
        distance = config.trailing_distance
        
        if position_side == PositionSide.LONG:
            return entry_price * (1 - distance)
        else:
            return entry_price * (1 + distance)
    
    def _calculate_fixed_stop(self, symbol: str, position_side: PositionSide,
                            entry_price: float, risk_amount: float, position_size: float) -> float:
        """计算固定止损价格"""
        # 基于风险金额计算止损距离
        if position_size > 0:
            stop_distance = risk_amount / position_size
            
            if position_side == PositionSide.LONG:
                return entry_price - stop_distance
            else:
                return entry_price + stop_distance
        else:
            # 如果没有仓位大小信息，使用默认百分比
            config = self.stop_configs.get(symbol, StopLossConfig(symbol))
            distance = config.min_stop_distance
            
            if position_side == PositionSide.LONG:
                return entry_price * (1 - distance)
            else:
                return entry_price * (1 + distance)
    
    async def _update_stop_level(self, symbol: str, current_price: float):
        """更新止损水平"""
        if symbol not in self.stop_levels:
            return
        
        stop_level = self.stop_levels[symbol]
        config = self.stop_configs.get(symbol, StopLossConfig(symbol))
        old_stop_price = stop_level.stop_price
        
        try:
            # 检查是否需要更新止损
            should_update = False
            new_stop_price = stop_level.stop_price
            
            if stop_level.stop_type == StopLossType.TRAILING:
                # 追踪止损逻辑
                if stop_level.position_side == PositionSide.LONG:
                    # 多头追踪止损：价格上涨时提高止损价格
                    if current_price > stop_level.initial_price * (1 + config.trailing_activation):
                        potential_stop = current_price * (1 - config.trailing_distance)
                        if potential_stop > stop_level.stop_price:
                            new_stop_price = potential_stop
                            should_update = True
                else:
                    # 空头追踪止损：价格下跌时降低止损价格
                    if current_price < stop_level.initial_price * (1 - config.trailing_activation):
                        potential_stop = current_price * (1 + config.trailing_distance)
                        if potential_stop < stop_level.stop_price:
                            new_stop_price = potential_stop
                            should_update = True
            
            elif stop_level.stop_type == StopLossType.ADAPTIVE:
                # 自适应止损：根据市场状态调整
                market_regime = self.market_regime_cache.get(symbol, MarketRegime.SIDEWAYS)
                
                # 在高波动率市场中放宽止损
                if market_regime == MarketRegime.HIGH_VOLATILITY:
                    current_distance = abs(current_price - stop_level.stop_price) / current_price
                    min_distance = config.min_stop_distance * 1.5
                    
                    if current_distance < min_distance:
                        if stop_level.position_side == PositionSide.LONG:
                            new_stop_price = current_price * (1 - min_distance)
                        else:
                            new_stop_price = current_price * (1 + min_distance)
                        should_update = True
            
            # 更新止损水平
            if should_update:
                stop_level.stop_price = new_stop_price
                stop_level.current_price = current_price
                stop_level.timestamp = datetime.now()
                stop_level.confidence = self._calculate_stop_confidence(
                    symbol, new_stop_price, current_price
                )
                
                self.logger.debug(f"更新止损: {symbol} {old_stop_price:.4f} -> {new_stop_price:.4f}")
                
                # 触发回调
                for callback in self.stop_updated_callbacks:
                    try:
                        await callback(stop_level, 'updated')
                    except Exception as e:
                        self.logger.error(f"止损更新回调执行失败: {e}")
            else:
                # 仅更新当前价格
                stop_level.current_price = current_price
                
        except Exception as e:
            self.logger.error(f"更新止损水平失败 {symbol}: {e}")
    
    async def _handle_stop_triggered(self, symbol: str, trigger_price: float):
        """处理止损触发"""
        if symbol not in self.stop_levels:
            return
        
        stop_level = self.stop_levels[symbol]
        
        try:
            # 记录止损触发历史
            trigger_record = {
                'symbol': symbol,
                'trigger_time': datetime.now(),
                'trigger_price': trigger_price,
                'stop_price': stop_level.stop_price,
                'initial_price': stop_level.initial_price,
                'position_side': stop_level.position_side,
                'stop_type': stop_level.stop_type,
                'loss_amount': self._calculate_loss_amount(stop_level, trigger_price)
            }
            
            self.stop_history.append(trigger_record)
            
            self.logger.warning(f"止损触发: {symbol} {stop_level.position_side.value} "
                              f"触发价:{trigger_price:.4f} 止损价:{stop_level.stop_price:.4f}")
            
            # 触发回调
            for callback in self.stop_triggered_callbacks:
                try:
                    await callback(stop_level, trigger_price)
                except Exception as e:
                    self.logger.error(f"止损触发回调执行失败: {e}")
            
            # 清除止损水平
            del self.stop_levels[symbol]
            
            # 更新性能统计
            self._update_performance_stats(trigger_record)
            
        except Exception as e:
            self.logger.error(f"处理止损触发失败 {symbol}: {e}")
    
    def _update_technical_indicators(self, symbol: str):
        """更新技术指标"""
        if len(self.price_history[symbol]) < self.atr_window:
            return
        
        try:
            # 获取最近的价格数据
            recent_data = list(self.price_history[symbol])[-self.atr_window:]
            prices = [data['price'] for data in recent_data]
            
            # 计算ATR
            if len(prices) >= 2:
                highs = prices[1:]  # 简化：使用价格作为高点
                lows = prices[1:]   # 简化：使用价格作为低点
                closes = prices[1:]
                prev_closes = prices[:-1]
                
                true_ranges = []
                for i in range(len(highs)):
                    tr1 = highs[i] - lows[i]
                    tr2 = abs(highs[i] - prev_closes[i])
                    tr3 = abs(lows[i] - prev_closes[i])
                    true_ranges.append(max(tr1, tr2, tr3))
                
                atr = np.mean(true_ranges) if true_ranges else 0
                self.atr_cache[symbol].append(atr)
            
            # 计算波动率
            if len(prices) >= self.volatility_window:
                returns = []
                for i in range(1, len(prices)):
                    returns.append((prices[i] - prices[i-1]) / prices[i-1])
                
                volatility = np.std(returns) if returns else 0
                self.volatility_cache[symbol].append(volatility)
                
        except Exception as e:
            self.logger.error(f"更新技术指标失败 {symbol}: {e}")
    
    def _update_market_regime(self, symbol: str):
        """更新市场状态"""
        if len(self.price_history[symbol]) < self.market_regime_window:
            return
        
        try:
            recent_data = list(self.price_history[symbol])[-self.market_regime_window:]
            prices = [data['price'] for data in recent_data]
            
            # 计算价格趋势
            start_price = prices[0]
            end_price = prices[-1]
            trend = (end_price - start_price) / start_price
            
            # 计算波动率
            returns = []
            for i in range(1, len(prices)):
                returns.append((prices[i] - prices[i-1]) / prices[i-1])
            
            volatility = np.std(returns) if len(returns) > 1 else 0
            
            # 判断市场状态
            if volatility > 0.03:  # 日波动率 > 3%
                regime = MarketRegime.HIGH_VOLATILITY
            elif volatility < 0.01:  # 日波动率 < 1%
                regime = MarketRegime.LOW_VOLATILITY
            elif trend > 0.05:  # 上涨超过5%
                regime = MarketRegime.TRENDING_UP
            elif trend < -0.05:  # 下跌超过5%
                regime = MarketRegime.TRENDING_DOWN
            else:
                regime = MarketRegime.SIDEWAYS
            
            self.market_regime_cache[symbol] = regime
            
        except Exception as e:
            self.logger.error(f"更新市场状态失败 {symbol}: {e}")
    
    def _get_current_volatility(self, symbol: str) -> float:
        """获取当前波动率"""
        if symbol not in self.volatility_cache or not self.volatility_cache[symbol]:
            return 0.02  # 默认2%波动率
        return self.volatility_cache[symbol][-1]
    
    def _get_current_atr(self, symbol: str) -> float:
        """获取当前ATR"""
        if symbol not in self.atr_cache or not self.atr_cache[symbol]:
            return 0.0
        return self.atr_cache[symbol][-1]
    
    def _calculate_stop_confidence(self, symbol: str, stop_price: float, 
                                 current_price: float) -> float:
        """计算止损有效性信心度"""
        try:
            # 基于市场状态和技术指标计算信心度
            base_confidence = 0.7
            
            # 获取市场状态
            market_regime = self.market_regime_cache.get(symbol, MarketRegime.SIDEWAYS)
            
            # 根据市场状态调整信心度
            if market_regime == MarketRegime.HIGH_VOLATILITY:
                base_confidence -= 0.2  # 高波动率降低信心度
            elif market_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
                base_confidence += 0.1  # 趋势市场提高信心度
            
            # 根据止损距离调整信心度
            distance = abs(current_price - stop_price) / current_price
            if distance < 0.01:  # 止损太近
                base_confidence -= 0.2
            elif distance > 0.05:  # 止损太远
                base_confidence -= 0.1
            
            return max(0.1, min(1.0, base_confidence))
            
        except Exception as e:
            self.logger.error(f"计算止损信心度失败 {symbol}: {e}")
            return 0.5
    
    def _calculate_loss_amount(self, stop_level: StopLossLevel, trigger_price: float) -> float:
        """计算损失金额"""
        if stop_level.position_side == PositionSide.LONG:
            price_diff = stop_level.initial_price - trigger_price
        else:
            price_diff = trigger_price - stop_level.initial_price
        
        return max(0, price_diff)
    
    def _update_performance_stats(self, trigger_record: Dict[str, Any]):
        """更新性能统计"""
        try:
            if 'total_stops' not in self.performance_stats:
                self.performance_stats = {
                    'total_stops': 0,
                    'total_loss': 0.0,
                    'avg_loss': 0.0,
                    'stop_types': {},
                    'market_regimes': {}
                }
            
            # 更新基本统计
            self.performance_stats['total_stops'] += 1
            self.performance_stats['total_loss'] += trigger_record['loss_amount']
            self.performance_stats['avg_loss'] = (
                self.performance_stats['total_loss'] / self.performance_stats['total_stops']
            )
            
            # 按止损类型统计
            stop_type = trigger_record['stop_type'].value
            if stop_type not in self.performance_stats['stop_types']:
                self.performance_stats['stop_types'][stop_type] = 0
            self.performance_stats['stop_types'][stop_type] += 1
            
        except Exception as e:
            self.logger.error(f"更新性能统计失败: {e}")
    
    def get_stop_level(self, symbol: str) -> Optional[StopLossLevel]:
        """获取止损水平"""
        return self.stop_levels.get(symbol)
    
    def remove_stop_loss(self, symbol: str) -> bool:
        """移除止损"""
        if symbol in self.stop_levels:
            del self.stop_levels[symbol]
            self.logger.info(f"移除止损: {symbol}")
            return True
        return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'active_stops': len(self.stop_levels),
            'symbols_tracked': len(self.price_history),
            'performance_stats': self.performance_stats,
            'market_regimes': {k: v.value for k, v in self.market_regime_cache.items()},
            'config': {
                'volatility_window': self.volatility_window,
                'atr_window': self.atr_window,
                'default_risk_per_trade': self.default_risk_per_trade
            }
        }
    
    def get_stop_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """获取止损历史"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            record for record in self.stop_history 
            if record['trigger_time'] >= cutoff_time
        ]