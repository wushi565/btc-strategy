"""
风险管理主控制器
实现高级风险管理系统的核心控制逻辑
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass
import numpy as np
import pandas as pd
from collections import defaultdict, deque


class RiskLevel(Enum):
    """风险级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskEvent:
    """风险事件数据结构"""
    event_type: str
    level: RiskLevel
    message: str
    timestamp: datetime
    symbol: str
    value: float
    threshold: float
    metadata: Dict[str, Any] = None


class RiskManager:
    """
    高级风险管理系统主控制器
    
    核心功能：
    - 动态风险监控
    - 多层次风险预算
    - 自动风险响应
    - 交易暂停机制
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化风险管理器
        
        Args:
            config: 风险管理配置
        """
        self.config = config
        self.logger = logging.getLogger('RiskManager')
        
        # 风险限制参数
        self.max_daily_loss = config.get('max_daily_loss', 1000)
        self.max_drawdown = config.get('max_drawdown', 0.1)
        self.max_position_value = config.get('max_position_value', 50000)
        self.max_leverage = config.get('max_leverage', 10)
        self.risk_per_trade = config.get('risk_per_trade', 0.02)
        
        # 风险监控状态
        self.current_drawdown = 0.0
        self.daily_pnl = 0.0
        self.total_exposure = 0.0
        self.active_positions = {}
        self.risk_budget_used = 0.0
        
        # 风险事件处理
        self.risk_events = deque(maxlen=1000)
        self.event_handlers = defaultdict(list)
        self.risk_alerts = []
        
        # 控制状态
        self.trading_paused = False
        self.emergency_mode = False
        self.last_risk_check = datetime.now()
        
        # 历史数据
        self.daily_pnl_history = deque(maxlen=30)
        self.drawdown_history = deque(maxlen=100)
        self.exposure_history = deque(maxlen=100)
        
        # 注册默认事件处理器
        self._register_default_handlers()
        
        self.logger.info("风险管理器初始化完成")
    
    def _register_default_handlers(self):
        """注册默认风险事件处理器"""
        self.register_handler('daily_loss_limit', self._handle_daily_loss_limit)
        self.register_handler('drawdown_limit', self._handle_drawdown_limit)
        self.register_handler('position_size_limit', self._handle_position_size_limit)
        self.register_handler('leverage_limit', self._handle_leverage_limit)
        self.register_handler('exposure_limit', self._handle_exposure_limit)
    
    def register_handler(self, event_type: str, handler: Callable):
        """注册风险事件处理器"""
        self.event_handlers[event_type].append(handler)
        self.logger.debug(f"注册风险事件处理器: {event_type}")
    
    async def check_risk_limits(self, portfolio_data: Dict[str, Any]) -> bool:
        """
        检查风险限制
        
        Args:
            portfolio_data: 投资组合数据
            
        Returns:
            bool: 是否通过风险检查
        """
        self.last_risk_check = datetime.now()
        risk_passed = True
        
        try:
            # 更新当前状态
            self._update_portfolio_state(portfolio_data)
            
            # 检查各项风险限制
            risk_checks = [
                self._check_daily_loss_limit(),
                self._check_drawdown_limit(),
                self._check_position_limits(),
                self._check_leverage_limits(),
                self._check_exposure_limits()
            ]
            
            for check_result in risk_checks:
                if not check_result:
                    risk_passed = False
            
            # 更新历史数据
            self._update_risk_history()
            
            # 如果风险检查失败，考虑暂停交易
            if not risk_passed and not self.trading_paused:
                await self._consider_trading_pause()
            
            return risk_passed
            
        except Exception as e:
            self.logger.error(f"风险检查失败: {e}")
            await self._emit_risk_event(
                'risk_check_error',
                RiskLevel.HIGH,
                f"风险检查系统错误: {e}",
                value=0,
                threshold=0
            )
            return False
    
    def _update_portfolio_state(self, portfolio_data: Dict[str, Any]):
        """更新投资组合状态"""
        self.daily_pnl = portfolio_data.get('daily_pnl', 0)
        self.current_drawdown = portfolio_data.get('current_drawdown', 0)
        self.total_exposure = portfolio_data.get('total_exposure', 0)
        self.active_positions = portfolio_data.get('positions', {})
        
        # 计算风险预算使用率
        if self.max_daily_loss > 0:
            self.risk_budget_used = abs(self.daily_pnl) / self.max_daily_loss
    
    def _check_daily_loss_limit(self) -> bool:
        """检查每日损失限制"""
        if self.daily_pnl <= -self.max_daily_loss:
            asyncio.create_task(self._emit_risk_event(
                'daily_loss_limit',
                RiskLevel.CRITICAL,
                f"每日损失超限: {self.daily_pnl:.2f} <= -{self.max_daily_loss}",
                value=abs(self.daily_pnl),
                threshold=self.max_daily_loss
            ))
            return False
        
        # 预警机制
        warning_threshold = self.max_daily_loss * 0.8
        if self.daily_pnl <= -warning_threshold:
            asyncio.create_task(self._emit_risk_event(
                'daily_loss_warning',
                RiskLevel.HIGH,
                f"每日损失接近限制: {self.daily_pnl:.2f}",
                value=abs(self.daily_pnl),
                threshold=warning_threshold
            ))
        
        return True
    
    def _check_drawdown_limit(self) -> bool:
        """检查回撤限制"""
        if self.current_drawdown >= self.max_drawdown:
            asyncio.create_task(self._emit_risk_event(
                'drawdown_limit',
                RiskLevel.CRITICAL,
                f"回撤超限: {self.current_drawdown:.2%} >= {self.max_drawdown:.2%}",
                value=self.current_drawdown,
                threshold=self.max_drawdown
            ))
            return False
        
        # 预警机制
        warning_threshold = self.max_drawdown * 0.8
        if self.current_drawdown >= warning_threshold:
            asyncio.create_task(self._emit_risk_event(
                'drawdown_warning',
                RiskLevel.HIGH,
                f"回撤接近限制: {self.current_drawdown:.2%}",
                value=self.current_drawdown,
                threshold=warning_threshold
            ))
        
        return True
    
    def _check_position_limits(self) -> bool:
        """检查仓位限制"""
        risk_passed = True
        
        for symbol, position in self.active_positions.items():
            position_value = abs(position.get('value', 0))
            
            if position_value > self.max_position_value:
                asyncio.create_task(self._emit_risk_event(
                    'position_size_limit',
                    RiskLevel.HIGH,
                    f"仓位过大: {symbol} {position_value:.2f} > {self.max_position_value}",
                    symbol=symbol,
                    value=position_value,
                    threshold=self.max_position_value
                ))
                risk_passed = False
        
        return risk_passed
    
    def _check_leverage_limits(self) -> bool:
        """检查杠杆限制"""
        risk_passed = True
        
        for symbol, position in self.active_positions.items():
            leverage = position.get('leverage', 1)
            
            if leverage > self.max_leverage:
                asyncio.create_task(self._emit_risk_event(
                    'leverage_limit',
                    RiskLevel.HIGH,
                    f"杠杆过高: {symbol} {leverage:.1f}x > {self.max_leverage}x",
                    symbol=symbol,
                    value=leverage,
                    threshold=self.max_leverage
                ))
                risk_passed = False
        
        return risk_passed
    
    def _check_exposure_limits(self) -> bool:
        """检查总风险敞口限制"""
        max_exposure = self.config.get('max_total_exposure', 100000)
        
        if self.total_exposure > max_exposure:
            asyncio.create_task(self._emit_risk_event(
                'exposure_limit',
                RiskLevel.HIGH,
                f"总风险敞口过大: {self.total_exposure:.2f} > {max_exposure}",
                value=self.total_exposure,
                threshold=max_exposure
            ))
            return False
        
        return True
    
    def _update_risk_history(self):
        """更新风险历史数据"""
        current_time = datetime.now()
        
        self.daily_pnl_history.append({
            'timestamp': current_time,
            'pnl': self.daily_pnl
        })
        
        self.drawdown_history.append({
            'timestamp': current_time,
            'drawdown': self.current_drawdown
        })
        
        self.exposure_history.append({
            'timestamp': current_time,
            'exposure': self.total_exposure
        })
    
    async def _consider_trading_pause(self):
        """考虑是否暂停交易"""
        critical_events = [e for e in self.risk_events if e.level == RiskLevel.CRITICAL]
        high_events = [e for e in self.risk_events if e.level == RiskLevel.HIGH]
        
        # 如果有关键风险事件，立即暂停交易
        if critical_events:
            await self.pause_trading("检测到关键风险事件")
        
        # 如果高风险事件过多，也考虑暂停
        elif len(high_events) >= 3:
            await self.pause_trading("高风险事件过多")
    
    async def pause_trading(self, reason: str = "风险管理要求"):
        """暂停交易"""
        if not self.trading_paused:
            self.trading_paused = True
            self.logger.warning(f"暂停交易: {reason}")
            
            await self._emit_risk_event(
                'trading_paused',
                RiskLevel.CRITICAL,
                f"交易已暂停: {reason}",
                value=1,
                threshold=1
            )
    
    async def resume_trading(self, reason: str = "风险状况改善"):
        """恢复交易"""
        if self.trading_paused:
            # 重新检查风险状况
            portfolio_data = self._get_current_portfolio_data()
            risk_ok = await self.check_risk_limits(portfolio_data)
            
            if risk_ok:
                self.trading_paused = False
                self.logger.info(f"恢复交易: {reason}")
                
                await self._emit_risk_event(
                    'trading_resumed',
                    RiskLevel.MEDIUM,
                    f"交易已恢复: {reason}",
                    value=0,
                    threshold=1
                )
            else:
                self.logger.warning("风险状况未改善，无法恢复交易")
    
    def _get_current_portfolio_data(self) -> Dict[str, Any]:
        """获取当前投资组合数据"""
        # 这里应该从实际的投资组合系统获取数据
        # 暂时返回模拟数据
        return {
            'daily_pnl': self.daily_pnl,
            'current_drawdown': self.current_drawdown,
            'total_exposure': self.total_exposure,
            'positions': self.active_positions
        }
    
    async def _emit_risk_event(self, event_type: str, level: RiskLevel, message: str, 
                             symbol: str = '', value: float = 0, threshold: float = 0):
        """发送风险事件"""
        event = RiskEvent(
            event_type=event_type,
            level=level,
            message=message,
            timestamp=datetime.now(),
            symbol=symbol,
            value=value,
            threshold=threshold
        )
        
        self.risk_events.append(event)
        self.logger.log(
            logging.CRITICAL if level == RiskLevel.CRITICAL else
            logging.ERROR if level == RiskLevel.HIGH else
            logging.WARNING if level == RiskLevel.MEDIUM else
            logging.INFO,
            f"风险事件 [{level.value}]: {message}"
        )
        
        # 调用事件处理器
        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                self.logger.error(f"风险事件处理器执行失败 {event_type}: {e}")
    
    # 默认风险事件处理器
    async def _handle_daily_loss_limit(self, event: RiskEvent):
        """处理每日损失限制事件"""
        await self.pause_trading("每日损失超限")
    
    async def _handle_drawdown_limit(self, event: RiskEvent):
        """处理回撤限制事件"""
        await self.pause_trading("回撤超限")
    
    async def _handle_position_size_limit(self, event: RiskEvent):
        """处理仓位大小限制事件"""
        # 可以实现自动减仓逻辑
        self.logger.warning(f"仓位 {event.symbol} 过大，建议减仓")
    
    async def _handle_leverage_limit(self, event: RiskEvent):
        """处理杠杆限制事件"""
        # 可以实现自动降杠杆逻辑
        self.logger.warning(f"仓位 {event.symbol} 杠杆过高，建议降低杠杆")
    
    async def _handle_exposure_limit(self, event: RiskEvent):
        """处理风险敞口限制事件"""
        self.logger.warning("总风险敞口过大，建议减少仓位")
    
    def get_risk_status(self) -> Dict[str, Any]:
        """获取风险状态"""
        return {
            'trading_paused': self.trading_paused,
            'emergency_mode': self.emergency_mode,
            'daily_pnl': self.daily_pnl,
            'current_drawdown': self.current_drawdown,
            'total_exposure': self.total_exposure,
            'risk_budget_used': self.risk_budget_used,
            'active_positions_count': len(self.active_positions),
            'recent_events_count': len([e for e in self.risk_events if 
                                      (datetime.now() - e.timestamp).seconds < 3600]),
            'last_risk_check': self.last_risk_check
        }
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """获取风险指标"""
        if not self.daily_pnl_history:
            return {}
        
        pnl_values = [h['pnl'] for h in self.daily_pnl_history]
        drawdown_values = [h['drawdown'] for h in self.drawdown_history]
        
        return {
            'daily_pnl_mean': np.mean(pnl_values),
            'daily_pnl_std': np.std(pnl_values),
            'daily_sharpe_ratio': np.mean(pnl_values) / np.std(pnl_values) if np.std(pnl_values) > 0 else 0,
            'max_drawdown': max(drawdown_values) if drawdown_values else 0,
            'current_drawdown': self.current_drawdown,
            'win_rate': len([p for p in pnl_values if p > 0]) / len(pnl_values) if pnl_values else 0,
            'profit_factor': sum([p for p in pnl_values if p > 0]) / abs(sum([p for p in pnl_values if p < 0])) 
                           if any(p < 0 for p in pnl_values) else float('inf')
        }
    
    def get_recent_events(self, hours: int = 24) -> List[RiskEvent]:
        """获取最近的风险事件"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [e for e in self.risk_events if e.timestamp >= cutoff_time]