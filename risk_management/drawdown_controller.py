"""
回撤控制系统
监控投资组合回撤并实施自动风险降低措施
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
import asyncio


class DrawdownSeverity(Enum):
    """回撤严重程度"""
    NORMAL = "normal"      # 正常回撤
    WARNING = "warning"    # 警告级别
    SERIOUS = "serious"    # 严重回撤
    CRITICAL = "critical"  # 关键回撤
    EMERGENCY = "emergency" # 紧急回撤


class DrawdownPhase(Enum):
    """回撤阶段"""
    NONE = "none"           # 无回撤
    BEGINNING = "beginning" # 回撤开始
    DEEPENING = "deepening" # 回撤加深
    STABILIZING = "stabilizing" # 回撤稳定
    RECOVERING = "recovering"   # 回撤恢复


class RiskReduction(Enum):
    """风险降低措施"""
    NONE = "none"
    REDUCE_POSITION = "reduce_position"  # 减少仓位
    CLOSE_POSITIONS = "close_positions"  # 关闭仓位
    PAUSE_TRADING = "pause_trading"      # 暂停交易
    EMERGENCY_EXIT = "emergency_exit"    # 紧急退出


@dataclass
class DrawdownEvent:
    """回撤事件数据结构"""
    start_time: datetime
    peak_value: float
    current_value: float
    current_drawdown: float
    max_drawdown: float
    duration_hours: float
    phase: DrawdownPhase
    severity: DrawdownSeverity
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_recovery_ratio(self) -> float:
        """获取回撤恢复比例"""
        if self.max_drawdown == 0:
            return 1.0
        return 1.0 - (self.current_drawdown / self.max_drawdown)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'start_time': self.start_time.isoformat(),
            'peak_value': self.peak_value,
            'current_value': self.current_value,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'duration_hours': self.duration_hours,
            'phase': self.phase.value,
            'severity': self.severity.value,
            'recovery_ratio': self.get_recovery_ratio(),
            'metadata': self.metadata
        }


@dataclass
class DrawdownConfig:
    """回撤控制配置"""
    warning_threshold: float = 0.05    # 警告阈值 5%
    serious_threshold: float = 0.10    # 严重阈值 10%
    critical_threshold: float = 0.15   # 关键阈值 15%
    emergency_threshold: float = 0.20  # 紧急阈值 20%
    
    # 时间阈值
    warning_duration_hours: float = 24    # 警告持续时间
    serious_duration_hours: float = 72    # 严重持续时间
    
    # 风险降低参数
    position_reduction_ratio: float = 0.5  # 仓位减少比例
    recovery_threshold: float = 0.7        # 恢复阈值
    
    # 监控参数
    lookback_hours: int = 168             # 回看时间(7天)
    update_frequency: int = 300           # 更新频率(秒)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'warning_threshold': self.warning_threshold,
            'serious_threshold': self.serious_threshold,
            'critical_threshold': self.critical_threshold,
            'emergency_threshold': self.emergency_threshold,
            'warning_duration_hours': self.warning_duration_hours,
            'serious_duration_hours': self.serious_duration_hours,
            'position_reduction_ratio': self.position_reduction_ratio,
            'recovery_threshold': self.recovery_threshold,
            'lookback_hours': self.lookback_hours,
            'update_frequency': self.update_frequency
        }


@dataclass
class PortfolioSnapshot:
    """投资组合快照"""
    timestamp: datetime
    total_value: float
    positions: Dict[str, Any]
    cash_balance: float
    unrealized_pnl: float
    realized_pnl: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_value': self.total_value,
            'positions': self.positions,
            'cash_balance': self.cash_balance,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl
        }


class DrawdownController:
    """
    回撤控制系统
    
    核心功能：
    - 实时回撤监控和计算
    - 多级回撤预警和响应
    - 自动风险降低措施
    - 回撤恢复检测
    - 历史回撤分析
    """
    
    def __init__(self, config: DrawdownConfig = None):
        """
        初始化回撤控制系统
        
        Args:
            config: 回撤控制配置
        """
        self.config = config or DrawdownConfig()
        self.logger = logging.getLogger('DrawdownController')
        
        # 数据存储
        self.portfolio_history: deque = deque(maxlen=10000)
        self.drawdown_events: List[DrawdownEvent] = []
        self.current_event: Optional[DrawdownEvent] = None
        
        # 状态跟踪
        self.peak_value: float = 0.0
        self.current_value: float = 0.0
        self.current_drawdown: float = 0.0
        self.is_in_drawdown: bool = False
        self.last_update: datetime = datetime.now()
        
        # 风险控制状态
        self.risk_reduction_active: bool = False
        self.current_reduction_level: RiskReduction = RiskReduction.NONE
        self.original_position_sizes: Dict[str, float] = {}
        
        # 回调函数
        self.drawdown_callbacks: List[Callable] = []
        self.risk_reduction_callbacks: List[Callable] = []
        self.recovery_callbacks: List[Callable] = []
        
        # 性能统计
        self.stats = {
            'total_drawdown_events': 0,
            'max_historical_drawdown': 0.0,
            'avg_drawdown_duration': 0.0,
            'avg_recovery_time': 0.0,
            'risk_reductions_triggered': 0
        }
        
        self.logger.info("回撤控制系统初始化完成")
    
    def register_drawdown_callback(self, callback: Callable):
        """注册回撤事件回调"""
        self.drawdown_callbacks.append(callback)
    
    def register_risk_reduction_callback(self, callback: Callable):
        """注册风险降低回调"""
        self.risk_reduction_callbacks.append(callback)
    
    def register_recovery_callback(self, callback: Callable):
        """注册恢复回调"""
        self.recovery_callbacks.append(callback)
    
    async def update_portfolio(self, snapshot: PortfolioSnapshot):
        """
        更新投资组合数据
        
        Args:
            snapshot: 投资组合快照
        """
        try:
            # 保存历史数据
            self.portfolio_history.append(snapshot)
            
            # 更新当前状态
            self.current_value = snapshot.total_value
            self.last_update = snapshot.timestamp
            
            # 更新峰值
            if snapshot.total_value > self.peak_value:
                self.peak_value = snapshot.total_value
            
            # 计算当前回撤
            if self.peak_value > 0:
                self.current_drawdown = (self.peak_value - snapshot.total_value) / self.peak_value
            else:
                self.current_drawdown = 0.0
            
            # 检查回撤状态
            await self._check_drawdown_status()
            
            # 更新回撤事件
            await self._update_drawdown_event()
            
            # 检查风险降低措施
            await self._check_risk_reduction()
            
        except Exception as e:
            self.logger.error(f"更新投资组合失败: {e}")
    
    async def _check_drawdown_status(self):
        """检查回撤状态"""
        was_in_drawdown = self.is_in_drawdown
        
        # 判断是否进入回撤
        if self.current_drawdown >= self.config.warning_threshold:
            if not self.is_in_drawdown:
                # 开始新的回撤事件
                self.is_in_drawdown = True
                await self._start_drawdown_event()
            else:
                # 更新现有回撤事件
                await self._update_drawdown_phase()
        else:
            if self.is_in_drawdown:
                # 回撤结束
                self.is_in_drawdown = False
                await self._end_drawdown_event()
    
    async def _start_drawdown_event(self):
        """开始新的回撤事件"""
        try:
            severity = self._calculate_drawdown_severity(self.current_drawdown)
            
            self.current_event = DrawdownEvent(
                start_time=self.last_update,
                peak_value=self.peak_value,
                current_value=self.current_value,
                current_drawdown=self.current_drawdown,
                max_drawdown=self.current_drawdown,
                duration_hours=0.0,
                phase=DrawdownPhase.BEGINNING,
                severity=severity
            )
            
            self.logger.warning(f"回撤开始: {self.current_drawdown:.2%} [{severity.value}]")
            
            # 触发回调
            for callback in self.drawdown_callbacks:
                try:
                    await callback(self.current_event, 'started')
                except Exception as e:
                    self.logger.error(f"回撤回调执行失败: {e}")
                    
        except Exception as e:
            self.logger.error(f"开始回撤事件失败: {e}")
    
    async def _update_drawdown_event(self):
        """更新回撤事件"""
        if not self.current_event:
            return
        
        try:
            # 更新事件数据
            self.current_event.current_value = self.current_value
            self.current_event.current_drawdown = self.current_drawdown
            self.current_event.max_drawdown = max(
                self.current_event.max_drawdown, self.current_drawdown
            )
            
            # 计算持续时间
            duration = self.last_update - self.current_event.start_time
            self.current_event.duration_hours = duration.total_seconds() / 3600
            
            # 更新严重程度
            old_severity = self.current_event.severity
            new_severity = self._calculate_drawdown_severity(self.current_drawdown)
            self.current_event.severity = new_severity
            
            # 如果严重程度发生变化，触发回调
            if new_severity != old_severity:
                self.logger.warning(f"回撤严重程度变化: {old_severity.value} -> {new_severity.value}")
                for callback in self.drawdown_callbacks:
                    try:
                        await callback(self.current_event, 'severity_changed')
                    except Exception as e:
                        self.logger.error(f"回撤回调执行失败: {e}")
                        
        except Exception as e:
            self.logger.error(f"更新回撤事件失败: {e}")
    
    async def _update_drawdown_phase(self):
        """更新回撤阶段"""
        if not self.current_event:
            return
        
        try:
            old_phase = self.current_event.phase
            new_phase = self._determine_drawdown_phase()
            
            if new_phase != old_phase:
                self.current_event.phase = new_phase
                self.logger.info(f"回撤阶段变化: {old_phase.value} -> {new_phase.value}")
                
                # 触发回调
                for callback in self.drawdown_callbacks:
                    try:
                        await callback(self.current_event, 'phase_changed')
                    except Exception as e:
                        self.logger.error(f"回撤回调执行失败: {e}")
                        
        except Exception as e:
            self.logger.error(f"更新回撤阶段失败: {e}")
    
    async def _end_drawdown_event(self):
        """结束回撤事件"""
        if not self.current_event:
            return
        
        try:
            # 设置为恢复阶段
            self.current_event.phase = DrawdownPhase.RECOVERING
            
            # 添加到历史记录
            self.drawdown_events.append(self.current_event)
            
            # 更新统计
            self._update_drawdown_stats(self.current_event)
            
            self.logger.info(f"回撤结束: 最大回撤 {self.current_event.max_drawdown:.2%}, "
                           f"持续时间 {self.current_event.duration_hours:.1f}小时")
            
            # 触发恢复回调
            for callback in self.recovery_callbacks:
                try:
                    await callback(self.current_event)
                except Exception as e:
                    self.logger.error(f"恢复回调执行失败: {e}")
            
            # 清除当前事件
            self.current_event = None
            
            # 检查是否需要取消风险降低措施
            await self._check_risk_reduction_recovery()
            
        except Exception as e:
            self.logger.error(f"结束回撤事件失败: {e}")
    
    async def _check_risk_reduction(self):
        """检查是否需要风险降低措施"""
        if not self.current_event:
            return
        
        try:
            required_reduction = self._calculate_required_risk_reduction()
            
            if required_reduction != self.current_reduction_level:
                await self._implement_risk_reduction(required_reduction)
                
        except Exception as e:
            self.logger.error(f"检查风险降低措施失败: {e}")
    
    def _calculate_required_risk_reduction(self) -> RiskReduction:
        """计算所需的风险降低措施"""
        if not self.current_event:
            return RiskReduction.NONE
        
        drawdown = self.current_event.current_drawdown
        duration = self.current_event.duration_hours
        
        # 基于回撤严重程度
        if drawdown >= self.config.emergency_threshold:
            return RiskReduction.EMERGENCY_EXIT
        elif drawdown >= self.config.critical_threshold:
            return RiskReduction.CLOSE_POSITIONS
        elif drawdown >= self.config.serious_threshold:
            return RiskReduction.REDUCE_POSITION
        elif (drawdown >= self.config.warning_threshold and 
              duration >= self.config.warning_duration_hours):
            return RiskReduction.PAUSE_TRADING
        
        return RiskReduction.NONE
    
    async def _implement_risk_reduction(self, reduction_level: RiskReduction):
        """实施风险降低措施"""
        if reduction_level == self.current_reduction_level:
            return
        
        try:
            old_level = self.current_reduction_level
            self.current_reduction_level = reduction_level
            
            if reduction_level != RiskReduction.NONE:
                self.risk_reduction_active = True
                self.stats['risk_reductions_triggered'] += 1
            
            self.logger.warning(f"实施风险降低措施: {old_level.value} -> {reduction_level.value}")
            
            # 触发风险降低回调
            for callback in self.risk_reduction_callbacks:
                try:
                    await callback(reduction_level, self.current_event)
                except Exception as e:
                    self.logger.error(f"风险降低回调执行失败: {e}")
                    
        except Exception as e:
            self.logger.error(f"实施风险降低措施失败: {e}")
    
    async def _check_risk_reduction_recovery(self):
        """检查风险降低措施恢复"""
        if not self.risk_reduction_active:
            return
        
        try:
            # 如果当前回撤已经低于恢复阈值，考虑取消风险降低措施
            if self.current_drawdown < self.config.warning_threshold * self.config.recovery_threshold:
                await self._recover_risk_reduction()
                
        except Exception as e:
            self.logger.error(f"检查风险降低恢复失败: {e}")
    
    async def _recover_risk_reduction(self):
        """恢复风险降低措施"""
        try:
            old_level = self.current_reduction_level
            self.current_reduction_level = RiskReduction.NONE
            self.risk_reduction_active = False
            
            self.logger.info(f"恢复风险降低措施: {old_level.value} -> NONE")
            
            # 触发恢复回调
            for callback in self.risk_reduction_callbacks:
                try:
                    await callback(RiskReduction.NONE, None)
                except Exception as e:
                    self.logger.error(f"风险降低恢复回调执行失败: {e}")
                    
        except Exception as e:
            self.logger.error(f"恢复风险降低措施失败: {e}")
    
    def _calculate_drawdown_severity(self, drawdown: float) -> DrawdownSeverity:
        """计算回撤严重程度"""
        if drawdown >= self.config.emergency_threshold:
            return DrawdownSeverity.EMERGENCY
        elif drawdown >= self.config.critical_threshold:
            return DrawdownSeverity.CRITICAL
        elif drawdown >= self.config.serious_threshold:
            return DrawdownSeverity.SERIOUS
        elif drawdown >= self.config.warning_threshold:
            return DrawdownSeverity.WARNING
        else:
            return DrawdownSeverity.NORMAL
    
    def _determine_drawdown_phase(self) -> DrawdownPhase:
        """确定回撤阶段"""
        if not self.current_event:
            return DrawdownPhase.NONE
        
        # 分析最近的回撤变化趋势
        if len(self.portfolio_history) < 10:
            return DrawdownPhase.BEGINNING
        
        recent_values = [snap.total_value for snap in list(self.portfolio_history)[-10:]]
        trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
        
        # 基于趋势和回撤深度判断阶段
        if trend > 0:
            if self.current_event.get_recovery_ratio() > 0.3:
                return DrawdownPhase.RECOVERING
            else:
                return DrawdownPhase.STABILIZING
        else:
            if self.current_event.duration_hours < 6:
                return DrawdownPhase.BEGINNING
            else:
                return DrawdownPhase.DEEPENING
    
    def _update_drawdown_stats(self, event: DrawdownEvent):
        """更新回撤统计"""
        try:
            self.stats['total_drawdown_events'] += 1
            self.stats['max_historical_drawdown'] = max(
                self.stats['max_historical_drawdown'], 
                event.max_drawdown
            )
            
            # 计算平均持续时间
            total_duration = sum(e.duration_hours for e in self.drawdown_events)
            self.stats['avg_drawdown_duration'] = total_duration / len(self.drawdown_events)
            
            # 计算平均恢复时间
            recovery_events = [e for e in self.drawdown_events if e.phase == DrawdownPhase.RECOVERING]
            if recovery_events:
                total_recovery_time = sum(e.duration_hours for e in recovery_events)
                self.stats['avg_recovery_time'] = total_recovery_time / len(recovery_events)
                
        except Exception as e:
            self.logger.error(f"更新回撤统计失败: {e}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            'is_in_drawdown': self.is_in_drawdown,
            'current_drawdown': self.current_drawdown,
            'peak_value': self.peak_value,
            'current_value': self.current_value,
            'risk_reduction_active': self.risk_reduction_active,
            'current_reduction_level': self.current_reduction_level.value,
            'current_event': self.current_event.to_dict() if self.current_event else None,
            'last_update': self.last_update.isoformat(),
            'config': self.config.to_dict()
        }
    
    def get_drawdown_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """获取回撤历史"""
        cutoff_time = datetime.now() - timedelta(days=days)
        return [
            event.to_dict() for event in self.drawdown_events
            if event.start_time >= cutoff_time
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'stats': self.stats,
            'current_drawdown': self.current_drawdown,
            'max_historical_drawdown': self.stats['max_historical_drawdown'],
            'total_events': len(self.drawdown_events),
            'portfolio_data_points': len(self.portfolio_history),
            'config': self.config.to_dict()
        }
    
    def calculate_risk_metrics(self) -> Dict[str, Any]:
        """计算风险指标"""
        if len(self.portfolio_history) < 30:
            return {}
        
        try:
            # 获取最近的投资组合价值
            values = [snap.total_value for snap in self.portfolio_history]
            
            # 计算收益率
            returns = []
            for i in range(1, len(values)):
                if values[i-1] > 0:
                    returns.append((values[i] - values[i-1]) / values[i-1])
            
            if not returns:
                return {}
            
            # 计算各项指标
            metrics = {
                'total_return': (values[-1] - values[0]) / values[0] if values[0] > 0 else 0,
                'volatility': np.std(returns) * np.sqrt(24 * 365),  # 年化波动率
                'max_drawdown': max([e.max_drawdown for e in self.drawdown_events], default=0),
                'current_drawdown': self.current_drawdown,
                'sharpe_ratio': 0,  # 需要无风险利率计算
                'calmar_ratio': 0,  # 年化收益/最大回撤
                'drawdown_frequency': len(self.drawdown_events) / max(1, len(self.portfolio_history) / (24 * 30)),  # 每月回撤次数
                'avg_drawdown_duration': self.stats['avg_drawdown_duration'],
                'avg_recovery_time': self.stats['avg_recovery_time']
            }
            
            # 计算卡尔玛比率
            max_dd = metrics['max_drawdown']
            if max_dd > 0:
                annual_return = metrics['total_return'] * (365 * 24) / len(self.portfolio_history)
                metrics['calmar_ratio'] = annual_return / max_dd
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"计算风险指标失败: {e}")
            return {}
    
    def export_data(self, start_date: datetime = None, end_date: datetime = None) -> Dict[str, Any]:
        """导出数据"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        # 过滤数据
        portfolio_data = [
            snap.to_dict() for snap in self.portfolio_history
            if start_date <= snap.timestamp <= end_date
        ]
        
        drawdown_data = [
            event.to_dict() for event in self.drawdown_events
            if start_date <= event.start_time <= end_date
        ]
        
        return {
            'portfolio_history': portfolio_data,
            'drawdown_events': drawdown_data,
            'statistics': self.get_statistics(),
            'risk_metrics': self.calculate_risk_metrics(),
            'export_time': datetime.now().isoformat(),
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }
        }