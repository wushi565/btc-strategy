"""
仓位管理系统
管理交易仓位的计算、跟踪和风险控制
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass, field
import threading
import time
from collections import defaultdict, deque
import math
import numpy as np


class PositionSide(Enum):
    """仓位方向"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class PositionType(Enum):
    """仓位类型"""
    SPOT = "spot"      # 现货
    MARGIN = "margin"  # 保证金
    FUTURES = "futures"  # 期货


@dataclass
class Position:
    """仓位数据结构"""
    symbol: str
    position_type: PositionType
    side: PositionSide = PositionSide.FLAT
    size: float = 0.0  # 仓位大小(绝对值)
    entry_price: float = 0.0  # 平均入场价格
    market_price: float = 0.0  # 当前市场价格
    
    # 盈亏信息
    unrealized_pnl: float = 0.0  # 未实现盈亏
    realized_pnl: float = 0.0    # 已实现盈亏
    
    # 保证金信息(期货)
    margin_required: float = 0.0  # 需要保证金
    leverage: float = 1.0         # 杠杆倍数
    
    # 风险管理
    stop_loss: Optional[float] = None    # 止损价格
    take_profit: Optional[float] = None  # 止盈价格
    liquidation_price: Optional[float] = None  # 强平价格
    
    # 时间信息
    open_time: Optional[datetime] = None
    last_update: datetime = field(default_factory=datetime.now)
    
    # 关联订单
    entry_orders: List[str] = field(default_factory=list)  # 开仓订单ID
    exit_orders: List[str] = field(default_factory=list)   # 平仓订单ID
    
    # 策略信息
    strategy_id: Optional[str] = None
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_long(self) -> bool:
        """是否为多头仓位"""
        return self.side == PositionSide.LONG
    
    @property
    def is_short(self) -> bool:
        """是否为空头仓位"""
        return self.side == PositionSide.SHORT
    
    @property
    def is_flat(self) -> bool:
        """是否为平仓状态"""
        return self.side == PositionSide.FLAT or self.size == 0
    
    @property
    def notional_value(self) -> float:
        """名义价值"""
        return self.size * self.market_price
    
    @property
    def pnl_percentage(self) -> float:
        """盈亏百分比"""
        if self.entry_price == 0 or self.size == 0:
            return 0.0
        
        entry_value = self.size * self.entry_price
        if entry_value == 0:
            return 0.0
        
        return (self.unrealized_pnl / entry_value) * 100
    
    @property
    def margin_ratio(self) -> float:
        """保证金比例"""
        if self.margin_required == 0:
            return 0.0
        return self.notional_value / self.margin_required
    
    def calculate_pnl(self, current_price: float) -> float:
        """计算当前未实现盈亏"""
        if self.size == 0 or self.entry_price == 0:
            return 0.0
        
        if self.is_long:
            return self.size * (current_price - self.entry_price)
        elif self.is_short:
            return self.size * (self.entry_price - current_price)
        else:
            return 0.0
    
    def update_market_price(self, price: float):
        """更新市场价格和未实现盈亏"""
        self.market_price = price
        self.unrealized_pnl = self.calculate_pnl(price)
        self.last_update = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'position_type': self.position_type.value,
            'side': self.side.value,
            'size': self.size,
            'entry_price': self.entry_price,
            'market_price': self.market_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'margin_required': self.margin_required,
            'leverage': self.leverage,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'liquidation_price': self.liquidation_price,
            'notional_value': self.notional_value,
            'pnl_percentage': self.pnl_percentage,
            'margin_ratio': self.margin_ratio,
            'open_time': self.open_time.isoformat() if self.open_time else None,
            'last_update': self.last_update.isoformat(),
            'strategy_id': self.strategy_id,
            'metadata': self.metadata
        }


@dataclass 
class Trade:
    """交易记录"""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    fee: float
    fee_currency: str
    timestamp: datetime
    order_id: str
    trade_id: str
    strategy_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PositionManager:
    """
    仓位管理系统
    负责跟踪和管理所有交易仓位
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化仓位管理系统
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 仓位存储
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.positions_by_strategy: Dict[str, List[str]] = defaultdict(list)
        
        # 交易历史
        self.trade_history: deque = deque(maxlen=config.get('max_trade_history', 1000))
        self.pnl_history: deque = deque(maxlen=config.get('max_pnl_history', 1000))
        
        # 风险管理参数
        self.max_position_size = config.get('max_position_size', 1000000)  # 最大仓位价值
        self.max_leverage = config.get('max_leverage', 10)  # 最大杠杆
        self.risk_free_rate = config.get('risk_free_rate', 0.02)  # 无风险利率
        
        # 事件处理
        self.event_handlers = defaultdict(list)
        
        # 监控数据
        self.total_unrealized_pnl = 0.0
        self.total_realized_pnl = 0.0
        self.total_margin_used = 0.0
        self.total_notional_value = 0.0
        
        # 运行控制
        self.running = False
        
        # 性能统计
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.largest_win = 0.0
        self.largest_loss = 0.0
        
        self.logger.info("PositionManager初始化完成")
    
    async def start(self):
        """启动仓位管理系统"""
        if self.running:
            raise RuntimeError("PositionManager已经在运行")
        
        self.logger.info("启动仓位管理系统...")
        self.running = True
        
        # 启动监控任务
        tasks = [
            asyncio.create_task(self._position_monitor()),
            asyncio.create_task(self._risk_monitor()),
            asyncio.create_task(self._pnl_calculator())
        ]
        
        self.logger.info("仓位管理系统启动完成")
        
        # 等待任务完成
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.logger.info("仓位管理任务被取消")
    
    async def stop(self):
        """停止仓位管理系统"""
        if not self.running:
            return
        
        self.logger.info("停止仓位管理系统...")
        self.running = False
        
        self.logger.info("仓位管理系统已停止")
    
    async def update_from_fill(self, fill_data: Dict[str, Any]):
        """
        根据订单成交更新仓位
        
        Args:
            fill_data: 成交数据
        """
        symbol = fill_data.get('symbol')
        side = fill_data.get('side', '').lower()
        quantity = float(fill_data.get('quantity', 0))
        price = float(fill_data.get('price', 0))
        fee = float(fill_data.get('fee', 0))
        fee_currency = fill_data.get('fee_currency', '')
        order_id = fill_data.get('order_id', '')
        trade_id = fill_data.get('trade_id', '')
        strategy_id = fill_data.get('strategy_id')
        
        if not symbol or quantity == 0 or price == 0:
            self.logger.error(f"无效的成交数据: {fill_data}")
            return
        
        # 创建交易记录
        trade = Trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            fee=fee,
            fee_currency=fee_currency,
            timestamp=datetime.now(),
            order_id=order_id,
            trade_id=trade_id,
            strategy_id=strategy_id
        )
        
        self.trade_history.append(trade)
        self.trade_count += 1
        
        # 更新仓位
        await self._update_position_from_trade(trade)
        
        # 发送事件
        await self._emit_event('position_updated', {
            'symbol': symbol,
            'trade': trade,
            'position': self.positions.get(symbol)
        })
        
        self.logger.info(f"仓位更新: {symbol} {side} {quantity}@{price}")
    
    async def _update_position_from_trade(self, trade: Trade):
        """根据交易更新仓位"""
        symbol = trade.symbol
        
        # 获取或创建仓位
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                position_type=PositionType.SPOT,  # 默认现货，可配置
                open_time=trade.timestamp
            )
        
        position = self.positions[symbol]
        
        # 根据交易方向更新仓位
        if trade.side == 'buy':
            await self._process_buy_trade(position, trade)
        elif trade.side == 'sell':
            await self._process_sell_trade(position, trade)
        
        # 更新策略关联
        if trade.strategy_id:
            if symbol not in self.positions_by_strategy[trade.strategy_id]:
                self.positions_by_strategy[trade.strategy_id].append(symbol)
            position.strategy_id = trade.strategy_id
        
        # 更新时间戳
        position.last_update = datetime.now()
    
    async def _process_buy_trade(self, position: Position, trade: Trade):
        """处理买入交易"""
        if position.is_flat or position.is_long:
            # 开多头仓位或加仓
            if position.size == 0:
                # 新开仓
                position.side = PositionSide.LONG
                position.size = trade.quantity
                position.entry_price = trade.price
                position.open_time = trade.timestamp
            else:
                # 加仓 - 计算新的平均价格
                total_cost = (position.size * position.entry_price) + (trade.quantity * trade.price)
                total_size = position.size + trade.quantity
                position.entry_price = total_cost / total_size
                position.size = total_size
                
        elif position.is_short:
            # 平空头仓位
            if trade.quantity >= position.size:
                # 完全平仓或反向开仓
                close_quantity = position.size
                close_pnl = close_quantity * (position.entry_price - trade.price)
                
                # 记录已实现盈亏
                position.realized_pnl += close_pnl
                self.total_realized_pnl += close_pnl
                
                # 更新统计
                if close_pnl > 0:
                    self.winning_trades += 1
                    self.largest_win = max(self.largest_win, close_pnl)
                else:
                    self.losing_trades += 1
                    self.largest_loss = min(self.largest_loss, close_pnl)
                
                # 剩余数量开新仓
                remaining_quantity = trade.quantity - position.size
                if remaining_quantity > 0:
                    position.side = PositionSide.LONG
                    position.size = remaining_quantity
                    position.entry_price = trade.price
                    position.open_time = trade.timestamp
                    position.unrealized_pnl = 0.0
                else:
                    # 完全平仓
                    position.side = PositionSide.FLAT
                    position.size = 0.0
                    position.entry_price = 0.0
                    position.unrealized_pnl = 0.0
            else:
                # 部分平仓
                close_pnl = trade.quantity * (position.entry_price - trade.price)
                position.realized_pnl += close_pnl
                position.size -= trade.quantity
                self.total_realized_pnl += close_pnl
    
    async def _process_sell_trade(self, position: Position, trade: Trade):
        """处理卖出交易"""
        if position.is_flat or position.is_short:
            # 开空头仓位或加仓
            if position.size == 0:
                # 新开仓
                position.side = PositionSide.SHORT
                position.size = trade.quantity
                position.entry_price = trade.price
                position.open_time = trade.timestamp
            else:
                # 加仓 - 计算新的平均价格
                total_cost = (position.size * position.entry_price) + (trade.quantity * trade.price)
                total_size = position.size + trade.quantity
                position.entry_price = total_cost / total_size
                position.size = total_size
                
        elif position.is_long:
            # 平多头仓位
            if trade.quantity >= position.size:
                # 完全平仓或反向开仓
                close_quantity = position.size
                close_pnl = close_quantity * (trade.price - position.entry_price)
                
                # 记录已实现盈亏
                position.realized_pnl += close_pnl
                self.total_realized_pnl += close_pnl
                
                # 更新统计
                if close_pnl > 0:
                    self.winning_trades += 1
                    self.largest_win = max(self.largest_win, close_pnl)
                else:
                    self.losing_trades += 1
                    self.largest_loss = min(self.largest_loss, close_pnl)
                
                # 剩余数量开新仓
                remaining_quantity = trade.quantity - position.size
                if remaining_quantity > 0:
                    position.side = PositionSide.SHORT
                    position.size = remaining_quantity
                    position.entry_price = trade.price
                    position.open_time = trade.timestamp
                    position.unrealized_pnl = 0.0
                else:
                    # 完全平仓
                    position.side = PositionSide.FLAT
                    position.size = 0.0
                    position.entry_price = 0.0
                    position.unrealized_pnl = 0.0
            else:
                # 部分平仓
                close_pnl = trade.quantity * (trade.price - position.entry_price)
                position.realized_pnl += close_pnl
                position.size -= trade.quantity
                self.total_realized_pnl += close_pnl
    
    async def update_market_price(self, symbol: str, price: float):
        """更新市场价格"""
        if symbol in self.positions:
            position = self.positions[symbol]
            position.update_market_price(price)
    
    async def _position_monitor(self):
        """仓位监控器"""
        while self.running:
            try:
                # 计算总体指标
                await self._calculate_portfolio_metrics()
                
                # 检查仓位风险
                await self._check_position_risks()
                
                await asyncio.sleep(5)  # 每5秒监控一次
                
            except Exception as e:
                self.logger.error(f"仓位监控器发生错误: {e}")
    
    async def _calculate_portfolio_metrics(self):
        """计算投资组合指标"""
        self.total_unrealized_pnl = 0.0
        self.total_margin_used = 0.0
        self.total_notional_value = 0.0
        
        for position in self.positions.values():
            if not position.is_flat:
                self.total_unrealized_pnl += position.unrealized_pnl
                self.total_margin_used += position.margin_required
                self.total_notional_value += position.notional_value
    
    async def _check_position_risks(self):
        """检查仓位风险"""
        for symbol, position in self.positions.items():
            if position.is_flat:
                continue
            
            # 检查仓位大小限制
            if position.notional_value > self.max_position_size:
                await self._emit_event('position_risk_warning', {
                    'symbol': symbol,
                    'risk_type': 'size_limit',
                    'current_value': position.notional_value,
                    'limit': self.max_position_size
                })
            
            # 检查杠杆限制
            if position.leverage > self.max_leverage:
                await self._emit_event('position_risk_warning', {
                    'symbol': symbol,
                    'risk_type': 'leverage_limit',
                    'current_leverage': position.leverage,
                    'limit': self.max_leverage
                })
            
            # 检查止损触发
            if position.stop_loss:
                if ((position.is_long and position.market_price <= position.stop_loss) or
                    (position.is_short and position.market_price >= position.stop_loss)):
                    
                    await self._emit_event('stop_loss_triggered', {
                        'symbol': symbol,
                        'position': position,
                        'current_price': position.market_price,
                        'stop_loss': position.stop_loss
                    })
            
            # 检查止盈触发
            if position.take_profit:
                if ((position.is_long and position.market_price >= position.take_profit) or
                    (position.is_short and position.market_price <= position.take_profit)):
                    
                    await self._emit_event('take_profit_triggered', {
                        'symbol': symbol,
                        'position': position,
                        'current_price': position.market_price,
                        'take_profit': position.take_profit
                    })
    
    async def _risk_monitor(self):
        """风险监控器"""
        while self.running:
            try:
                # 计算风险指标
                risk_metrics = await self._calculate_risk_metrics()
                
                # 检查风险阈值
                await self._check_risk_thresholds(risk_metrics)
                
                await asyncio.sleep(10)  # 每10秒监控一次
                
            except Exception as e:
                self.logger.error(f"风险监控器发生错误: {e}")
    
    async def _calculate_risk_metrics(self) -> Dict[str, float]:
        """计算风险指标"""
        active_positions = [p for p in self.positions.values() if not p.is_flat]
        
        if not active_positions:
            return {}
        
        # 计算VaR (简化版本)
        pnl_values = [p.unrealized_pnl for p in active_positions]
        var_95 = np.percentile(pnl_values, 5) if pnl_values else 0.0
        
        # 计算最大回撤
        max_drawdown = self._calculate_max_drawdown()
        
        # 计算夏普比率
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        return {
            'var_95': var_95,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_exposure': self.total_notional_value,
            'margin_utilization': self.total_margin_used
        }
    
    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if len(self.pnl_history) < 2:
            return 0.0
        
        pnl_values = [record['total_pnl'] for record in self.pnl_history]
        peak = pnl_values[0]
        max_dd = 0.0
        
        for pnl in pnl_values:
            if pnl > peak:
                peak = pnl
            drawdown = (peak - pnl) / peak if peak != 0 else 0
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_sharpe_ratio(self) -> float:
        """计算夏普比率"""
        if len(self.pnl_history) < 2:
            return 0.0
        
        pnl_values = [record['total_pnl'] for record in self.pnl_history]
        returns = [(pnl_values[i] - pnl_values[i-1]) / abs(pnl_values[i-1]) 
                  for i in range(1, len(pnl_values)) if pnl_values[i-1] != 0]
        
        if not returns:
            return 0.0
        
        avg_return = sum(returns) / len(returns)
        return_std = math.sqrt(sum((r - avg_return) ** 2 for r in returns) / len(returns))
        
        if return_std == 0:
            return 0.0
        
        return (avg_return - self.risk_free_rate / 252) / return_std  # 假设日收益率
    
    async def _check_risk_thresholds(self, risk_metrics: Dict[str, float]):
        """检查风险阈值"""
        # 可配置的风险阈值
        max_drawdown_limit = self.config.get('max_drawdown_limit', 0.2)  # 20%
        min_sharpe_ratio = self.config.get('min_sharpe_ratio', 0.5)
        max_var_limit = self.config.get('max_var_limit', -10000)  # 最大损失
        
        # 检查最大回撤
        if risk_metrics.get('max_drawdown', 0) > max_drawdown_limit:
            await self._emit_event('risk_limit_breach', {
                'risk_type': 'max_drawdown',
                'current_value': risk_metrics['max_drawdown'],
                'limit': max_drawdown_limit
            })
        
        # 检查VaR
        if risk_metrics.get('var_95', 0) < max_var_limit:
            await self._emit_event('risk_limit_breach', {
                'risk_type': 'var_95',
                'current_value': risk_metrics['var_95'],
                'limit': max_var_limit
            })
    
    async def _pnl_calculator(self):
        """盈亏计算器"""
        while self.running:
            try:
                # 记录当前盈亏状态
                total_pnl = self.total_realized_pnl + self.total_unrealized_pnl
                
                pnl_record = {
                    'timestamp': datetime.now(),
                    'total_pnl': total_pnl,
                    'realized_pnl': self.total_realized_pnl,
                    'unrealized_pnl': self.total_unrealized_pnl,
                    'total_notional': self.total_notional_value,
                    'position_count': len([p for p in self.positions.values() if not p.is_flat])
                }
                
                self.pnl_history.append(pnl_record)
                
                await asyncio.sleep(60)  # 每分钟记录一次
                
            except Exception as e:
                self.logger.error(f"盈亏计算器发生错误: {e}")
    
    async def close_position(self, symbol: str, reason: str = "手动平仓"):
        """
        平仓指定仓位
        
        Args:
            symbol: 交易对
            reason: 平仓原因
        """
        position = self.positions.get(symbol)
        if not position or position.is_flat:
            self.logger.warning(f"没有找到可平仓的仓位: {symbol}")
            return
        
        # 发送平仓事件
        await self._emit_event('position_close_requested', {
            'symbol': symbol,
            'position': position,
            'reason': reason
        })
        
        self.logger.info(f"请求平仓: {symbol} - {reason}")
    
    async def close_all_positions(self, reason: str = "全部平仓"):
        """平仓所有仓位"""
        active_positions = [symbol for symbol, pos in self.positions.items() if not pos.is_flat]
        
        self.logger.info(f"请求平仓所有仓位: {len(active_positions)} 个")
        
        for symbol in active_positions:
            await self.close_position(symbol, reason)
    
    def set_stop_loss(self, symbol: str, stop_price: float):
        """设置止损价格"""
        if symbol in self.positions:
            self.positions[symbol].stop_loss = stop_price
            self.logger.info(f"设置止损: {symbol} @ {stop_price}")
    
    def set_take_profit(self, symbol: str, target_price: float):
        """设置止盈价格"""
        if symbol in self.positions:
            self.positions[symbol].take_profit = target_price
            self.logger.info(f"设置止盈: {symbol} @ {target_price}")
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """发送事件"""
        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                await handler(data)
            except Exception as e:
                self.logger.error(f"事件处理器执行失败 {event_type}: {e}")
    
    def register_handler(self, event_type: str, handler: Callable):
        """注册事件处理器"""
        self.event_handlers[event_type].append(handler)
        self.logger.debug(f"注册仓位事件处理器: {event_type}")
    
    # 查询方法
    def get_position(self, symbol: str) -> Optional[Position]:
        """获取指定仓位"""
        return self.positions.get(symbol)
    
    async def get_all_positions(self) -> List[Position]:
        """获取所有仓位"""
        return list(self.positions.values())
    
    async def get_active_positions(self) -> List[Position]:
        """获取活跃仓位"""
        return [p for p in self.positions.values() if not p.is_flat]
    
    def get_positions_by_strategy(self, strategy_id: str) -> List[Position]:
        """根据策略获取仓位"""
        symbols = self.positions_by_strategy.get(strategy_id, [])
        return [self.positions[symbol] for symbol in symbols if symbol in self.positions]
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """获取投资组合摘要"""
        active_positions = [p for p in self.positions.values() if not p.is_flat]
        
        return {
            'total_positions': len(active_positions),
            'total_realized_pnl': self.total_realized_pnl,
            'total_unrealized_pnl': self.total_unrealized_pnl,
            'total_pnl': self.total_realized_pnl + self.total_unrealized_pnl,
            'total_notional_value': self.total_notional_value,
            'total_margin_used': self.total_margin_used,
            'trade_count': self.trade_count,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': (self.winning_trades / self.trade_count * 100) if self.trade_count > 0 else 0,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'max_drawdown': self._calculate_max_drawdown(),
            'sharpe_ratio': self._calculate_sharpe_ratio()
        }
    
    async def is_healthy(self) -> bool:
        """检查系统健康状况"""
        return True  # 基本健康检查
    
    def get_trade_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Trade]:
        """获取交易历史"""
        trades = list(self.trade_history)
        
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        
        return trades[-limit:] if len(trades) > limit else trades
    
    def get_pnl_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取盈亏历史"""
        history = list(self.pnl_history)
        return history[-limit:] if len(history) > limit else history