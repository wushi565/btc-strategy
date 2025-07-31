"""
订单管理系统(OMS)
管理订单的整个生命周期，支持多种订单类型和高级订单功能
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass, field
import threading
import time
from collections import defaultdict, deque
import ccxt
import json


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"
    TRAILING_STOP = "trailing_stop"
    OCO = "oco"  # One-Cancels-Other


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"         # 待提交
    SUBMITTED = "submitted"     # 已提交
    OPEN = "open"              # 挂单中
    PARTIALLY_FILLED = "partially_filled"  # 部分成交
    FILLED = "filled"          # 完全成交
    CANCELED = "canceled"      # 已取消
    REJECTED = "rejected"      # 被拒绝
    EXPIRED = "expired"        # 已过期
    ERROR = "error"            # 错误状态


class TimeInForce(Enum):
    """订单有效期"""
    GTC = "GTC"  # Good Till Canceled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    GTD = "GTD"  # Good Till Date


@dataclass
class Order:
    """订单数据结构"""
    symbol: str
    order_type: OrderType
    side: OrderSide
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    
    # 标识信息
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    client_order_id: str = field(default_factory=lambda: f"order_{int(time.time() * 1000)}")
    exchange_order_id: Optional[str] = None
    
    # 状态信息
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: Optional[float] = None
    average_price: Optional[float] = None
    
    # 时间信息
    created_time: datetime = field(default_factory=datetime.now)
    submitted_time: Optional[datetime] = None
    updated_time: Optional[datetime] = None
    
    # 策略信息
    strategy_id: Optional[str] = None
    parent_order_id: Optional[str] = None  # 用于OCO等复合订单
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    fees: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity
    
    @property
    def is_buy(self) -> bool:
        """是否为买单"""
        return self.side == OrderSide.BUY
    
    @property
    def is_sell(self) -> bool:
        """是否为卖单"""
        return self.side == OrderSide.SELL
    
    @property
    def is_active(self) -> bool:
        """是否为活跃订单"""
        return self.status in [OrderStatus.SUBMITTED, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]
    
    @property
    def is_terminal(self) -> bool:
        """是否为终态订单"""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED, OrderStatus.ERROR]
    
    @property
    def fill_percentage(self) -> float:
        """成交百分比"""
        if self.quantity == 0:
            return 0.0
        return (self.filled_quantity / self.quantity) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'order_id': self.order_id,
            'client_order_id': self.client_order_id,
            'exchange_order_id': self.exchange_order_id,
            'symbol': self.symbol,
            'order_type': self.order_type.value,
            'side': self.side.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force.value,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'remaining_quantity': self.remaining_quantity,
            'average_price': self.average_price,
            'created_time': self.created_time.isoformat() if self.created_time else None,
            'submitted_time': self.submitted_time.isoformat() if self.submitted_time else None,
            'updated_time': self.updated_time.isoformat() if self.updated_time else None,
            'strategy_id': self.strategy_id,
            'metadata': self.metadata,
            'fees': self.fees
        }


@dataclass
class OrderFill:
    """订单成交数据"""
    order_id: str
    exchange_order_id: str
    trade_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    fee: float
    fee_currency: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrderManagementSystem:
    """
    订单管理系统(OMS)
    提供订单的完整生命周期管理
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化订单管理系统
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 交易所配置
        self.exchange_name = config.get('exchange', 'binance')
        self.api_key = config.get('api_key', '')
        self.api_secret = config.get('api_secret', '')
        self.testnet = config.get('testnet', True)
        self.passphrase = config.get('passphrase', '')  # OKX需要
        
        # 初始化交易所客户端
        self.exchange = self._init_exchange()
        
        # 订单存储
        self.orders: Dict[str, Order] = {}  # order_id -> Order
        self.orders_by_exchange_id: Dict[str, str] = {}  # exchange_order_id -> order_id
        self.orders_by_symbol: Dict[str, List[str]] = defaultdict(list)
        self.orders_by_strategy: Dict[str, List[str]] = defaultdict(list)
        
        # 订单历史
        self.order_history: deque = deque(maxlen=config.get('max_history', 1000))
        self.fill_history: deque = deque(maxlen=config.get('max_fill_history', 1000))
        
        # 事件处理
        self.event_handlers = defaultdict(list)
        self.order_queue = asyncio.Queue()
        self.cancel_queue = asyncio.Queue()
        
        # 监控参数
        self.max_orders_per_symbol = config.get('max_orders_per_symbol', 10)
        self.max_order_amount = config.get('max_order_amount', 100000)
        self.order_timeout = config.get('order_timeout', 300)  # 5分钟
        
        # 运行控制
        self.running = False
        self.order_processing_enabled = True
        
        # 性能监控
        self.order_count = 0
        self.fill_count = 0
        self.error_count = 0
        self.last_order_time = None
        
        self.logger.info(f"OrderManagementSystem初始化完成 - 交易所: {self.exchange_name}")
    
    def _init_exchange(self) -> ccxt.Exchange:
        """初始化交易所客户端"""
        exchange_config = {
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'sandbox': self.testnet,
            'enableRateLimit': True,
            'timeout': 30000,
        }
        
        if self.exchange_name.lower() == 'binance':
            if self.testnet:
                exchange_config['urls'] = {
                    'api': {
                        'public': 'https://testnet.binance.vision/api',
                        'private': 'https://testnet.binance.vision/api',
                    }
                }
            return ccxt.binance(exchange_config)
            
        elif self.exchange_name.lower() == 'okx':
            exchange_config['password'] = self.passphrase
            if self.testnet:
                exchange_config['sandbox'] = True
            return ccxt.okx(exchange_config)
            
        else:
            raise ValueError(f"不支持的交易所: {self.exchange_name}")
    
    async def start(self):
        """启动订单管理系统"""
        if self.running:
            raise RuntimeError("OrderManagementSystem已经在运行")
        
        self.logger.info("启动订单管理系统...")
        self.running = True
        
        # 启动各个任务
        tasks = [
            asyncio.create_task(self._order_processor()),
            asyncio.create_task(self._cancel_processor()),
            asyncio.create_task(self._order_monitor()),
            asyncio.create_task(self._status_updater())
        ]
        
        self.logger.info("订单管理系统启动完成")
        
        # 等待任务完成
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.logger.info("订单管理任务被取消")
    
    async def stop(self):
        """停止订单管理系统"""
        if not self.running:
            return
        
        self.logger.info("停止订单管理系统...")
        self.running = False
        
        # 取消所有未完成的订单
        await self.cancel_all_orders()
        
        # 清空队列
        while not self.order_queue.empty():
            try:
                self.order_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        while not self.cancel_queue.empty():
            try:
                self.cancel_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        self.logger.info("订单管理系统已停止")
    
    async def submit_order(self, order: Order) -> str:
        """
        提交订单
        
        Args:
            order: 订单对象
            
        Returns:
            str: 订单ID
            
        Raises:
            ValueError: 订单验证失败
            RuntimeError: 系统错误
        """
        if not self.order_processing_enabled:
            raise RuntimeError("订单处理已禁用")
        
        # 验证订单
        self._validate_order(order)
        
        # 添加到系统
        self.orders[order.order_id] = order
        self.orders_by_symbol[order.symbol].append(order.order_id)
        
        if order.strategy_id:
            self.orders_by_strategy[order.strategy_id].append(order.order_id)
        
        # 加入处理队列
        await self.order_queue.put(order.order_id)
        
        self.logger.info(f"订单已提交到队列: {order.order_id}")
        return order.order_id
    
    def _validate_order(self, order: Order):
        """验证订单"""
        # 基本参数验证
        if not order.symbol:
            raise ValueError("交易对不能为空")
        
        if order.quantity <= 0:
            raise ValueError("数量必须大于0")
        
        if order.quantity > self.max_order_amount:
            raise ValueError(f"订单金额超过限制: {self.max_order_amount}")
        
        # 价格验证
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_LIMIT]:
            if not order.price or order.price <= 0:
                raise ValueError("限价订单必须设置有效价格")
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if not order.stop_price or order.stop_price <= 0:
                raise ValueError("止损订单必须设置有效止损价格")
        
        # 数量限制检查
        symbol_orders = len([o for o in self.orders_by_symbol[order.symbol] 
                           if self.orders[o].is_active])
        if symbol_orders >= self.max_orders_per_symbol:
            raise ValueError(f"交易对 {order.symbol} 活跃订单数量超过限制")
    
    async def _order_processor(self):
        """订单处理器"""
        while self.running:
            try:
                # 获取待处理订单
                order_id = await self.order_queue.get()
                order = self.orders.get(order_id)
                
                if not order:
                    self.logger.error(f"订单不存在: {order_id}")
                    continue
                
                # 处理订单
                await self._process_order(order)
                
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.01)
            except Exception as e:
                self.logger.error(f"订单处理器发生错误: {e}")
                self.error_count += 1
    
    async def _process_order(self, order: Order):
        """处理单个订单"""
        try:
            self.logger.info(f"处理订单: {order.order_id} {order.symbol} {order.side.value} {order.quantity}")
            
            # 构建交易所订单参数
            params = self._build_exchange_params(order)
            
            # 提交到交易所
            result = await self._submit_to_exchange(order, params)
            
            if result:
                # 更新订单状态
                order.exchange_order_id = result.get('id')
                order.status = OrderStatus.OPEN
                order.submitted_time = datetime.now()
                order.updated_time = datetime.now()
                
                # 添加到交易所ID映射
                if order.exchange_order_id:
                    self.orders_by_exchange_id[order.exchange_order_id] = order.order_id
                
                self.order_count += 1
                self.last_order_time = time.time()
                
                # 发送事件
                await self._emit_event('order_submitted', {
                    'order': order,
                    'exchange_result': result
                })
                
                self.logger.info(f"订单提交成功: {order.order_id} -> {order.exchange_order_id}")
            
        except Exception as e:
            # 订单失败
            order.status = OrderStatus.ERROR
            order.updated_time = datetime.now()
            order.metadata['error'] = str(e)
            
            self.error_count += 1
            
            # 发送错误事件
            await self._emit_event('order_error', {
                'order': order,
                'error': e
            })
            
            self.logger.error(f"订单处理失败 {order.order_id}: {e}")
    
    def _build_exchange_params(self, order: Order) -> Dict[str, Any]:
        """构建交易所订单参数"""
        params = {
            'symbol': order.symbol,
            'type': order.order_type.value,
            'side': order.side.value,
            'amount': order.quantity,
            'params': {
                'timeInForce': order.time_in_force.value
            }
        }
        
        # 设置价格参数
        if order.price:
            params['price'] = order.price
        
        if order.stop_price:
            params['stopPrice'] = order.stop_price
        
        # 客户端订单ID
        if self.exchange_name.lower() == 'binance':
            params['params']['newClientOrderId'] = order.client_order_id
        elif self.exchange_name.lower() == 'okx':
            params['params']['clOrdId'] = order.client_order_id
        
        return params
    
    async def _submit_to_exchange(self, order: Order, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """提交订单到交易所"""
        try:
            # 使用asyncio执行同步交易所操作
            loop = asyncio.get_event_loop()
            
            if order.order_type == OrderType.MARKET:
                result = await loop.run_in_executor(
                    None, 
                    self.exchange.create_market_order,
                    params['symbol'], params['side'], params['amount']
                )
            elif order.order_type == OrderType.LIMIT:
                result = await loop.run_in_executor(
                    None,
                    self.exchange.create_limit_order,
                    params['symbol'], params['side'], params['amount'], params['price']
                )
            else:
                # 其他订单类型使用通用方法
                result = await loop.run_in_executor(
                    None,
                    self.exchange.create_order,
                    params['symbol'], params['type'], params['side'], 
                    params['amount'], params.get('price'), params.get('params', {})
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"交易所订单提交失败: {e}")
            raise
    
    async def cancel_order(self, order_id: str, reason: str = "用户取消") -> bool:
        """
        取消订单
        
        Args:
            order_id: 订单ID
            reason: 取消原因
            
        Returns:
            bool: 是否成功取消
        """
        order = self.orders.get(order_id)
        if not order:
            self.logger.error(f"订单不存在: {order_id}")
            return False
        
        if not order.is_active:
            self.logger.warning(f"订单不是活跃状态，无法取消: {order_id}")
            return False
        
        # 加入取消队列
        await self.cancel_queue.put({
            'order_id': order_id,
            'reason': reason
        })
        
        self.logger.info(f"订单取消请求已提交: {order_id}")
        return True
    
    async def _cancel_processor(self):
        """取消订单处理器"""
        while self.running:
            try:
                # 获取取消请求
                cancel_request = await self.cancel_queue.get()
                
                order_id = cancel_request['order_id']
                reason = cancel_request['reason']
                
                await self._process_cancel(order_id, reason)
                
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.01)
            except Exception as e:
                self.logger.error(f"取消处理器发生错误: {e}")
    
    async def _process_cancel(self, order_id: str, reason: str):
        """处理取消订单"""
        order = self.orders.get(order_id)
        if not order or not order.exchange_order_id:
            return
        
        try:
            self.logger.info(f"取消订单: {order_id} ({order.exchange_order_id})")
            
            # 调用交易所取消接口
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.exchange.cancel_order,
                order.exchange_order_id,
                order.symbol
            )
            
            # 更新订单状态
            order.status = OrderStatus.CANCELED
            order.updated_time = datetime.now()
            order.metadata['cancel_reason'] = reason
            
            # 发送事件
            await self._emit_event('order_cancelled', {
                'order': order,
                'reason': reason,
                'exchange_result': result
            })
            
            self.logger.info(f"订单取消成功: {order_id}")
            
        except Exception as e:
            self.logger.error(f"取消订单失败 {order_id}: {e}")
            order.metadata['cancel_error'] = str(e)
    
    async def cancel_all_orders(self, symbol: Optional[str] = None):
        """取消所有订单"""
        orders_to_cancel = []
        
        for order_id, order in self.orders.items():
            if order.is_active and (symbol is None or order.symbol == symbol):
                orders_to_cancel.append(order_id)
        
        self.logger.info(f"批量取消订单: {len(orders_to_cancel)} 个")
        
        for order_id in orders_to_cancel:
            await self.cancel_order(order_id, "批量取消")
    
    async def _order_monitor(self):
        """订单监控器"""
        while self.running:
            try:
                # 监控超时订单
                await self._check_order_timeouts()
                
                # 监控异常订单
                await self._check_order_anomalies()
                
                await asyncio.sleep(30)  # 每30秒检查一次
                
            except Exception as e:
                self.logger.error(f"订单监控器发生错误: {e}")
    
    async def _check_order_timeouts(self):
        """检查订单超时"""
        current_time = time.time()
        
        for order_id, order in self.orders.items():
            if (order.is_active and order.submitted_time and
                (current_time - order.submitted_time.timestamp()) > self.order_timeout):
                
                self.logger.warning(f"订单超时: {order_id}")
                await self.cancel_order(order_id, "订单超时")
    
    async def _check_order_anomalies(self):
        """检查订单异常"""
        # 检查是否有过多活跃订单
        active_orders = [o for o in self.orders.values() if o.is_active]
        
        if len(active_orders) > 100:  # 可配置阈值
            self.logger.warning(f"活跃订单数量过多: {len(active_orders)}")
    
    async def _status_updater(self):
        """状态更新器"""
        while self.running:
            try:
                # 更新活跃订单状态
                await self._update_active_orders()
                
                await asyncio.sleep(5)  # 每5秒更新一次
                
            except Exception as e:
                self.logger.error(f"状态更新器发生错误: {e}")
    
    async def _update_active_orders(self):
        """更新活跃订单状态"""
        active_orders = [o for o in self.orders.values() 
                        if o.is_active and o.exchange_order_id]
        
        if not active_orders:
            return
        
        try:
            # 批量查询订单状态
            loop = asyncio.get_event_loop()
            
            for order in active_orders[:10]:  # 限制每次查询数量
                try:
                    result = await loop.run_in_executor(
                        None,
                        self.exchange.fetch_order,
                        order.exchange_order_id,
                        order.symbol
                    )
                    
                    # 更新订单状态
                    await self._update_order_from_exchange(order, result)
                    
                except Exception as e:
                    self.logger.error(f"查询订单状态失败 {order.order_id}: {e}")
                    
                await asyncio.sleep(0.1)  # 避免请求过快
                
        except Exception as e:
            self.logger.error(f"批量更新订单状态失败: {e}")
    
    async def _update_order_from_exchange(self, order: Order, exchange_data: Dict[str, Any]):
        """根据交易所数据更新订单"""
        old_status = order.status
        
        # 更新状态
        status_map = {
            'open': OrderStatus.OPEN,
            'closed': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELED,
            'cancelled': OrderStatus.CANCELED,
            'rejected': OrderStatus.REJECTED,
            'expired': OrderStatus.EXPIRED
        }
        
        exchange_status = exchange_data.get('status', '').lower()
        if exchange_status in status_map:
            order.status = status_map[exchange_status]
        
        # 更新成交信息
        filled = exchange_data.get('filled', 0)
        if filled > order.filled_quantity:
            order.filled_quantity = filled
            order.remaining_quantity = order.quantity - filled
            
            if filled == order.quantity:
                order.status = OrderStatus.FILLED
            elif filled > 0:
                order.status = OrderStatus.PARTIALLY_FILLED
        
        # 更新平均价格
        if exchange_data.get('average'):
            order.average_price = exchange_data['average']
        
        # 更新费用
        if exchange_data.get('fee'):
            fee_info = exchange_data['fee']
            if fee_info.get('cost'):
                order.fees[fee_info.get('currency', 'UNKNOWN')] = fee_info['cost']
        
        order.updated_time = datetime.now()
        
        # 如果状态有变化，发送事件
        if old_status != order.status:
            await self._emit_event('order_status_changed', {
                'order': order,
                'old_status': old_status,
                'new_status': order.status,
                'exchange_data': exchange_data
            })
            
            # 如果订单完成，发送特定事件
            if order.status == OrderStatus.FILLED:
                await self._emit_event('order_filled', {
                    'order': order,
                    'exchange_data': exchange_data
                })
                
                # 创建成交记录
                fill = OrderFill(
                    order_id=order.order_id,
                    exchange_order_id=order.exchange_order_id,
                    trade_id=exchange_data.get('id', ''),
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.filled_quantity,
                    price=order.average_price or order.price,
                    fee=sum(order.fees.values()),
                    fee_currency=list(order.fees.keys())[0] if order.fees else '',
                    timestamp=datetime.now()
                )
                
                self.fill_history.append(fill)
                self.fill_count += 1
    
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
        self.logger.debug(f"注册订单事件处理器: {event_type}")
    
    # 查询方法
    def get_order(self, order_id: str) -> Optional[Order]:
        """获取订单"""
        return self.orders.get(order_id)
    
    def get_order_by_exchange_id(self, exchange_order_id: str) -> Optional[Order]:
        """通过交易所订单ID获取订单"""
        order_id = self.orders_by_exchange_id.get(exchange_order_id)
        return self.orders.get(order_id) if order_id else None
    
    async def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """获取活跃订单"""
        orders = []
        for order in self.orders.values():
            if order.is_active and (symbol is None or order.symbol == symbol):
                orders.append(order)
        return orders
    
    async def get_order_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Order]:
        """获取订单历史"""
        orders = []
        for order in self.orders.values():
            if order.is_terminal and (symbol is None or order.symbol == symbol):
                orders.append(order)
        
        # 按时间排序
        orders.sort(key=lambda x: x.created_time, reverse=True)
        return orders[:limit]
    
    def get_fills(self, limit: int = 100) -> List[OrderFill]:
        """获取成交历史"""
        fills = list(self.fill_history)
        return fills[-limit:] if len(fills) > limit else fills
    
    async def is_healthy(self) -> bool:
        """检查系统健康状况"""
        try:
            # 检查交易所连接
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.exchange.fetch_balance)
            return True
        except Exception as e:
            self.logger.error(f"健康检查失败: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        active_orders = [o for o in self.orders.values() if o.is_active]
        
        return {
            'total_orders': len(self.orders),
            'active_orders': len(active_orders),
            'filled_orders': len([o for o in self.orders.values() if o.status == OrderStatus.FILLED]),
            'canceled_orders': len([o for o in self.orders.values() if o.status == OrderStatus.CANCELED]),
            'error_orders': len([o for o in self.orders.values() if o.status == OrderStatus.ERROR]),
            'total_fills': self.fill_count,
            'error_count': self.error_count,
            'last_order_time': self.last_order_time,
            'order_processing_enabled': self.order_processing_enabled
        }
    
    def enable_order_processing(self):
        """启用订单处理"""
        self.order_processing_enabled = True
        self.logger.info("订单处理已启用")
    
    def disable_order_processing(self):
        """禁用订单处理"""
        self.order_processing_enabled = False
        self.logger.warning("订单处理已禁用")