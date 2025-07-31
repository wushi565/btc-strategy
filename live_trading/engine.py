"""
实时交易执行引擎核心模块
LiveTradingEngine - 实时交易系统的主控制器
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass
import threading
import time
from collections import defaultdict

from .websocket_manager import WebSocketDataManager
from .order_management import OrderManagementSystem, Order, OrderStatus, OrderType
from .position_manager import PositionManager, Position
from .state_synchronizer import StateSynchronizer
from .trailing_stop_manager import TrailingStopManager
from .trading_logger import TradingLogger


class EngineState(Enum):
    """交易引擎状态"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused" 
    STOPPING = "stopping"
    ERROR = "error"


class EmergencyLevel(Enum):
    """紧急情况级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TradingSignal:
    """交易信号数据结构"""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    price: float
    quantity: float
    confidence: float
    timestamp: datetime
    strategy_id: str
    metadata: Dict[str, Any] = None


class LiveTradingEngine:
    """
    实时交易执行引擎主控制器
    采用事件驱动架构，协调各个子系统的运行
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化交易引擎
        
        Args:
            config: 引擎配置参数
        """
        self.config = config
        self.state = EngineState.STOPPED
        self.start_time = None
        
        # 核心组件初始化
        self.logger = TradingLogger(config.get('logging', {}))
        self.websocket_manager = WebSocketDataManager(config.get('websocket', {}))
        self.order_manager = OrderManagementSystem(config.get('orders', {}))
        self.position_manager = PositionManager(config.get('positions', {}))
        self.state_synchronizer = StateSynchronizer(config.get('sync', {}))
        self.trailing_stop_manager = TrailingStopManager(config.get('trailing_stops', {}))
        
        # 事件处理器
        self.event_handlers = defaultdict(list)
        self.signal_queue = asyncio.Queue()
        self.emergency_queue = asyncio.Queue()
        
        # 运行时数据
        self.active_strategies = {}
        self.market_data = {}
        self.performance_metrics = {}
        
        # 控制标志
        self.running = False
        self.emergency_stop = False
        self.manual_override = False
        
        # 注册默认事件处理器
        self._register_event_handlers()
        
        self.logger.info("LiveTradingEngine初始化完成")
    
    def _register_event_handlers(self):
        """注册默认事件处理器"""
        # 市场数据事件
        self.register_handler('market_data', self._handle_market_data)
        self.register_handler('price_update', self._handle_price_update)
        
        # 订单事件
        self.register_handler('order_filled', self._handle_order_filled)
        self.register_handler('order_cancelled', self._handle_order_cancelled)
        self.register_handler('order_error', self._handle_order_error)
        
        # 系统事件
        self.register_handler('connection_lost', self._handle_connection_lost)
        self.register_handler('system_error', self._handle_system_error)
        self.register_handler('emergency_stop', self._handle_emergency_stop)
    
    def register_handler(self, event_type: str, handler: Callable):
        """注册事件处理器"""
        self.event_handlers[event_type].append(handler)
        self.logger.debug(f"注册事件处理器: {event_type}")
    
    async def start(self):
        """启动交易引擎"""
        if self.state != EngineState.STOPPED:
            raise RuntimeError(f"引擎当前状态为 {self.state.value}，无法启动")
        
        self.logger.info("开始启动实时交易引擎...")
        self.state = EngineState.STARTING
        
        try:
            # 启动各个子系统
            await self._start_subsystems()
            
            # 启动主事件循环
            self.running = True
            self.start_time = datetime.now()
            self.state = EngineState.RUNNING
            
            # 创建主要任务
            tasks = [
                asyncio.create_task(self._main_trading_loop()),
                asyncio.create_task(self._emergency_monitor()),
                asyncio.create_task(self._health_monitor()),
                asyncio.create_task(self._performance_monitor())
            ]
            
            self.logger.info("实时交易引擎启动成功")
            
            # 等待任务完成
            await asyncio.gather(*tasks)
            
        except Exception as e:
            self.logger.error(f"引擎启动失败: {e}")
            self.state = EngineState.ERROR
            raise
    
    async def _start_subsystems(self):
        """启动各个子系统"""
        self.logger.info("启动子系统...")
        
        # 启动WebSocket数据管理器
        await self.websocket_manager.start()
        
        # 启动订单管理系统
        await self.order_manager.start()
        
        # 启动仓位管理器  
        await self.position_manager.start()
        
        # 启动状态同步器
        await self.state_synchronizer.start()
        
        # 启动追踪止损管理器
        await self.trailing_stop_manager.start()
        
        self.logger.info("所有子系统启动完成")
    
    async def stop(self):
        """停止交易引擎"""
        if self.state == EngineState.STOPPED:
            return
        
        self.logger.info("开始停止实时交易引擎...")
        self.state = EngineState.STOPPING
        self.running = False
        
        try:
            # 停止所有子系统
            await self._stop_subsystems()
            
            # 清理资源
            await self._cleanup_resources()
            
            self.state = EngineState.STOPPED
            self.logger.info("实时交易引擎已停止")
            
        except Exception as e:
            self.logger.error(f"引擎停止时发生错误: {e}")
            self.state = EngineState.ERROR
            raise
    
    async def _stop_subsystems(self):
        """停止各个子系统"""
        self.logger.info("停止子系统...")
        
        # 停止追踪止损管理器
        await self.trailing_stop_manager.stop()
        
        # 停止状态同步器
        await self.state_synchronizer.stop()
        
        # 停止仓位管理器
        await self.position_manager.stop()
        
        # 停止订单管理系统
        await self.order_manager.stop()
        
        # 停止WebSocket数据管理器
        await self.websocket_manager.stop()
        
        self.logger.info("所有子系统已停止")
    
    async def _cleanup_resources(self):
        """清理资源"""
        # 清空队列
        while not self.signal_queue.empty():
            try:
                self.signal_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        while not self.emergency_queue.empty():
            try:
                self.emergency_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
    
    async def _main_trading_loop(self):
        """主交易循环"""
        self.logger.info("主交易循环开始运行")
        
        while self.running:
            try:
                # 处理交易信号
                await self._process_trading_signals()
                
                # 更新市场数据
                await self._update_market_data()
                
                # 检查订单状态
                await self._check_order_status()
                
                # 更新仓位
                await self._update_positions()
                
                # 检查止损条件
                await self._check_stop_conditions()
                
                # 同步状态
                await self._sync_state()
                
                # 短暂休息
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"主交易循环发生错误: {e}")
                await self._handle_system_error({'error': e, 'source': 'main_loop'})
    
    async def _process_trading_signals(self):
        """处理交易信号"""
        try:
            # 非阻塞地获取信号
            while not self.signal_queue.empty():
                signal = await self.signal_queue.get()
                await self._execute_trading_signal(signal)
        except Exception as e:
            self.logger.error(f"处理交易信号时发生错误: {e}")
    
    async def _execute_trading_signal(self, signal: TradingSignal):
        """执行交易信号"""
        if self.manual_override:
            self.logger.warning(f"手动干预模式，跳过信号执行: {signal.symbol}")
            return
        
        self.logger.info(f"执行交易信号: {signal.symbol} {signal.action} {signal.quantity}@{signal.price}")
        
        try:
            # 创建订单
            order = Order(
                symbol=signal.symbol,
                order_type=OrderType.MARKET if signal.action in ['BUY', 'SELL'] else OrderType.LIMIT,
                side=signal.action.lower(),
                quantity=signal.quantity,
                price=signal.price if signal.action not in ['BUY', 'SELL'] else None,
                strategy_id=signal.strategy_id,
                metadata=signal.metadata
            )
            
            # 提交订单
            order_id = await self.order_manager.submit_order(order)
            
            self.logger.info(f"订单已提交: {order_id}")
            
        except Exception as e:
            self.logger.error(f"执行交易信号失败: {e}")
            await self._emit_event('signal_execution_error', {
                'signal': signal,
                'error': e
            })
    
    async def _emergency_monitor(self):
        """紧急情况监控"""
        self.logger.info("紧急情况监控开始运行")
        
        while self.running:
            try:
                # 检查紧急情况队列
                if not self.emergency_queue.empty():
                    emergency = await self.emergency_queue.get()
                    await self._handle_emergency(emergency)
                
                # 检查系统健康状况
                await self._check_system_health()
                
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"紧急情况监控发生错误: {e}")
    
    async def _handle_emergency(self, emergency: Dict[str, Any]):
        """处理紧急情况"""
        level = emergency.get('level', EmergencyLevel.MEDIUM)
        message = emergency.get('message', '未知紧急情况')
        
        self.logger.critical(f"紧急情况 [{level.value}]: {message}")
        
        if level == EmergencyLevel.CRITICAL:
            await self._emergency_stop_all()
        elif level == EmergencyLevel.HIGH:
            await self._pause_trading()
    
    async def _emergency_stop_all(self):
        """紧急停止所有交易"""
        self.logger.critical("执行紧急停止操作")
        self.emergency_stop = True
        
        # 取消所有挂单
        await self.order_manager.cancel_all_orders()
        
        # 平仓所有持仓(如果配置允许)
        if self.config.get('emergency', {}).get('close_all_positions', False):
            await self.position_manager.close_all_positions()
        
        # 暂停交易
        await self._pause_trading()
    
    async def _pause_trading(self):
        """暂停交易"""
        self.logger.warning("暂停交易")
        self.state = EngineState.PAUSED
        self.manual_override = True
    
    async def resume_trading(self):
        """恢复交易"""
        if self.state != EngineState.PAUSED:
            raise RuntimeError(f"当前状态 {self.state.value} 无法恢复交易")
        
        self.logger.info("恢复交易")
        self.manual_override = False
        self.emergency_stop = False
        self.state = EngineState.RUNNING
    
    async def submit_signal(self, signal: TradingSignal):
        """提交交易信号"""
        await self.signal_queue.put(signal)
        self.logger.debug(f"交易信号已提交: {signal.symbol} {signal.action}")
    
    async def emergency_stop(self, level: EmergencyLevel = EmergencyLevel.HIGH, message: str = "手动紧急停止"):
        """触发紧急停止"""
        emergency = {
            'level': level,
            'message': message,
            'timestamp': datetime.now()
        }
        await self.emergency_queue.put(emergency)
    
    # 事件处理器方法
    async def _handle_market_data(self, data: Dict[str, Any]):
        """处理市场数据更新"""
        symbol = data.get('symbol')
        if symbol:
            self.market_data[symbol] = data
    
    async def _handle_price_update(self, data: Dict[str, Any]):
        """处理价格更新"""
        symbol = data.get('symbol')
        price = data.get('price')
        
        if symbol and price:
            # 通知追踪止损管理器
            await self.trailing_stop_manager.update_price(symbol, price)
    
    async def _handle_order_filled(self, data: Dict[str, Any]):
        """处理订单成交"""
        order_id = data.get('order_id')
        self.logger.info(f"订单成交: {order_id}")
        
        # 更新仓位
        await self.position_manager.update_from_fill(data)
    
    async def _handle_order_cancelled(self, data: Dict[str, Any]):
        """处理订单取消"""
        order_id = data.get('order_id')
        self.logger.info(f"订单已取消: {order_id}")
    
    async def _handle_order_error(self, data: Dict[str, Any]):
        """处理订单错误"""
        order_id = data.get('order_id')
        error = data.get('error')
        self.logger.error(f"订单错误 {order_id}: {error}")
    
    async def _handle_connection_lost(self, data: Dict[str, Any]):
        """处理连接丢失"""
        source = data.get('source', 'unknown')
        self.logger.error(f"连接丢失: {source}")
        
        # 触发重连
        if source == 'websocket':
            await self.websocket_manager.reconnect()
    
    async def _handle_system_error(self, data: Dict[str, Any]):
        """处理系统错误"""
        error = data.get('error')
        source = data.get('source', 'unknown')
        
        self.logger.error(f"系统错误 from {source}: {error}")
        
        # 根据错误类型决定处理方式
        if self._is_critical_error(error):
            await self.emergency_stop(EmergencyLevel.CRITICAL, f"关键系统错误: {error}")
    
    async def _handle_emergency_stop(self, data: Dict[str, Any]):
        """处理紧急停止事件"""
        await self._emergency_stop_all()
    
    def _is_critical_error(self, error: Exception) -> bool:
        """判断是否为关键错误"""
        critical_errors = [
            'ConnectionError',
            'TimeoutError',
            'AuthenticationError'
        ]
        return any(err in str(type(error)) for err in critical_errors)
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """发送事件"""
        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                await handler(data)
            except Exception as e:
                self.logger.error(f"事件处理器执行失败 {event_type}: {e}")
    
    # 监控方法
    async def _health_monitor(self):
        """健康状况监控"""
        while self.running:
            try:
                await self._check_system_health()
                await asyncio.sleep(5)
            except Exception as e:
                self.logger.error(f"健康监控发生错误: {e}")
    
    async def _performance_monitor(self):
        """性能监控"""
        while self.running:
            try:
                await self._update_performance_metrics()
                await asyncio.sleep(10)
            except Exception as e:
                self.logger.error(f"性能监控发生错误: {e}")
    
    async def _check_system_health(self):
        """检查系统健康状况"""
        # 检查各个子系统状态
        health_status = {
            'websocket': await self.websocket_manager.is_healthy(),
            'orders': await self.order_manager.is_healthy(),
            'positions': await self.position_manager.is_healthy(),
            'sync': await self.state_synchronizer.is_healthy(),
            'trailing_stops': await self.trailing_stop_manager.is_healthy()
        }
        
        # 检查是否有不健康的子系统
        unhealthy = [k for k, v in health_status.items() if not v]
        if unhealthy:
            self.logger.warning(f"不健康的子系统: {unhealthy}")
    
    async def _update_performance_metrics(self):
        """更新性能指标"""
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()
            self.performance_metrics['uptime'] = uptime
        
        self.performance_metrics['signal_queue_size'] = self.signal_queue.qsize()
        self.performance_metrics['emergency_queue_size'] = self.emergency_queue.qsize()
        self.performance_metrics['active_orders'] = len(await self.order_manager.get_active_orders())
        self.performance_metrics['positions'] = len(await self.position_manager.get_all_positions())
    
    # 其他抽象方法(需要在子系统中实现)
    async def _update_market_data(self):
        """更新市场数据 - 委托给WebSocket管理器"""
        pass
    
    async def _check_order_status(self):
        """检查订单状态 - 委托给订单管理系统"""
        pass
    
    async def _update_positions(self):
        """更新仓位 - 委托给仓位管理器"""
        pass
    
    async def _check_stop_conditions(self):
        """检查止损条件 - 委托给追踪止损管理器"""
        pass
    
    async def _sync_state(self):
        """同步状态 - 委托给状态同步器"""
        pass
    
    # 状态查询方法
    def get_state(self) -> EngineState:
        """获取引擎状态"""
        return self.state
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return self.performance_metrics.copy()
    
    def is_running(self) -> bool:
        """检查引擎是否运行中"""
        return self.state == EngineState.RUNNING
    
    def is_manual_override(self) -> bool:
        """检查是否处于手动干预模式"""
        return self.manual_override