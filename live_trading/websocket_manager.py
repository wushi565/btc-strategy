"""
WebSocket数据管理器
处理与交易所的WebSocket连接和实时市场数据流
"""

import asyncio
import json
import logging
import websockets
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import time
from dataclasses import dataclass
import threading
from collections import deque, defaultdict


class ConnectionState(Enum):
    """连接状态"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class MessageType(Enum):
    """消息类型"""
    TICKER = "ticker"
    ORDER_BOOK = "orderbook" 
    TRADES = "trades"
    KLINE = "kline"
    ACCOUNT = "account"
    ORDER = "order"
    HEARTBEAT = "heartbeat"


@dataclass
class MarketData:
    """市场数据结构"""
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    change_24h: Optional[float] = None
    metadata: Dict[str, Any] = None


@dataclass
class OrderBookData:
    """订单簿数据结构"""
    symbol: str
    bids: List[List[float]]  # [[price, quantity], ...]
    asks: List[List[float]]  # [[price, quantity], ...]
    timestamp: datetime
    sequence: Optional[int] = None


class WebSocketDataManager:
    """
    WebSocket数据管理器
    管理与交易所的WebSocket连接，处理实时数据流
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化WebSocket数据管理器
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 连接配置
        self.exchange = config.get('exchange', 'binance')
        self.api_key = config.get('api_key', '')
        self.api_secret = config.get('api_secret', '')
        self.testnet = config.get('testnet', True)
        
        # WebSocket连接
        self.websocket = None
        self.connection_state = ConnectionState.DISCONNECTED
        self.last_heartbeat = None
        self.connection_attempts = 0
        self.max_reconnect_attempts = config.get('max_reconnect_attempts', 10)
        
        # 数据订阅
        self.subscribed_symbols = set()
        self.subscribed_channels = set()
        self.data_handlers = defaultdict(list)
        
        # 数据缓存
        self.market_data_cache = {}
        self.orderbook_cache = {}
        self.recent_trades = defaultdict(lambda: deque(maxlen=100))
        
        # 性能监控
        self.message_count = 0
        self.last_message_time = None
        self.connection_quality = 1.0
        self.latency_history = deque(maxlen=100)
        
        # 控制标志
        self.running = False
        self.auto_reconnect = config.get('auto_reconnect', True)
        self.heartbeat_interval = config.get('heartbeat_interval', 30)
        
        # 任务队列
        self.message_queue = asyncio.Queue(maxsize=1000)
        self.outbound_queue = asyncio.Queue(maxsize=100)
        
        self.logger.info(f"WebSocketDataManager初始化完成 - 交易所: {self.exchange}")
    
    async def start(self):
        """启动WebSocket数据管理器"""
        if self.running:
            raise RuntimeError("WebSocketDataManager已经在运行")
        
        self.logger.info("启动WebSocket数据管理器...")
        self.running = True
        
        # 启动各个任务
        tasks = [
            asyncio.create_task(self._connection_manager()),
            asyncio.create_task(self._message_processor()),
            asyncio.create_task(self._outbound_processor()),
            asyncio.create_task(self._heartbeat_monitor()),
            asyncio.create_task(self._quality_monitor())
        ]
        
        # 建立初始连接
        await self._connect()
        
        self.logger.info("WebSocket数据管理器启动完成")
        
        # 等待任务完成
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.logger.info("WebSocket任务被取消")
    
    async def stop(self):
        """停止WebSocket数据管理器"""
        if not self.running:
            return
        
        self.logger.info("停止WebSocket数据管理器...")
        self.running = False
        
        # 关闭WebSocket连接
        if self.websocket:
            await self.websocket.close()
        
        # 清空队列
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        while not self.outbound_queue.empty():
            try:
                self.outbound_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        self.connection_state = ConnectionState.DISCONNECTED
        self.logger.info("WebSocket数据管理器已停止")
    
    async def _connect(self):
        """建立WebSocket连接"""
        if self.connection_state == ConnectionState.CONNECTING:
            return
        
        self.connection_state = ConnectionState.CONNECTING
        self.logger.info(f"连接到 {self.exchange} WebSocket...")
        
        try:
            # 获取WebSocket URL
            ws_url = self._get_websocket_url()
            
            # 建立连接
            self.websocket = await websockets.connect(
                ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            
            self.connection_state = ConnectionState.CONNECTED
            self.connection_attempts = 0
            self.last_heartbeat = time.time()
            
            self.logger.info("WebSocket连接建立成功")
            
            # 重新订阅之前的频道
            await self._resubscribe_channels()
            
        except Exception as e:
            self.connection_state = ConnectionState.ERROR
            self.connection_attempts += 1
            self.logger.error(f"WebSocket连接失败: {e}")
            
            if self.auto_reconnect and self.connection_attempts < self.max_reconnect_attempts:
                await asyncio.sleep(min(2 ** self.connection_attempts, 30))
                await self._connect()
            else:
                raise
    
    def _get_websocket_url(self) -> str:
        """获取WebSocket URL"""
        if self.exchange.lower() == 'binance':
            if self.testnet:
                return "wss://testnet.binance.vision/ws"
            else:
                return "wss://stream.binance.com:9443/ws"
        elif self.exchange.lower() == 'okx':
            if self.testnet:
                return "wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999"
            else:
                return "wss://ws.okx.com:8443/ws/v5/public"
        else:
            raise ValueError(f"不支持的交易所: {self.exchange}")
    
    async def _connection_manager(self):
        """连接管理器"""
        while self.running:
            try:
                if self.websocket and not self.websocket.closed:
                    # 接收消息
                    message = await self.websocket.recv()
                    await self.message_queue.put(message)
                    
                    self.message_count += 1
                    self.last_message_time = time.time()
                    
                else:
                    # 连接断开，尝试重连
                    if self.auto_reconnect and self.running:
                        self.connection_state = ConnectionState.RECONNECTING
                        await self._connect()
                    else:
                        break
                        
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning("WebSocket连接关闭")
                if self.auto_reconnect and self.running:
                    self.connection_state = ConnectionState.RECONNECTING
                    await self._connect()
                else:
                    break
                    
            except Exception as e:
                self.logger.error(f"连接管理器发生错误: {e}")
                if self.running:
                    await asyncio.sleep(1)
    
    async def _message_processor(self):
        """消息处理器"""
        while self.running:
            try:
                # 获取消息
                message = await self.message_queue.get()
                
                # 解析消息
                data = json.loads(message)
                
                # 处理消息
                await self._process_message(data)
                
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.01)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON解析错误: {e}")
            except Exception as e:
                self.logger.error(f"消息处理器发生错误: {e}")
    
    async def _process_message(self, data: Dict[str, Any]):
        """处理收到的消息"""
        # 根据交易所格式处理消息
        if self.exchange.lower() == 'binance':
            await self._process_binance_message(data)
        elif self.exchange.lower() == 'okx':
            await self._process_okx_message(data)
        else:
            self.logger.warning(f"未知的交易所消息格式: {self.exchange}")
    
    async def _process_binance_message(self, data: Dict[str, Any]):
        """处理币安WebSocket消息"""
        stream = data.get('stream', '')
        event_data = data.get('data', {})
        
        if '@ticker' in stream:
            # 价格行情数据
            market_data = MarketData(
                symbol=event_data.get('s', ''),
                price=float(event_data.get('c', 0)),
                volume=float(event_data.get('v', 0)),
                timestamp=datetime.fromtimestamp(event_data.get('E', 0) / 1000),
                bid=float(event_data.get('b', 0)),
                ask=float(event_data.get('a', 0)),
                high_24h=float(event_data.get('h', 0)),
                low_24h=float(event_data.get('l', 0)),
                change_24h=float(event_data.get('P', 0))
            )
            
            await self._handle_market_data(market_data)
            
        elif '@depth' in stream:
            # 订单簿数据
            orderbook = OrderBookData(
                symbol=event_data.get('s', ''),
                bids=[[float(bid[0]), float(bid[1])] for bid in event_data.get('b', [])],
                asks=[[float(ask[0]), float(ask[1])] for ask in event_data.get('a', [])],
                timestamp=datetime.fromtimestamp(event_data.get('E', 0) / 1000),
                sequence=event_data.get('u', 0)
            )
            
            await self._handle_orderbook_data(orderbook)
            
        elif '@trade' in stream:
            # 交易数据
            trade_data = {
                'symbol': event_data.get('s', ''),
                'price': float(event_data.get('p', 0)),
                'quantity': float(event_data.get('q', 0)),
                'timestamp': datetime.fromtimestamp(event_data.get('T', 0) / 1000),
                'is_buyer_maker': event_data.get('m', False)
            }
            
            await self._handle_trade_data(trade_data)
    
    async def _process_okx_message(self, data: Dict[str, Any]):
        """处理OKX WebSocket消息"""
        # 根据OKX格式处理消息
        arg = data.get('arg', {})
        channel = arg.get('channel', '')
        
        if channel == 'tickers':
            # 价格行情数据
            for item in data.get('data', []):
                market_data = MarketData(
                    symbol=item.get('instId', ''),
                    price=float(item.get('last', 0)),
                    volume=float(item.get('vol24h', 0)),
                    timestamp=datetime.fromtimestamp(int(item.get('ts', 0)) / 1000),
                    bid=float(item.get('bidPx', 0)),
                    ask=float(item.get('askPx', 0)),
                    high_24h=float(item.get('high24h', 0)),
                    low_24h=float(item.get('low24h', 0)),
                    change_24h=float(item.get('chgUtc', 0))
                )
                
                await self._handle_market_data(market_data)
    
    async def _handle_market_data(self, data: MarketData):
        """处理市场数据"""
        # 缓存数据
        self.market_data_cache[data.symbol] = data
        
        # 通知数据处理器
        handlers = self.data_handlers.get(MessageType.TICKER, [])
        for handler in handlers:
            try:
                await handler(data)
            except Exception as e:
                self.logger.error(f"市场数据处理器错误: {e}")
    
    async def _handle_orderbook_data(self, data: OrderBookData):
        """处理订单簿数据"""
        # 缓存数据
        self.orderbook_cache[data.symbol] = data
        
        # 通知数据处理器
        handlers = self.data_handlers.get(MessageType.ORDER_BOOK, [])
        for handler in handlers:
            try:
                await handler(data)
            except Exception as e:
                self.logger.error(f"订单簿数据处理器错误: {e}")
    
    async def _handle_trade_data(self, data: Dict[str, Any]):
        """处理交易数据"""
        symbol = data['symbol']
        
        # 缓存最近交易
        self.recent_trades[symbol].append(data)
        
        # 通知数据处理器
        handlers = self.data_handlers.get(MessageType.TRADES, [])
        for handler in handlers:
            try:
                await handler(data)
            except Exception as e:
                self.logger.error(f"交易数据处理器错误: {e}")
    
    async def _outbound_processor(self):
        """出站消息处理器"""
        while self.running:
            try:
                message = await self.outbound_queue.get()
                
                if self.websocket and not self.websocket.closed:
                    await self.websocket.send(json.dumps(message))
                    self.logger.debug(f"发送消息: {message}")
                else:
                    self.logger.warning("WebSocket未连接，无法发送消息")
                    
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.01)
            except Exception as e:
                self.logger.error(f"出站消息处理器错误: {e}")
    
    async def _heartbeat_monitor(self):
        """心跳监控"""
        while self.running:
            try:
                current_time = time.time()
                
                # 检查是否需要发送心跳
                if (self.last_heartbeat and 
                    current_time - self.last_heartbeat > self.heartbeat_interval):
                    
                    await self._send_heartbeat()
                
                # 检查连接质量
                if (self.last_message_time and 
                    current_time - self.last_message_time > 60):
                    
                    self.logger.warning("长时间未收到消息，连接可能有问题")
                    self.connection_quality *= 0.9
                
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"心跳监控错误: {e}")
    
    async def _send_heartbeat(self):
        """发送心跳消息"""
        heartbeat_msg = {"op": "ping"}
        await self.outbound_queue.put(heartbeat_msg)
        self.last_heartbeat = time.time()
    
    async def _quality_monitor(self):
        """连接质量监控"""
        while self.running:
            try:
                # 计算连接质量指标
                if self.last_message_time:
                    current_time = time.time()
                    time_since_last = current_time - self.last_message_time
                    
                    if time_since_last < 5:
                        self.connection_quality = min(1.0, self.connection_quality + 0.01)
                    elif time_since_last > 30:
                        self.connection_quality = max(0.0, self.connection_quality - 0.05)
                
                await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"质量监控错误: {e}")
    
    async def subscribe_ticker(self, symbol: str):
        """订阅价格行情"""
        self.subscribed_symbols.add(symbol)
        
        if self.exchange.lower() == 'binance':
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": [f"{symbol.lower()}@ticker"],
                "id": int(time.time())
            }
        elif self.exchange.lower() == 'okx':
            subscribe_msg = {
                "op": "subscribe",
                "args": [{"channel": "tickers", "instId": symbol}]
            }
        else:
            raise ValueError(f"不支持的交易所: {self.exchange}")
        
        await self.outbound_queue.put(subscribe_msg)
        self.logger.info(f"订阅价格行情: {symbol}")
    
    async def subscribe_orderbook(self, symbol: str, depth: int = 20):
        """订阅订单簿"""
        self.subscribed_symbols.add(symbol)
        
        if self.exchange.lower() == 'binance':
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": [f"{symbol.lower()}@depth{depth}@100ms"],
                "id": int(time.time())
            }
        elif self.exchange.lower() == 'okx':
            subscribe_msg = {
                "op": "subscribe",
                "args": [{"channel": "books", "instId": symbol}]
            }
        else:
            raise ValueError(f"不支持的交易所: {self.exchange}")
        
        await self.outbound_queue.put(subscribe_msg)
        self.logger.info(f"订阅订单簿: {symbol}")
    
    async def subscribe_trades(self, symbol: str):
        """订阅实时交易"""
        self.subscribed_symbols.add(symbol)
        
        if self.exchange.lower() == 'binance':
            subscribe_msg = {
                "method": "SUBSCRIBE", 
                "params": [f"{symbol.lower()}@trade"],
                "id": int(time.time())
            }
        elif self.exchange.lower() == 'okx':
            subscribe_msg = {
                "op": "subscribe",
                "args": [{"channel": "trades", "instId": symbol}]
            }
        else:
            raise ValueError(f"不支持的交易所: {self.exchange}")
        
        await self.outbound_queue.put(subscribe_msg)
        self.logger.info(f"订阅实时交易: {symbol}")
    
    async def unsubscribe(self, symbol: str, channel: str):
        """取消订阅"""
        if symbol in self.subscribed_symbols:
            self.subscribed_symbols.remove(symbol)
        
        if self.exchange.lower() == 'binance':
            unsubscribe_msg = {
                "method": "UNSUBSCRIBE",
                "params": [f"{symbol.lower()}@{channel}"],
                "id": int(time.time())
            }
        elif self.exchange.lower() == 'okx':
            unsubscribe_msg = {
                "op": "unsubscribe",
                "args": [{"channel": channel, "instId": symbol}]
            }
        else:
            raise ValueError(f"不支持的交易所: {self.exchange}")
        
        await self.outbound_queue.put(unsubscribe_msg)
        self.logger.info(f"取消订阅: {symbol}@{channel}")
    
    async def _resubscribe_channels(self):
        """重新订阅频道"""
        if self.subscribed_symbols:
            self.logger.info("重新订阅频道...")
            
            for symbol in self.subscribed_symbols.copy():
                try:
                    await self.subscribe_ticker(symbol)
                    await self.subscribe_orderbook(symbol)
                    await self.subscribe_trades(symbol)
                    await asyncio.sleep(0.1)  # 避免订阅过快
                except Exception as e:
                    self.logger.error(f"重新订阅失败 {symbol}: {e}")
    
    def register_handler(self, message_type: MessageType, handler: Callable):
        """注册数据处理器"""
        self.data_handlers[message_type].append(handler)
        self.logger.debug(f"注册数据处理器: {message_type.value}")
    
    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """获取最新市场数据"""
        return self.market_data_cache.get(symbol)
    
    def get_orderbook(self, symbol: str) -> Optional[OrderBookData]:
        """获取最新订单簿数据"""
        return self.orderbook_cache.get(symbol)
    
    def get_recent_trades(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近交易数据"""
        trades = list(self.recent_trades.get(symbol, []))
        return trades[-limit:] if trades else []
    
    async def reconnect(self):
        """手动重连"""
        self.logger.info("手动触发重连...")
        
        if self.websocket:
            await self.websocket.close()
        
        await self._connect()
    
    async def is_healthy(self) -> bool:
        """检查健康状况"""
        return (self.connection_state == ConnectionState.CONNECTED and
                self.connection_quality > 0.5 and
                self.websocket and not self.websocket.closed)
    
    def get_connection_info(self) -> Dict[str, Any]:
        """获取连接信息"""
        return {
            'state': self.connection_state.value,
            'quality': self.connection_quality,
            'message_count': self.message_count,
            'subscribed_symbols': list(self.subscribed_symbols),
            'connection_attempts': self.connection_attempts,
            'last_message_time': self.last_message_time,
            'uptime': time.time() - (self.last_heartbeat or time.time())
        }