"""
追踪止损管理器
管理动态追踪止损订单
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass
import math


@dataclass
class TrailingStop:
    """追踪止损配置"""
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    trailing_distance: float  # 追踪距离(价格或百分比)
    is_percentage: bool = True
    stop_price: float = 0.0
    highest_price: float = 0.0  # 多头最高价
    lowest_price: float = 0.0   # 空头最低价
    is_active: bool = True
    created_time: datetime = None
    
    def __post_init__(self):
        if self.created_time is None:
            self.created_time = datetime.now()


class TrailingStopManager:
    """
    追踪止损管理器
    管理动态止损策略
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 追踪止损存储
        self.trailing_stops: Dict[str, TrailingStop] = {}
        
        # 事件处理器
        self.event_handlers = []
        
        # 运行控制
        self.running = False
        
        self.logger.info("TrailingStopManager初始化完成")
    
    async def start(self):
        """启动追踪止损管理器"""
        if self.running:
            return
        
        self.logger.info("启动追踪止损管理器...")
        self.running = True
        
        # 启动监控任务
        asyncio.create_task(self._monitor_loop())
    
    async def stop(self):
        """停止追踪止损管理器"""
        self.running = False
        self.logger.info("追踪止损管理器已停止")
    
    async def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                await self._update_trailing_stops()
                await asyncio.sleep(1)  # 每秒检查一次
            except Exception as e:
                self.logger.error(f"追踪止损监控错误: {e}")
                await asyncio.sleep(1)
    
    async def _update_trailing_stops(self):
        """更新所有追踪止损"""
        for stop_id, trailing_stop in self.trailing_stops.items():
            if trailing_stop.is_active:
                # 这里应该根据最新价格更新止损价格
                pass
    
    async def update_price(self, symbol: str, price: float):
        """更新价格并调整追踪止损"""
        for stop_id, trailing_stop in self.trailing_stops.items():
            if trailing_stop.symbol == symbol and trailing_stop.is_active:
                await self._adjust_trailing_stop(trailing_stop, price)
    
    async def _adjust_trailing_stop(self, trailing_stop: TrailingStop, current_price: float):
        """调整追踪止损价格"""
        if trailing_stop.side == 'long':
            # 多头追踪止损
            if current_price > trailing_stop.highest_price:
                trailing_stop.highest_price = current_price
                
                # 计算新的止损价格
                if trailing_stop.is_percentage:
                    new_stop_price = current_price * (1 - trailing_stop.trailing_distance / 100)
                else:
                    new_stop_price = current_price - trailing_stop.trailing_distance
                
                if new_stop_price > trailing_stop.stop_price:
                    trailing_stop.stop_price = new_stop_price
                    self.logger.info(f"更新多头追踪止损: {trailing_stop.symbol} @ {new_stop_price}")
        
        elif trailing_stop.side == 'short':
            # 空头追踪止损
            if current_price < trailing_stop.lowest_price or trailing_stop.lowest_price == 0:
                trailing_stop.lowest_price = current_price
                
                # 计算新的止损价格
                if trailing_stop.is_percentage:
                    new_stop_price = current_price * (1 + trailing_stop.trailing_distance / 100)
                else:
                    new_stop_price = current_price + trailing_stop.trailing_distance
                
                if new_stop_price < trailing_stop.stop_price or trailing_stop.stop_price == 0:
                    trailing_stop.stop_price = new_stop_price
                    self.logger.info(f"更新空头追踪止损: {trailing_stop.symbol} @ {new_stop_price}")
    
    def add_trailing_stop(self, symbol: str, side: str, entry_price: float, 
                         trailing_distance: float, is_percentage: bool = True) -> str:
        """添加追踪止损"""
        stop_id = f"{symbol}_{side}_{int(datetime.now().timestamp())}"
        
        trailing_stop = TrailingStop(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            trailing_distance=trailing_distance,
            is_percentage=is_percentage,
            highest_price=entry_price if side == 'long' else 0,
            lowest_price=entry_price if side == 'short' else 0
        )
        
        self.trailing_stops[stop_id] = trailing_stop
        self.logger.info(f"添加追踪止损: {stop_id}")
        
        return stop_id
    
    def remove_trailing_stop(self, stop_id: str):
        """移除追踪止损"""
        if stop_id in self.trailing_stops:
            del self.trailing_stops[stop_id]
            self.logger.info(f"移除追踪止损: {stop_id}")
    
    async def is_healthy(self) -> bool:
        """检查健康状况"""
        return True