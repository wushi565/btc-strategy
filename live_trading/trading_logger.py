"""
交易日志系统
记录和管理所有交易相关的日志信息
"""

import logging
import logging.handlers
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass


class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(Enum):
    """日志分类"""
    TRADING = "trading"
    ORDER = "order"
    POSITION = "position"
    RISK = "risk"
    SYSTEM = "system"
    WEBSOCKET = "websocket"
    SYNC = "sync"


@dataclass
class LogEntry:
    """日志条目"""
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    message: str
    data: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'category': self.category.value,
            'message': self.message,
            'data': self.data or {}
        }


class TradingLogger:
    """
    交易日志系统
    提供结构化的交易日志记录和管理
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 日志配置
        self.log_dir = config.get('log_dir', 'logs')
        self.max_file_size = config.get('max_file_size', 10 * 1024 * 1024)  # 10MB
        self.backup_count = config.get('backup_count', 5)
        self.log_level = config.get('log_level', 'INFO')
        
        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 设置日志记录器
        self.logger = self._setup_logger()
        
        # 内存日志缓存
        self.log_buffer = []
        self.max_buffer_size = config.get('max_buffer_size', 1000)
        
        self.logger.info("TradingLogger初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('trading_engine')
        logger.setLevel(getattr(logging, self.log_level))
        
        # 清除现有处理器
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 文件处理器
        log_file = os.path.join(self.log_dir, 'trading.log')
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        
        # 日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log(self, level: LogLevel, category: LogCategory, message: str, data: Dict[str, Any] = None):
        """记录日志"""
        # 创建日志条目
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            category=category,
            message=message,
            data=data
        )
        
        # 添加到缓存
        self.log_buffer.append(entry)
        if len(self.log_buffer) > self.max_buffer_size:
            self.log_buffer.pop(0)
        
        # 记录到文件
        log_msg = f"[{category.value.upper()}] {message}"
        if data:
            log_msg += f" | Data: {json.dumps(data, default=str)}"
        
        getattr(self.logger, level.value.lower())(log_msg)
    
    def debug(self, category: LogCategory, message: str, data: Dict[str, Any] = None):
        """记录调试日志"""
        self.log(LogLevel.DEBUG, category, message, data)
    
    def info(self, category: LogCategory, message: str, data: Dict[str, Any] = None):
        """记录信息日志"""
        self.log(LogLevel.INFO, category, message, data)
    
    def warning(self, category: LogCategory, message: str, data: Dict[str, Any] = None):
        """记录警告日志"""
        self.log(LogLevel.WARNING, category, message, data)
    
    def error(self, category: LogCategory, message: str, data: Dict[str, Any] = None):
        """记录错误日志"""
        self.log(LogLevel.ERROR, category, message, data)
    
    def critical(self, category: LogCategory, message: str, data: Dict[str, Any] = None):
        """记录严重错误日志"""
        self.log(LogLevel.CRITICAL, category, message, data)
    
    # 便捷方法
    def log_order(self, message: str, data: Dict[str, Any] = None):
        """记录订单相关日志"""
        self.info(LogCategory.ORDER, message, data)
    
    def log_trade(self, message: str, data: Dict[str, Any] = None):
        """记录交易相关日志"""
        self.info(LogCategory.TRADING, message, data)
    
    def log_position(self, message: str, data: Dict[str, Any] = None):
        """记录仓位相关日志"""
        self.info(LogCategory.POSITION, message, data)
    
    def log_risk(self, message: str, data: Dict[str, Any] = None):
        """记录风险相关日志"""
        self.warning(LogCategory.RISK, message, data)
    
    def log_system(self, message: str, data: Dict[str, Any] = None):
        """记录系统相关日志"""
        self.info(LogCategory.SYSTEM, message, data)
    
    def get_recent_logs(self, category: Optional[LogCategory] = None, 
                       level: Optional[LogLevel] = None, 
                       limit: int = 100) -> List[LogEntry]:
        """获取最近的日志"""
        logs = self.log_buffer.copy()
        
        # 过滤条件
        if category:
            logs = [log for log in logs if log.category == category]
        
        if level:
            logs = [log for log in logs if log.level == level]
        
        # 返回最近的日志
        return logs[-limit:] if len(logs) > limit else logs
    
    def export_logs(self, start_time: datetime, end_time: datetime, 
                   output_file: str) -> bool:
        """导出指定时间范围的日志"""
        try:
            filtered_logs = [
                log for log in self.log_buffer
                if start_time <= log.timestamp <= end_time
            ]
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for log in filtered_logs:
                    f.write(json.dumps(log.to_dict(), ensure_ascii=False) + '\n')
            
            return True
        except Exception as e:
            self.error(LogCategory.SYSTEM, f"导出日志失败: {e}")
            return False