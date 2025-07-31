"""
风险指标计算器
实现VaR、CVaR、夏普比率等高级风险指标计算
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
import warnings


class RiskMetricType(Enum):
    """风险指标类型"""
    VALUE_AT_RISK = "var"  # 风险价值
    CONDITIONAL_VAR = "cvar"  # 条件风险价值
    SHARPE_RATIO = "sharpe"  # 夏普比率
    SORTINO_RATIO = "sortino"  # 索提诺比率
    MAXIMUM_DRAWDOWN = "max_dd"  # 最大回撤
    CALMAR_RATIO = "calmar"  # 卡尔玛比率
    VOLATILITY = "volatility"  # 波动率
    BETA = "beta"  # 贝塔系数
    CORRELATION = "correlation"  # 相关性
    INFORMATION_RATIO = "info_ratio"  # 信息比率


class ConfidenceLevel(Enum):
    """置信水平"""
    NINETY = 0.90
    NINETY_FIVE = 0.95
    NINETY_NINE = 0.99


@dataclass
class RiskMetric:
    """风险指标数据结构"""
    metric_type: RiskMetricType
    value: float
    confidence_level: Optional[ConfidenceLevel] = None
    period: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'metric_type': self.metric_type.value,
            'value': self.value,
            'confidence_level': self.confidence_level.value if self.confidence_level else None,
            'period': self.period,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class RiskReport:
    """风险报告数据结构"""
    symbol: str
    timestamp: datetime
    metrics: List[RiskMetric]
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def get_metric(self, metric_type: RiskMetricType) -> Optional[RiskMetric]:
        """获取指定类型的风险指标"""
        for metric in self.metrics:
            if metric.metric_type == metric_type:
                return metric
        return None


class RiskMetricsCalculator:
    """
    风险指标计算器
    
    核心功能：
    - VaR和CVaR计算
    - 夏普比率、索提诺比率计算
    - 最大回撤和波动率分析
    - 相关性和贝塔系数计算
    - 历史数据管理和缓存
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化风险指标计算器
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger('RiskMetricsCalculator')
        
        # 配置参数
        self.default_confidence_level = ConfidenceLevel(
            config.get('default_confidence_level', 0.95)
        )
        self.default_period = config.get('default_period', 252)  # 一年交易日
        self.risk_free_rate = config.get('risk_free_rate', 0.02)  # 无风险利率
        self.max_history_size = config.get('max_history_size', 10000)
        
        # 历史数据存储
        self.price_history: Dict[str, deque] = {}
        self.return_history: Dict[str, deque] = {}
        self.pnl_history: Dict[str, deque] = {}
        
        # 基准数据（用于贝塔和相关性计算）
        self.benchmark_returns: deque = deque(maxlen=self.max_history_size)
        
        # 缓存计算结果
        self.metrics_cache: Dict[str, RiskReport] = {}
        self.cache_ttl = config.get('cache_ttl', 300)  # 5分钟缓存
        
        self.logger.info("风险指标计算器初始化完成")
    
    def update_price_data(self, symbol: str, price: float, timestamp: datetime = None):
        """
        更新价格数据
        
        Args:
            symbol: 交易对符号
            price: 价格
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # 初始化历史数据存储
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.max_history_size)
            self.return_history[symbol] = deque(maxlen=self.max_history_size)
        
        # 添加价格数据
        self.price_history[symbol].append({
            'price': price,
            'timestamp': timestamp
        })
        
        # 计算收益率
        if len(self.price_history[symbol]) >= 2:
            prev_price = self.price_history[symbol][-2]['price']
            if prev_price > 0:
                return_rate = (price - prev_price) / prev_price
                self.return_history[symbol].append({
                    'return': return_rate,
                    'timestamp': timestamp
                })
        
        # 清除相关缓存
        self._invalidate_cache(symbol)
    
    def update_pnl_data(self, symbol: str, pnl: float, timestamp: datetime = None):
        """
        更新损益数据
        
        Args:
            symbol: 交易对符号
            pnl: 损益
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if symbol not in self.pnl_history:
            self.pnl_history[symbol] = deque(maxlen=self.max_history_size)
        
        self.pnl_history[symbol].append({
            'pnl': pnl,
            'timestamp': timestamp
        })
        
        self._invalidate_cache(symbol)
    
    def update_benchmark_data(self, return_rate: float, timestamp: datetime = None):
        """
        更新基准数据
        
        Args:
            return_rate: 基准收益率
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.benchmark_returns.append({
            'return': return_rate,
            'timestamp': timestamp
        })
    
    def calculate_var(self, symbol: str, confidence_level: ConfidenceLevel = None,
                     period: int = None, method: str = 'historical') -> float:
        """
        计算风险价值(VaR)
        
        Args:
            symbol: 交易对符号
            confidence_level: 置信水平
            period: 计算周期
            method: 计算方法 ('historical', 'parametric', 'monte_carlo')
            
        Returns:
            VaR值
        """
        if confidence_level is None:
            confidence_level = self.default_confidence_level
        if period is None:
            period = min(self.default_period, len(self.return_history.get(symbol, [])))
        
        returns = self._get_recent_returns(symbol, period)
        if len(returns) < 30:  # 最少需要30个数据点
            self.logger.warning(f"数据不足，无法计算VaR: {symbol}")
            return 0.0
        
        try:
            if method == 'historical':
                return self._calculate_historical_var(returns, confidence_level)
            elif method == 'parametric':
                return self._calculate_parametric_var(returns, confidence_level)
            elif method == 'monte_carlo':
                return self._calculate_monte_carlo_var(returns, confidence_level)
            else:
                raise ValueError(f"不支持的VaR计算方法: {method}")
        except Exception as e:
            self.logger.error(f"VaR计算失败 {symbol}: {e}")
            return 0.0
    
    def calculate_cvar(self, symbol: str, confidence_level: ConfidenceLevel = None,
                      period: int = None) -> float:
        """
        计算条件风险价值(CVaR/Expected Shortfall)
        
        Args:
            symbol: 交易对符号
            confidence_level: 置信水平
            period: 计算周期
            
        Returns:
            CVaR值
        """
        if confidence_level is None:
            confidence_level = self.default_confidence_level
        if period is None:
            period = min(self.default_period, len(self.return_history.get(symbol, [])))
        
        returns = self._get_recent_returns(symbol, period)
        if len(returns) < 30:
            self.logger.warning(f"数据不足，无法计算CVaR: {symbol}")
            return 0.0
        
        try:
            var_threshold = np.percentile(returns, (1 - confidence_level.value) * 100)
            tail_losses = [r for r in returns if r <= var_threshold]
            return abs(np.mean(tail_losses)) if tail_losses else 0.0
        except Exception as e:
            self.logger.error(f"CVaR计算失败 {symbol}: {e}")
            return 0.0
    
    def calculate_sharpe_ratio(self, symbol: str, period: int = None) -> float:
        """
        计算夏普比率
        
        Args:
            symbol: 交易对符号
            period: 计算周期
            
        Returns:
            夏普比率
        """
        if period is None:
            period = min(self.default_period, len(self.return_history.get(symbol, [])))
        
        returns = self._get_recent_returns(symbol, period)
        if len(returns) < 30:
            self.logger.warning(f"数据不足，无法计算夏普比率: {symbol}")
            return 0.0
        
        try:
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
            
            if std_return == 0:
                return 0.0
            
            # 年化处理
            annualized_return = mean_return * np.sqrt(252)
            annualized_std = std_return * np.sqrt(252)
            
            return (annualized_return - self.risk_free_rate) / annualized_std
        except Exception as e:
            self.logger.error(f"夏普比率计算失败 {symbol}: {e}")
            return 0.0
    
    def calculate_sortino_ratio(self, symbol: str, period: int = None) -> float:
        """
        计算索提诺比率
        
        Args:
            symbol: 交易对符号
            period: 计算周期
            
        Returns:
            索提诺比率
        """
        if period is None:
            period = min(self.default_period, len(self.return_history.get(symbol, [])))
        
        returns = self._get_recent_returns(symbol, period)
        if len(returns) < 30:
            self.logger.warning(f"数据不足，无法计算索提诺比率: {symbol}")
            return 0.0
        
        try:
            mean_return = np.mean(returns)
            negative_returns = [r for r in returns if r < 0]
            
            if not negative_returns:
                return float('inf') if mean_return > 0 else 0.0
            
            downside_deviation = np.std(negative_returns, ddof=1)
            
            if downside_deviation == 0:
                return 0.0
            
            # 年化处理
            annualized_return = mean_return * np.sqrt(252)
            annualized_downside = downside_deviation * np.sqrt(252)
            
            return (annualized_return - self.risk_free_rate) / annualized_downside
        except Exception as e:
            self.logger.error(f"索提诺比率计算失败 {symbol}: {e}")
            return 0.0
    
    def calculate_maximum_drawdown(self, symbol: str, period: int = None) -> Tuple[float, int]:
        """
        计算最大回撤
        
        Args:
            symbol: 交易对符号
            period: 计算周期
            
        Returns:
            (最大回撤, 回撤持续天数)
        """
        if period is None:
            period = min(self.default_period, len(self.price_history.get(symbol, [])))
        
        prices = self._get_recent_prices(symbol, period)
        if len(prices) < 10:
            self.logger.warning(f"数据不足，无法计算最大回撤: {symbol}")
            return 0.0, 0
        
        try:
            cumulative_returns = np.cumprod(1 + np.array(self._get_recent_returns(symbol, period)))
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak) / peak
            
            max_drawdown = abs(np.min(drawdown))
            
            # 计算回撤持续时间
            max_dd_idx = np.argmin(drawdown)
            recovery_idx = np.where(cumulative_returns[max_dd_idx:] >= peak[max_dd_idx])[0]
            duration = len(recovery_idx) if len(recovery_idx) > 0 else len(drawdown) - max_dd_idx
            
            return max_drawdown, duration
        except Exception as e:
            self.logger.error(f"最大回撤计算失败 {symbol}: {e}")
            return 0.0, 0
    
    def calculate_volatility(self, symbol: str, period: int = None, 
                           annualized: bool = True) -> float:
        """
        计算波动率
        
        Args:
            symbol: 交易对符号
            period: 计算周期
            annualized: 是否年化
            
        Returns:
            波动率
        """
        if period is None:
            period = min(self.default_period, len(self.return_history.get(symbol, [])))
        
        returns = self._get_recent_returns(symbol, period)
        if len(returns) < 10:
            self.logger.warning(f"数据不足，无法计算波动率: {symbol}")
            return 0.0
        
        try:
            volatility = np.std(returns, ddof=1)
            return volatility * np.sqrt(252) if annualized else volatility
        except Exception as e:
            self.logger.error(f"波动率计算失败 {symbol}: {e}")
            return 0.0
    
    def calculate_beta(self, symbol: str, period: int = None) -> float:
        """
        计算贝塔系数
        
        Args:
            symbol: 交易对符号
            period: 计算周期
            
        Returns:
            贝塔系数
        """
        if period is None:
            period = min(self.default_period, len(self.return_history.get(symbol, [])))
        
        returns = self._get_recent_returns(symbol, period)
        benchmark_returns = self._get_recent_benchmark_returns(period)
        
        if len(returns) < 30 or len(benchmark_returns) < 30:
            self.logger.warning(f"数据不足，无法计算贝塔系数: {symbol}")
            return 1.0  # 默认贝塔为1
        
        try:
            # 确保数据长度一致
            min_length = min(len(returns), len(benchmark_returns))
            returns = returns[-min_length:]
            benchmark_returns = benchmark_returns[-min_length:]
            
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns, ddof=1)
            
            if benchmark_variance == 0:
                return 1.0
            
            return covariance / benchmark_variance
        except Exception as e:
            self.logger.error(f"贝塔系数计算失败 {symbol}: {e}")
            return 1.0
    
    def calculate_correlation(self, symbol: str, period: int = None) -> float:
        """
        计算与基准的相关系数
        
        Args:
            symbol: 交易对符号
            period: 计算周期
            
        Returns:
            相关系数
        """
        if period is None:
            period = min(self.default_period, len(self.return_history.get(symbol, [])))
        
        returns = self._get_recent_returns(symbol, period)
        benchmark_returns = self._get_recent_benchmark_returns(period)
        
        if len(returns) < 30 or len(benchmark_returns) < 30:
            self.logger.warning(f"数据不足，无法计算相关系数: {symbol}")
            return 0.0
        
        try:
            # 确保数据长度一致
            min_length = min(len(returns), len(benchmark_returns))
            returns = returns[-min_length:]
            benchmark_returns = benchmark_returns[-min_length:]
            
            correlation = np.corrcoef(returns, benchmark_returns)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except Exception as e:
            self.logger.error(f"相关系数计算失败 {symbol}: {e}")
            return 0.0
    
    def generate_risk_report(self, symbol: str, period: int = None) -> RiskReport:
        """
        生成综合风险报告
        
        Args:
            symbol: 交易对符号
            period: 计算周期
            
        Returns:
            风险报告
        """
        # 检查缓存
        cache_key = f"{symbol}_{period or self.default_period}"
        cached_report = self.metrics_cache.get(cache_key)
        if cached_report and self._is_cache_valid(cached_report.timestamp):
            return cached_report
        
        if period is None:
            period = min(self.default_period, len(self.return_history.get(symbol, [])))
        
        try:
            # 计算各项风险指标
            metrics = []
            
            # VaR (95%置信度)
            var_95 = self.calculate_var(symbol, ConfidenceLevel.NINETY_FIVE, period)
            metrics.append(RiskMetric(
                RiskMetricType.VALUE_AT_RISK,
                var_95,
                ConfidenceLevel.NINETY_FIVE,
                period
            ))
            
            # CVaR (95%置信度)
            cvar_95 = self.calculate_cvar(symbol, ConfidenceLevel.NINETY_FIVE, period)
            metrics.append(RiskMetric(
                RiskMetricType.CONDITIONAL_VAR,
                cvar_95,
                ConfidenceLevel.NINETY_FIVE,
                period
            ))
            
            # 夏普比率
            sharpe = self.calculate_sharpe_ratio(symbol, period)
            metrics.append(RiskMetric(
                RiskMetricType.SHARPE_RATIO,
                sharpe,
                period=period
            ))
            
            # 索提诺比率
            sortino = self.calculate_sortino_ratio(symbol, period)
            metrics.append(RiskMetric(
                RiskMetricType.SORTINO_RATIO,
                sortino,
                period=period
            ))
            
            # 最大回撤
            max_dd, dd_duration = self.calculate_maximum_drawdown(symbol, period)
            metrics.append(RiskMetric(
                RiskMetricType.MAXIMUM_DRAWDOWN,
                max_dd,
                period=period,
                metadata={'duration_days': dd_duration}
            ))
            
            # 波动率
            volatility = self.calculate_volatility(symbol, period)
            metrics.append(RiskMetric(
                RiskMetricType.VOLATILITY,
                volatility,
                period=period
            ))
            
            # 贝塔系数
            beta = self.calculate_beta(symbol, period)
            metrics.append(RiskMetric(
                RiskMetricType.BETA,
                beta,
                period=period
            ))
            
            # 相关系数
            correlation = self.calculate_correlation(symbol, period)
            metrics.append(RiskMetric(
                RiskMetricType.CORRELATION,
                correlation,
                period=period
            ))
            
            # 计算卡尔玛比率
            calmar = self._calculate_calmar_ratio(symbol, period)
            metrics.append(RiskMetric(
                RiskMetricType.CALMAR_RATIO,
                calmar,
                period=period
            ))
            
            # 生成报告摘要
            summary = self._generate_report_summary(metrics, symbol)
            
            # 创建报告
            report = RiskReport(
                symbol=symbol,
                timestamp=datetime.now(),
                metrics=metrics,
                summary=summary
            )
            
            # 缓存报告
            self.metrics_cache[cache_key] = report
            
            return report
            
        except Exception as e:
            self.logger.error(f"生成风险报告失败 {symbol}: {e}")
            return RiskReport(symbol, datetime.now(), [])
    
    def _calculate_historical_var(self, returns: List[float], 
                                confidence_level: ConfidenceLevel) -> float:
        """计算历史法VaR"""
        percentile = (1 - confidence_level.value) * 100
        return abs(np.percentile(returns, percentile))
    
    def _calculate_parametric_var(self, returns: List[float], 
                                confidence_level: ConfidenceLevel) -> float:
        """计算参数法VaR"""
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # 正态分布分位数
        from scipy.stats import norm
        z_score = norm.ppf(1 - confidence_level.value)
        
        return abs(mean_return + z_score * std_return)
    
    def _calculate_monte_carlo_var(self, returns: List[float], 
                                 confidence_level: ConfidenceLevel, 
                                 simulations: int = 10000) -> float:
        """计算蒙特卡洛法VaR"""
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # 蒙特卡洛模拟
        simulated_returns = np.random.normal(mean_return, std_return, simulations)
        
        percentile = (1 - confidence_level.value) * 100
        return abs(np.percentile(simulated_returns, percentile))
    
    def _calculate_calmar_ratio(self, symbol: str, period: int) -> float:
        """计算卡尔玛比率"""
        try:
            returns = self._get_recent_returns(symbol, period)
            if len(returns) < 30:
                return 0.0
            
            annual_return = np.mean(returns) * 252
            max_drawdown, _ = self.calculate_maximum_drawdown(symbol, period)
            
            if max_drawdown == 0:
                return float('inf') if annual_return > 0 else 0.0
            
            return annual_return / max_drawdown
        except Exception as e:
            self.logger.error(f"卡尔玛比率计算失败 {symbol}: {e}")
            return 0.0
    
    def _generate_report_summary(self, metrics: List[RiskMetric], symbol: str) -> Dict[str, Any]:
        """生成报告摘要"""
        summary = {
            'symbol': symbol,
            'data_points': len(self.return_history.get(symbol, [])),
            'risk_assessment': 'unknown'
        }
        
        # 风险评估
        var_metric = next((m for m in metrics if m.metric_type == RiskMetricType.VALUE_AT_RISK), None)
        sharpe_metric = next((m for m in metrics if m.metric_type == RiskMetricType.SHARPE_RATIO), None)
        volatility_metric = next((m for m in metrics if m.metric_type == RiskMetricType.VOLATILITY), None)
        
        risk_score = 0
        if var_metric and var_metric.value > 0.05:  # 日VaR > 5%
            risk_score += 2
        if sharpe_metric and sharpe_metric.value < 0.5:  # 夏普比率 < 0.5
            risk_score += 1
        if volatility_metric and volatility_metric.value > 0.3:  # 年化波动率 > 30%
            risk_score += 2
        
        if risk_score >= 4:
            summary['risk_assessment'] = 'high'
        elif risk_score >= 2:
            summary['risk_assessment'] = 'medium'
        else:
            summary['risk_assessment'] = 'low'
        
        return summary
    
    def _get_recent_returns(self, symbol: str, period: int) -> List[float]:
        """获取最近的收益率数据"""
        if symbol not in self.return_history:
            return []
        
        recent_data = list(self.return_history[symbol])[-period:]
        return [data['return'] for data in recent_data]
    
    def _get_recent_prices(self, symbol: str, period: int) -> List[float]:
        """获取最近的价格数据"""
        if symbol not in self.price_history:
            return []
        
        recent_data = list(self.price_history[symbol])[-period:]
        return [data['price'] for data in recent_data]
    
    def _get_recent_benchmark_returns(self, period: int) -> List[float]:
        """获取最近的基准收益率数据"""
        recent_data = list(self.benchmark_returns)[-period:]
        return [data['return'] for data in recent_data]
    
    def _invalidate_cache(self, symbol: str):
        """清除指定符号的缓存"""
        keys_to_remove = [key for key in self.metrics_cache.keys() if key.startswith(symbol)]
        for key in keys_to_remove:
            del self.metrics_cache[key]
    
    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """检查缓存是否有效"""
        return (datetime.now() - timestamp).total_seconds() < self.cache_ttl
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标计算器状态摘要"""
        return {
            'symbols_tracked': len(self.price_history),
            'total_data_points': sum(len(history) for history in self.price_history.values()),
            'cache_entries': len(self.metrics_cache),
            'benchmark_data_points': len(self.benchmark_returns),
            'config': {
                'default_confidence_level': self.default_confidence_level.value,
                'default_period': self.default_period,
                'risk_free_rate': self.risk_free_rate,
                'cache_ttl': self.cache_ttl
            }
        }