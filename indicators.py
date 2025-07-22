# =============================================================================
# 优化版技术指标计算器
# 功能：使用向量化操作和缓存机制计算Supertrend、DEMA、ATR、ADX等指标
# 优化：支持批量计算、内存优化、性能监控
# 版本：2.0 (优化版)
# =============================================================================

import pandas as pd         # 数据处理和分析
import numpy as np          # 数值计算和数组操作
import pandas_ta as ta      # 技术指标库
import time                 # 时间测量
from datetime import datetime  # 日期时间处理
from functools import lru_cache  # 缓存装饰器
import warnings             # 警告控制

# 性能优化设置 - 忽略pandas性能警告以提高运行速度
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

class OptimizedIndicatorCalculator:
    """优化的指标计算器，使用缓存和向量化操作提高性能"""
    
    def __init__(self, config):
        """初始化指标计算器"""
        self.config = config
        self.indicators_config = config.get("indicators", {})
        
        # 获取参数
        self.dema144_len = self.indicators_config.get("dema144_len", 144)
        self.dema169_len = self.indicators_config.get("dema169_len", 169)
        self.atr_period = self.indicators_config.get("atr_period", 34)
        self.multiplier = self.indicators_config.get("atr_multiplier", 3.0)
        self.adx_period = self.indicators_config.get("adx_period", 14)
        self.adx_threshold = self.indicators_config.get("adx_threshold", 20)
        
        # 缓存
        self._cache = {}
        self._last_data_hash = None
    
    def calculate_all_indicators_optimized(self, df):
        """
        优化的指标计算，使用缓存和向量化操作
        
        参数:
            df (pd.DataFrame): 原始数据
            
        返回:
            pd.DataFrame: 添加了指标的数据
        """
        start_time = time.time()
        print(f"开始优化指标计算... 数据大小: {len(df)} 行 x {len(df.columns)} 列")
        
        # 检查数据是否变化
        data_hash = hash(str(df.values.tobytes()) + str(df.index.values.tobytes()))
        if data_hash == self._last_data_hash and 'result' in self._cache:
            print("使用缓存的指标计算结果")
            return self._cache['result'].copy()
        
        # 验证必要列
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"缺少必要的列: {missing_columns}")
        
        # 创建副本
        result_df = df.copy()
        
        # 批量计算指标
        indicators = self._calculate_indicators_batch(result_df)
        
        # 合并结果
        for name, values in indicators.items():
            result_df[name] = values
        
        # 缓存结果
        self._cache['result'] = result_df.copy()
        self._last_data_hash = data_hash
        
        total_time = time.time() - start_time
        memory_usage = result_df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        
        print(f"优化指标计算完成! 总耗时: {total_time:.2f} 秒")
        print(f"结果数据大小: {len(result_df)} 行 x {len(result_df.columns)} 列")
        print(f"内存占用: {memory_usage:.2f} MB")
        print(f"新增指标列: {', '.join(indicators.keys())}")
        
        return result_df
    
    def _calculate_indicators_batch(self, df):
        """批量计算所有指标"""
        indicators = {}
        
        # 1. 计算DEMA (向量化)
        print("  计算DEMA指标...")
        indicators['dema144'] = self._calculate_dema_vectorized(df['close'], self.dema144_len)
        indicators['dema169'] = self._calculate_dema_vectorized(df['close'], self.dema169_len)
        
        # 2. 计算ATR (使用pandas_ta)
        print("  计算ATR指标...")
        indicators['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)
        
        # 3. 计算ADX (使用pandas_ta)
        print("  计算ADX指标...")
        adx_result = ta.adx(df['high'], df['low'], df['close'], length=self.adx_period)
        if adx_result is not None:
            indicators['adx'] = adx_result[f'ADX_{self.adx_period}']
            indicators['plus_di'] = adx_result[f'DMP_{self.adx_period}']
            indicators['minus_di'] = adx_result[f'DMN_{self.adx_period}']
        else:
            # 备用计算
            indicators['adx'] = pd.Series(np.nan, index=df.index)
            indicators['plus_di'] = pd.Series(np.nan, index=df.index)
            indicators['minus_di'] = pd.Series(np.nan, index=df.index)
        
        # 4. 计算Supertrend (优化版本)
        print("  计算Supertrend指标...")
        st_result = self._calculate_supertrend_vectorized(df, self.atr_period, self.multiplier)
        indicators.update(st_result)
        
        return indicators
    
    @lru_cache(maxsize=32)
    def _calculate_dema_vectorized(self, price_series_tuple, period):
        """向量化计算DEMA，使用缓存"""
        # 将元组转换回Series
        price_series = pd.Series(price_series_tuple[0], index=price_series_tuple[1])
        
        # 使用pandas的ewm函数计算DEMA
        ema1 = price_series.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        dema = 2 * ema1 - ema2
        
        return dema
    
    def _calculate_dema_vectorized(self, price_series, period):
        """向量化计算DEMA"""
        # 使用pandas的ewm函数计算DEMA
        ema1 = price_series.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        dema = 2 * ema1 - ema2
        
        return dema
    
    def _calculate_supertrend_vectorized(self, df, length, multiplier):
        """向量化计算Supertrend指标"""
        # 计算ATR如果还没有
        if 'atr' not in df.columns:
            atr = ta.atr(df['high'], df['low'], df['close'], length=length)
        else:
            atr = df['atr']
        
        # 计算HL2
        hl2 = (df['high'] + df['low']) / 2.0
        
        # 计算上下轨
        upper_basic = hl2 - multiplier * atr
        lower_basic = hl2 + multiplier * atr
        
        # 使用numpy进行向量化计算
        close_values = df['close'].values
        upper_values = upper_basic.values
        lower_values = lower_basic.values
        
        # 计算平滑轨
        upper_smooth = self._calculate_smooth_band(upper_values, close_values, is_upper=True)
        lower_smooth = self._calculate_smooth_band(lower_values, close_values, is_upper=False)
        
        # 计算趋势方向
        trend = self._calculate_trend_vectorized(close_values, upper_smooth, lower_smooth)
        
        # 计算买卖信号
        buy_signals = np.zeros(len(trend), dtype=bool)
        sell_signals = np.zeros(len(trend), dtype=bool)
        
        # 向量化计算信号
        trend_change = np.diff(trend, prepend=trend[0])
        buy_signals[1:] = (trend[1:] == 1) & (trend_change[1:] == 2)  # 从-1变为1
        sell_signals[1:] = (trend[1:] == -1) & (trend_change[1:] == -2)  # 从1变为-1
        
        return {
            'supertrend_upper': pd.Series(upper_smooth, index=df.index),
            'supertrend_lower': pd.Series(lower_smooth, index=df.index),
            'supertrend_direction': pd.Series(trend, index=df.index),
            'supertrend_buy': pd.Series(buy_signals, index=df.index),
            'supertrend_sell': pd.Series(sell_signals, index=df.index)
        }
    
    def _calculate_smooth_band(self, basic_band, close_values, is_upper):
        """向量化计算平滑轨道"""
        smooth_band = basic_band.copy()
        
        for i in range(1, len(basic_band)):
            if is_upper:
                # 上轨平滑逻辑
                if close_values[i-1] > smooth_band[i-1]:
                    smooth_band[i] = max(basic_band[i], smooth_band[i-1])
                else:
                    smooth_band[i] = basic_band[i]
            else:
                # 下轨平滑逻辑
                if close_values[i-1] < smooth_band[i-1]:
                    smooth_band[i] = min(basic_band[i], smooth_band[i-1])
                else:
                    smooth_band[i] = basic_band[i]
        
        return smooth_band
    
    def _calculate_trend_vectorized(self, close_values, upper_smooth, lower_smooth):
        """向量化计算趋势方向"""
        trend = np.ones(len(close_values), dtype=int)
        
        for i in range(1, len(close_values)):
            prev_trend = trend[i-1]
            
            if prev_trend == -1 and close_values[i] > lower_smooth[i-1]:
                trend[i] = 1
            elif prev_trend == 1 and close_values[i] < upper_smooth[i-1]:
                trend[i] = -1
            else:
                trend[i] = prev_trend
        
        return trend


class IndicatorCache:
    """指标计算缓存管理器"""
    
    def __init__(self, max_size=100):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
    
    def get(self, key):
        """获取缓存值"""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key, value):
        """设置缓存值"""
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def _evict_oldest(self):
        """淘汰最老的缓存项"""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_times.clear()


class FastIndicators:
    """快速指标计算工具类"""
    
    @staticmethod
    def fast_ema(prices, period):
        """快速EMA计算"""
        alpha = 2.0 / (period + 1.0)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    @staticmethod
    def fast_sma(prices, period):
        """快速SMA计算"""
        return pd.Series(prices).rolling(window=period).mean().values
    
    @staticmethod
    def fast_rsi(prices, period=14):
        """快速RSI计算"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = pd.Series(gains).rolling(window=period).mean()
        avg_losses = pd.Series(losses).rolling(window=period).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.values
    
    @staticmethod
    def fast_bollinger_bands(prices, period=20, std_dev=2):
        """快速布林带计算"""
        sma = pd.Series(prices).rolling(window=period).mean()
        std = pd.Series(prices).rolling(window=period).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return {
            'upper': upper.values,
            'middle': sma.values,
            'lower': lower.values
        }


# 向后兼容性包装器
class IndicatorCalculator:
    """原有接口的包装器，保持向后兼容性"""
    
    def __init__(self, config):
        self.optimized_calculator = OptimizedIndicatorCalculator(config)
    
    def calculate_all_indicators(self, df):
        """调用优化版本的计算方法"""
        return self.optimized_calculator.calculate_all_indicators_optimized(df)
