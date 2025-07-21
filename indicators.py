import pandas as pd
import numpy as np
import pandas_ta as ta
import time
from datetime import datetime

class IndicatorCalculator:
    """指标计算器，负责计算Supertrend和DEMA技术指标"""
    
    def __init__(self, config):
        """
        初始化指标计算器
        
        参数:
            config (dict): 配置信息
        """
        self.config = config
        self.indicators_config = config.get("indicators", {})
        
        # 获取DEMA参数
        self.dema144_len = self.indicators_config.get("dema144_len", 144)
        self.dema169_len = self.indicators_config.get("dema169_len", 169)
     
        # 获取ATR参数
        self.atr_period = self.indicators_config.get("atr_period", 34)
        
        # 获取乘数参数(用于Supertrend)
        self.multiplier = self.indicators_config.get("atr_multiplier", 3.0)
        
        # Supertrend长度使用与ATR相同的周期
        self.supertrend_length = self.atr_period
        # Supertrend乘数使用与ATR相同的乘数
        self.supertrend_multiplier = self.multiplier
        
        # 获取ADX参数
        self.adx_period = self.indicators_config.get("adx_period", 14)
        self.adx_threshold = self.indicators_config.get("adx_threshold", 20)
    
    def calculate_dema(self, price_series, timeperiod):
        """
        计算DEMA (双指数移动平均线)
        
        参数:
            price_series (pd.Series): 价格序列
            timeperiod (int): 周期
            
        返回:
            pd.Series: DEMA值
        """
        print(f"  正在计算DEMA(周期={timeperiod})...")
        start_time = time.time()
        
        # 使用TradingView风格的DEMA计算
        ema1 = price_series.ewm(span=timeperiod, adjust=False).mean()
        ema2 = ema1.ewm(span=timeperiod, adjust=False).mean()
        dema = 2 * ema1 - ema2
        
        elapsed = time.time() - start_time
        print(f"  DEMA(周期={timeperiod})计算完成，耗时 {elapsed:.2f} 秒")
        return dema
    
    def calculate_atr(self, df, period):
        """
        计算ATR (平均真实波幅)
        
        参数:
            df (pd.DataFrame): 数据
            period (int): 周期
            
        返回:
            pd.Series: ATR值
        """
        print(f"  正在计算ATR(周期={period})...")
        start_time = time.time()
        
        # 使用pandas_ta计算ATR
        atr = ta.atr(df["high"], df["low"], df["close"], length=period)
        
        elapsed = time.time() - start_time
        print(f"  ATR计算完成，耗时 {elapsed:.2f} 秒")
        return atr
    
    def calculate_manual_supertrend(self, df, length=10, multiplier=3.0):
        """
        按照 Pine Script 逻辑计算 Supertrend 指标
        """
        print(f"  正在计算Supertrend指标(周期={length}, 乘数={multiplier}) [PineScript对齐版本]...")
        start_time = time.time()

        df_copy = df.copy()
        n = len(df_copy)

        # ATR 与 HL2
        df_copy["atr"] = ta.atr(df_copy["high"], df_copy["low"], df_copy["close"], length=length)
        df_copy["hl2"] = (df_copy["high"] + df_copy["low"]) / 2.0

        # 注意此处方向反转（完全对齐你的Pine脚本）
        df_copy["upper"] = df_copy["hl2"] - multiplier * df_copy["atr"]
        df_copy["lower"] = df_copy["hl2"] + multiplier * df_copy["atr"]

        upper = df_copy["upper"].values
        lower = df_copy["lower"].values
        close = df_copy["close"].values

        # 初始化平滑轨
        upper_smoothed = np.copy(upper)
        lower_smoothed = np.copy(lower)

        for i in range(1, n):
            if close[i - 1] > upper_smoothed[i - 1]:
                upper_smoothed[i] = max(upper[i], upper_smoothed[i - 1])
            else:
                upper_smoothed[i] = upper[i]

            if close[i - 1] < lower_smoothed[i - 1]:
                lower_smoothed[i] = min(lower[i], lower_smoothed[i - 1])
            else:
                lower_smoothed[i] = lower[i]

        # 计算趋势
        trend = np.ones(n, dtype=int)
        for i in range(1, n):
            prev_trend = trend[i - 1]
            prev_lower = lower_smoothed[i - 1]
            prev_upper = upper_smoothed[i - 1]

            if prev_trend == -1 and close[i] > prev_lower:
                trend[i] = 1
            elif prev_trend == 1 and close[i] < prev_upper:
                trend[i] = -1
            else:
                trend[i] = prev_trend

        # 生成买卖信号
        buy_signal = (trend == 1) & (np.roll(trend, 1) == -1)
        sell_signal = (trend == -1) & (np.roll(trend, 1) == 1)
        buy_signal[0] = False
        sell_signal[0] = False

        # 返回DataFrame格式
        df_copy["supertrend_upper"] = upper_smoothed
        df_copy["supertrend_lower"] = lower_smoothed
        df_copy["supertrend_direction"] = trend
        df_copy["supertrend_buy"] = buy_signal
        df_copy["supertrend_sell"] = sell_signal

        print(f"  Supertrend计算完成，总耗时: {time.time() - start_time:.2f} 秒")
        return (
            df_copy["supertrend_direction"],
            df_copy["supertrend_upper"],
            df_copy["supertrend_lower"],
            df_copy["supertrend_buy"],
            df_copy["supertrend_sell"]
        )
    
    def calculate_adx(self, df, period=14):
        """
        计算ADX (平均趋向指数)
        
        参数:
            df (pd.DataFrame): 数据
            period (int): 周期
            
        返回:
            tuple: (adx, plus_di, minus_di)
        """
        print(f"  正在计算ADX(周期={period})...")
        start_time = time.time()
        
        # 使用pandas_ta计算ADX
        adx_result = ta.adx(df["high"], df["low"], df["close"], length=period)
        
        # 从结果中提取ADX, +DI, -DI
        adx = adx_result["ADX_"+str(period)]
        plus_di = adx_result["DMP_"+str(period)]
        minus_di = adx_result["DMN_"+str(period)]
        
        elapsed = time.time() - start_time
        print(f"  ADX计算完成，耗时 {elapsed:.2f} 秒")
        return adx, plus_di, minus_di
        
    def calculate_all_indicators(self, df):
        """
        计算Supertrend和DEMA指标
        
        参数:
            df (pd.DataFrame): 原始数据
            
        返回:
            pd.DataFrame: 添加了指标的数据
        """
        start_time = time.time()
        print(f"开始计算技术指标... 数据大小: {len(df)} 行 x {len(df.columns)} 列")
        print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 创建副本避免修改原始数据
        indicators_df = df.copy()
        
        # 检查是否有必要的列
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in indicators_df.columns:
                print(f"错误: 缺少必要的列 '{col}'")
                return None
        
        # 计算DEMA指标
        indicators_df["dema144"] = self.calculate_dema(indicators_df["close"], self.dema144_len)
        indicators_df["dema169"] = self.calculate_dema(indicators_df["close"], self.dema169_len)
        
        # 计算ATR
        indicators_df["atr"] = self.calculate_atr(indicators_df, self.atr_period)
        
        # 计算ADX
        adx, plus_di, minus_di = self.calculate_adx(indicators_df, period=self.adx_period)
        indicators_df["adx"] = adx
        indicators_df["plus_di"] = plus_di
        indicators_df["minus_di"] = minus_di
        
        # 计算Supertrend
        try:
            trend, upper_smoothed, lower_smoothed, buy_signal, sell_signal = self.calculate_manual_supertrend(
                indicators_df, 
                length=self.supertrend_length, 
                multiplier=self.supertrend_multiplier
            )
            
            # 添加Supertrend结果到DataFrame
            indicators_df["supertrend_direction"] = trend
            indicators_df["supertrend_upper"] = upper_smoothed
            indicators_df["supertrend_lower"] = lower_smoothed
            indicators_df["supertrend_buy"] = buy_signal
            indicators_df["supertrend_sell"] = sell_signal
        except Exception as e:
            print(f"Supertrend计算错误: {e}")
            print("使用备选方法计算Supertrend...")
            # 使用备选方法
            indicators_df["hl2"] = (indicators_df["high"] + indicators_df["low"]) / 2
            # 注意此处方向与Pine Script对齐
            indicators_df["upper"] = indicators_df["hl2"] - self.supertrend_multiplier * indicators_df["atr"]
            indicators_df["lower"] = indicators_df["hl2"] + self.supertrend_multiplier * indicators_df["atr"]
            indicators_df["trend"] = 1
            # 使用简单的备用逻辑计算信号
            for i in range(1, len(indicators_df)):
                # 平滑计算
                if indicators_df["close"].iloc[i-1] > indicators_df["upper"].iloc[i-1]:
                    indicators_df.loc[indicators_df.index[i], "upper"] = max(
                        indicators_df["upper"].iloc[i], 
                        indicators_df["upper"].iloc[i-1]
                    )
                    
                if indicators_df["close"].iloc[i-1] < indicators_df["lower"].iloc[i-1]:
                    indicators_df.loc[indicators_df.index[i], "lower"] = min(
                        indicators_df["lower"].iloc[i], 
                        indicators_df["lower"].iloc[i-1]
                    )
                
                # 趋势计算
                prev_trend = indicators_df["trend"].iloc[i-1]
                if prev_trend == -1 and indicators_df["close"].iloc[i] > indicators_df["lower"].iloc[i-1]:
                    indicators_df.loc[indicators_df.index[i], "trend"] = 1
                elif prev_trend == 1 and indicators_df["close"].iloc[i] < indicators_df["upper"].iloc[i-1]:
                    indicators_df.loc[indicators_df.index[i], "trend"] = -1
                else:
                    indicators_df.loc[indicators_df.index[i], "trend"] = prev_trend
            
            indicators_df["supertrend_direction"] = indicators_df["trend"]
            indicators_df["supertrend_buy"] = (indicators_df["trend"] == 1) & (indicators_df["trend"].shift(1) == -1)
            indicators_df["supertrend_sell"] = (indicators_df["trend"] == -1) & (indicators_df["trend"].shift(1) == 1)
        
        # 计算内存使用情况
        memory_usage = indicators_df.memory_usage().sum() / 1024 / 1024  # MB
        
        total_time = time.time() - start_time
        print(f"技术指标计算完成! 总耗时: {total_time:.2f} 秒")
        print(f"结果数据大小: {len(indicators_df)} 行 x {len(indicators_df.columns)} 列")
        print(f"内存占用: {memory_usage:.2f} MB")
        print(f"已添加的指标列: {', '.join(indicators_df.columns.difference(df.columns))}")
        
        return indicators_df