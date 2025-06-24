import pandas as pd
import numpy as np
import streamlit as st

class IndicatorCalculator:
    """指标计算器类，负责计算各种技术指标"""
    
    def __init__(self, config):
        """
        初始化指标计算器
        
        参数:
            config (dict): 配置信息
        """
        self.config = config
        self.indicator_config = config.get("indicators", {})
        
        # 获取指标参数
        self.dema144_len = self.indicator_config.get("dema144_len", 144)
        self.dema169_len = self.indicator_config.get("dema169_len", 169)
        self.ema120_len = self.indicator_config.get("ema120_len", 120)
        self.ema200_len = self.indicator_config.get("ema200_len", 200)
        self.atr_period = self.indicator_config.get("atr_period", 34)
        self.atr_multiplier = self.indicator_config.get("atr_multiplier", 3.0)
    
    def calculate_ema(self, series, length):
        """
        计算指数移动平均线 (EMA)
        
        参数:
            series (pd.Series): 价格序列
            length (int): EMA周期
            
        返回:
            pd.Series: EMA值
        """
        return series.ewm(span=length, adjust=False).mean()
    
    def calculate_dema(self, series, length):
        """
        计算双指数移动平均线 (DEMA)
        
        参数:
            series (pd.Series): 价格序列
            length (int): DEMA周期
            
        返回:
            pd.Series: DEMA值
        """
        ema1 = self.calculate_ema(series, length)
        ema2 = self.calculate_ema(ema1, length)
        dema = 2 * ema1 - ema2
        return dema
    
    def calculate_atr(self, df, period):
        """
        计算平均真实范围 (ATR)
        
        参数:
            df (pd.DataFrame): 包含'high', 'low', 'close'列的DataFrame
            period (int): ATR周期
            
        返回:
            pd.Series: ATR值
        """
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calculate_trend_channels(self, df, atr):
        """
        计算趋势通道 (上下轨道)
        
        参数:
            df (pd.DataFrame): 价格数据
            atr (pd.Series): ATR值
            
        返回:
            tuple: (upper, lower) 上轨道和下轨道
        """
        hl2 = (df['high'] + df['low']) / 2
        
        # 初始化上下轨道
        upper = hl2 - self.atr_multiplier * atr
        lower = hl2 + self.atr_multiplier * atr
        
        # 创建副本以避免链式赋值警告
        upper_values = upper.copy()
        lower_values = lower.copy()
        
        # 计算趋势轨道
        for i in range(1, len(df)):
            if df['close'].iloc[i] > upper.shift(1).iloc[i]:
                upper_values.iloc[i] = max(upper_values.iloc[i], upper.shift(1).iloc[i])
            
            if df['close'].iloc[i] < lower.shift(1).iloc[i]:
                lower_values.iloc[i] = min(lower_values.iloc[i], lower.shift(1).iloc[i])
        
        return upper_values, lower_values
    
    def calculate_trend_direction(self, df, upper, lower):
        """
        计算趋势方向
        
        参数:
            df (pd.DataFrame): 价格数据
            upper (pd.Series): 上轨道
            lower (pd.Series): 下轨道
            
        返回:
            pd.Series: 趋势方向 (1: 上升, -1: 下降)
        """
        trend = pd.Series(1, index=df.index)  # 初始化为上升趋势
        
        for i in range(1, len(df)):
            prev_trend = trend.iloc[i-1]
            
            if prev_trend == -1 and df['close'].iloc[i] > lower.iloc[i-1]:
                trend.iloc[i] = 1  # 转为上升趋势
            elif prev_trend == 1 and df['close'].iloc[i] < upper.iloc[i-1]:
                trend.iloc[i] = -1  # 转为下降趋势
            else:
                trend.iloc[i] = prev_trend  # 保持当前趋势
        
        return trend
    
    def calculate_all_indicators(self, df):
        """
        计算所有技术指标
        
        参数:
            df (pd.DataFrame): 价格数据
            
        返回:
            pd.DataFrame: 带有所有指标的DataFrame
        """
        with st.spinner("正在计算技术指标..."):
            # 创建副本避免修改原始数据
            result_df = df.copy()
            
            # 计算移动平均线
            result_df['ema_120'] = self.calculate_ema(result_df['close'], self.ema120_len)
            result_df['ema_200'] = self.calculate_ema(result_df['close'], self.ema200_len)
            
            # 计算DEMA
            result_df['ema_first_144'] = self.calculate_ema(result_df['close'], self.dema144_len)
            result_df['ema_second_144'] = self.calculate_ema(result_df['ema_first_144'], self.dema144_len)
            result_df['dema_144'] = 2 * result_df['ema_first_144'] - result_df['ema_second_144']
            
            result_df['ema_first_169'] = self.calculate_ema(result_df['close'], self.dema169_len)
            result_df['ema_second_169'] = self.calculate_ema(result_df['ema_first_169'], self.dema169_len)
            result_df['dema_169'] = 2 * result_df['ema_first_169'] - result_df['ema_second_169']
            
            # 计算ATR
            result_df['atr'] = self.calculate_atr(result_df, self.atr_period)
            
            # 计算hl2 (high+low)/2
            result_df['hl2'] = (result_df['high'] + result_df['low']) / 2
            
            # 计算趋势通道
            upper, lower = self.calculate_trend_channels(result_df, result_df['atr'])
            result_df['upper'] = upper
            result_df['lower'] = lower
            
            # 计算趋势方向
            result_df['trend'] = self.calculate_trend_direction(result_df, upper, lower)
            
            st.success("技术指标计算完成")
            
            return result_df

def render_indicator_ui(config, klines_data=None):
    """
    渲染指标计算UI界面
    
    参数:
        config (dict): 配置信息
        klines_data (pd.DataFrame): K线数据
        
    返回:
        pd.DataFrame: 带有指标的DataFrame
    """
    st.header("指标计算")
    
    if klines_data is None:
        if 'klines_data' in st.session_state:
            klines_data = st.session_state.klines_data
        else:
            st.warning("请先获取K线数据")
            return None
    
    # 创建指标计算器
    indicator_calculator = IndicatorCalculator(config)
    
    # 显示当前配置
    indicator_config = config.get("indicators", {})
    st.info(f"DEMA参数: {indicator_config.get('dema144_len')}/{indicator_config.get('dema169_len')} | " +
            f"EMA参数: {indicator_config.get('ema120_len')}/{indicator_config.get('ema200_len')} | " +
            f"ATR参数: {indicator_config.get('atr_period')}周期/{indicator_config.get('atr_multiplier')}倍")
    
    # 计算指标按钮
    if st.button("计算技术指标"):
        indicators_df = indicator_calculator.calculate_all_indicators(klines_data)
        
        # 保存到session_state
        st.session_state.indicators_df = indicators_df
        
        # 显示指标预览
        st.subheader("指标预览")
        
        # 显示部分列
        preview_columns = ['close', 'dema_144', 'dema_169', 'ema_120', 'ema_200', 'atr', 'upper', 'lower', 'trend']
        st.dataframe(indicators_df[preview_columns].tail())
        
        return indicators_df
    
    # 如果已经计算过指标
    if 'indicators_df' in st.session_state:
        indicators_df = st.session_state.indicators_df
        st.success(f"已加载指标数据 ({len(indicators_df)} 条记录)")
        
        # 显示指标预览按钮
        if st.button("显示指标预览"):
            preview_columns = ['close', 'dema_144', 'dema_169', 'ema_120', 'ema_200', 'atr', 'upper', 'lower', 'trend']
            st.dataframe(indicators_df[preview_columns].tail())
        
        return indicators_df
    
    return None

if __name__ == "__main__":
    # 测试指标计算UI
    import yaml
    
    st.set_page_config(page_title="阿翔趋势交易系统 - 指标计算", layout="wide")
    
    # 加载配置
    with open("config.yaml", 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    # 创建测试数据
    if 'klines_data' not in st.session_state:
        # 创建模拟数据
        date_range = pd.date_range(start='2023-01-01', periods=200, freq='H')
        data = {
            'open': np.random.normal(10000, 500, len(date_range)),
            'high': np.random.normal(10100, 500, len(date_range)),
            'low': np.random.normal(9900, 500, len(date_range)),
            'close': np.random.normal(10050, 500, len(date_range)),
            'volume': np.random.normal(100, 20, len(date_range))
        }
        df = pd.DataFrame(data, index=date_range)
        st.session_state.klines_data = df
    
    render_indicator_ui(config) 