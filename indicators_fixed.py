import pandas as pd
import numpy as np
import streamlit as st

class IndicatorCalculator:
    """鎸囨爣璁＄畻鍣ㄧ被锛岃礋璐ｈ绠楀悇绉嶆妧鏈寚鏍?""
    
    def __init__(self, config):
        """
        鍒濆鍖栨寚鏍囪绠楀櫒
        
        鍙傛暟:
            config (dict): 閰嶇疆淇℃伅
        """
        self.config = config
        self.indicator_config = config.get("indicators", {})
        
        # 鑾峰彇鎸囨爣鍙傛暟
        self.dema144_len = self.indicator_config.get("dema144_len", 144)
        self.dema169_len = self.indicator_config.get("dema169_len", 169)
        self.ema120_len = self.indicator_config.get("ema120_len", 120)
        self.ema200_len = self.indicator_config.get("ema200_len", 200)
        self.atr_period = self.indicator_config.get("atr_period", 34)
        self.atr_multiplier = self.indicator_config.get("atr_multiplier", 3.0)
    
    def calculate_ema(self, series, length):
        """
        璁＄畻鎸囨暟绉诲姩骞冲潎绾?(EMA)
        
        鍙傛暟:
            series (pd.Series): 浠锋牸搴忓垪
            length (int): EMA鍛ㄦ湡
            
        杩斿洖:
            pd.Series: EMA鍊?
        """
        return series.ewm(span=length, adjust=False).mean()
    
    def calculate_dema(self, series, length):
        """
        璁＄畻鍙屾寚鏁扮Щ鍔ㄥ钩鍧囩嚎 (DEMA)
        
        鍙傛暟:
            series (pd.Series): 浠锋牸搴忓垪
            length (int): DEMA鍛ㄦ湡
            
        杩斿洖:
            pd.Series: DEMA鍊?
        """
        ema1 = self.calculate_ema(series, length)
        ema2 = self.calculate_ema(ema1, length)
        dema = 2 * ema1 - ema2
        return dema
    
    def calculate_atr(self, df, period):
        """
        璁＄畻骞冲潎鐪熷疄鑼冨洿 (ATR)
        
        鍙傛暟:
            df (pd.DataFrame): 鍖呭惈'high', 'low', 'close'鍒楃殑DataFrame
            period (int): ATR鍛ㄦ湡
            
        杩斿洖:
            pd.Series: ATR鍊?
        """
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calculate_trend_channels(self, df, atr):
        """
        璁＄畻瓒嬪娍閫氶亾 (涓婁笅杞ㄩ亾)
        
        鍙傛暟:
            df (pd.DataFrame): 浠锋牸鏁版嵁
            atr (pd.Series): ATR鍊?
            
        杩斿洖:
            tuple: (upper, lower) 涓婅建閬撳拰涓嬭建閬?
        """
        hl2 = (df['high'] + df['low']) / 2
        
        # 鍒濆鍖栦笂涓嬭建閬?
        upper = hl2 - self.atr_multiplier * atr
        lower = hl2 + self.atr_multiplier * atr
        
        # 鍒涘缓鍓湰浠ラ伩鍏嶉摼寮忚祴鍊艰鍛?
        upper_values = upper.copy()
        lower_values = lower.copy()
        
        # 璁＄畻瓒嬪娍杞ㄩ亾
        for i in range(1, len(df)):
            if df['close'].iloc[i] > upper.shift(1).iloc[i]:
                upper_values.iloc[i] = max(upper_values.iloc[i], upper.shift(1).iloc[i])
            
            if df['close'].iloc[i] < lower.shift(1).iloc[i]:
                lower_values.iloc[i] = min(lower_values.iloc[i], lower.shift(1).iloc[i])
        
        return upper_values, lower_values
    
    def calculate_trend_direction(self, df, upper, lower):
        """
        璁＄畻瓒嬪娍鏂瑰悜
        
        鍙傛暟:
            df (pd.DataFrame): 浠锋牸鏁版嵁
            upper (pd.Series): 涓婅建閬?
            lower (pd.Series): 涓嬭建閬?
            
        杩斿洖:
            pd.Series: 瓒嬪娍鏂瑰悜 (1: 涓婂崌, -1: 涓嬮檷)
        """
        trend = pd.Series(1, index=df.index)  # 鍒濆鍖栦负涓婂崌瓒嬪娍
        
        for i in range(1, len(df)):
            prev_trend = trend.iloc[i-1]
            
            if prev_trend == -1 and df['close'].iloc[i] > lower.iloc[i-1]:
                trend.iloc[i] = 1  # 杞负涓婂崌瓒嬪娍
            elif prev_trend == 1 and df['close'].iloc[i] < upper.iloc[i-1]:
                trend.iloc[i] = -1  # 杞负涓嬮檷瓒嬪娍
            else:
                trend.iloc[i] = prev_trend  # 淇濇寔褰撳墠瓒嬪娍
        
        return trend
    
    def calculate_all_indicators(self, df):
        """
        璁＄畻鎵€鏈夋妧鏈寚鏍?
        
        鍙傛暟:
            df (pd.DataFrame): 浠锋牸鏁版嵁
            
        杩斿洖:
            pd.DataFrame: 甯︽湁鎵€鏈夋寚鏍囩殑DataFrame
        """
        with st.spinner("姝ｅ湪璁＄畻鎶€鏈寚鏍?.."):
            # 鍒涘缓鍓湰閬垮厤淇敼鍘熷鏁版嵁
            result_df = df.copy()
            
            # 璁＄畻绉诲姩骞冲潎绾?
            result_df['ema_120'] = self.calculate_ema(result_df['close'], self.ema120_len)
            result_df['ema_200'] = self.calculate_ema(result_df['close'], self.ema200_len)
            
            # 璁＄畻DEMA
            result_df['ema_first_144'] = self.calculate_ema(result_df['close'], self.dema144_len)
            result_df['ema_second_144'] = self.calculate_ema(result_df['ema_first_144'], self.dema144_len)
            result_df['dema_144'] = 2 * result_df['ema_first_144'] - result_df['ema_second_144']
            
            result_df['ema_first_169'] = self.calculate_ema(result_df['close'], self.dema169_len)
            result_df['ema_second_169'] = self.calculate_ema(result_df['ema_first_169'], self.dema169_len)
            result_df['dema_169'] = 2 * result_df['ema_first_169'] - result_df['ema_second_169']
            
            # 璁＄畻ATR
            result_df['atr'] = self.calculate_atr(result_df, self.atr_period)
            
            # 璁＄畻hl2 (high+low)/2
            result_df['hl2'] = (result_df['high'] + result_df['low']) / 2
            
            # 璁＄畻瓒嬪娍閫氶亾
            upper, lower = self.calculate_trend_channels(result_df, result_df['atr'])
            result_df['upper'] = upper
            result_df['lower'] = lower
            
            # 璁＄畻瓒嬪娍鏂瑰悜
            result_df['trend'] = self.calculate_trend_direction(result_df, upper, lower)
            
            st.success("鎶€鏈寚鏍囪绠楀畬鎴?)
            
            return result_df

def render_indicator_ui(config, klines_data=None):
    """
    娓叉煋鎸囨爣璁＄畻UI鐣岄潰
    
    鍙傛暟:
        config (dict): 閰嶇疆淇℃伅
        klines_data (pd.DataFrame): K绾挎暟鎹?
        
    杩斿洖:
        pd.DataFrame: 甯︽湁鎸囨爣鐨凞ataFrame
    """
    st.header("鎸囨爣璁＄畻")
    
    if klines_data is None:
        if 'klines_data' in st.session_state:
            klines_data = st.session_state.klines_data
        else:
            st.warning("璇峰厛鑾峰彇K绾挎暟鎹?)
            return None
    
    # 鍒涘缓鎸囨爣璁＄畻鍣?
    indicator_calculator = IndicatorCalculator(config)
    
    # 鏄剧ず褰撳墠閰嶇疆
    indicator_config = config.get("indicators", {})
    st.info(f"DEMA鍙傛暟: {indicator_config.get('dema144_len')}/{indicator_config.get('dema169_len')} | " +
            f"EMA鍙傛暟: {indicator_config.get('ema120_len')}/{indicator_config.get('ema200_len')} | " +
            f"ATR鍙傛暟: {indicator_config.get('atr_period')}鍛ㄦ湡/{indicator_config.get('atr_multiplier')}鍊?)
    
    # 璁＄畻鎸囨爣鎸夐挳
    if st.button("璁＄畻鎶€鏈寚鏍?):
        indicators_df = indicator_calculator.calculate_all_indicators(klines_data)
        
        # 淇濆瓨鍒皊ession_state
        st.session_state.indicators_df = indicators_df
        
        # 鏄剧ず鎸囨爣棰勮
        st.subheader("鎸囨爣棰勮")
        
        # 鏄剧ず閮ㄥ垎鍒?
        preview_columns = ['close', 'dema_144', 'dema_169', 'ema_120', 'ema_200', 'atr', 'upper', 'lower', 'trend']
        st.dataframe(indicators_df[preview_columns].tail())
        
        return indicators_df
    
    # 濡傛灉宸茬粡璁＄畻杩囨寚鏍?
    if 'indicators_df' in st.session_state:
        indicators_df = st.session_state.indicators_df
        st.success(f"宸插姞杞芥寚鏍囨暟鎹?({len(indicators_df)} 鏉¤褰?")
        
        # 鏄剧ず鎸囨爣棰勮鎸夐挳
        if st.button("鏄剧ず鎸囨爣棰勮"):
            preview_columns = ['close', 'dema_144', 'dema_169', 'ema_120', 'ema_200', 'atr', 'upper', 'lower', 'trend']
            st.dataframe(indicators_df[preview_columns].tail())
        
        return indicators_df
    
    return None

if __name__ == "__main__":
    # 娴嬭瘯鎸囨爣璁＄畻UI
    import yaml
    
    st.set_page_config(page_title="闃跨繑瓒嬪娍浜ゆ槗绯荤粺 - 鎸囨爣璁＄畻", layout="wide")
    
    # 鍔犺浇閰嶇疆
    with open("config.yaml", 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    # 鍒涘缓娴嬭瘯鏁版嵁
    if 'klines_data' not in st.session_state:
        # 鍒涘缓妯℃嫙鏁版嵁
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
