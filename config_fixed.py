import streamlit as st
import yaml
from utils import load_config, save_config, test_proxy, set_proxy

class ConfigManager:
    """閰嶇疆绠＄悊绫伙紝璐熻矗鍔犺浇銆佷慨鏀瑰拰淇濆瓨閰嶇疆"""
    
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = load_config(config_path)
        
    def get_config(self):
        """鑾峰彇褰撳墠閰嶇疆"""
        return self.config
        
    def update_config(self, new_config):
        """鏇存柊閰嶇疆"""
        self.config = new_config
        
    def save_config(self):
        """淇濆瓨閰嶇疆鍒版枃浠?""
        return save_config(self.config, self.config_path)
    
    def get_trading_params(self):
        """鑾峰彇浜ゆ槗鍙傛暟"""
        return self.config.get("trading", {})
    
    def get_capital_params(self):
        """鑾峰彇璧勯噾鍙傛暟"""
        return self.config.get("capital", {})
    
    def get_indicator_params(self):
        """鑾峰彇鎸囨爣鍙傛暟"""
        return self.config.get("indicators", {})
    
    def get_network_params(self):
        """鑾峰彇缃戠粶鍙傛暟"""
        return self.config.get("network", {})
    
    def get_data_params(self):
        """鑾峰彇鏁版嵁鍙傛暟"""
        return self.config.get("data", {})

def render_config_ui():
    """娓叉煋閰嶇疆UI鐣岄潰"""
    
    st.header("閰嶇疆绠＄悊")
    
    # 鍒濆鍖栭厤缃鐞嗗櫒
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # 鍒涘缓琛ㄥ崟浠ユ敹闆嗘墍鏈夐厤缃?
    with st.form("config_form"):
        st.subheader("浜ゆ槗璁剧疆")
        col1, col2 = st.columns(2)
        
        with col1:
            trading_config = config.get("trading", {})
            symbol = st.text_input("浜ゆ槗瀵?, 
                                 value=trading_config.get("symbol", "BTC/USDT:USDT"))
            is_futures = st.checkbox("鏈熻揣浜ゆ槗", 
                                   value=trading_config.get("is_futures", True))
        
        with col2:
            timeframe_options = ["1m", "5m", "15m", "30m", "1h", "4h", "6h", "12h", "1d", "3d", "1w", "1M"]
            timeframe = st.selectbox("K绾垮懆鏈?, 
                                    options=timeframe_options,
                                    index=timeframe_options.index(trading_config.get("timeframe", "1h")))
        
        st.subheader("璧勯噾璁剧疆")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            capital_config = config.get("capital", {})
            initial_capital = st.number_input("鍒濆璧勯噾(USDT)", 
                                           min_value=100.0, max_value=1000000.0, 
                                           value=float(capital_config.get("initial_capital", 10000)))
        
        with col2:
            leverage = st.number_input("鏉犳潌鍊嶆暟", 
                                     min_value=1.0, max_value=125.0, 
                                     value=float(capital_config.get("leverage", 3)))
        
        with col3:
            risk_percent = st.number_input("椋庨櫓姣斾緥(%)", 
                                         min_value=0.1, max_value=10.0, 
                                         value=float(capital_config.get("risk_per_trade", 0.02))*100)
            risk_per_trade = risk_percent / 100
        
        st.subheader("鎸囨爣鍙傛暟")
        col1, col2 = st.columns(2)
        
        with col1:
            indicator_config = config.get("indicators", {})
            dema144_len = st.number_input("DEMA144鍧囩嚎鍛ㄦ湡", 
                                        min_value=10, max_value=500, 
                                        value=indicator_config.get("dema144_len", 144))
            dema169_len = st.number_input("DEMA169鍧囩嚎鍛ㄦ湡", 
                                        min_value=10, max_value=500, 
                                        value=indicator_config.get("dema169_len", 169))
            ema120_len = st.number_input("EMA120鍧囩嚎鍛ㄦ湡", 
                                       min_value=10, max_value=500, 
                                       value=indicator_config.get("ema120_len", 120))
        
        with col2:
            ema200_len = st.number_input("EMA200鍧囩嚎鍛ㄦ湡", 
                                       min_value=10, max_value=500, 
                                       value=indicator_config.get("ema200_len", 200))
            atr_period = st.number_input("ATR璁＄畻鍛ㄦ湡", 
                                       min_value=5, max_value=100, 
                                       value=indicator_config.get("atr_period", 34))
            atr_multiplier = st.number_input("ATR鍊嶆暟", 
                                          min_value=0.5, max_value=10.0, 
                                          value=float(indicator_config.get("atr_multiplier", 3.0)))
        
        st.subheader("缃戠粶璁剧疆")
        network_config = config.get("network", {})
        http_proxy = st.text_input("HTTP浠ｇ悊", 
                                 value=network_config.get("http_proxy", "http://127.0.0.1:7897"))
        
        # 浠ｇ悊娴嬭瘯鎸夐挳鍦ㄨ〃鍗曞
        
        st.subheader("鏁版嵁璁剧疆")
        data_config = config.get("data", {})
        max_candles = st.number_input("鏈€澶绾挎暟閲?, 
                                    min_value=100, max_value=5000, 
                                    value=data_config.get("max_candles", 1000))
        
        submitted = st.form_submit_button("淇濆瓨閰嶇疆")
        
        if submitted:
            # 鏇存柊閰嶇疆
            new_config = {
                "trading": {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "is_futures": is_futures
                },
                "capital": {
                    "initial_capital": initial_capital,
                    "leverage": leverage,
                    "risk_per_trade": risk_per_trade
                },
                "indicators": {
                    "dema144_len": dema144_len,
                    "dema169_len": dema169_len,
                    "ema120_len": ema120_len,
                    "ema200_len": ema200_len,
                    "atr_period": atr_period,
                    "atr_multiplier": atr_multiplier
                },
                "network": {
                    "http_proxy": http_proxy,
                    "https_proxy": http_proxy
                },
                "data": {
                    "default_lookback_days": data_config.get("default_lookback_days", 30),
                    "max_candles": max_candles
                }
            }
            
            config_manager.update_config(new_config)
            config_manager.save_config()
            
    # 浠ｇ悊娴嬭瘯鎸夐挳锛堝湪琛ㄥ崟澶栵級
    if st.button("娴嬭瘯浠ｇ悊杩炴帴"):
        proxy_url = config["network"]["http_proxy"]
        set_proxy(proxy_url)
        test_proxy(proxy_url)
    
    return config_manager.get_config()

if __name__ == "__main__":
    # 娴嬭瘯閰嶇疆UI
    st.set_page_config(page_title="闃跨繑瓒嬪娍浜ゆ槗绯荤粺 - 閰嶇疆", layout="wide")
    render_config_ui() 
