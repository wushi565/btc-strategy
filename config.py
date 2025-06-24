import streamlit as st
import yaml
from utils import load_config, save_config, test_proxy, set_proxy

class ConfigManager:
    """配置管理类，负责加载、修改和保存配置"""
    
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = load_config(config_path)
        
    def get_config(self):
        """获取当前配置"""
        return self.config
        
    def update_config(self, new_config):
        """更新配置"""
        self.config = new_config
        
    def save_config(self):
        """保存配置到文件"""
        return save_config(self.config, self.config_path)
    
    def get_trading_params(self):
        """获取交易参数"""
        return self.config.get("trading", {})
    
    def get_capital_params(self):
        """获取资金参数"""
        return self.config.get("capital", {})
    
    def get_indicator_params(self):
        """获取指标参数"""
        return self.config.get("indicators", {})
    
    def get_network_params(self):
        """获取网络参数"""
        return self.config.get("network", {})
    
    def get_data_params(self):
        """获取数据参数"""
        return self.config.get("data", {})

def render_config_ui():
    """渲染配置UI界面"""
    
    st.header("配置管理")
    
    # 初始化配置管理器
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # 创建表单以收集所有配置
    with st.form("config_form"):
        st.subheader("交易设置")
        col1, col2 = st.columns(2)
        
        with col1:
            trading_config = config.get("trading", {})
            symbol = st.text_input("交易对", 
                                 value=trading_config.get("symbol", "BTC/USDT:USDT"))
            is_futures = st.checkbox("期货交易", 
                                   value=trading_config.get("is_futures", True))
        
        with col2:
            timeframe_options = ["1m", "5m", "15m", "30m", "1h", "4h", "6h", "12h", "1d", "3d", "1w", "1M"]
            timeframe = st.selectbox("K线周期", 
                                    options=timeframe_options,
                                    index=timeframe_options.index(trading_config.get("timeframe", "1h")))
        
        st.subheader("资金设置")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            capital_config = config.get("capital", {})
            initial_capital = st.number_input("初始资金(USDT)", 
                                           min_value=100.0, max_value=1000000.0, 
                                           value=float(capital_config.get("initial_capital", 10000)))
        
        with col2:
            leverage = st.number_input("杠杆倍数", 
                                     min_value=1.0, max_value=125.0, 
                                     value=float(capital_config.get("leverage", 3)))
        
        with col3:
            risk_percent = st.number_input("风险比例(%)", 
                                         min_value=0.1, max_value=100.0, 
                                         value=float(capital_config.get("risk_per_trade", 0.02))*100)
            risk_per_trade = risk_percent / 100
        
        st.subheader("指标参数")
        col1, col2 = st.columns(2)
        
        with col1:
            indicator_config = config.get("indicators", {})
            dema144_len = st.number_input("DEMA144均线周期", 
                                        min_value=10, max_value=500, 
                                        value=indicator_config.get("dema144_len", 144))
            dema169_len = st.number_input("DEMA169均线周期", 
                                        min_value=10, max_value=500, 
                                        value=indicator_config.get("dema169_len", 169))
            ema120_len = st.number_input("EMA120均线周期", 
                                       min_value=10, max_value=500, 
                                       value=indicator_config.get("ema120_len", 120))
        
        with col2:
            ema200_len = st.number_input("EMA200均线周期", 
                                       min_value=10, max_value=500, 
                                       value=indicator_config.get("ema200_len", 200))
            atr_period = st.number_input("ATR计算周期", 
                                       min_value=5, max_value=100, 
                                       value=indicator_config.get("atr_period", 34))
            atr_multiplier = st.number_input("ATR倍数", 
                                          min_value=0.5, max_value=10.0, 
                                          value=float(indicator_config.get("atr_multiplier", 3.0)))
        
        st.subheader("网络设置")
        network_config = config.get("network", {})
        http_proxy = st.text_input("HTTP代理", 
                                 value=network_config.get("http_proxy", "http://127.0.0.1:7897"))
        
        # 代理测试按钮在表单外
        
        st.subheader("数据设置")
        data_config = config.get("data", {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            default_lookback = st.number_input("默认回溯天数", 
                                          min_value=1, max_value=3650, 
                                          value=data_config.get("default_lookback_days", 1095),
                                          help="获取数据时，如果未指定开始时间，默认回溯的天数。设置为1095可获取3年数据，1825可获取5年数据。")
        
        with col2:
            max_candles = st.number_input("最大K线数量", 
                                       min_value=1000, max_value=10000000, 
                                       value=data_config.get("max_candles", 1000000),
                                       help="获取数据的最大K线数量限制，设置较大值可确保获取完整的历史数据。1小时K线，3年约有26280根，5年约有43800根。")
        
        submitted = st.form_submit_button("保存配置")
        
        if submitted:
            # 更新配置
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
                    "default_lookback_days": default_lookback,
                    "max_candles": max_candles
                }
            }
            
            config_manager.update_config(new_config)
            config_manager.save_config()
            
    # 代理测试按钮（在表单外）
    if st.button("测试代理连接"):
        proxy_url = config["network"]["http_proxy"]
        set_proxy(proxy_url)
        test_proxy(proxy_url)
    
    return config_manager.get_config()

if __name__ == "__main__":
    # 测试配置UI
    st.set_page_config(page_title="阿翔趋势交易系统 - 配置", layout="wide")
    render_config_ui() 