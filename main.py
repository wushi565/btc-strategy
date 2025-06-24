import streamlit as st
import yaml
import os
import pandas as pd
import numpy as np
from config import render_config_ui
from data import render_data_ui
from indicators import render_indicator_ui
from signals import render_signal_ui
from backtest import render_backtest_ui

# 设置页面配置
st.set_page_config(
    page_title="阿翔趋势交易系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 应用标题
st.title("阿翔趋势交易系统")

# 加载配置
@st.cache_data
def load_config():
    """加载配置文件"""
    try:
        if os.path.exists("config.yaml"):
            with open("config.yaml", "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
                return config
        else:
            # 返回默认配置
            return {
                "trading": {
                    "symbol": "BTC/USDT",
                    "timeframe": "1h",
                    "is_futures": True
                },
                "proxy": {
                    "http_proxy": "http://127.0.0.1:7890",
                    "https_proxy": "http://127.0.0.1:7890"
                },
                "indicators": {
                    "dema144_len": 144,
                    "dema169_len": 169,
                    "ema120_len": 120,
                    "ema200_len": 200,
                    "atr_period": 34,
                    "atr_multiplier": 3.0
                },
                "signals": {
                    "risk_reward_ratio": 3.0
                },
                "backtest": {
                    "initial_capital": 10000,
                    "leverage": 3,
                    "risk_per_trade": 0.02
                }
            }
    except Exception as e:
        st.error(f"加载配置文件失败: {e}")
        return {}

# 主界面
def main():
    """主函数"""
    # 加载配置
    config = load_config()
    
    # 创建边栏菜单
    st.sidebar.title("阿翔趋势交易系统")
    
    # 菜单选项
    menu_options = ["配置管理", "数据获取", "指标计算", "信号分析", "回测分析"]
    selected_menu = st.sidebar.radio("导航菜单", menu_options)
    
    # 根据选择显示不同页面
    if selected_menu == "配置管理":
        updated_config = render_config_ui()
        if updated_config:
            config = updated_config
    
    elif selected_menu == "数据获取":
        klines_data = render_data_ui(config)
    
    elif selected_menu == "指标计算":
        indicators_df = render_indicator_ui(config)
    
    elif selected_menu == "信号分析":
        signals_df = render_signal_ui(config)
    
    elif selected_menu == "回测分析":
        backtest_df = render_backtest_ui(config)
    
    # 在底部显示版本信息
    st.sidebar.markdown("---")
    st.sidebar.info("阿翔趋势交易系统 v1.0.0")
    st.sidebar.text("Copyright © 2023")

if __name__ == "__main__":
    main()
