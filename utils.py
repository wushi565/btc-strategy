import os
import requests
import yaml
import streamlit as st
from datetime import datetime, timedelta
import pytz

def load_config(config_path="config.yaml"):
    """
    加载配置文件
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except Exception as e:
        st.error(f"加载配置文件失败: {e}")
        # 返回默认配置
        return {
            "trading": {"symbol": "BTC/USDT:USDT", "timeframe": "1h", "is_futures": True},
            "capital": {"initial_capital": 10000, "leverage": 3, "risk_per_trade": 0.02},
            "indicators": {
                "dema144_len": 144, 
                "dema169_len": 169, 
                "ema120_len": 120, 
                "ema200_len": 200,
                "atr_period": 34,
                "atr_multiplier": 3.0
            },
            "network": {
                "http_proxy": "http://127.0.0.1:7897",
                "https_proxy": "http://127.0.0.1:7897"
            },
            "data": {
                "default_lookback_days": 30,
                "max_candles": 1000
            }
        }

def save_config(config, config_path="config.yaml"):
    """
    保存配置到文件
    """
    try:
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.dump(config, file, default_flow_style=False)
        st.success("配置已保存")
        return True
    except Exception as e:
        st.error(f"保存配置失败: {e}")
        return False

def set_proxy(proxy_url):
    """
    全局设置代理
    """
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url

def test_proxy(proxy_url):
    """
    测试代理是否可用
    """
    try:
        with st.spinner(f"测试代理 {proxy_url} 是否可用..."):
            proxies = {
                'http': proxy_url,
                'https': proxy_url
            }
            # 尝试访问Binance API
            r = requests.get('https://fapi.binance.com/fapi/v1/ping', proxies=proxies, timeout=10)
            if r.status_code == 200:
                st.success("代理连接到Binance期货成功！")
                return True
            else:
                st.error(f"代理测试返回状态码: {r.status_code}")
                return False
    except Exception as e:
        st.error(f"代理测试失败: {e}")
        return False

def get_default_time_range():
    """
    获取默认时间范围（当前时间和30天前）
    """
    beijing_tz = pytz.timezone("Asia/Shanghai")
    now = datetime.now(beijing_tz)
    thirty_days_ago = now - timedelta(days=30)
    
    return thirty_days_ago, now

def format_datetime(dt):
    """
    格式化日期时间为字符串
    """
    if isinstance(dt, str):
        return dt
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def parse_datetime(dt_str):
    """
    解析日期时间字符串为datetime对象
    """
    if isinstance(dt_str, datetime):
        return dt_str
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")

def format_money(value):
    """
    格式化货币数值
    """
    return f"{value:,.2f}"

def format_percent(value):
    """
    格式化百分比
    """
    return f"{value:.2%}"

def display_status(message, status="info"):
    """
    在Streamlit中显示状态消息
    """
    if status == "success":
        st.success(message)
    elif status == "error":
        st.error(message)
    elif status == "warning":
        st.warning(message)
    else:
        st.info(message) 