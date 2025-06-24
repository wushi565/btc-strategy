import ccxt
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import streamlit as st
import time
import traceback
from utils import format_datetime, set_proxy, test_proxy

class DataManager:
    """数据管理类，负责从交易所获取数据"""
    
    def __init__(self, config):
        """
        初始化数据管理器
        
        参数:
            config (dict): 配置信息
        """
        self.config = config
        self.trading_config = config.get("trading", {})
        self.network_config = config.get("network", {})
        self.data_config = config.get("data", {})
        
        self.symbol = self.trading_config.get("symbol", "BTC/USDT:USDT")
        self.timeframe = self.trading_config.get("timeframe", "1h")
        self.is_futures = self.trading_config.get("is_futures", True)
        
        self.http_proxy = self.network_config.get("http_proxy", "")
        self.https_proxy = self.network_config.get("https_proxy", "")
        
        self.max_candles = self.data_config.get("max_candles", 1000)
        self.default_lookback_days = self.data_config.get("default_lookback_days", 30)
        
        # 设置代理
        if self.http_proxy:
            set_proxy(self.http_proxy)
        
        # 初始化交易所
        self._init_exchange()
        
        # 时区设置
        self.beijing_tz = pytz.timezone("Asia/Shanghai")
    
    def _init_exchange(self):
        """初始化交易所连接"""
        try:
            # 基本配置
            config = {
                'enableRateLimit': True,
                'timeout': 60000,  # 60秒超时
                'options': {
                    'defaultType': 'future' if self.is_futures else 'spot'
                }
            }
            
            # 创建交易所对象 - 使用binanceusdm而不是binance来获取合约数据
            exchange_id = 'binanceusdm' if self.is_futures else 'binance'
            self.exchange = getattr(ccxt, exchange_id)(config)
            
            # 设置代理
            if self.http_proxy:
                self.exchange.proxies = {
                    'http': self.http_proxy,
                    'https': self.https_proxy or self.http_proxy
                }
                
        except Exception as e:
            st.error(f"初始化交易所失败: {e}")
            raise
    
    def test_connection(self):
        """测试交易所连接"""
        try:
            with st.spinner("正在测试交易所连接..."):
                self.exchange.fetch_time()
                st.success("交易所连接成功!")
                return True
        except Exception as e:
            st.error(f"交易所连接失败: {e}")
            return False
    
    def fetch_klines(self, start_time=None, end_time=None, progress_bar=None):
        """
        获取K线数据
        
        参数:
            start_time (str|datetime): 开始时间 (例如 '2023-01-01 00:00:00')
            end_time (str|datetime): 结束时间 (例如 '2023-01-31 23:59:59')
            progress_bar (st.progress): Streamlit进度条对象
            
        返回:
            pandas.DataFrame: K线数据
        """
        try:
            market_type = "合约" if self.is_futures else "现货"
            st.info(f"正在获取{market_type}交易对 {self.symbol} 的 {self.timeframe} K线数据...")
            
            # 时间转换
            if start_time:
                if isinstance(start_time, str):
                    start_dt = self.beijing_tz.localize(datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S"))
                else:
                    start_dt = start_time.astimezone(self.beijing_tz)
                since = int(start_dt.astimezone(pytz.UTC).timestamp() * 1000)
                st.info(f"开始时间(北京): {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                # 默认获取过去30天的数据
                now = datetime.now(self.beijing_tz)
                start_dt = now - timedelta(days=self.default_lookback_days)
                since = int(start_dt.astimezone(pytz.UTC).timestamp() * 1000)
                st.info(f"未指定开始时间，默认获取过去{self.default_lookback_days}天数据")
                st.info(f"开始时间(北京): {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
                
            if end_time:
                if isinstance(end_time, str):
                    end_dt = self.beijing_tz.localize(datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S"))
                else:
                    end_dt = end_time.astimezone(self.beijing_tz)
                end_ts = int(end_dt.astimezone(pytz.UTC).timestamp() * 1000)
                st.info(f"结束时间(北京): {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                end_ts = None
                st.info("未指定结束时间，获取至最新数据")
            
            # 验证交易对是否支持
            try:
                markets = self.exchange.load_markets()
                if self.symbol not in markets:
                    st.warning(f"警告: 交易对 {self.symbol} 不在支持列表中!")
                    available_symbols = list(markets.keys())[:10]  # 只显示前10个
                    st.warning(f"支持的部分交易对: {available_symbols}...")
                else:
                    market_info = markets[self.symbol]
                    st.success(f"交易对 {self.symbol} 验证通过! {market_info.get('type', '未知')} 市场, 基础货币:{market_info.get('base')}, 计价货币:{market_info.get('quote')}")
            except Exception as e:
                st.error(f"验证交易对时出错: {e}")
            
            # 获取数据
            all_ohlcv = []
            fetch_since = since
            batch_count = 0
            batch_limit = 500  # 每批获取的K线数量
            
            # 创建进度条
            if progress_bar is None:
                progress_bar = st.progress(0)
                
            st.info("开始获取K线数据...")
            
            while True:
                batch_count += 1
                progress_text = f"获取第{batch_count}批数据..."
                st.text(progress_text)
                
                try:
                    ohlcv = self.exchange.fetch_ohlcv(
                        self.symbol, 
                        self.timeframe, 
                        since=fetch_since, 
                        limit=batch_limit
                    )
                    
                    if not ohlcv:
                        st.info("没有更多数据")
                        break
                        
                    # 显示首条和末条数据时间
                    first_time = datetime.fromtimestamp(ohlcv[0][0]/1000, tz=pytz.UTC).astimezone(self.beijing_tz)
                    last_time = datetime.fromtimestamp(ohlcv[-1][0]/1000, tz=pytz.UTC).astimezone(self.beijing_tz)
                    
                    st.text(f"批次 {batch_count}: {first_time.strftime('%Y-%m-%d %H:%M:%S')} 至 {last_time.strftime('%Y-%m-%d %H:%M:%S')}, 共{len(ohlcv)}条")
                    
                    # 过滤超出end_time的数据
                    if end_ts:
                        ohlcv = [x for x in ohlcv if x[0] <= end_ts]
                    
                    all_ohlcv.extend(ohlcv)
                    
                    # 更新进度条 - 估算进度
                    if end_ts:
                        progress = min(1.0, (ohlcv[-1][0] - since) / (end_ts - since)) if ohlcv else 1.0
                    else:
                        # 如果没有结束时间，则假设最多获取100批
                        progress = min(1.0, batch_count / 100)
                    progress_bar.progress(progress)
                    
                    if len(ohlcv) < batch_limit:
                        st.info("返回数据少于请求限制，已获取全部数据")
                        break
                        
                    last_ts = ohlcv[-1][0]
                    # 防止死循环
                    if fetch_since is not None and last_ts <= fetch_since:
                        st.warning("数据时间戳未前进，停止获取")
                        break
                        
                    fetch_since = last_ts + 1
                    
                    if end_ts and fetch_since > end_ts:
                        st.info("已达到结束时间，停止获取")
                        break
                        
                    # 检查是否已超过最大K线数量
                    if len(all_ohlcv) >= self.max_candles:
                        st.warning(f"已达到最大K线数量限制 ({self.max_candles})，停止获取")
                        break
                        
                    # 稍作延迟，避免API请求过快
                    time.sleep(0.5)
                    
                except Exception as e:
                    st.error(f"获取数据时出错: {e}")
                    st.error(traceback.format_exc())
                    break
            
            # 完成进度条
            progress_bar.progress(1.0)
            
            if not all_ohlcv:
                st.error("未获取到K线数据，请检查时间区间或网络/代理设置！")
                return None
                
            st.success(f"总计获取了 {len(all_ohlcv)} 条K线数据")
            
            # 转换为DataFrame
            df = pd.DataFrame(all_ohlcv)
            df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
            
            # 时间戳转换为北京时间
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(self.beijing_tz)
            df.set_index("timestamp", inplace=True)
            
            # 显示数据范围
            st.info(f"数据时间范围: {df.index.min()} 至 {df.index.max()}")
            
            return df
            
        except Exception as e:
            st.error(f"获取K线数据失败: {e}")
            st.error(traceback.format_exc())
            return None

def render_data_ui(config):
    """渲染数据获取UI界面"""
    
    st.header("数据获取")
    
    # 初始化数据管理器
    data_manager = DataManager(config)
    
    # 显示当前配置
    trading_config = config.get("trading", {})
    st.info(f"交易对: {trading_config.get('symbol')} ({trading_config.get('timeframe')}) - {'合约' if trading_config.get('is_futures') else '现货'}")
    
    # 设置时间范围
    st.subheader("时间范围设置")
    col1, col2 = st.columns(2)
    
    with col1:
        # 默认开始时间为30天前
        default_start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
        start_time = st.text_input("开始时间 (北京时间)", value=default_start, 
                                 help="格式: YYYY-MM-DD HH:MM:SS")
    
    with col2:
        # 默认结束时间为当前时间
        default_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        end_time = st.text_input("结束时间 (北京时间)", value=default_end,
                               help="格式: YYYY-MM-DD HH:MM:SS")
    
    # 数据获取按钮
    if st.button("获取K线数据"):
        # 测试连接
        if data_manager.test_connection():
            # 创建进度条
            progress_bar = st.progress(0)
            
            # 获取数据
            df = data_manager.fetch_klines(start_time, end_time, progress_bar)
            
            if df is not None and not df.empty:
                # 保存数据到session_state
                st.session_state.klines_data = df
                
                # 显示数据预览
                st.subheader("数据预览")
                st.dataframe(df.head())
                
                # 显示数据统计
                st.subheader("数据统计")
                st.write(f"数据条数: {len(df)}")
                st.write(f"开始时间: {df.index.min()}")
                st.write(f"结束时间: {df.index.max()}")
                
                return df
            else:
                st.error("获取数据失败或数据为空")
                return None
    
    # 如果session_state中已有数据，显示数据统计
    if 'klines_data' in st.session_state and st.session_state.klines_data is not None:
        df = st.session_state.klines_data
        st.success(f"已加载 {len(df)} 条K线数据 ({df.index.min()} 至 {df.index.max()})")
        
        # 显示数据预览按钮
        if st.button("显示数据预览"):
            st.dataframe(df.head())
            
        return df
    
    return None

if __name__ == "__main__":
    # 测试数据获取UI
    import yaml
    
    st.set_page_config(page_title="阿翔趋势交易系统 - 数据获取", layout="wide")
    
    # 加载配置
    with open("config.yaml", 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    render_data_ui(config) 