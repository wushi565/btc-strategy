import ccxt
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import os
import requests
import sys
import traceback
# xlsxwriter用于更稳定的Excel导出
try:
    import xlsxwriter
except ImportError:
    print("警告: xlsxwriter未安装, 将使用备选Excel引擎")
    print("建议安装 xlsxwriter: pip install xlsxwriter")

# 自动安装缺失的依赖包
def install_missing_packages(packages):
    """
    尝试安装缺失的包
    
    参数:
        packages (list): 要安装的包名列表
    """
    import subprocess
    import sys
    
    print(f"正在检查并安装缺失的依赖包: {', '.join(packages)}")
    
    for package in packages:
        try:
            # 尝试导入包以检查是否已安装
            __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            print(f"正在安装 {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ {package} 安装成功")
            except Exception as e:
                print(f"✗ 安装 {package} 失败: {e}")

# 检查并安装依赖包
required_packages = ["ccxt", "pandas", "numpy", "pytz", "openpyxl", "xlsxwriter"]

# 设置全局代理环境变量
def set_proxy(proxy_url):
    """全局设置代理"""
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url
    # 不要直接设置ccxt.Exchange.proxies，会导致类型错误

# 测试代理是否可用
def test_proxy(proxy_url):
    try:
        print(f"测试代理 {proxy_url} 是否可用...")
        proxies = {
            'http': proxy_url,
            'https': proxy_url
        }
        # 尝试访问Binance API
        r = requests.get('https://fapi.binance.com/fapi/v1/ping', proxies=proxies, timeout=10)
        if r.status_code == 200:
            print("代理连接到Binance期货成功！")
            return True
        else:
            print(f"代理测试返回状态码: {r.status_code}")
            return False
    except Exception as e:
        print(f"代理测试失败: {e}")
        return False

class AxiangTrendSystem:
    def __init__(self, symbol, timeframe="1h", dema144_len=144, dema169_len=169, 
                 ema120_len=120, ema200_len=200, atr_period=34, atr_multiplier=3.0,
                 http_proxy=None, https_proxy=None, is_futures=True,
                 initial_capital=10000, leverage=1):
        """
        初始化阿翔趋势交易系统
        
        参数:
            symbol (str): 交易对 (例如 "BTC/USDT")
            timeframe (str): 时间周期 (例如 "1h", "4h", "1d")
            dema144_len (int): DEMA144均线周期
            dema169_len (int): DEMA169均线周期
            ema120_len (int): EMA120均线周期
            ema200_len (int): EMA200均线周期
            atr_period (int): ATR计算周期
            atr_multiplier (float): ATR倍数
            http_proxy (str): HTTP代理URL，例如 'http://127.0.0.1:7890'
            https_proxy (str): HTTPS代理URL，例如 'http://127.0.0.1:7890'
            is_futures (bool): 是否为合约数据
            initial_capital (float): 初始资金（USDT）
            leverage (float): 杠杆倍数
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.dema144_len = dema144_len
        self.dema169_len = dema169_len
        self.ema120_len = ema120_len
        self.ema200_len = ema200_len
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.is_futures = is_futures
        
        # 交易相关参数
        self.initial_capital = initial_capital  # 初始资金
        self.leverage = leverage                # 杠杆倍数
        self.risk_per_trade = 0.02              # 每笔交易固定风险比例为2%
        
        # 初始化交易所接口，使用简单的配置方式
        config = {
            'enableRateLimit': True,
            'timeout': 60000,  # 延长超时时间到60秒
            'options': {
                'defaultType': 'future' if is_futures else 'spot'
            }
        }
        
        # 创建交易所对象 - 使用binanceusdm而不是binance来获取合约数据
        exchange_id = 'binanceusdm' if is_futures else 'binance'
        self.exchange = getattr(ccxt, exchange_id)(config)
        
        # 手动设置代理
        if http_proxy:
            try:
                self.exchange.proxies = {
                    'http': http_proxy,
                    'https': https_proxy or http_proxy
                }
            except Exception as e:
                print(f"设置代理时出错: {e}")
        
        # 时区转换
        self.beijing_tz = pytz.timezone("Asia/Shanghai")
        
    def fetch_ohlcv(self, start_time=None, end_time=None, limit=500):
        """
        获取K线数据，支持指定开始和结束时间（北京时间字符串），如 '2024-05-01 00:00:00'
        """
        print(f"正在获取{'合约' if self.is_futures else '现货'}交易对数据...")
        print(f"交易所ID: {self.exchange.id}")
        print(f"市场类型: {self.exchange.options.get('defaultType', '未知')}")
        
        # 时间转换
        if start_time:
            start_dt = self.beijing_tz.localize(datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S"))
            since = int(start_dt.astimezone(pytz.UTC).timestamp() * 1000)
            print(f"开始时间(北京): {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"开始时间(UTC): {start_dt.astimezone(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            since = None
            print("未指定开始时间，获取最近数据")
            
        if end_time:
            end_dt = self.beijing_tz.localize(datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S"))
            end_ts = int(end_dt.astimezone(pytz.UTC).timestamp() * 1000)
            print(f"结束时间(北京): {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"结束时间(UTC): {end_dt.astimezone(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            end_ts = None
            print("未指定结束时间，获取至最新数据")
        
        # 验证交易对是否支持
        try:
            markets = self.exchange.load_markets()
            if self.symbol not in markets:
                print(f"警告: 交易对 {self.symbol} 不在支持列表中!")
                available_symbols = list(markets.keys())[:10]  # 只显示前10个
                print(f"支持的部分交易对: {available_symbols}...")
            else:
                print(f"交易对 {self.symbol} 验证通过!")
                market_info = markets[self.symbol]
                print(f"交易对信息: {market_info.get('type', '未知')} 市场, 基础货币:{market_info.get('base')}, 计价货币:{market_info.get('quote')}")
        except Exception as e:
            print(f"验证交易对时出错: {e}")
        
        # 获取数据
        all_ohlcv = []
        fetch_since = since
        batch_count = 0
        
        print("开始获取K线数据...")
        while True:
            batch_count += 1
            print(f"获取第{batch_count}批数据...")
            try:
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since=fetch_since, limit=limit)
                if not ohlcv:
                    print("没有更多数据")
                    break
                    
                # 显示首条和末条数据时间
                first_time = datetime.fromtimestamp(ohlcv[0][0]/1000, tz=pytz.UTC).astimezone(self.beijing_tz)
                last_time = datetime.fromtimestamp(ohlcv[-1][0]/1000, tz=pytz.UTC).astimezone(self.beijing_tz)
                print(f"本批数据时间范围: {first_time.strftime('%Y-%m-%d %H:%M:%S')} 至 {last_time.strftime('%Y-%m-%d %H:%M:%S')}, 共{len(ohlcv)}条")
                
                # 过滤超出end_time的数据
                if end_ts:
                    ohlcv = [x for x in ohlcv if x[0] <= end_ts]
                
                all_ohlcv.extend(ohlcv)
                
                if len(ohlcv) < limit:
                    print("返回数据少于请求限制，已获取全部数据")
                    break
                    
                last_ts = ohlcv[-1][0]
                # 防止死循环
                if fetch_since is not None and last_ts <= fetch_since:
                    print("数据时间戳未前进，停止获取")
                    break
                    
                fetch_since = last_ts + 1
                
                if end_ts and fetch_since > end_ts:
                    print("已达到结束时间，停止获取")
                    break
                    
            except Exception as e:
                print(f"获取数据时出错: {e}")
                break
        
        if not all_ohlcv:
            raise Exception("未获取到K线数据，请检查时间区间或网络/代理设置！")
            
        print(f"总计获取了 {len(all_ohlcv)} 条K线数据")
        
        # 转换为DataFrame
        df = pd.DataFrame(all_ohlcv)
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        
        # 时间戳转换为北京时间
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(self.beijing_tz)
        df.set_index("timestamp", inplace=True)
        
        # 显示数据范围
        print(f"数据时间范围: {df.index.min()} 至 {df.index.max()}")
        print(f"数据列: {df.columns.tolist()}")
        print(f"数据前5条:\n{df.head()}")
        
        return df
    
    def calculate_indicators(self, df):
        """
        计算所有技术指标
        """
        # 均线计算
        # DEMA = 2 * EMA - EMA(EMA)
        df["ema_first_144"] = df["close"].ewm(span=self.dema144_len, adjust=False).mean()
        df["ema_second_144"] = df["ema_first_144"].ewm(span=self.dema144_len, adjust=False).mean()
        df["dema_144"] = 2 * df["ema_first_144"] - df["ema_second_144"]
        
        df["ema_first_169"] = df["close"].ewm(span=self.dema169_len, adjust=False).mean()
        df["ema_second_169"] = df["ema_first_169"].ewm(span=self.dema169_len, adjust=False).mean()
        df["dema_169"] = 2 * df["ema_first_169"] - df["ema_second_169"]
        
        df["ema_120"] = df["close"].ewm(span=self.ema120_len, adjust=False).mean()
        df["ema_200"] = df["close"].ewm(span=self.ema200_len, adjust=False).mean()
        
        # ATR计算
        df["high_low"] = df["high"] - df["low"]
        df["high_close"] = abs(df["high"] - df["close"].shift())
        df["low_close"] = abs(df["low"] - df["close"].shift())
        df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)
        df["atr"] = df["tr"].rolling(window=self.atr_period).mean()
        
        # 计算hl2 (high+low)/2
        df["hl2"] = (df["high"] + df["low"]) / 2
        
        # 计算上下轨道
        df["upper"] = df["hl2"] - self.atr_multiplier * df["atr"]
        df["lower"] = df["hl2"] + self.atr_multiplier * df["atr"]
        
        # 创建上下轨道的副本，以防止链式赋值问题
        upper_values = df["upper"].copy()
        lower_values = df["lower"].copy()
        upper_shifted = df["upper"].shift(1).fillna(df["upper"].iloc[0])
        lower_shifted = df["lower"].shift(1).fillna(df["lower"].iloc[0])
        
        # 计算趋势 - 使用安全的方法更新值
        for i in range(1, len(df)):
            if df["close"].iloc[i] > upper_shifted.iloc[i]:
                upper_values.iloc[i] = max(upper_values.iloc[i], upper_shifted.iloc[i])
            
            if df["close"].iloc[i] < lower_shifted.iloc[i]:
                lower_values.iloc[i] = min(lower_values.iloc[i], lower_shifted.iloc[i])
        
        # 将计算结果赋值回DataFrame
        df["upper"] = upper_values
        df["lower"] = lower_values
        
        # 初始化趋势列
        trend = np.ones(len(df), dtype=int)  # 初始化全部为1
        
        # 计算趋势 - 避免使用df["trend"].iloc[i]这种方式
        for i in range(1, len(df)):
            prev_trend = trend[i-1]
            if prev_trend == -1 and df["close"].iloc[i] > df["lower"].iloc[i-1]:
                trend[i] = 1
            elif prev_trend == 1 and df["close"].iloc[i] < df["upper"].iloc[i-1]:
                trend[i] = -1
            else:
                trend[i] = prev_trend
        
        # 将趋势数组赋值给DataFrame
        df["trend"] = trend
        
        # 计算买卖信号 - 使用numpy初始化并赋值，避免链式赋值
        buy_signal = np.zeros(len(df), dtype=bool)
        sell_signal = np.zeros(len(df), dtype=bool)
        
        for i in range(1, len(df)):
            # 计算买入信号
            buy_signal[i] = (
                trend[i] == 1 and 
                trend[i-1] == -1 and
                df["close"].iloc[i] > df["dema_144"].iloc[i] and 
                df["close"].iloc[i] > df["dema_169"].iloc[i]
            )
            
            # 计算卖出信号
            sell_signal[i] = (
                trend[i] == -1 and 
                trend[i-1] == 1 and
                df["close"].iloc[i] < df["dema_144"].iloc[i] and 
                df["close"].iloc[i] < df["dema_169"].iloc[i]
            )
        
        # 将计算结果赋值回DataFrame
        df["buy_signal"] = buy_signal
        df["sell_signal"] = sell_signal
        
        return df
    
    def calculate_risk_reward(self, df):
        """
        计算风险收益比
        """
        df["stop_loss_buy"] = df["dema_169"]
        df["stop_loss_sell"] = df["dema_144"]
        
        # 初始化为NaN的数组
        take_profit_buy = np.full(len(df), np.nan)
        take_profit_sell = np.full(len(df), np.nan)
        
        # 计算买入时的止盈价格 (风险收益比为3) - 使用数组操作
        for i in range(len(df)):
            if df["buy_signal"].iloc[i]:
                close_price = df["close"].iloc[i]
                stop_loss = df["stop_loss_buy"].iloc[i]
                risk = close_price - stop_loss
                take_profit_buy[i] = close_price + 3 * risk
        
        # 计算卖出时的止盈价格 (风险收益比为3) - 使用数组操作
        for i in range(len(df)):
            if df["sell_signal"].iloc[i]:
                close_price = df["close"].iloc[i]
                stop_loss = df["stop_loss_sell"].iloc[i]
                risk = stop_loss - close_price
                take_profit_sell[i] = close_price - 3 * risk
        
        # 将计算结果赋值回DataFrame
        df["take_profit_buy"] = take_profit_buy
        df["take_profit_sell"] = take_profit_sell
        
        return df
    
    def calculate_position_size(self, entry_price, stop_loss, capital, leverage=1, risk_ratio=0.02):
        """
        计算头寸大小
        
        参数:
            entry_price (float): 入场价格
            stop_loss (float): 止损价格
            capital (float): 可用资金
            leverage (float): 杠杆倍数
            risk_ratio (float): 风险比例(占总资金的百分比)
            
        返回:
            float: 头寸大小(以基础货币计量)
        """
        # 计算愿意承担的风险金额
        risk_amount = capital * risk_ratio
        
        # 计算价格差异(风险)百分比
        price_risk_percent = abs(entry_price - stop_loss) / entry_price
        
        # 根据杠杆调整风险百分比
        effective_risk_percent = price_risk_percent / leverage
        
        # 计算头寸大小(以基础货币计量)
        if effective_risk_percent > 0:
            position_size = risk_amount / effective_risk_percent
            # 调整为杠杆后的头寸大小
            position_size = position_size * leverage / entry_price
        else:
            position_size = 0
            
        return position_size
    
    def calculate_trade_metrics(self, df):
        """
        计算交易指标，包括头寸大小、盈亏和资金变化
        """
        # 复制DataFrame以避免修改原始数据
        trade_df = df.copy()
        
        # 添加资金列并初始化为初始资金
        trade_df["capital"] = self.initial_capital
        
        # 添加头寸大小列
        trade_df["position_size"] = 0.0
        
        # 添加交易盈亏列
        trade_df["trade_pnl"] = 0.0
        
        # 添加累计盈亏列
        trade_df["cumulative_pnl"] = 0.0
        
        # 遍历数据计算交易指标
        current_capital = self.initial_capital
        current_position = 0.0
        entry_price = 0.0
        position_type = None  # "long"或"short"
        cumulative_pnl = 0.0
        win_count = 0
        loss_count = 0
        
        for i in range(len(trade_df)):
            trade_df.loc[trade_df.index[i], "capital"] = current_capital
            
            # 检查是否有买入信号
            buy_signal_value = bool(trade_df.iloc[i]["buy_signal"] == True)
            
            if buy_signal_value:
                # 如果有空头头寸，先平仓
                if position_type == "short" and current_position > 0:
                    # 计算空头平仓盈亏
                    exit_price = trade_df.iloc[i]["close"]
                    pnl = current_position * (entry_price - exit_price) * self.leverage
                    current_capital += pnl
                    cumulative_pnl += pnl
                    
                    # 更新统计
                    if pnl > 0:
                        win_count += 1
                    elif pnl < 0:
                        loss_count += 1
                    
                    # 记录平仓结果
                    trade_df.loc[trade_df.index[i], "trade_pnl"] = pnl
                    
                # 计算新的多头头寸大小
                entry_price = trade_df.iloc[i]["close"]
                stop_loss = trade_df.iloc[i]["stop_loss_buy"]
                
                current_position = self.calculate_position_size(
                    entry_price, 
                    stop_loss, 
                    current_capital, 
                    self.leverage, 
                    self.risk_per_trade
                )
                
                # 更新状态
                position_type = "long"
                trade_df.loc[trade_df.index[i], "position_size"] = current_position
                
            # 检查是否有卖出信号
            elif bool(trade_df.iloc[i]["sell_signal"] == True):
                # 如果有多头头寸，先平仓
                if position_type == "long" and current_position > 0:
                    # 计算多头平仓盈亏
                    exit_price = trade_df.iloc[i]["close"]
                    pnl = current_position * (exit_price - entry_price) * self.leverage
                    current_capital += pnl
                    cumulative_pnl += pnl
                    
                    # 更新统计
                    if pnl > 0:
                        win_count += 1
                    elif pnl < 0:
                        loss_count += 1
                    
                    # 记录平仓结果
                    trade_df.loc[trade_df.index[i], "trade_pnl"] = pnl
                
                # 计算新的空头头寸大小
                entry_price = trade_df.iloc[i]["close"]
                stop_loss = trade_df.iloc[i]["stop_loss_sell"]
                
                current_position = self.calculate_position_size(
                    entry_price, 
                    stop_loss, 
                    current_capital, 
                    self.leverage, 
                    self.risk_per_trade
                )
                
                # 更新状态
                position_type = "short"
                trade_df.loc[trade_df.index[i], "position_size"] = current_position
            
            # 更新累计盈亏
            trade_df.loc[trade_df.index[i], "cumulative_pnl"] = cumulative_pnl
        
        # 计算胜率
        total_trades = win_count + loss_count
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # 计算净值增长率
        growth_rate = (current_capital - self.initial_capital) / self.initial_capital
        
        # 保存汇总数据
        self.summary = {
            "初始资金": self.initial_capital,
            "当前资金": current_capital,
            "净利润": cumulative_pnl,
            "杠杆倍数": self.leverage,
            "交易次数": total_trades,
            "盈利次数": win_count,
            "亏损次数": loss_count,
            "胜率": win_rate,
            "净值增长率": growth_rate
        }
        
        return trade_df
    
    def run_strategy(self, start_time=None, end_time=None):
        """
        运行策略并返回带有信号的DataFrame
        """
        df = self.fetch_ohlcv(start_time=start_time, end_time=end_time)
        df = self.calculate_indicators(df)
        df = self.calculate_risk_reward(df)
        df = self.calculate_trade_metrics(df)
        
        return df
    
    def export_to_excel(self, df, filename="trading_signals.xlsx"):
        """
        将交易信号导出到Excel
        """
        try:
            # 创建买入信号数据
            buy_signals = df[df["buy_signal"] == True].copy()
            if not buy_signals.empty:
                # 为买入信号创建一个DataFrame
                buy_data = pd.DataFrame({
                    "信号类型": "买入",
                    "时间": buy_signals.index,
                    "开仓价": buy_signals["close"],
                    "DEMA144": buy_signals["dema_144"],
                    "DEMA169": buy_signals["dema_169"],
                    "止损价": buy_signals["stop_loss_buy"],
                    "止盈价": buy_signals["take_profit_buy"],
                    "风险": buy_signals["close"] - buy_signals["stop_loss_buy"],
                    "收益": buy_signals["take_profit_buy"] - buy_signals["close"],
                    "风险收益比": (buy_signals["take_profit_buy"] - buy_signals["close"]) / 
                              (buy_signals["close"] - buy_signals["stop_loss_buy"]),
                    "杠杆倍数": self.leverage,
                    "头寸大小": buy_signals["position_size"],
                    "投入资金": buy_signals["position_size"] * buy_signals["close"],
                    "资金余额": buy_signals["capital"],
                    "交易盈亏": buy_signals["trade_pnl"],
                    "累计盈亏": buy_signals["cumulative_pnl"]
                })
                
                # 格式化风险收益比为"1:X"格式
                buy_data["风险收益比"] = buy_data["风险收益比"].apply(lambda x: f"1:{x:.2f}")
                
                # 将时间列转换为字符串格式，去除时区信息
                buy_data["时间"] = buy_data["时间"].dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                buy_data = pd.DataFrame()
            
            # 创建卖出信号数据
            sell_signals = df[df["sell_signal"] == True].copy()
            if not sell_signals.empty:
                # 为卖出信号创建一个DataFrame
                sell_data = pd.DataFrame({
                    "信号类型": "卖出",
                    "时间": sell_signals.index,
                    "开仓价": sell_signals["close"],
                    "DEMA144": sell_signals["dema_144"],
                    "DEMA169": sell_signals["dema_169"],
                    "止损价": sell_signals["stop_loss_sell"],
                    "止盈价": sell_signals["take_profit_sell"],
                    "风险": sell_signals["stop_loss_sell"] - sell_signals["close"],
                    "收益": sell_signals["close"] - sell_signals["take_profit_sell"],
                    "风险收益比": abs((sell_signals["close"] - sell_signals["take_profit_sell"]) / 
                                  (sell_signals["stop_loss_sell"] - sell_signals["close"])),
                    "杠杆倍数": self.leverage,
                    "头寸大小": sell_signals["position_size"],
                    "投入资金": sell_signals["position_size"] * sell_signals["close"],
                    "资金余额": sell_signals["capital"],
                    "交易盈亏": sell_signals["trade_pnl"],
                    "累计盈亏": sell_signals["cumulative_pnl"]
                })
                
                # 格式化风险收益比为"1:X"格式
                sell_data["风险收益比"] = sell_data["风险收益比"].apply(lambda x: f"1:{x:.2f}")
                
                # 将时间列转换为字符串格式，去除时区信息
                sell_data["时间"] = sell_data["时间"].dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                sell_data = pd.DataFrame()
            
            # 合并买入和卖出信号
            all_signals = pd.concat([buy_data, sell_data]).sort_values("时间")
            
            # 导出到Excel
            if not all_signals.empty:
                # 处理可能导致问题的列
                for col in all_signals.columns:
                    # 检查是否有非法值（如NaN、Inf）
                    if (all_signals[col].isnull().values.any()):
                        print(f"警告: 列 '{col}' 包含缺失值，将填充为0")
                        all_signals[col] = all_signals[col].fillna(0)
                
                # 检查并删除可能存在的文件
                if os.path.exists(filename):
                    try:
                        os.remove(filename)
                        print(f"已删除现有文件: {filename}")
                    except Exception as e:
                        print(f"无法删除现有文件 {filename}: {e}")
                        # 使用不同的文件名
                        base, ext = os.path.splitext(filename)
                        filename = f"{base}_{datetime.now().strftime('%Y%m%d%H%M%S')}{ext}"
                        print(f"将使用新文件名: {filename}")
                
                # 先导出为CSV (更稳定的格式)
                csv_filename = os.path.splitext(filename)[0] + '.csv'
                all_signals.to_csv(csv_filename, index=False, encoding='utf-8-sig')
                print(f"交易信号已导出到CSV文件: {csv_filename}")
                
                try:
                    # 尝试使用xlsxwriter引擎导出到Excel
                    excel_engine = 'xlsxwriter'
                    writer = pd.ExcelWriter(filename, engine=excel_engine)
                    all_signals.to_excel(writer, sheet_name='交易信号', index=False)
                    
                    # 创建交易汇总表
                    summary_df = pd.DataFrame({
                        "指标": list(self.summary.keys()),
                        "数值": list(self.summary.values())
                    })
                    summary_df.to_excel(writer, sheet_name='交易汇总', index=False)
                    
                    # 导出最后30条K线的关键指标
                    recent_data = df.tail(30).copy()
                    recent_data_reset = recent_data.reset_index()
                    recent_data_reset['timestamp'] = recent_data_reset['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    recent_data_reset.to_excel(writer, sheet_name='最近K线数据', index=False)
                    
                    # 获取workbook和worksheet
                    workbook = writer.book
                    worksheet = writer.sheets['交易信号']
                    summary_sheet = writer.sheets['交易汇总']
                    recent_sheet = writer.sheets['最近K线数据']
                    
                    # 根据所选引擎设置列宽
                    if excel_engine == 'xlsxwriter':
                        for idx, col in enumerate(all_signals.columns):
                            max_len = max(all_signals[col].astype(str).map(len).max(), len(col)) + 2
                            worksheet.set_column(idx, idx, max_len)
                        
                        # 设置汇总表格式
                        summary_sheet.set_column(0, 0, 15)
                        summary_sheet.set_column(1, 1, 12)
                        
                        # 设置最近K线数据表格式
                        for idx, col in enumerate(recent_data_reset.columns):
                            max_len = max(recent_data_reset[col].astype(str).map(len).max(), len(col)) + 2
                            recent_sheet.set_column(idx, idx, max_len)
                        
                        # 添加格式
                        percent_format = workbook.add_format({'num_format': '0.00%'})
                        money_format = workbook.add_format({'num_format': '#,##0.00'})
                        
                        # 设置胜率和净值增长率的百分比格式
                        summary_sheet.write(7, 1, self.summary["胜率"], percent_format)
                        summary_sheet.write(8, 1, self.summary["净值增长率"], percent_format)
                    
                    # 确保文件被保存并关闭
                    writer.close()
                    print(f"交易信号也已导出到Excel文件: {filename}")
                    return filename
                except Exception as e:
                    print(f"导出到Excel失败: {e}")
                    print(f"请使用CSV文件: {csv_filename}")
                    return csv_filename
            else:
                print("没有找到交易信号")
                return None
                
        except Exception as e:
            print(f"导出数据时出错: {e}")
            traceback.print_exc()
            return None
    
    def analyze_signals(self, df):
        """
        分析交易信号（简化控制台输出）
        """
        # 只输出交易汇总信息
        if hasattr(self, 'summary'):
            print("\n==== 交易汇总 ====")
            print(f"初始资金: {self.summary['初始资金']:.2f} USDT")
            print(f"当前资金: {self.summary['当前资金']:.2f} USDT")
            print(f"净利润: {self.summary['净利润']:.2f} USDT")
            print(f"杠杆倍数: {self.summary['杠杆倍数']}倍")
            print(f"交易次数: {self.summary['交易次数']}次")
            print(f"盈利次数: {self.summary['盈利次数']}次")
            print(f"亏损次数: {self.summary['亏损次数']}次")
            print(f"胜率: {self.summary['胜率']:.2%}")
            print(f"净值增长率: {self.summary['净值增长率']:.2%}")
            print("==============")
            print("详细信息请查看Excel文件")

    def export_raw_data_to_excel(self, df, filename=None):
        """
        将原始K线数据导出到Excel
        """
        if filename is None:
            filename = f"{self.symbol.replace('/', '_').replace(':', '_')}_{self.timeframe}_raw_data.xlsx"
        
        try:
            # 复制数据框以避免修改原始数据
            export_df = df.reset_index().copy()
            
            # 验证数据有效性
            if export_df.empty:
                print("警告: 没有数据可导出")
                return None
                
            # 限制行数，防止Excel崩溃（如果数据太多）
            max_rows = 500000  # Excel的最大行数约为1,048,576
            if len(export_df) > max_rows:
                print(f"警告: 数据过多 ({len(export_df)} 行)，将仅导出最近的 {max_rows} 行")
                export_df = export_df.tail(max_rows)
            
            # 处理包含时区信息的日期时间列 - 转换为字符串格式
            export_df['timestamp'] = export_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # 检查并删除可能存在的文件
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                    print(f"已删除现有文件: {filename}")
                except Exception as e:
                    print(f"无法删除现有文件 {filename}: {e}")
                    # 使用不同的文件名
                    base, ext = os.path.splitext(filename)
                    filename = f"{base}_{datetime.now().strftime('%Y%m%d%H%M%S')}{ext}"
                    print(f"将使用新文件名: {filename}")
            
            # 处理可能导致问题的列
            for col in export_df.columns:
                # 检查是否有非法值（如NaN、Inf）
                if export_df[col].isna().any():
                    print(f"警告: 列 '{col}' 包含缺失值，将填充为0")
                    export_df[col] = export_df[col].fillna(0)
                
                # 检查数据类型
                if export_df[col].dtype == 'object' and col != 'timestamp':
                    print(f"警告: 列 '{col}' 为对象类型，将转换为字符串")
                    export_df[col] = export_df[col].astype(str)
            
            # 导出到CSV而不是Excel，CSV更稳定且几乎所有表格软件都能打开
            csv_filename = os.path.splitext(filename)[0] + '.csv'
            export_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            print(f"原始K线数据已导出到CSV文件: {csv_filename}")
            
            # 尝试导出到Excel (XLSX格式)
            try:
                # 使用xlsxwriter引擎，可能比openpyxl更稳定
                writer = pd.ExcelWriter(filename, engine='xlsxwriter')
                export_df.to_excel(writer, sheet_name='K线数据', index=False)
                
                # 获取xlsxwriter工作簿和工作表对象
                workbook = writer.book
                worksheet = writer.sheets['K线数据']
                
                # 调整列宽
                for idx, col in enumerate(export_df.columns):
                    # 获取列的最大长度
                    max_len = max(export_df[col].astype(str).map(len).max(), len(col)) + 2
                    # xlsxwriter使用像素，需要转换
                    worksheet.set_column(idx, idx, max_len)
                
                # 确保文件被保存并关闭
                writer.close()
                print(f"原始K线数据也已导出到Excel文件: {filename}")
                return filename
            except Exception as e:
                print(f"导出到Excel失败: {e}")
                print(f"请使用CSV文件: {csv_filename}")
                return csv_filename
                
        except Exception as e:
            print(f"导出数据时出错: {e}")
            traceback.print_exc()
            return None

# 使用示例
if __name__ == "__main__":
    print("阿翔趋势交易系统 - 支持合约和代理访问")
    
    # 获取参数，默认使用已配置的代理
    print(f"默认使用代理: http://127.0.0.1:7897")
    proxy_url = 'http://127.0.0.1:7897'
    
    # 全局设置代理
    set_proxy(proxy_url)
    
    # 测试代理连接
    if not test_proxy(proxy_url):
        use_anyway = input("代理测试失败！是否仍要继续? (y/n): ").lower() or 'y'
        if use_anyway != 'y':
            print("程序退出！")
            sys.exit(0)
    
    # 默认选择合约市场
    is_futures = True
    market_name = "合约"
    print(f"使用{market_name}交易")
    
    # 默认使用BTC/USDT:USDT交易对（合约市场格式）
    symbol = "BTC/USDT:USDT" if is_futures else "BTC/USDT"
    print(f"交易对: {symbol}")
    
    # 默认K线周期为1h
    timeframe = "1h"
    print(f"K线周期: {timeframe}")
    
    # 设置初始资金和杠杆
    initial_capital_str = input("请输入初始资金(USDT)[默认10000]: ") or "10000"
    leverage_str = input("请输入杠杆倍数[默认3]: ") or "3"
    
    initial_capital = float(initial_capital_str)
    leverage = float(leverage_str)
    
    print(f"初始资金: {initial_capital} USDT")
    print(f"杠杆倍数: {leverage}倍")
    
    # 设置时间范围
    use_time_range = input("是否设置时间范围? (y/n)[默认n]: ").lower() == 'y'
    
    start_time = None
    end_time = None
    
    if use_time_range:
        # 获取当前时间和30天前的时间作为默认值
        now = datetime.now()
        thirty_days_ago = now - timedelta(days=30)
        
        default_start = thirty_days_ago.strftime("%Y-%m-%d %H:%M:%S")
        default_end = now.strftime("%Y-%m-%d %H:%M:%S")
        
        start_time_str = input(f"请输入开始时间(北京时间, 格式: YYYY-MM-DD HH:MM:SS)[默认{default_start}]: ") or default_start
        end_time_str = input(f"请输入结束时间(北京时间, 格式: YYYY-MM-DD HH:MM:SS)[默认{default_end}]: ") or default_end
        
        start_time = start_time_str
        end_time = end_time_str
        
        print(f"时间范围: {start_time} 至 {end_time}")
    else:
        print("获取最新数据...")
    
    # 创建交易系统实例
    trading_system = AxiangTrendSystem(
        symbol=symbol,
        timeframe=timeframe,
        dema144_len=144,
        dema169_len=169,
        ema120_len=120,
        ema200_len=200,
        atr_period=34,
        atr_multiplier=3.0,
        http_proxy=proxy_url,
        https_proxy=proxy_url,
        is_futures=is_futures,
        initial_capital=initial_capital,
        leverage=leverage
    )
    
    excel_file = None
    raw_data_file = None
    csv_file = None
    
    try:
        print(f"正在获取 {trading_system.symbol} {trading_system.timeframe} {market_name}数据...")
        
        # 尝试简单的API调用测试连接
        print("测试API连接...")
        try:
            trading_system.exchange.fetch_time()
            print("API连接成功!")
        except Exception as e:
            print(f"API连接测试失败: {e}")
            retry = input("连接测试失败，是否继续尝试获取数据? (y/n): ").lower() or 'y'
            if retry != 'y':
                print("程序退出！")
                sys.exit(0)
        
        # 获取并处理数据
        print(f"正在获取和处理数据，请稍候...")
        df = trading_system.run_strategy(start_time=start_time, end_time=end_time)
        print(f"数据处理完成，共获取{len(df)}条记录")
        
        # 导出原始K线数据
        raw_data_file = trading_system.export_raw_data_to_excel(df)
        if raw_data_file and raw_data_file.endswith('.csv'):
            csv_file = raw_data_file
            print(f"原始K线数据已保存至CSV: {csv_file}")
        else:
            print(f"原始K线数据已保存至: {raw_data_file}")
        
        # 分析交易信号
        trading_system.analyze_signals(df)
        
        # 导出分析结果
        file_prefix = f"{trading_system.symbol.replace('/', '_').replace(':', '_')}_{trading_system.timeframe}_{'futures' if is_futures else 'spot'}"
        excel_file = trading_system.export_to_excel(df, f"{file_prefix}_signals.xlsx")
        if excel_file and excel_file.endswith('.csv'):
            csv_file = excel_file
            print(f"\n分析完成! 结果已保存至CSV: {csv_file}")
        else:
            print(f"\n分析完成! 结果已保存至: {excel_file}")
        
        # 如果有CSV文件，提示用户
        if csv_file:
            print("\n提示: 如果Excel文件无法打开，请使用CSV文件，几乎所有表格软件都能打开CSV文件。")
        
        # 增加提示，确保文件已关闭并可以安全退出
        print("\n注意: 请确保已关闭所有导出的文件，然后按任意键退出...")
        input("按Enter键退出程序...")
        
    except Exception as e:
        print(f"发生错误: {e}")
        traceback.print_exc()
        print("\n请检查以下问题:")
        print("1. 代理服务器是否启动并正常工作")
        print("2. 代理地址和端口是否正确")
        print("3. 网络连接是否正常")
        print("4. 检查VPN是否正常连接")
        print("5. 合约交易对是否存在")
        print("6. 日期格式是否正确(YYYY-MM-DD HH:MM:SS)")
        
        # 增加提示，确保文件已关闭并可以安全退出
        print("\n按Enter键退出程序...")
        input()
    
    # 在程序结束前，显式地进行垃圾回收，帮助释放文件句柄
    import gc
    gc.collect()
    
    # 添加一个小延迟，确保所有文件操作都已完成
    import time
    time.sleep(1)
