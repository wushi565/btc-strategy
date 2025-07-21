import ccxt
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import time
import traceback
import os
import glob
import requests
from tqdm import tqdm

def set_proxy(proxy):
    """设置全局代理"""
    os.environ['http_proxy'] = proxy
    os.environ['https_proxy'] = proxy
    print(f"已设置全局代理: {proxy}")

def test_proxy(proxy=None):
    """测试代理连接"""
    import requests
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            print(f"代理测试尝试 {attempt}/{max_retries}...")
            if proxy:
                proxies = {'http': proxy, 'https': proxy}
                response = requests.get('https://api.binance.com/api/v3/time', proxies=proxies, timeout=15)
            else:
                response = requests.get('https://api.binance.com/api/v3/time', timeout=15)
            
            if response.status_code == 200:
                print(f"代理测试成功: {response.json()}")
                return True
            print(f"代理测试失败，状态码: {response.status_code}")
            if attempt < max_retries:
                wait_time = attempt * 2
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                return False
        except Exception as e:
            print(f"代理测试失败 (尝试 {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                wait_time = attempt * 2
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print("所有代理测试尝试均失败")
                return False

def format_datetime(dt, format_str="%Y-%m-%d %H:%M:%S"):
    """格式化日期时间"""
    if dt is None:
        return "None"
    if isinstance(dt, (int, float)):
        dt = datetime.fromtimestamp(dt / 1000)
    return dt.strftime(format_str)

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
        
        self.symbol = self.trading_config.get("symbol", "BTC/USDT:USDT")
        self.timeframe = self.trading_config.get("timeframe", "1h")
        self.is_futures = self.trading_config.get("is_futures", True)
        
        # 代理配置
        self.enable_proxy = self.network_config.get("enable_proxy", False)
        self.http_proxy = self.network_config.get("http_proxy", "")
        self.https_proxy = self.network_config.get("https_proxy", "")

        # 缓存目录
        self.data_config = config.get("data", {})
        self.cache_dir = self.data_config.get("cache_dir", "data")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 设置代理 (如果启用)
        if self.enable_proxy and self.http_proxy:
            print(f"代理功能已启用")
            set_proxy(self.http_proxy)
            print(f"正在测试代理连接...")
            if test_proxy(self.http_proxy):
                print("代理连接正常!")
            else:
                print("警告: 代理连接测试失败，可能会影响数据获取!")
        else:
            print(f"代理功能已关闭，使用系统网络环境")
        
        # 时区设置
        self.beijing_tz = pytz.timezone("Asia/Shanghai")
        
        # 延迟初始化交易所，只有在需要时才连接
        self.exchange = None
    
    def _init_exchange(self):
        """初始化交易所连接"""
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                # 创建交易所实例
                exchange_class = getattr(ccxt, "binance")
                
                # 设置代理
                options = {}
                if self.enable_proxy and self.http_proxy:
                    # 确保两个代理一致
                    print(f"设置交易所代理 HTTP: {self.http_proxy}, HTTPS: {self.https_proxy}")
                    options["proxies"] = {
                        "http": self.http_proxy,
                        "https": self.https_proxy
                    }
                
                # 设置额外选项
                options["verbose"] = False  # 不显示详细HTTP请求信息
                
                # 创建交易所实例
                if self.is_futures:
                    print(f"初始化币安合约交易所 (尝试 {attempt}/{max_retries})...")
                    options["defaultType"] = "future"
                    # 为币安期货API特别设置
                    extra_options = {
                        "options": options,
                        "enableRateLimit": True,
                        "timeout": 60000,  # 超时时间60秒
                        "headers": {
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                        }
                    }
                    
                    # 确保代理配置正确
                    if self.enable_proxy and self.http_proxy:
                        extra_options["proxies"] = {
                            "http": self.http_proxy,
                            "https": self.http_proxy  # 使用相同的代理地址
                        }
                    
                    self.exchange = exchange_class(extra_options)
                else:
                    print(f"初始化币安现货交易所 (尝试 {attempt}/{max_retries})...")
                    self.exchange = exchange_class({
                        "options": options,
                        "enableRateLimit": True,
                        "timeout": 60000  # 增加超时时间到60秒
                    })
                
                # 测试连接
                print("测试交易所连接...")
                time_data = self.exchange.fetch_time()
                server_time = datetime.fromtimestamp(time_data/1000, tz=pytz.UTC)
                print(f"交易所连接成功! 服务器时间: {server_time}")
                
                return self.exchange
                
            except Exception as e:
                print(f"交易所初始化失败 (尝试 {attempt}/{max_retries}): {e}")
                if "timeout" in str(e).lower():
                    print("连接超时，可能是网络问题或代理设置有误")
                elif "blocked" in str(e).lower() or "forbidden" in str(e).lower() or "restricted" in str(e).lower():
                    print("访问被阻止，可能需要使用代理或更换IP")
                
                if attempt < max_retries:
                    wait_time = attempt * 5
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print("所有交易所连接尝试均失败")
                    traceback.print_exc()
                    return None
    
    def list_local_data(self):
        """
        列出本地所有缓存的数据文件
        
        返回:
            list: 包含所有缓存文件信息的列表，每个元素是一个字典，包含文件路径、交易对、时间范围等信息
        """
        print("正在搜索本地数据文件...")
        cache_files = glob.glob(os.path.join(self.cache_dir, "*.csv"))
        
        if not cache_files:
            print("未找到本地数据文件")
            return []
        
        result = []
        for idx, file_path in enumerate(cache_files):
            try:
                file_name = os.path.basename(file_path)
                parts = file_name.replace('.csv', '').split('_')
                
                # 尝试解析文件名中的信息
                if len(parts) >= 5:
                    symbol_parts = parts[0:len(parts)-4]  # 交易对名称可能包含多个下划线
                    symbol = '_'.join(symbol_parts)
                    timeframe = parts[-4]
                    start_date = parts[-3]
                    end_date = parts[-1]
                    
                    # 尝试获取文件的行数
                    row_count = 0
                    try:
                        # 计算行数（减去标题行）
                        with open(file_path, 'r') as f:
                            row_count = sum(1 for _ in f) - 1
                    except Exception as e:
                        print(f"读取文件{file_path}内容时出错: {e}")
                    
                    # 添加到结果列表
                    result.append({
                        "file_path": file_path,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "start_date": start_date,
                        "end_date": end_date,
                        "row_count": row_count
                    })
            except Exception as e:
                print(f"解析文件{file_path}时出错: {e}")
        
        # 按照文件名排序
        result.sort(key=lambda x: x.get("file_path", ""))
        
        return result
    
    def _get_cache_file(self, start_dt, end_dt):
        """生成缓存文件路径"""
        start_str = start_dt.strftime("%Y%m%d") if start_dt else "start"
        end_str = end_dt.strftime("%Y%m%d") if end_dt else "latest"
        symbol = self.symbol.replace('/', '_').replace(':', '_')
        filename = f"{symbol}_{self.timeframe}_{start_str}_to_{end_str}.csv"
        return os.path.join(self.cache_dir, filename)
    
    def test_connection(self):
        """测试交易所连接"""
        try:
            print("正在测试交易所连接...")
            # 确保交易所已初始化
            if self.exchange is None:
                self._init_exchange()
            
            if self.exchange is not None:
                self.exchange.fetch_time()
                print("交易所连接成功!")
                return True
            else:
                print("交易所初始化失败")
                return False
        except Exception as e:
            print(f"交易所连接失败: {e}")
            traceback.print_exc()
            return False
    
    def check_local_data(self, start_time=None, end_time=None):
        """
        检查是否有本地缓存数据
        
        参数:
            start_time (str|datetime): 开始时间 (例如 '2023-01-01' 或 '2023-01-01 00:00:00')
            end_time (str|datetime): 结束时间 (例如 '2023-01-31' 或 '2023-01-31 23:59:59')
            
        返回:
            tuple: (bool, str) - (是否存在本地数据, 缓存文件路径)
        """
        # 时间转换
        start_dt = None
        end_dt = None
        
        if start_time:
            if isinstance(start_time, str):
                try:
                    # 尝试使用完整的日期时间格式解析
                    start_dt = self.beijing_tz.localize(datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S"))
                except ValueError:
                    # 如果失败，尝试只解析日期部分，并设置时间为00:00:00
                    start_dt = self.beijing_tz.localize(datetime.strptime(start_time, "%Y-%m-%d"))
            else:
                start_dt = start_time.astimezone(self.beijing_tz)
                
        if end_time:
            if isinstance(end_time, str):
                try:
                    # 尝试使用完整的日期时间格式解析
                    end_dt = self.beijing_tz.localize(datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S"))
                except ValueError:
                    # 如果失败，尝试只解析日期部分，并设置时间为23:59:59
                    end_dt = self.beijing_tz.localize(datetime.strptime(end_time, "%Y-%m-%d").replace(hour=23, minute=59, second=59))
            else:
                end_dt = end_time.astimezone(self.beijing_tz)

        # 检查本地缓存
        cache_file = self._get_cache_file(start_dt if start_time else None, end_dt if end_time else None)
        exists = os.path.isfile(cache_file)
        
        if exists:
            print(f"发现本地缓存数据: {cache_file}")
        else:
            print(f"未找到本地缓存数据，需要从交易所获取")
            
        return exists, cache_file
    
    def load_local_data(self, cache_file):
        """
        从本地缓存加载数据
        
        参数:
            cache_file (str): 缓存文件路径
            
        返回:
            pandas.DataFrame: 从缓存加载的数据
        """
        try:
            print(f"从缓存加载数据: {cache_file}")
            
            # 读取CSV文件，将timestamp列转为日期时间索引
            df = pd.read_csv(cache_file, index_col="timestamp", parse_dates=["timestamp"])
            
            # 处理时区 - 使用简单的尝试方法
            try:
                # 直接尝试转换为北京时间，如果已经有时区信息会自动处理
                df.index = pd.to_datetime(df.index).tz_convert(self.beijing_tz)
            except Exception:
                # 如果失败，可能是缺少时区信息，尝试先添加UTC时区再转换
                try:
                    df.index = pd.to_datetime(df.index).tz_localize(pytz.UTC).tz_convert(self.beijing_tz)
                except Exception:
                    # 如果还是失败，尝试先移除时区信息再重新添加
                    df.index = pd.to_datetime(df.index).tz_localize(None).tz_localize(pytz.UTC).tz_convert(self.beijing_tz)
                
            print(f"成功加载本地数据: {len(df)} 行 ({df.index.min()} 至 {df.index.max()})")
            return df
        except Exception as e:
            print(f"加载本地数据失败: {e}")
            return None
    
    def fetch_klines(self, start_time=None, end_time=None, use_local_data=None, cache_file=None):
        """
        获取K线数据
        
        参数:
            start_time (str|datetime): 开始时间 (例如 '2023-01-01' 或 '2023-01-01 00:00:00')
            end_time (str|datetime): 结束时间 (例如 '2023-01-31' 或 '2023-01-31 23:59:59')
            use_local_data (bool|None): 是否使用本地数据，None表示自动检测
            cache_file (str): 指定要加载的缓存文件路径，如果提供则直接加载此文件
            
        返回:
            pandas.DataFrame: K线数据
        """
        try:
            # 如果指定了缓存文件，直接加载
            if cache_file and os.path.exists(cache_file):
                print(f"直接加载指定的缓存文件: {cache_file}")
                return self.load_local_data(cache_file)
            
            market_type = "合约" if self.is_futures else "现货"
            print(f"正在获取{market_type}交易对 {self.symbol} 的 {self.timeframe} K线数据...")

            # 时间转换
            start_dt = None
            end_dt = None
            since = None
            end_ts = None
            
            if start_time:
                if isinstance(start_time, str):
                    try:
                        # 尝试使用完整的日期时间格式解析
                        start_dt = self.beijing_tz.localize(datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S"))
                    except ValueError:
                        # 如果失败，尝试只解析日期部分，并设置时间为00:00:00
                        start_dt = self.beijing_tz.localize(datetime.strptime(start_time, "%Y-%m-%d"))
                else:
                    start_dt = start_time.astimezone(self.beijing_tz)
                
                # 转换为UTC时间戳（毫秒）
                since = int(start_dt.timestamp() * 1000)
            
            if end_time:
                if isinstance(end_time, str):
                    try:
                        # 尝试使用完整的日期时间格式解析
                        end_dt = self.beijing_tz.localize(datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S"))
                    except ValueError:
                        # 如果失败，尝试只解析日期部分，并设置时间为23:59:59
                        end_dt = self.beijing_tz.localize(datetime.strptime(end_time, "%Y-%m-%d").replace(hour=23, minute=59, second=59))
                else:
                    end_dt = end_time.astimezone(self.beijing_tz)
                
                # 转换为UTC时间戳（毫秒）
                end_ts = int(end_dt.timestamp() * 1000)

            # 检查本地缓存
            has_local_data, cache_file = self.check_local_data(start_time, end_time)
            
            # 如果有本地数据且用户选择使用本地数据
            if has_local_data and (use_local_data is True):
                return self.load_local_data(cache_file)
            
            # 如果use_local_data是None且有本地数据，则询问用户
            if use_local_data is None and has_local_data:
                response = input("发现本地缓存数据，是否使用? (y/n): ").lower()
                if response in ['y', 'yes']:
                    return self.load_local_data(cache_file)
            
            # 如果没有本地数据或用户选择不使用本地数据，则从交易所获取
            print("将从交易所获取数据...")
            
            # 确保交易所已初始化
            if self.exchange is None:
                self._init_exchange()
            
            if self.exchange is None:
                print("交易所初始化失败，无法获取数据")
                return None
            
            # 验证交易对是否支持
            try:
                print("正在验证交易对...")
                markets = self.exchange.load_markets()
                if self.symbol not in markets:
                    print(f"警告: 交易对 {self.symbol} 不在支持列表中!")
                    available_symbols = list(markets.keys())[:10]  # 只显示前10个
                    print(f"支持的部分交易对: {available_symbols}...")
                else:
                    market_info = markets[self.symbol]
                    print(f"交易对 {self.symbol} 验证通过! {market_info.get('type', '未知')} 市场, 基础货币:{market_info.get('base')}, 计价货币:{market_info.get('quote')}")
            except Exception as e:
                print(f"验证交易对时出错: {e}")
                traceback.print_exc()
                
                # 尝试重新连接
                print("尝试重新初始化交易所连接...")
                self.exchange = None
                self._init_exchange()
                
                if self.exchange is None:
                    print("交易所重新初始化失败，无法获取数据")
                    return None
            
            # 获取数据
            all_ohlcv = []
            fetch_since = since
            batch_count = 0
            batch_limit = 1000  # 增加每批获取的K线数量为1000
            
            # 估算总批次数量（用于进度条）
            total_batches = None
            if start_dt and end_dt:
                # 根据时间间隔和时间框架估算总批次
                timeframe_minutes = {
                    '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
                    '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480,
                    '12h': 720, '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
                }
                
                minutes_diff = int((end_dt - start_dt).total_seconds() / 60)
                tf_minutes = timeframe_minutes.get(self.timeframe, 60)
                estimated_candles = minutes_diff / tf_minutes
                total_batches = max(1, int(estimated_candles / batch_limit) + 1)
                
                print(f"预计需要获取约 {estimated_candles:.0f} 根K线，分 {total_batches} 批获取")
            
            print("开始获取K线数据...")
            
            # 创建进度条
            with tqdm(total=total_batches, desc="获取数据进度", unit="批") as pbar:
                retry_count = 0
                max_retries = 5
                
                while True:
                    batch_count += 1
                    
                    try:
                        # 确保交易所实例有效
                        if self.exchange is None:
                            print("\n交易所实例无效，尝试重新初始化...")
                            self._init_exchange()
                            if self.exchange is None:
                                print("无法重新初始化交易所，中止数据获取")
                                break
                        
                        ohlcv = self.exchange.fetch_ohlcv(
                            self.symbol, 
                            self.timeframe, 
                            since=fetch_since, 
                            limit=batch_limit
                        )
                        
                        # 重置重试计数
                        retry_count = 0
                        
                        if not ohlcv:
                            print("没有更多数据")
                            break
                            
                        # 显示首条和末条数据时间
                        first_time = datetime.fromtimestamp(ohlcv[0][0]/1000, tz=pytz.UTC).astimezone(self.beijing_tz)
                        last_time = datetime.fromtimestamp(ohlcv[-1][0]/1000, tz=pytz.UTC).astimezone(self.beijing_tz)
                        
                        pbar.set_postfix({"开始": first_time.strftime('%m-%d %H:%M'), 
                                         "结束": last_time.strftime('%m-%d %H:%M'),
                                         "数量": len(ohlcv)})
                        pbar.update(1)
                        
                        # 过滤超出end_time的数据
                        if end_ts:
                            ohlcv = [x for x in ohlcv if x[0] <= end_ts]
                        
                        all_ohlcv.extend(ohlcv)
                        
                        if len(ohlcv) < batch_limit:
                            print("\n返回数据少于请求限制，已获取全部数据")
                            break
                            
                        last_ts = ohlcv[-1][0]
                        # 防止死循环
                        if fetch_since is not None and last_ts <= fetch_since:
                            print("\n数据时间戳未前进，停止获取")
                            break
                            
                        # 设置下一批的起始时间
                        fetch_since = last_ts + 1
                        
                        # 如果已经超过结束时间，退出循环
                        if end_ts and fetch_since >= end_ts:
                            print("\n已达到结束时间，停止获取")
                            break
                            
                        # 在大量数据请求中添加小延迟，以避免过度请求
                        time.sleep(0.5)
                    
                    except Exception as e:
                        retry_count += 1
                        print(f"\n获取K线数据出错 (重试 {retry_count}/{max_retries}): {e}")
                        
                        if retry_count >= max_retries:
                            print(f"已达到最大重试次数 ({max_retries})，跳过当前批次")
                            # 跳过当前批次，尝试获取下一批
                            if fetch_since is not None:
                                # 根据timeframe计算跳过的时间
                                timeframe_seconds = {
                                    '1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800,
                                    '1h': 3600, '2h': 7200, '4h': 14400, '6h': 21600, '8h': 28800,
                                    '12h': 43200, '1d': 86400, '3d': 259200, '1w': 604800, '1M': 2592000
                                }
                                
                                tf_seconds = timeframe_seconds.get(self.timeframe, 3600)
                                fetch_since += batch_limit * tf_seconds * 1000  # 跳过一批数据的时间
                                print(f"尝试跳过错误区域，继续获取下一批数据...")
                                retry_count = 0
                            else:
                                break
                        else:
                            # 添加递增的等待时间
                            wait_time = 2 * retry_count
                            print(f"等待 {wait_time} 秒后重试...")
                            time.sleep(wait_time)
                            
                            # 如果是网络或连接问题，尝试重新初始化交易所
                            if retry_count == 3:
                                print("尝试重新初始化交易所连接...")
                                self.exchange = None
                                if self._init_exchange() is None:
                                    print("交易所重新初始化失败")
                                    break
            
            print(f"共获取到 {len(all_ohlcv)} 条K线数据")
            
            if not all_ohlcv:
                print("未获取到数据，请检查参数和网络")
                return None
                
            # 转换为DataFrame
            columns_list = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df = pd.DataFrame(all_ohlcv, columns=columns_list)
            
            # 将timestamp转换为日期时间类型，并设置为索引
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(self.beijing_tz)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # 打印数据范围
            print(f"数据范围: {df.index.min()} 至 {df.index.max()}, 共{len(df)}条")
            
            # 保存到本地缓存
            if cache_file:
                print(f"保存数据到本地缓存: {cache_file}")
                df.to_csv(cache_file)
                print("保存成功!")
            
            return df
            
        except Exception as e:
            print(f"获取K线数据失败: {e}")
            traceback.print_exc()
            return None 