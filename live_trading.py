import ccxt
import pandas as pd
import numpy as np
import yaml
import time
import os
from datetime import datetime, timedelta
import threading
import logging
from tabulate import tabulate
import traceback

# 导入项目模块
from data import DataManager
from indicators import IndicatorCalculator
from signals import SignalGenerator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('live_trading')

class LiveTrader:
    def __init__(self, config):
        self.config = config
        self.trading_config = config.get('trading', {})
        self.backtest_config = config.get('backtest', {})
        self.symbol = self.trading_config.get('symbol', 'BTC/USDT')
        self.timeframe = self.trading_config.get('timeframe', '1h')
        self.is_futures = self.trading_config.get('is_futures', True)
        self.risk_per_trade = self.backtest_config.get('risk_per_trade', 0.02)
        self.leverage = self.backtest_config.get('leverage', 1.0)
        
        # 交易状态
        self.current_position = None  # 'long', 'short', None
        self.position_size = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.target_price = 0
        self.unrealized_pnl = 0
        
        # 交易所连接
        self.exchange = None
        self.api_config = self._load_api_config()
        
        # 数据管理
        self.data_manager = DataManager(config)
        self.indicator_calculator = IndicatorCalculator(config)
        self.signal_generator = SignalGenerator(config)
        
        # 运行状态
        self.is_running = False
        self.last_check_time = None
        self.last_data_update = None
        
        # 交易统计
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0

    def _load_api_config(self):
        """加载API配置文件"""
        try:
            if os.path.exists('api_config.yaml'):
                with open('api_config.yaml', 'r', encoding='utf-8') as file:
                    api_config = yaml.safe_load(file)
                    logger.info('成功加载API配置文件')
                    return api_config
            else:
                logger.error('找不到API配置文件 api_config.yaml')
                return None
        except Exception as e:
            logger.error(f'加载API配置失败: {e}')
            return None

    def connect_exchange(self):
        """连接交易所"""
        if not self.api_config:
            logger.error('缺少API配置，无法连接交易所')
            return False
        
        try:
            exchange_options = {
                'apiKey': self.api_config.get('api_key'),
                'secret': self.api_config.get('api_secret'),
                'enableRateLimit': True,
                'timeout': 30000,
            }
            
            if self.is_futures:
                exchange_options['options'] = {'defaultType': 'future'}
            
            self.exchange = ccxt.binance(exchange_options)
            
            # 测试连接
            balance = self.exchange.fetch_balance()
            logger.info(f'成功连接到交易所API，账户余额: {balance.get("total", {})}')
            
            # 设置杠杆（如果是期货）
            if self.is_futures and self.leverage != 1.0:
                try:
                    self.exchange.set_leverage(self.leverage, self.symbol)
                    logger.info(f'设置杠杆为: {self.leverage}')
                except Exception as e:
                    logger.warning(f'设置杠杆失败: {e}')
            
            return True
        except Exception as e:
            logger.error(f'连接交易所失败: {e}')
            return False

    def get_current_price(self):
        """获取当前价格"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f'获取当前价格失败: {e}')
            return None

    def get_market_data(self, limit=100):
        """获取市场数据"""
        try:
            # 获取K线数据
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 计算指标
            df = self.indicator_calculator.calculate_all_indicators(df)
            
            # 生成信号
            df = self.signal_generator.generate_signals(df)
            
            return df
        except Exception as e:
            logger.error(f'获取市场数据失败: {e}')
            return None

    def calculate_position_size(self, entry_price, stop_loss):
        """计算仓位大小"""
        try:
            balance = self.exchange.fetch_balance()
            account_value = balance['total']['USDT'] if 'USDT' in balance['total'] else 10000
            
            # 计算风险金额
            risk_amount = account_value * self.risk_per_trade
            
            # 计算每点风险
            if self.current_position == 'long':
                risk_per_unit = entry_price - stop_loss
            else:  # short
                risk_per_unit = stop_loss - entry_price
            
            if risk_per_unit <= 0:
                logger.warning('止损价格设置错误')
                return 0
            
            # 计算仓位大小
            position_size = risk_amount / risk_per_unit
            
            # 如果是期货，需要考虑杠杆
            if self.is_futures:
                position_size = position_size * self.leverage
            
            return position_size
        except Exception as e:
            logger.error(f'计算仓位大小失败: {e}')
            return 0

    def place_order(self, side, amount, price=None, order_type='market'):
        """下单"""
        try:
            order_params = {}
            if self.is_futures:
                order_params['type'] = 'market'
            
            order = self.exchange.create_order(
                symbol=self.symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=order_params
            )
            
            logger.info(f'下单成功: {side} {amount} {self.symbol} @ {price or "市价"}')
            return order
        except Exception as e:
            logger.error(f'下单失败: {e}')
            return None

    def close_position(self):
        """平仓"""
        if not self.current_position or self.position_size == 0:
            return True
        
        try:
            side = 'sell' if self.current_position == 'long' else 'buy'
            order = self.place_order(side, self.position_size)
            
            if order:
                logger.info(f'平仓成功: {self.current_position} 仓位')
                self.current_position = None
                self.position_size = 0
                self.entry_price = 0
                self.stop_loss = 0
                self.target_price = 0
                return True
            return False
        except Exception as e:
            logger.error(f'平仓失败: {e}')
            return False

    def check_stop_loss_and_target(self, current_price):
        """检查止损和目标价格"""
        if not self.current_position or self.position_size == 0:
            return
        
        try:
            # 检查止损
            if self.current_position == 'long' and current_price <= self.stop_loss:
                logger.info(f'触发止损: 当前价格 {current_price} <= 止损价格 {self.stop_loss}')
                self.close_position()
                self.losing_trades += 1
            elif self.current_position == 'short' and current_price >= self.stop_loss:
                logger.info(f'触发止损: 当前价格 {current_price} >= 止损价格 {self.stop_loss}')
                self.close_position()
                self.losing_trades += 1
            
            # 检查目标价格
            elif self.current_position == 'long' and current_price >= self.target_price:
                logger.info(f'达到目标价格: 当前价格 {current_price} >= 目标价格 {self.target_price}')
                self.close_position()
                self.winning_trades += 1
            elif self.current_position == 'short' and current_price <= self.target_price:
                logger.info(f'达到目标价格: 当前价格 {current_price} <= 目标价格 {self.target_price}')
                self.close_position()
                self.winning_trades += 1
        except Exception as e:
            logger.error(f'检查止损和目标价格失败: {e}')

    def execute_trading_logic(self):
        """执行交易逻辑"""
        try:
            # 获取市场数据
            df = self.get_market_data()
            if df is None or df.empty:
                logger.warning('无法获取市场数据')
                return
            
            current_price = df['close'].iloc[-1]
            current_time = df.index[-1]
            
            # 检查止损和目标价格
            self.check_stop_loss_and_target(current_price)
            
            # 如果已有仓位，跳过开仓逻辑
            if self.current_position:
                return
            
            # 获取最新信号
            latest_signal = df.iloc[-1]
            
            # 开仓逻辑
            if latest_signal['buy_signal']:
                # 开多仓
                stop_loss = latest_signal['stop_loss_buy']
                target_price = latest_signal['target_price_buy']
                
                position_size = self.calculate_position_size(current_price, stop_loss)
                
                if position_size > 0:
                    order = self.place_order('buy', position_size)
                    if order:
                        self.current_position = 'long'
                        self.position_size = position_size
                        self.entry_price = current_price
                        self.stop_loss = stop_loss
                        self.target_price = target_price
                        self.total_trades += 1
                        logger.info(f'开多仓成功: 价格={current_price}, 止损={stop_loss}, 目标={target_price}')
            
            elif latest_signal['sell_signal']:
                # 开空仓
                stop_loss = latest_signal['stop_loss_sell']
                target_price = latest_signal['target_price_sell']
                
                position_size = self.calculate_position_size(current_price, stop_loss)
                
                if position_size > 0:
                    order = self.place_order('sell', position_size)
                    if order:
                        self.current_position = 'short'
                        self.position_size = position_size
                        self.entry_price = current_price
                        self.stop_loss = stop_loss
                        self.target_price = target_price
                        self.total_trades += 1
                        logger.info(f'开空仓成功: 价格={current_price}, 止损={stop_loss}, 目标={target_price}')
        
        except Exception as e:
            logger.error(f'执行交易逻辑失败: {e}')
            traceback.print_exc()

    def print_status(self):
        """打印交易状态"""
        try:
            current_price = self.get_current_price()
            if current_price is None:
                return
            
            # 计算未实现盈亏
            if self.current_position and self.position_size > 0:
                if self.current_position == 'long':
                    self.unrealized_pnl = (current_price - self.entry_price) * self.position_size
                else:  # short
                    self.unrealized_pnl = (self.entry_price - current_price) * self.position_size
            else:
                self.unrealized_pnl = 0
            
            # 计算胜率
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            
            status_data = [
                ['当前时间', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['交易对', self.symbol],
                ['当前价格', f'{current_price:.2f}'],
                ['当前仓位', self.current_position or '无'],
                ['仓位大小', f'{self.position_size:.4f}'],
                ['入场价格', f'{self.entry_price:.2f}'],
                ['止损价格', f'{self.stop_loss:.2f}'],
                ['目标价格', f'{self.target_price:.2f}'],
                ['未实现盈亏', f'{self.unrealized_pnl:.2f}'],
                ['总交易次数', self.total_trades],
                ['胜率', f'{win_rate:.1f}%'],
                ['总盈亏', f'{self.total_pnl:.2f}']
            ]
            
            print('\n' + '='*50)
            print('实时交易状态')
            print('='*50)
            print(tabulate(status_data, headers=['项目', '数值'], tablefmt='grid'))
            print('='*50)
        
        except Exception as e:
            logger.error(f'打印状态失败: {e}')

    def start_trading(self):
        """开始实时交易"""
        if self.is_running:
            logger.warning('交易已经在运行中')
            return
        
        if not self.connect_exchange():
            logger.error('无法连接交易所，交易启动失败')
            return
        
        self.is_running = True
        logger.info('开始实时交易')
        
        try:
            while self.is_running:
                try:
                    # 执行交易逻辑
                    self.execute_trading_logic()
                    
                    # 打印状态
                    self.print_status()
                    
                    # 等待下一个周期
                    time.sleep(60)  # 每分钟检查一次
                    
                except KeyboardInterrupt:
                    logger.info('用户中断，停止交易')
                    break
                except Exception as e:
                    logger.error(f'交易循环出错: {e}')
                    time.sleep(30)  # 出错后等待30秒再继续
        
        finally:
            self.is_running = False
            # 关闭所有仓位
            if self.current_position:
                logger.info('关闭所有仓位')
                self.close_position()
            logger.info('交易已停止')

def start_live_trading(config):
    """启动实时交易"""
    trader = LiveTrader(config)
    trader.start_trading()
    return trader

if __name__ == '__main__':
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print('启动实时交易...')
        trader = start_live_trading(config)
    except KeyboardInterrupt:
        print('用户中断，停止交易')
    except Exception as e:
        print(f'程序运行失败: {e}')
        traceback.print_exc() 