import pandas as pd
import numpy as np
from datetime import datetime
from abc import ABC, abstractmethod

__all__ = ['Strategy', 'SupertrendDEMAStrategy', 'SupertrendDEMAAdxStrategy', 'StrategyFactory', 'SignalGenerator', 'analyze_signals']

class Strategy(ABC):
    """策略抽象基类"""
    
    def __init__(self, config):
        """
        初始化策略
        
        参数:
            config (dict): 配置信息
        """
        self.config = config
        self.signals_config = config.get("signals", {})
        # 获取风险回报比
        self.risk_reward_ratio = self.signals_config.get("risk_reward_ratio", 3.0)
        
    @abstractmethod
    def generate_signals(self, df):
        """生成交易信号"""
        pass
    
    def calculate_risk_reward(self, signals_df):
        """
        计算风险回报比
        
        参数:
            signals_df (pd.DataFrame): 包含信号和止损的数据
            
        返回:
            pd.DataFrame: 添加了目标价格的数据
        """
            # 创建副本避免修改原始数据
        signals_df = signals_df.copy()
            
        # 为多头信号计算目标价格
        # 目标价格 = 入场价格 + (入场价格 - 止损价格) * 风险回报比
        buy_signal_mask = signals_df["buy_signal"]
        if buy_signal_mask.any():
            entry_price = signals_df.loc[buy_signal_mask, "close"]
            stop_loss = signals_df.loc[buy_signal_mask, "stop_loss_buy"]
            risk_distance = entry_price - stop_loss
            target_price = entry_price + risk_distance * self.risk_reward_ratio
            signals_df.loc[buy_signal_mask, "target_price_buy"] = target_price
        
        # 为空头信号计算目标价格
        # 目标价格 = 入场价格 - (止损价格 - 入场价格) * 风险回报比
        sell_signal_mask = signals_df["sell_signal"]
        if sell_signal_mask.any():
            entry_price = signals_df.loc[sell_signal_mask, "close"]
            stop_loss = signals_df.loc[sell_signal_mask, "stop_loss_sell"]
            risk_distance = stop_loss - entry_price
            target_price = entry_price - risk_distance * self.risk_reward_ratio
            signals_df.loc[sell_signal_mask, "target_price_sell"] = target_price
        
        return signals_df

class SupertrendDEMAStrategy(Strategy):
    """基于Supertrend和DEMA的策略"""
    
    def generate_signals(self, df):
        """
        生成交易信号：
        1. Supertrend出现买入信号
        2. 价格突破DEMA144和DEMA169做多
        3. 止损在DEMA169线上
        4. 做空信号则相反，止损在DEMA144线上
        
        参数:
            df (pd.DataFrame): 包含指标数据的DataFrame
            
        返回:
            pd.DataFrame: 添加了信号的DataFrame
        """
            # 创建副本避免修改原始数据
        signals_df = df.copy()
            
        # 初始化信号列
        signals_df["buy_signal"] = False
        signals_df["sell_signal"] = False
        signals_df["position"] = 0  # 0表示无持仓，1表示多头，-1表示空头
        signals_df["trade_action"] = None  # 记录交易动作
        
        # 获取当前状态
        current_position = None
        
        # 遍历数据生成信号
        for i in range(1, len(signals_df)):
            # 读取当前行和前一行的数据
            prev = signals_df.iloc[i-1]
            curr = signals_df.iloc[i]
            
            # 买入信号: 
            # 1. Supertrend出现买入信号
            # 2. 价格突破DEMA144和DEMA169
            buy_condition = (
                # Supertrend买入信号
                curr["supertrend_buy"] and
                # 价格突破DEMA144和DEMA169
                (curr["close"] > curr["dema144"]) and (curr["close"] > curr["dema169"])
            )
            
            # 卖出信号:
            # 1. Supertrend出现卖出信号
            # 2. 价格跌破DEMA144和DEMA169
            sell_condition = (
                # Supertrend卖出信号
                curr["supertrend_sell"] and
                # 价格跌破DEMA144和DEMA169
                (curr["close"] < curr["dema144"]) and (curr["close"] < curr["dema169"])
            )
            
            # 设置信号，考虑当前持仓状态
            # 只有当没有持仓时才生成新的开仓信号
            if buy_condition and current_position is None:
                signals_df.loc[signals_df.index[i], "buy_signal"] = True
                signals_df.loc[signals_df.index[i], "position"] = 1
                signals_df.loc[signals_df.index[i], "trade_action"] = "开多"
                current_position = "long"
            elif sell_condition and current_position is None:
                signals_df.loc[signals_df.index[i], "sell_signal"] = True
                signals_df.loc[signals_df.index[i], "position"] = -1
                signals_df.loc[signals_df.index[i], "trade_action"] = "开空"
                current_position = "short"
            
            # 止损逻辑：如果持有多头仓位且价格跌破DEMA169
            elif current_position == "long" and curr["close"] < curr["dema169"]:
                signals_df.loc[signals_df.index[i], "sell_signal"] = True
                signals_df.loc[signals_df.index[i], "position"] = 0
                signals_df.loc[signals_df.index[i], "trade_action"] = "平多止损"
                current_position = None  # 平仓
            # 止损逻辑：如果持有空头仓位且价格突破DEMA144
            elif current_position == "short" and curr["close"] > curr["dema144"]:
                signals_df.loc[signals_df.index[i], "buy_signal"] = True
                signals_df.loc[signals_df.index[i], "position"] = 0
                signals_df.loc[signals_df.index[i], "trade_action"] = "平空止损"
                current_position = None  # 平仓
            else:
                # 延续前一个状态
                signals_df.loc[signals_df.index[i], "position"] = signals_df.loc[signals_df.index[i-1], "position"]
        
        # 计算多头止损位 (设置在DEMA169上)
        signals_df["stop_loss_buy"] = signals_df["dema169"]
        
        # 计算空头止损位 (设置在DEMA144上)
        signals_df["stop_loss_sell"] = signals_df["dema144"]
                
        return signals_df

class SupertrendDEMAAdxStrategy(Strategy):
    """基于Supertrend、DEMA和ADX的策略"""
    
    def generate_signals(self, df):
        """
        生成交易信号：
        1. Supertrend出现买入信号
        2. 价格突破DEMA144和DEMA169做多
        3. ADX大于阈值(默认20)，表示趋势强度足够
        4. 止损在DEMA169线上
        5. 做空信号则相反，止损在DEMA144线上
        
        参数:
            df (pd.DataFrame): 包含指标数据的DataFrame
            
        返回:
            pd.DataFrame: 添加了信号的DataFrame
        """
        # 创建副本避免修改原始数据
        signals_df = df.copy()
            
        # 初始化信号列
        signals_df["buy_signal"] = False
        signals_df["sell_signal"] = False
        signals_df["position"] = 0  # 0表示无持仓，1表示多头，-1表示空头
        signals_df["trade_action"] = None  # 记录交易动作
        
        # ADX阈值
        adx_threshold = self.signals_config.get("adx_threshold", 20)
        
        # 获取当前状态
        current_position = None
        
        # 遍历数据生成信号
        for i in range(1, len(signals_df)):
            # 读取当前行和前一行的数据
            prev = signals_df.iloc[i-1]
            curr = signals_df.iloc[i]
            
            # 检查ADX是否大于阈值(表示趋势强度足够)
            adx_strong_trend = curr["adx"] > adx_threshold
            
            # 买入信号: 
            # 1. Supertrend出现买入信号
            # 2. 价格突破DEMA144和DEMA169
            # 3. ADX大于阈值，表示趋势强度足够
            buy_condition = (
                # Supertrend买入信号
                curr["supertrend_buy"] and
                # 价格突破DEMA144和DEMA169
                (curr["close"] > curr["dema144"]) and (curr["close"] > curr["dema169"]) and
                # ADX大于阈值，表示趋势强度足够
                adx_strong_trend
            )
            
            # 卖出信号:
            # 1. Supertrend出现卖出信号
            # 2. 价格跌破DEMA144和DEMA169
            # 3. ADX大于阈值，表示趋势强度足够
            sell_condition = (
                # Supertrend卖出信号
                curr["supertrend_sell"] and
                # 价格跌破DEMA144和DEMA169
                (curr["close"] < curr["dema144"]) and (curr["close"] < curr["dema169"]) and
                # ADX大于阈值，表示趋势强度足够
                adx_strong_trend
            )
            
            # 设置信号，考虑当前持仓状态
            # 只有当没有持仓时才生成新的开仓信号
            if buy_condition and current_position is None:
                signals_df.loc[signals_df.index[i], "buy_signal"] = True
                signals_df.loc[signals_df.index[i], "position"] = 1
                signals_df.loc[signals_df.index[i], "trade_action"] = "开多"
                current_position = "long"
            elif sell_condition and current_position is None:
                signals_df.loc[signals_df.index[i], "sell_signal"] = True
                signals_df.loc[signals_df.index[i], "position"] = -1
                signals_df.loc[signals_df.index[i], "trade_action"] = "开空"
                current_position = "short"
            
            # 止损逻辑：如果持有多头仓位且价格跌破DEMA169
            elif current_position == "long" and curr["close"] < curr["dema169"]:
                signals_df.loc[signals_df.index[i], "sell_signal"] = True
                signals_df.loc[signals_df.index[i], "position"] = 0
                signals_df.loc[signals_df.index[i], "trade_action"] = "平多止损"
                current_position = None  # 平仓
            # 止损逻辑：如果持有空头仓位且价格突破DEMA144
            elif current_position == "short" and curr["close"] > curr["dema144"]:
                signals_df.loc[signals_df.index[i], "buy_signal"] = True
                signals_df.loc[signals_df.index[i], "position"] = 0
                signals_df.loc[signals_df.index[i], "trade_action"] = "平空止损"
                current_position = None  # 平仓
            else:
                # 延续前一个状态
                signals_df.loc[signals_df.index[i], "position"] = signals_df.loc[signals_df.index[i-1], "position"]
        
        # 计算多头止损位 (设置在DEMA169上)
        signals_df["stop_loss_buy"] = signals_df["dema169"]
        
        # 计算空头止损位 (设置在DEMA144上)
        signals_df["stop_loss_sell"] = signals_df["dema144"]
                
        return signals_df


class StrategyFactory:
    """策略工厂，用于创建和管理策略"""
    
    @staticmethod
    def get_strategy_list():
        """
        获取可用策略列表
        
        返回:
            list: 策略列表
        """
        return ["Supertrend和DEMA策略", "SupertrendDEMAAdx策略"]
    
    @staticmethod
    def create_strategy(strategy_name, config):
        """
        创建策略
        
        参数:
            strategy_name (str): 策略名称
            config (dict): 配置信息
            
        返回:
            Strategy: 策略实例
        """
        if strategy_name == "Supertrend和DEMA策略":
            return SupertrendDEMAStrategy(config)
        elif strategy_name == "SupertrendDEMAAdx策略":
            return SupertrendDEMAAdxStrategy(config)
        else:
            # 默认使用Supertrend和DEMA策略
            print(f"未知策略 '{strategy_name}'，使用默认Supertrend和DEMA策略")
            return SupertrendDEMAStrategy(config)

class SignalGenerator:
    """信号生成器，负责根据指标生成交易信号"""
    
    def __init__(self, config):
        """
        初始化信号生成器
        
        参数:
            config (dict): 配置信息
        """
        self.config = config
        self.signals_config = config.get("signals", {})
        
        # 创建策略
        strategy_name = self.signals_config.get("strategy", "Supertrend和DEMA策略")
        print(f"SignalGenerator初始化使用策略: {strategy_name}")
        self.strategy = StrategyFactory.create_strategy(strategy_name, config)
    
    def generate_signals(self, df):
        """
        生成交易信号
        
        参数:
            df (pd.DataFrame): 包含指标数据的DataFrame
            
        返回:
            pd.DataFrame: 添加了信号的DataFrame
        """
        return self.strategy.generate_signals(df)
    
    def calculate_risk_reward(self, signals_df):
        """
        计算风险回报比
        
        参数:
            signals_df (pd.DataFrame): 包含信号和止损的数据
            
        返回:
            pd.DataFrame: 添加了目标价格的数据
        """
        return self.strategy.calculate_risk_reward(signals_df)

def analyze_signals(signals_df):
    """
    分析交易信号
    
    参数:
        signals_df (pd.DataFrame): 包含信号的数据
        
    返回:
        dict: 分析结果
    """
    # 买入信号数量
    buy_signals = signals_df["buy_signal"].sum()
    
    # 卖出信号数量
    sell_signals = signals_df["sell_signal"].sum()
    
    # 总信号数量
    total_signals = buy_signals + sell_signals
    
    # 信号分布
    signal_distribution = {
        "买入信号": int(buy_signals),
        "卖出信号": int(sell_signals),
        "总信号数": int(total_signals)
    }
    
    return signal_distribution
