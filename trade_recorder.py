import pandas as pd
import numpy as np
import os
from datetime import datetime
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

class TradeRecorder:
    """交易记录器，负责记录交易并导出到Excel"""
    
    def __init__(self, config):
        """
        初始化交易记录器
        
        参数:
            config (dict): 配置信息
        """
        self.config = config
        self.backtest_config = config.get("backtest", {})
        self.initial_capital = self.backtest_config.get("initial_capital", 10000)
        self.leverage = self.backtest_config.get("leverage", 1.0)
        self.risk_per_trade = self.backtest_config.get("risk_per_trade", 0.02)
        
        # 交易记录
        self.trades = []
        self.current_capital = self.initial_capital
        self.equity = self.initial_capital
        self.max_equity = self.initial_capital
        self.max_drawdown = 0
        
        # 当前持仓信息
        self.current_position = None  # 可能值: None, "long", "short"
        self.position_size = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.target_price = 0
        self.entry_time = None
        
        # 创建输出目录
        self.output_dir = "trades"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def process_signals(self, signals_df):
        """
        处理信号并执行模拟交易
        
        参数:
            signals_df (pd.DataFrame): 包含信号的数据
            
        返回:
            pd.DataFrame: 添加了交易结果的数据
        """
        # 复制数据，避免修改原始数据
        df = signals_df.copy()
        
        # 添加结果列
        df["capital"] = self.initial_capital  # 资金
        df["position_size"] = 0.0  # 持仓大小
        df["leverage"] = self.leverage  # 杠杆
        df["trade_pnl"] = 0.0  # 单笔交易盈亏
        df["cumulative_pnl"] = 0.0  # 累计盈亏
        df["equity"] = self.initial_capital  # 净值
        df["drawdown"] = 0.0  # 回撤
        df["trade_action"] = None  # 交易动作
        
        # 遍历数据进行模拟交易
        for i in range(1, len(df)):
            # 默认继承前一行的状态
            df.loc[df.index[i], "capital"] = df.loc[df.index[i-1], "capital"]
            df.loc[df.index[i], "position_size"] = df.loc[df.index[i-1], "position_size"]
            df.loc[df.index[i], "leverage"] = self.leverage
            df.loc[df.index[i], "cumulative_pnl"] = df.loc[df.index[i-1], "cumulative_pnl"]
            df.loc[df.index[i], "equity"] = df.loc[df.index[i-1], "equity"]
            
            # 当前行和前一行
            prev_row = df.iloc[i-1]
            curr_row = df.iloc[i]
            
            # 记录当前持仓状态和位置大小 - 在执行任何操作前保存，用于平仓后可能的开仓
            original_position = self.current_position
            original_position_size = self.position_size
            closed_position = False  # 标记是否已经平仓
            
            # 交易处理逻辑：先处理止盈止损，再处理信号平仓，最后处理开仓
            
            # 步骤1：检查是否触发止盈止损 - 需要先于信号处理
            if self.current_position == "long":
                # 检查止盈
                if curr_row["high"] >= self.target_price:
                    # 计算平仓盈亏 (使用目标价格平仓)
                    exit_price = self.target_price
                    pnl = (exit_price - self.entry_price) * self.position_size * self.leverage
                    
                    # 更新资金和净值
                    self.current_capital += pnl
                    self.equity = self.current_capital
                    
                    # 更新最大回撤
                    if self.equity > self.max_equity:
                        self.max_equity = self.equity
                    drawdown = (self.max_equity - self.equity) / self.max_equity
                    self.max_drawdown = max(self.max_drawdown, drawdown)
                    
                    # 记录交易
                    trade = {
                        "交易类型": "平多(止盈)",
                        "开仓时间": self.entry_time,
                        "平仓时间": curr_row.name,
                        "开仓价格": self.entry_price,
                        "平仓价格": exit_price,
                        "持仓大小": self.position_size,
                        "杠杆": self.leverage,
                        "盈亏": pnl,
                        "盈亏率": pnl / (self.current_capital - pnl) if self.current_capital - pnl > 0 else float('inf'),
                        "止盈价": self.target_price,
                        "止损价": self.stop_loss
                    }
                    self.trades.append(trade)
                    
                    # 更新数据框
                    df.loc[df.index[i], "trade_pnl"] = pnl
                    df.loc[df.index[i], "cumulative_pnl"] = prev_row["cumulative_pnl"] + pnl
                    df.loc[df.index[i], "capital"] = self.current_capital
                    df.loc[df.index[i], "equity"] = self.equity
                    df.loc[df.index[i], "drawdown"] = drawdown
                    df.loc[df.index[i], "position_size"] = 0
                    df.loc[df.index[i], "trade_action"] = "止盈平多"
                    
                    # 重置当前持仓
                    self.current_position = None
                    self.position_size = 0
                    self.entry_price = 0
                    self.stop_loss = 0
                    self.target_price = 0
                    self.entry_time = None
                    closed_position = True
                    
                # 检查止损
                elif curr_row["low"] <= self.stop_loss:
                    # 计算平仓盈亏 (使用止损价格平仓)
                    exit_price = self.stop_loss
                    pnl = (exit_price - self.entry_price) * self.position_size * self.leverage
                    
                    # 更新资金和净值
                    self.current_capital += pnl
                    self.equity = self.current_capital
                    
                    # 更新最大回撤
                    if self.equity > self.max_equity:
                        self.max_equity = self.equity
                    drawdown = (self.max_equity - self.equity) / self.max_equity
                    self.max_drawdown = max(self.max_drawdown, drawdown)
                    
                    # 记录交易
                    trade = {
                        "交易类型": "平多(止损)",
                        "开仓时间": self.entry_time,
                        "平仓时间": curr_row.name,
                        "开仓价格": self.entry_price,
                        "平仓价格": exit_price,
                        "持仓大小": self.position_size,
                        "杠杆": self.leverage,
                        "盈亏": pnl,
                        "盈亏率": pnl / (self.current_capital - pnl) if self.current_capital - pnl > 0 else float('inf'),
                        "止盈价": self.target_price,
                        "止损价": self.stop_loss
                    }
                    self.trades.append(trade)
                    
                    # 更新数据框
                    df.loc[df.index[i], "trade_pnl"] = pnl
                    df.loc[df.index[i], "cumulative_pnl"] = prev_row["cumulative_pnl"] + pnl
                    df.loc[df.index[i], "capital"] = self.current_capital
                    df.loc[df.index[i], "equity"] = self.equity
                    df.loc[df.index[i], "drawdown"] = drawdown
                    df.loc[df.index[i], "position_size"] = 0
                    df.loc[df.index[i], "trade_action"] = "止损平多"
                    
                    # 重置当前持仓
                    self.current_position = None
                    self.position_size = 0
                    self.entry_price = 0
                    self.stop_loss = 0
                    self.target_price = 0
                    self.entry_time = None
                    closed_position = True
                    
            elif self.current_position == "short":
                # 检查止盈
                if curr_row["low"] <= self.target_price:
                    # 计算平仓盈亏 (使用目标价格平仓)
                    exit_price = self.target_price
                    pnl = (self.entry_price - exit_price) * self.position_size * self.leverage
                    
                    # 更新资金和净值
                    self.current_capital += pnl
                    self.equity = self.current_capital
                    
                    # 更新最大回撤
                    if self.equity > self.max_equity:
                        self.max_equity = self.equity
                    drawdown = (self.max_equity - self.equity) / self.max_equity
                    self.max_drawdown = max(self.max_drawdown, drawdown)
                    
                    # 记录交易
                    trade = {
                        "交易类型": "平空(止盈)",
                        "开仓时间": self.entry_time,
                        "平仓时间": curr_row.name,
                        "开仓价格": self.entry_price,
                        "平仓价格": exit_price,
                        "持仓大小": self.position_size,
                        "杠杆": self.leverage,
                        "盈亏": pnl,
                        "盈亏率": pnl / (self.current_capital - pnl) if self.current_capital - pnl > 0 else float('inf'),
                        "止盈价": self.target_price,
                        "止损价": self.stop_loss
                    }
                    self.trades.append(trade)
                    
                    # 更新数据框
                    df.loc[df.index[i], "trade_pnl"] = pnl
                    df.loc[df.index[i], "cumulative_pnl"] = prev_row["cumulative_pnl"] + pnl
                    df.loc[df.index[i], "capital"] = self.current_capital
                    df.loc[df.index[i], "equity"] = self.equity
                    df.loc[df.index[i], "drawdown"] = drawdown
                    df.loc[df.index[i], "position_size"] = 0
                    df.loc[df.index[i], "trade_action"] = "止盈平空"
                    
                    # 重置当前持仓
                    self.current_position = None
                    self.position_size = 0
                    self.entry_price = 0
                    self.stop_loss = 0
                    self.target_price = 0
                    self.entry_time = None
                    closed_position = True
                    
                # 检查止损
                elif curr_row["high"] >= self.stop_loss:
                    # 计算平仓盈亏 (使用止损价格平仓)
                    exit_price = self.stop_loss
                    pnl = (self.entry_price - exit_price) * self.position_size * self.leverage
                    
                    # 更新资金和净值
                    self.current_capital += pnl
                    self.equity = self.current_capital
                    
                    # 更新最大回撤
                    if self.equity > self.max_equity:
                        self.max_equity = self.equity
                    drawdown = (self.max_equity - self.equity) / self.max_equity
                    self.max_drawdown = max(self.max_drawdown, drawdown)
                    
                    # 记录交易
                    trade = {
                        "交易类型": "平空(止损)",
                        "开仓时间": self.entry_time,
                        "平仓时间": curr_row.name,
                        "开仓价格": self.entry_price,
                        "平仓价格": exit_price,
                        "持仓大小": self.position_size,
                        "杠杆": self.leverage,
                        "盈亏": pnl,
                        "盈亏率": pnl / (self.current_capital - pnl) if self.current_capital - pnl > 0 else float('inf'),
                        "止盈价": self.target_price,
                        "止损价": self.stop_loss
                    }
                    self.trades.append(trade)
                    
                    # 更新数据框
                    df.loc[df.index[i], "trade_pnl"] = pnl
                    df.loc[df.index[i], "cumulative_pnl"] = prev_row["cumulative_pnl"] + pnl
                    df.loc[df.index[i], "capital"] = self.current_capital
                    df.loc[df.index[i], "equity"] = self.equity
                    df.loc[df.index[i], "drawdown"] = drawdown
                    df.loc[df.index[i], "position_size"] = 0
                    df.loc[df.index[i], "trade_action"] = "止损平空"
                    
                    # 重置当前持仓
                    self.current_position = None
                    self.position_size = 0
                    self.entry_price = 0
                    self.stop_loss = 0
                    self.target_price = 0
                    self.entry_time = None
                    closed_position = True
            
            # 步骤2：处理信号平仓 - 如果有信号且与当前持仓方向相反，则平仓
            if not closed_position:
                if curr_row["buy_signal"] and self.current_position == "short":
                    # 平空仓
                    exit_price = curr_row["close"]
                    pnl = (self.entry_price - exit_price) * self.position_size * self.leverage
                    
                    # 更新资金和净值
                    self.current_capital += pnl
                    self.equity = self.current_capital
                    
                    # 更新最大回撤
                    if self.equity > self.max_equity:
                        self.max_equity = self.equity
                    drawdown = (self.max_equity - self.equity) / self.max_equity
                    self.max_drawdown = max(self.max_drawdown, drawdown)
                    
                    # 记录交易
                    trade = {
                        "交易类型": "平空(信号)",
                        "开仓时间": self.entry_time,
                        "平仓时间": curr_row.name,
                        "开仓价格": self.entry_price,
                        "平仓价格": exit_price,
                        "持仓大小": self.position_size,
                        "杠杆": self.leverage,
                        "盈亏": pnl,
                        "盈亏率": pnl / (self.current_capital - pnl) if self.current_capital - pnl > 0 else float('inf'),
                        "止盈价": self.target_price,
                        "止损价": self.stop_loss
                    }
                    self.trades.append(trade)
                    
                    # 更新数据框
                    df.loc[df.index[i], "trade_pnl"] = pnl
                    df.loc[df.index[i], "cumulative_pnl"] = prev_row["cumulative_pnl"] + pnl
                    df.loc[df.index[i], "capital"] = self.current_capital
                    df.loc[df.index[i], "equity"] = self.equity
                    df.loc[df.index[i], "drawdown"] = drawdown
                    df.loc[df.index[i], "position_size"] = 0
                    df.loc[df.index[i], "trade_action"] = "信号平空"
                    
                    # 重置当前持仓
                    self.current_position = None
                    self.position_size = 0
                    self.entry_price = 0
                    self.stop_loss = 0
                    self.target_price = 0
                    self.entry_time = None
                    closed_position = True
                    
                elif curr_row["sell_signal"] and self.current_position == "long":
                    # 平多仓
                    exit_price = curr_row["close"]
                    pnl = (exit_price - self.entry_price) * self.position_size * self.leverage
                    
                    # 更新资金和净值
                    self.current_capital += pnl
                    self.equity = self.current_capital
                    
                    # 更新最大回撤
                    if self.equity > self.max_equity:
                        self.max_equity = self.equity
                    drawdown = (self.max_equity - self.equity) / self.max_equity
                    self.max_drawdown = max(self.max_drawdown, drawdown)
                    
                    # 记录交易
                    trade = {
                        "交易类型": "平多(信号)",
                        "开仓时间": self.entry_time,
                        "平仓时间": curr_row.name,
                        "开仓价格": self.entry_price,
                        "平仓价格": exit_price,
                        "持仓大小": self.position_size,
                        "杠杆": self.leverage,
                        "盈亏": pnl,
                        "盈亏率": pnl / (self.current_capital - pnl) if self.current_capital - pnl > 0 else float('inf'),
                        "止盈价": self.target_price,
                        "止损价": self.stop_loss
                    }
                    self.trades.append(trade)
                    
                    # 更新数据框
                    df.loc[df.index[i], "trade_pnl"] = pnl
                    df.loc[df.index[i], "cumulative_pnl"] = prev_row["cumulative_pnl"] + pnl
                    df.loc[df.index[i], "capital"] = self.current_capital
                    df.loc[df.index[i], "equity"] = self.equity
                    df.loc[df.index[i], "drawdown"] = drawdown
                    df.loc[df.index[i], "position_size"] = 0
                    df.loc[df.index[i], "trade_action"] = "信号平多"
                    
                    # 重置当前持仓
                    self.current_position = None
                    self.position_size = 0
                    self.entry_price = 0
                    self.stop_loss = 0
                    self.target_price = 0
                    self.entry_time = None
                    closed_position = True
            
            # 步骤3：处理开仓信号 - 只有在没有持仓的情况下才开仓
            if self.current_position is None:
                if curr_row["buy_signal"]:
                    # 开多仓
                    self.entry_price = curr_row["close"]
                    self.stop_loss = curr_row["stop_loss_buy"]
                    risk_amount = self.current_capital * self.risk_per_trade
                    risk_per_unit = (self.entry_price - self.stop_loss) * self.leverage
                    self.position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
                    
                    # 更新目标价格
                    if "target_price_buy" in curr_row:
                        self.target_price = curr_row["target_price_buy"]
                    else:
                        risk_distance = self.entry_price - self.stop_loss
                        self.target_price = self.entry_price + risk_distance * 3.0  # 默认风险回报比为3
                    
                    # 更新持仓状态
                    self.current_position = "long"
                    self.entry_time = curr_row.name
                    
                    # 更新数据框
                    df.loc[df.index[i], "position_size"] = self.position_size
                    df.loc[df.index[i], "trade_action"] = "开多"
                
                elif curr_row["sell_signal"]:
                    # 开空仓
                    self.entry_price = curr_row["close"]
                    self.stop_loss = curr_row["stop_loss_sell"]
                    risk_amount = self.current_capital * self.risk_per_trade
                    risk_per_unit = (self.stop_loss - self.entry_price) * self.leverage
                    self.position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
                    
                    # 更新目标价格
                    if "target_price_sell" in curr_row:
                        self.target_price = curr_row["target_price_sell"]
                    else:
                        risk_distance = self.stop_loss - self.entry_price
                        self.target_price = self.entry_price - risk_distance * 3.0  # 默认风险回报比为3
                    
                    # 更新持仓状态
                    self.current_position = "short"
                    self.entry_time = curr_row.name
                    
                    # 更新数据框
                    df.loc[df.index[i], "position_size"] = self.position_size
                    df.loc[df.index[i], "trade_action"] = "开空"

            # 更新净值和回撤
            current_price = curr_row["close"]
            if self.current_position == "long":
                unrealized_pnl = (current_price - self.entry_price) * self.position_size * self.leverage
                df.loc[df.index[i], "equity"] = self.current_capital + unrealized_pnl
            elif self.current_position == "short":
                unrealized_pnl = (self.entry_price - current_price) * self.position_size * self.leverage
                df.loc[df.index[i], "equity"] = self.current_capital + unrealized_pnl
            
            # 更新最大净值和回撤
            if df.loc[df.index[i], "equity"] > self.max_equity:
                self.max_equity = df.loc[df.index[i], "equity"]
            drawdown = (self.max_equity - df.loc[df.index[i], "equity"]) / self.max_equity if self.max_equity > 0 else 0
            df.loc[df.index[i], "drawdown"] = drawdown
            self.max_drawdown = max(self.max_drawdown, drawdown)
        
        return df
    
    def get_trade_summary(self):
        """
        获取交易统计摘要
        
        返回:
            dict: 交易统计
        """
        if not self.trades:
            return {
                "初始资金": self.initial_capital,
                "当前资金": self.current_capital,
                "净利润": 0,
                "净利润率": 0,
                "交易次数": 0,
                "胜率": 0,
                "平均盈利": 0,
                "平均亏损": 0,
                "盈亏比": 0,
                "最大回撤": 0,
                "平均杠杆": self.leverage,
            }
        
        # 计算统计数据
        profit_trades = [t for t in self.trades if t["盈亏"] > 0]
        loss_trades = [t for t in self.trades if t["盈亏"] <= 0]
        
        total_trades = len(self.trades)
        win_trades = len(profit_trades)
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        
        total_profit = sum(t["盈亏"] for t in profit_trades) if profit_trades else 0
        total_loss = sum(t["盈亏"] for t in loss_trades) if loss_trades else 0
        
        avg_profit = total_profit / win_trades if win_trades > 0 else 0
        avg_loss = total_loss / len(loss_trades) if loss_trades else 0
        
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        return {
            "初始资金": self.initial_capital,
            "当前资金": self.current_capital,
            "净利润": self.current_capital - self.initial_capital,
            "净利润率": (self.current_capital - self.initial_capital) / self.initial_capital,
            "交易次数": total_trades,
            "胜率": win_rate,
            "平均盈利": avg_profit,
            "平均亏损": avg_loss,
            "盈亏比": abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf'),
            "最大回撤": self.max_drawdown,
            "平均杠杆": self.leverage,
        }
    
    def export_to_excel(self, signals_df, filename=None):
        """
        导出交易记录到Excel
        
        参数:
            signals_df (pd.DataFrame): 包含信号和交易结果的数据
            filename (str, optional): 导出的文件名，不提供则自动生成
            
        返回:
            str: 导出的文件路径
        """
        if filename is None:
            # 生成文件名
            strategy_name = self.config.get("signals", {}).get("strategy", "Supertrend和DEMA策略")
            timeframe = self.config.get("trading", {}).get("timeframe", "1h")
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/{strategy_name}_{timeframe}_{now}.xlsx"
        
        try:
            # 创建Excel写入器
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # 导出详细交易记录 - 修改为将每笔交易拆分为开仓和平仓两行
                if self.trades:
                    # 创建新的交易记录数组
                    trade_records = []
                    trade_id = 1
                    
                    for trade in self.trades:
                        # 从交易类型中识别交易方向
                        is_long = "平多" in trade["交易类型"]
                        direction = "多" if is_long else "空"
                        close_reason = trade["交易类型"].split("(")[1].split(")")[0] if "(" in trade["交易类型"] else "信号"
                        
                        # 计算持仓时间（小时）
                        entry_time = trade["开仓时间"]
                        exit_time = trade["平仓时间"]
                        holding_hours = (exit_time - entry_time).total_seconds() / 3600
                        
                        # 计算每小时收益
                        hourly_profit = trade["盈亏"] / holding_hours if holding_hours > 0 else 0
                        
                        # 计算投资回报率ROI
                        position_value = trade["开仓价格"] * trade["持仓大小"]
                        roi = (trade["盈亏"] / position_value) * 100 if position_value > 0 else 0
                        
                        # 创建开仓记录
                        open_record = {
                            "交易ID": trade_id,
                            "操作类型": f"开{direction}",
                            "时间": trade["开仓时间"],
                            "价格": trade["开仓价格"],
                            "持仓大小": trade["持仓大小"],
                            "杠杆": trade["杠杆"],
                            "止盈价": trade["止盈价"] if "止盈价" in trade else "-",
                            "止损价": trade["止损价"] if "止损价" in trade else "-",
                            "交易状态": "已平仓"
                        }
                        
                        # 创建平仓记录
                        close_record = {
                            "交易ID": trade_id,
                            "操作类型": f"平{direction}({close_reason})",
                            "时间": trade["平仓时间"],
                            "价格": trade["平仓价格"],
                            "持仓大小": trade["持仓大小"],
                            "杠杆": trade["杠杆"],
                            "盈亏": trade["盈亏"],
                            "盈亏率": trade["盈亏率"] if isinstance(trade["盈亏率"], (int, float)) else 0,
                            "持仓时间(小时)": round(holding_hours, 2),
                            "每小时收益": round(hourly_profit, 2),
                            "ROI(%)": round(roi, 2),
                            "资金效率": round(trade["盈亏"] / (self.initial_capital * (holding_hours/24)), 4) if holding_hours > 0 else 0,
                            "交易状态": "盈利" if trade["盈亏"] > 0 else "亏损"
                        }
                        
                        # 添加到记录数组
                        trade_records.append(open_record)
                        trade_records.append(close_record)
                        trade_id += 1
                    
                    # 创建DataFrame并按时间排序
                    trades_df = pd.DataFrame(trade_records)
                    # 按照交易ID和操作类型排序，确保相同交易ID内部开仓记录在前，平仓记录在后
                    trades_df['操作顺序'] = trades_df['操作类型'].apply(lambda x: 1 if '开' in x else 2)
                    trades_df = trades_df.sort_values(by=['交易ID', '操作顺序']).drop(columns=['操作顺序'])
                    
                    # 处理日期时间列
                    if "时间" in trades_df.columns:
                        if pd.api.types.is_datetime64_dtype(trades_df["时间"]):
                            # 检查是否有时区信息
                            if hasattr(trades_df["时间"].dtype, 'tz') and trades_df["时间"].dtype.tz is not None:
                                trades_df["时间"] = trades_df["时间"].dt.tz_localize(None)
                        elif len(trades_df) > 0 and isinstance(trades_df["时间"].iloc[0], datetime):
                            # 转换单个datetime对象
                            trades_df["时间"] = trades_df["时间"].apply(
                                lambda x: x.replace(tzinfo=None) if x is not None and hasattr(x, 'tzinfo') and x.tzinfo is not None else x
                            )
                    
                    # 导出到Excel
                    trades_df.to_excel(writer, sheet_name='交易记录', index=False)
                    
                    # 设置列宽和格式
                    worksheet = writer.sheets['交易记录']
                    for idx, col in enumerate(trades_df.columns):
                        column_width = max(len(str(col)), trades_df[col].astype(str).str.len().max())
                        worksheet.column_dimensions[get_column_letter(idx+1)].width = column_width + 4
                    
                    # 设置条件格式 - 为盈亏列和操作类型添加条件格式
                    from openpyxl.styles import PatternFill, Font
                    from openpyxl.formatting.rule import CellIsRule
                    
                    # 应用数据行的格式
                    for row_idx, row in enumerate(trades_df.itertuples(), start=2):  # Excel行从2开始（跳过标题行）
                        # 获取行数据
                        row_data = row._asdict()
                        
                        # 找到操作类型和交易状态的单元格索引
                        operation_cell = None
                        status_cell = None
                        profit_cell = None
                        
                        for col_idx, col_name in enumerate(trades_df.columns, start=1):
                            cell = worksheet.cell(row=row_idx, column=col_idx)
                            
                            if col_name == "操作类型":
                                operation_cell = cell
                            elif col_name == "交易状态":
                                status_cell = cell
                            elif col_name == "盈亏":
                                profit_cell = cell
                        
                        # 设置开仓行颜色（绿色或红色）
                        if operation_cell and "开多" in str(operation_cell.value):
                            operation_cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                            operation_cell.font = Font(color="006100", bold=True)
                        elif operation_cell and "开空" in str(operation_cell.value):
                            operation_cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
                            operation_cell.font = Font(color="9C0006", bold=True)
                        
                        # 设置平仓行颜色（绿色或红色）
                        if status_cell and profit_cell:
                            if "盈利" in str(status_cell.value) or (isinstance(profit_cell.value, (int, float)) and profit_cell.value > 0):
                                status_cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                                status_cell.font = Font(color="006100", bold=True)
                                if profit_cell:
                                    profit_cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                                    profit_cell.font = Font(color="006100", bold=True)
                            elif "亏损" in str(status_cell.value) or (isinstance(profit_cell.value, (int, float)) and profit_cell.value < 0):
                                status_cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
                                status_cell.font = Font(color="9C0006", bold=True)
                                if profit_cell:
                                    profit_cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
                                    profit_cell.font = Font(color="9C0006", bold=True)
                        
                        # 为止盈止损平仓添加特殊格式
                        if operation_cell:
                            if "止盈" in str(operation_cell.value):
                                operation_cell.fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
                                operation_cell.font = Font(color="9C6500", bold=True)
                            elif "止损" in str(operation_cell.value):
                                operation_cell.fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
                                operation_cell.font = Font(color="9C6500", bold=True)
                else:
                    # 如果没有交易记录，创建一个空的表格
                    empty_df = pd.DataFrame({'无数据': ['无交易记录']})
                    empty_df.to_excel(writer, sheet_name='交易记录', index=False)
                
                # 导出交易统计
                summary = self.get_trade_summary()
                summary_data = []
                
                # 添加更多的统计指标
                if self.trades:
                    total_profit = sum(t["盈亏"] for t in self.trades if t["盈亏"] > 0)
                    total_loss = sum(t["盈亏"] for t in self.trades if t["盈亏"] < 0)
                    summary["总盈利金额"] = total_profit
                    summary["总亏损金额"] = total_loss
                    summary["盈亏金额比"] = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
                    
                    # 计算最大连续盈利和亏损次数
                    results = [1 if t["盈亏"] > 0 else 0 for t in self.trades]
                    max_win_streak = max_consecutive(results, 1)
                    max_loss_streak = max_consecutive(results, 0)
                    summary["最大连续盈利次数"] = max_win_streak
                    summary["最大连续亏损次数"] = max_loss_streak
                    
                    # 计算最大单笔盈利和亏损
                    max_profit = max([t["盈亏"] for t in self.trades if t["盈亏"] > 0], default=0)
                    max_loss = min([t["盈亏"] for t in self.trades if t["盈亏"] < 0], default=0)
                    summary["最大单笔盈利"] = max_profit
                    summary["最大单笔亏损"] = max_loss
                    
                    # 计算持仓时间相关统计
                    holding_hours = [(t["平仓时间"] - t["开仓时间"]).total_seconds() / 3600 for t in self.trades]
                    avg_holding_hours = sum(holding_hours) / len(holding_hours) if holding_hours else 0
                    max_holding_hours = max(holding_hours) if holding_hours else 0
                    min_holding_hours = min(holding_hours) if holding_hours else 0
                    summary["平均持仓时间(小时)"] = round(avg_holding_hours, 2)
                    summary["最长持仓时间(小时)"] = round(max_holding_hours, 2)
                    summary["最短持仓时间(小时)"] = round(min_holding_hours, 2)
                    
                    # 计算不同持仓方向的统计
                    long_trades = [t for t in self.trades if "平多" in t["交易类型"]]
                    short_trades = [t for t in self.trades if "平空" in t["交易类型"]]
                    
                    # 多头统计
                    if long_trades:
                        long_wins = [t for t in long_trades if t["盈亏"] > 0]
                        long_win_rate = len(long_wins) / len(long_trades) if long_trades else 0
                        long_profit = sum(t["盈亏"] for t in long_trades)
                        summary["多头交易次数"] = len(long_trades)
                        summary["多头胜率"] = long_win_rate
                        summary["多头总盈亏"] = round(long_profit, 2)
                    
                    # 空头统计
                    if short_trades:
                        short_wins = [t for t in short_trades if t["盈亏"] > 0]
                        short_win_rate = len(short_wins) / len(short_trades) if short_trades else 0
                        short_profit = sum(t["盈亏"] for t in short_trades)
                        summary["空头交易次数"] = len(short_trades)
                        summary["空头胜率"] = short_win_rate
                        summary["空头总盈亏"] = round(short_profit, 2)
                    
                    # 日内/隔夜交易统计
                    intraday_trades = [t for t in self.trades if (t["平仓时间"] - t["开仓时间"]).total_seconds() <= 24*3600]
                    multiday_trades = [t for t in self.trades if (t["平仓时间"] - t["开仓时间"]).total_seconds() > 24*3600]
                    
                    if intraday_trades:
                        intraday_profit = sum(t["盈亏"] for t in intraday_trades)
                        intraday_win_rate = len([t for t in intraday_trades if t["盈亏"] > 0]) / len(intraday_trades)
                        summary["日内交易次数"] = len(intraday_trades)
                        summary["日内交易胜率"] = intraday_win_rate
                        summary["日内交易总盈亏"] = round(intraday_profit, 2)
                    
                    if multiday_trades:
                        multiday_profit = sum(t["盈亏"] for t in multiday_trades)
                        multiday_win_rate = len([t for t in multiday_trades if t["盈亏"] > 0]) / len(multiday_trades)
                        summary["隔夜交易次数"] = len(multiday_trades)
                        summary["隔夜交易胜率"] = multiday_win_rate
                        summary["隔夜交易总盈亏"] = round(multiday_profit, 2)
                        
                    # 计算系统效率和性能指标
                    total_days = (self.trades[-1]["平仓时间"] - self.trades[0]["开仓时间"]).total_seconds() / (24*3600) if len(self.trades) > 1 else 1
                    summary["日均交易次数"] = round(len(self.trades) / total_days, 2)
                    summary["日均盈亏"] = round((self.current_capital - self.initial_capital) / total_days, 2)
                    summary["资金效率(每日收益率)"] = round((self.current_capital - self.initial_capital) / (self.initial_capital * total_days) * 100, 4)
                
                # 格式化并排序显示统计数据
                for key, value in summary.items():
                    if isinstance(value, float):
                        if key in ['胜率', '最大回撤', '净利润率']:
                            formatted_value = f"{value:.2%}"
                        else:
                            formatted_value = f"{value:.2f}"
                    else:
                        formatted_value = value
                    summary_data.append([key, formatted_value])
                
                summary_df = pd.DataFrame(summary_data, columns=['指标', '数值'])
                summary_df.to_excel(writer, sheet_name='交易统计', index=False)
                
                # 设置列宽
                worksheet = writer.sheets['交易统计']
                worksheet.column_dimensions['A'].width = 20
                worksheet.column_dimensions['B'].width = 15
            
            print(f"交易记录已成功导出到: {filename}")
            return filename
        except Exception as e:
            print(f"导出到Excel失败: {e}")
            import traceback
            traceback.print_exc()
            return None

def max_consecutive(lst, val):
    """计算列表中最大连续出现val的次数"""
    max_count = 0
    current_count = 0
    
    for item in lst:
        if item == val:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0
    
    return max_count 