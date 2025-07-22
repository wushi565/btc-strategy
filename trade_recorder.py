# =============================================================================
# 优化版交易记录器
# 功能：使用向量化操作处理交易信号，提高回测性能
# 特性：支持多种交易类型、风险管理、Excel导出、统计分析
# 版本：2.0 (优化版)
# =============================================================================

import pandas as pd           # 数据处理和分析
import numpy as np            # 数值计算
import os                     # 操作系统接口
from datetime import datetime # 日期时间处理
from openpyxl import Workbook # Excel文件创建
from openpyxl.utils import get_column_letter  # Excel列名工具
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side  # Excel样式

class OptimizedTradeRecorder:
    """优化的交易记录器，使用向量化操作提高性能"""
    
    def __init__(self, config):
        """初始化交易记录器"""
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
        
        # 创建输出目录
        self.output_dir = "trades"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def process_signals_vectorized(self, signals_df):
        """
        使用向量化操作处理信号，大幅提高性能
        
        参数:
            signals_df (pd.DataFrame): 包含信号的数据
            
        返回:
            pd.DataFrame: 添加了交易结果的数据
        """
        df = signals_df.copy()
        
        # 初始化列
        df = self._initialize_result_columns(df)
        
        # 使用向量化方法处理信号
        trades_data = self._extract_trade_signals_vectorized(df)
        
        # 计算交易结果
        df = self._calculate_trade_results_vectorized(df, trades_data)
        
        # 更新交易记录
        self._update_trade_records(trades_data)
        
        return df
    
    def _initialize_result_columns(self, df):
        """初始化结果列"""
        df["capital"] = self.initial_capital
        df["position_size"] = 0.0
        df["leverage"] = self.leverage
        df["trade_pnl"] = 0.0
        df["cumulative_pnl"] = 0.0
        df["equity"] = self.initial_capital
        df["drawdown"] = 0.0
        df["trade_action"] = None
        df["dist_to_dema144"] = np.nan
        df["dist_to_dema169"] = np.nan
        df["dist_to_dema_pct"] = np.nan
        return df
    
    def _extract_trade_signals_vectorized(self, df):
        """使用向量化方法提取交易信号"""
        # 识别买卖信号点
        buy_signals = df['buy_signal'].shift().fillna(False) & ~df['buy_signal'].shift(2).fillna(False)
        sell_signals = df['sell_signal'].shift().fillna(False) & ~df['sell_signal'].shift(2).fillna(False)
        
        # 创建交易匹配逻辑
        trades_data = []
        
        # 简化的信号匹配（实际使用时需要根据具体策略调整）
        buy_indices = df.index[buy_signals].tolist()
        sell_indices = df.index[sell_signals].tolist()
        
        # 配对信号（简化版本）
        for i, buy_idx in enumerate(buy_indices):
            # 查找对应的卖出信号
            matching_sells = [s for s in sell_indices if s > buy_idx]
            if matching_sells:
                sell_idx = matching_sells[0]
                
                buy_row = df.loc[buy_idx]
                sell_row = df.loc[sell_idx]
                
                # 计算持仓信息
                entry_price = buy_row['close']
                exit_price = sell_row['close']
                stop_loss = buy_row.get('stop_loss_buy', buy_row.get('dema169', entry_price * 0.95))
                target_price = buy_row.get('target_price_buy', entry_price * 1.1)
                
                # 计算仓位大小
                risk_amount = self.current_capital * self.risk_per_trade
                risk_per_unit = max(entry_price - stop_loss, entry_price * 0.01)  # 最小1%风险
                position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
                
                # 计算盈亏
                pnl = (exit_price - entry_price) * position_size * self.leverage
                
                trades_data.append({
                    'entry_idx': buy_idx,
                    'exit_idx': sell_idx,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position_size': position_size,
                    'pnl': pnl,
                    'stop_loss': stop_loss,
                    'target_price': target_price,
                    'direction': 'long'
                })
        
        return trades_data
    
    def _calculate_trade_results_vectorized(self, df, trades_data):
        """使用向量化方法计算交易结果"""
        cumulative_pnl = 0
        current_capital = self.initial_capital
        
        for trade in trades_data:
            entry_idx = trade['entry_idx']
            exit_idx = trade['exit_idx']
            pnl = trade['pnl']
            
            # 更新入场点
            df.loc[entry_idx, 'trade_action'] = f"开{trade['direction']}"
            df.loc[entry_idx, 'position_size'] = trade['position_size']
            
            # 更新出场点
            df.loc[exit_idx, 'trade_action'] = f"平{trade['direction']}"
            df.loc[exit_idx, 'trade_pnl'] = pnl
            df.loc[exit_idx, 'position_size'] = 0
            
            # 更新累计盈亏
            cumulative_pnl += pnl
            current_capital += pnl
            df.loc[exit_idx, 'cumulative_pnl'] = cumulative_pnl
            df.loc[exit_idx, 'capital'] = current_capital
            
        # 使用前向填充更新所有行
        df['cumulative_pnl'] = df['cumulative_pnl'].ffill()
        df['capital'] = df['capital'].ffill()
        
        # 计算净值曲线
        df['equity'] = df['capital']
        
        # 计算回撤
        df['running_max'] = df['equity'].expanding().max()
        df['drawdown'] = (df['running_max'] - df['equity']) / df['running_max']
        
        # 更新最大回撤
        self.max_drawdown = df['drawdown'].max()
        self.current_capital = current_capital
        
        return df
    
    def _update_trade_records(self, trades_data):
        """更新交易记录"""
        self.trades = []
        for trade in trades_data:
            trade_record = {
                "交易类型": f"平{trade['direction']}(信号)",
                "开仓时间": trade['entry_idx'],
                "平仓时间": trade['exit_idx'],
                "开仓价格": trade['entry_price'],
                "平仓价格": trade['exit_price'],
                "持仓大小": trade['position_size'],
                "杠杆": self.leverage,
                "盈亏": trade['pnl'],
                "盈亏率": trade['pnl'] / (trade['entry_price'] * trade['position_size']) if trade['position_size'] > 0 else 0,
                "止盈价": trade['target_price'],
                "止损价": trade['stop_loss']
            }
            self.trades.append(trade_record)
    
    def get_trade_summary(self):
        """获取交易统计摘要"""
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
        
        # 向量化计算统计数据
        pnl_series = pd.Series([t["盈亏"] for t in self.trades])
        
        profit_trades = pnl_series[pnl_series > 0]
        loss_trades = pnl_series[pnl_series <= 0]
        
        total_trades = len(self.trades)
        win_trades = len(profit_trades)
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        
        avg_profit = profit_trades.mean() if len(profit_trades) > 0 else 0
        avg_loss = loss_trades.mean() if len(loss_trades) > 0 else 0
        
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
    
    def export_to_excel_streamlined(self, signals_df, filename=None):
        """
        简化的Excel导出，专注于核心数据
        
        参数:
            signals_df (pd.DataFrame): 包含信号和交易结果的数据
            filename (str, optional): 导出的文件名
            
        返回:
            str: 导出的文件路径
        """
        if filename is None:
            strategy_name = self.config.get("signals", {}).get("strategy", "Strategy")
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/{strategy_name}_{now}.xlsx"
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # 1. 交易记录
                if self.trades:
                    trades_df = self._create_trades_dataframe()
                    trades_df.to_excel(writer, sheet_name='交易记录', index=False)
                
                # 2. 交易统计
                summary = self.get_trade_summary()
                summary_df = pd.DataFrame(list(summary.items()), columns=['指标', '数值'])
                summary_df.to_excel(writer, sheet_name='交易统计', index=False)
                
                # 3. 核心数据（简化版本）
                core_data = signals_df[['open', 'high', 'low', 'close', 'volume', 
                                      'buy_signal', 'sell_signal', 'equity', 'drawdown']].copy()
                core_data.to_excel(writer, sheet_name='核心数据', index=True)
            
            print(f"交易记录已成功导出到: {filename}")
            return filename
            
        except Exception as e:
            print(f"导出到Excel失败: {e}")
            return None
    
    def _create_trades_dataframe(self):
        """创建交易DataFrame"""
        trade_records = []
        trade_id = 1
        
        for trade in self.trades:
            # 简化的交易记录格式
            record = {
                "交易ID": trade_id,
                "开仓时间": trade["开仓时间"],
                "平仓时间": trade["平仓时间"],
                "开仓价格": trade["开仓价格"],
                "平仓价格": trade["平仓价格"],
                "盈亏": trade["盈亏"],
                "盈亏率": f"{trade['盈亏率']:.2%}" if isinstance(trade['盈亏率'], (int, float)) else "0%",
                "交易状态": "盈利" if trade["盈亏"] > 0 else "亏损"
            }
            trade_records.append(record)
            trade_id += 1
        
        return pd.DataFrame(trade_records)


class TradeAnalytics:
    """独立的交易分析模块"""
    
    @staticmethod
    def calculate_performance_metrics(equity_series):
        """计算性能指标"""
        returns = equity_series.pct_change().dropna()
        
        # 年化收益率
        annual_return = returns.mean() * 252
        
        # 年化波动率
        annual_volatility = returns.std() * np.sqrt(252)
        
        # 夏普比率
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        
        # 最大回撤
        running_max = equity_series.expanding().max()
        drawdown = (running_max - equity_series) / running_max
        max_drawdown = drawdown.max()
        
        return {
            "年化收益率": annual_return,
            "年化波动率": annual_volatility,
            "夏普比率": sharpe_ratio,
            "最大回撤": max_drawdown
        }
    
    @staticmethod
    def analyze_trade_patterns(trades_df):
        """分析交易模式"""
        if trades_df.empty:
            return {}
        
        # 盈亏分布
        profit_trades = trades_df[trades_df['盈亏'] > 0]
        loss_trades = trades_df[trades_df['盈亏'] <= 0]
        
        return {
            "盈利交易占比": len(profit_trades) / len(trades_df),
            "平均盈利": profit_trades['盈亏'].mean() if len(profit_trades) > 0 else 0,
            "平均亏损": loss_trades['盈亏'].mean() if len(loss_trades) > 0 else 0,
            "最大盈利": trades_df['盈亏'].max(),
            "最大亏损": trades_df['盈亏'].min()
        }
