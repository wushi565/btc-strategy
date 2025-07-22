# =============================================================================
# 优化版交易记录器
# 功能：使用向量化操作处理交易信号，提高回测性能
# 特性：支持多种交易类型、风险管理、Excel导出、统计分析
# 版本：2.0 (优化版)
# =============================================================================

import pandas as pd           # 数据处理和分析
import numpy as np            # 数值计算
import os                     # 操作系统接口
from datetime import datetime, timedelta # 日期时间处理
from signal_pattern_analyzer import SignalPatternAnalyzer  # 信号模式分析器

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
        
        # 交易指标
        self.trade_metrics = {}
    
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
    
    def _calculate_dema_distance(self, entry_price, dema144, dema169, trade_type='long'):
        """
        计算开仓价格与DEMA线的距离
        
        参数:
            entry_price: 开仓价格
            dema144: DEMA144值
            dema169: DEMA169值
            trade_type: 交易类型 ('long' 或 'short')
            
        返回:
            tuple: (距离DEMA144百分比, 距离DEMA169百分比)
        """
        if trade_type == 'long':
            # 开多：价格突破DEMA，计算突破幅度
            dist_144 = ((entry_price - dema144) / dema144) * 100 if dema144 > 0 else 0
            dist_169 = ((entry_price - dema169) / dema169) * 100 if dema169 > 0 else 0
        else:  # short
            # 开空：价格跌破DEMA，计算跌破幅度
            dist_144 = ((dema144 - entry_price) / dema144) * 100 if dema144 > 0 else 0
            dist_169 = ((dema169 - entry_price) / dema169) * 100 if dema169 > 0 else 0
            
        return dist_144, dist_169
    
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
                
                # 计算DEMA距离
                dema144 = buy_row.get('dema144', entry_price)
                dema169 = buy_row.get('dema169', entry_price)
                dist_144, dist_169 = self._calculate_dema_distance(entry_price, dema144, dema169, 'long')
                
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
                    'direction': 'long',
                    'dist_to_dema144': dist_144,
                    'dist_to_dema169': dist_169,
                    'dema144_value': dema144,
                    'dema169_value': dema169
                })
        
        # 处理卖出信号（做空）
        for i, sell_idx in enumerate(sell_indices):
            # 查找对应的买入信号来平仓
            matching_buys = [b for b in buy_indices if b > sell_idx]
            if matching_buys:
                buy_idx = matching_buys[0]
                
                sell_row = df.loc[sell_idx]
                buy_row = df.loc[buy_idx]
                
                # 计算持仓信息
                entry_price = sell_row['close']  # 做空入场价
                exit_price = buy_row['close']    # 平仓价
                stop_loss = sell_row.get('stop_loss_sell', sell_row.get('dema144', entry_price * 1.05))
                target_price = sell_row.get('target_price_sell', entry_price * 0.9)
                
                # 计算DEMA距离（做空）
                dema144 = sell_row.get('dema144', entry_price)
                dema169 = sell_row.get('dema169', entry_price)
                dist_144, dist_169 = self._calculate_dema_distance(entry_price, dema144, dema169, 'short')
                
                # 计算仓位大小
                risk_amount = self.current_capital * self.risk_per_trade
                risk_per_unit = max(stop_loss - entry_price, entry_price * 0.01)  # 最小1%风险
                position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
                
                # 计算盈亏（做空）
                pnl = (entry_price - exit_price) * position_size * self.leverage
                
                trades_data.append({
                    'entry_idx': sell_idx,
                    'exit_idx': buy_idx,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position_size': position_size,
                    'pnl': pnl,
                    'stop_loss': stop_loss,
                    'target_price': target_price,
                    'direction': 'short',
                    'dist_to_dema144': dist_144,
                    'dist_to_dema169': dist_169,
                    'dema144_value': dema144,
                    'dema169_value': dema169
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
            # 获取实际的开仓和平仓时间
            entry_idx = trade['entry_idx']
            exit_idx = trade['exit_idx']
            
            # 将索引转换为实际的日期时间对象（如果索引本身不是日期时间）
            if isinstance(entry_idx, pd.Timestamp):
                entry_time = entry_idx
            else:
                # 处理非日期时间索引的情况，这里假设entry_idx是一个有效的索引
                # 实际实现中可能需要根据数据结构调整
                entry_time = entry_idx
            
            if isinstance(exit_idx, pd.Timestamp):
                exit_time = exit_idx
            else:
                exit_time = exit_idx
                
            trade_record = {
                "交易类型": f"平{trade['direction']}(信号)",
                "开仓时间": entry_time,
                "平仓时间": exit_time,
                "开仓价格": trade['entry_price'],
                "平仓价格": trade['exit_price'],
                "持仓大小": trade['position_size'],
                "杠杆": self.leverage,
                "盈亏": trade['pnl'],
                "盈亏率": trade['pnl'] / (trade['entry_price'] * trade['position_size']) if trade['position_size'] > 0 else 0,
                "止盈价": trade['target_price'],
                "止损价": trade['stop_loss'],
                "DEMA144距离%": trade.get('dist_to_dema144', 0),
                "DEMA169距离%": trade.get('dist_to_dema169', 0),
                "DEMA144值": trade.get('dema144_value', 0),
                "DEMA169值": trade.get('dema169_value', 0),
                "交易方向": trade['direction']
            }
            
            # 计算持仓时间（如果是日期时间对象）
            if isinstance(entry_time, pd.Timestamp) and isinstance(exit_time, pd.Timestamp):
                holding_time = exit_time - entry_time
                trade_record["持仓时间"] = holding_time
            else:
                trade_record["持仓时间"] = None
                
            self.trades.append(trade_record)
        
        # 计算交易指标
        self._calculate_trade_metrics()
    
    def _calculate_trade_metrics(self):
        """计算交易指标，包括开仓频率和持仓时间"""
        if not self.trades:
            self.trade_metrics = {
                "平均开仓频率": None,
                "平均持仓时间": None,
                "盈利交易平均持仓时间": None,
                "亏损交易平均持仓时间": None,
                "持仓时间中位数": None,
            }
            return
        
        # 确保交易按时间排序
        sorted_trades = sorted(self.trades, key=lambda x: x["开仓时间"] if isinstance(x["开仓时间"], pd.Timestamp) else pd.Timestamp('1970-01-01'))
        
        # 计算开仓频率
        entry_times = [t["开仓时间"] for t in sorted_trades if isinstance(t["开仓时间"], pd.Timestamp)]
        
        if len(entry_times) > 1:
            # 计算相邻开仓之间的时间间隔
            time_diffs = [(entry_times[i] - entry_times[i-1]) for i in range(1, len(entry_times))]
            avg_entry_interval = sum(time_diffs, timedelta()) / len(time_diffs)
        else:
            avg_entry_interval = None
            
        # 计算持仓时间统计
        holding_times = [t["持仓时间"] for t in self.trades if t["持仓时间"] is not None]
        
        if holding_times:
            avg_holding_time = sum(holding_times, timedelta()) / len(holding_times)
            median_idx = len(holding_times) // 2
            holding_times_sorted = sorted(holding_times)
            median_holding_time = holding_times_sorted[median_idx]
        else:
            avg_holding_time = None
            median_holding_time = None
            
        # 区分盈利和亏损交易的持仓时间
        profit_holding_times = [t["持仓时间"] for t in self.trades 
                              if t["持仓时间"] is not None and t["盈亏"] > 0]
        
        loss_holding_times = [t["持仓时间"] for t in self.trades 
                             if t["持仓时间"] is not None and t["盈亏"] <= 0]
                             
        if profit_holding_times:
            avg_profit_holding_time = sum(profit_holding_times, timedelta()) / len(profit_holding_times)
        else:
            avg_profit_holding_time = None
            
        if loss_holding_times:
            avg_loss_holding_time = sum(loss_holding_times, timedelta()) / len(loss_holding_times)
        else:
            avg_loss_holding_time = None
            
        self.trade_metrics = {
            "平均开仓频率": avg_entry_interval,
            "平均持仓时间": avg_holding_time,
            "盈利交易平均持仓时间": avg_profit_holding_time,
            "亏损交易平均持仓时间": avg_loss_holding_time,
            "持仓时间中位数": median_holding_time,
        }
    
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
                "平均开仓频率": None,
                "平均持仓时间": None,
                "盈利交易平均持仓时间": None,
                "亏损交易平均持仓时间": None,
                "持仓时间中位数": None,
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
        
        # 整合交易指标
        result = {
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
        
        # 添加交易频率和持仓时间相关指标
        result.update(self.trade_metrics)
        
        return result
    
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
                
                # 格式化时间间隔为易读的格式
                for key in ["平均开仓频率", "平均持仓时间", "盈利交易平均持仓时间", 
                           "亏损交易平均持仓时间", "持仓时间中位数"]:
                    if key in summary and summary[key] is not None:
                        # 将timedelta转换为更易读的格式
                        hours = summary[key].total_seconds() / 3600
                        summary[key] = f"{hours:.2f}小时"
                
                summary_df = pd.DataFrame(list(summary.items()), columns=['指标', '数值'])
                summary_df.to_excel(writer, sheet_name='交易统计', index=False)
                
                # 3. 核心数据（简化版本）
                core_data = signals_df[['open', 'high', 'low', 'close', 'volume', 
                                      'buy_signal', 'sell_signal', 'equity', 'drawdown']].copy()
                # 移除时区信息以支持Excel导出
                if hasattr(core_data.index, 'tz') and core_data.index.tz is not None:
                    core_data.index = core_data.index.tz_localize(None)
                core_data.to_excel(writer, sheet_name='核心数据', index=True)
                
                # 4. DEMA距离分析
                if self.trades:
                    distance_analysis = self._create_distance_analysis()
                    distance_analysis.to_excel(writer, sheet_name='DEMA距离分析', index=False)
                
                # 5. 信号模式分析
                if self.trades:
                    signal_analysis = self._create_signal_pattern_analysis()
                    for sheet_name, df in signal_analysis.items():
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        
                # 6. 持仓时间分析
                if self.trades:
                    holding_time_analysis = self._create_holding_time_analysis()
                    holding_time_analysis.to_excel(writer, sheet_name='持仓时间分析', index=False)
            
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
            # 格式化持仓时间为易读的格式
            holding_time_str = "未知"
            if trade.get("持仓时间") is not None:
                hours = trade["持仓时间"].total_seconds() / 3600
                holding_time_str = f"{hours:.2f}小时"
            
            # 确保日期时间正确处理
            entry_time = trade["开仓时间"]
            if hasattr(entry_time, 'tz') and entry_time.tz is not None:
                entry_time = entry_time.tz_localize(None)
                
            exit_time = trade["平仓时间"]
            if hasattr(exit_time, 'tz') and exit_time.tz is not None:
                exit_time = exit_time.tz_localize(None)
            
            # 简化的交易记录格式
            record = {
                "交易ID": trade_id,
                "开仓时间": entry_time,
                "平仓时间": exit_time,
                "开仓价格": trade["开仓价格"],
                "平仓价格": trade["平仓价格"],
                "盈亏": trade["盈亏"],
                "盈亏率": f"{trade['盈亏率']:.2%}" if isinstance(trade['盈亏率'], (int, float)) else "0%",
                "交易状态": "盈利" if trade["盈亏"] > 0 else "亏损",
                "持仓时间": holding_time_str
            }
            trade_records.append(record)
            trade_id += 1
        
        return pd.DataFrame(trade_records)
    
    def _create_distance_analysis(self):
        """创建DEMA距离分析DataFrame"""
        analysis_data = []
        
        # 按距离范围分组统计
        distance_ranges = [
            (0, 0.5, "0-0.5%"),
            (0.5, 1.0, "0.5-1.0%"), 
            (1.0, 2.0, "1.0-2.0%"),
            (2.0, 3.0, "2.0-3.0%"),
            (3.0, 5.0, "3.0-5.0%"),
            (5.0, float('inf'), "5.0%+")
        ]
        
        for min_dist, max_dist, range_name in distance_ranges:
            # DEMA144分析
            dema144_trades = []
            dema169_trades = []
            
            for trade in self.trades:
                dist_144 = abs(trade.get('DEMA144距离%', 0))
                dist_169 = abs(trade.get('DEMA169距离%', 0))
                
                if min_dist <= dist_144 < max_dist:
                    dema144_trades.append(trade)
                    
                if min_dist <= dist_169 < max_dist:
                    dema169_trades.append(trade)
            
            # 计算DEMA144统计
            if dema144_trades:
                profitable_144 = sum(1 for t in dema144_trades if t.get('盈亏', 0) > 0)
                win_rate_144 = profitable_144 / len(dema144_trades)
                total_pnl_144 = sum(t.get('盈亏', 0) for t in dema144_trades)
                avg_pnl_144 = total_pnl_144 / len(dema144_trades)
            else:
                win_rate_144 = 0
                total_pnl_144 = 0
                avg_pnl_144 = 0
            
            # 计算DEMA169统计
            if dema169_trades:
                profitable_169 = sum(1 for t in dema169_trades if t.get('盈亏', 0) > 0)
                win_rate_169 = profitable_169 / len(dema169_trades)
                total_pnl_169 = sum(t.get('盈亏', 0) for t in dema169_trades)
                avg_pnl_169 = total_pnl_169 / len(dema169_trades)
            else:
                win_rate_169 = 0
                total_pnl_169 = 0
                avg_pnl_169 = 0
            
            analysis_data.append({
                '距离范围': range_name,
                'DEMA144交易数': len(dema144_trades),
                'DEMA144胜率': f"{win_rate_144:.2%}",
                'DEMA144总盈亏': f"{total_pnl_144:.2f}",
                'DEMA144平均盈亏': f"{avg_pnl_144:.2f}",
                'DEMA169交易数': len(dema169_trades),
                'DEMA169胜率': f"{win_rate_169:.2%}",
                'DEMA169总盈亏': f"{total_pnl_169:.2f}",
                'DEMA169平均盈亏': f"{avg_pnl_169:.2f}"
            })
        
        return pd.DataFrame(analysis_data)
    
    def _create_signal_pattern_analysis(self):
        """创建信号模式分析DataFrame字典"""
        analyzer = SignalPatternAnalyzer()
        analysis_results = analyzer.analyze_signal_patterns(self.trades)
        
        analysis_sheets = {}
        
        # 1. 连续亏损分析
        consecutive = analysis_results.get('consecutive_losses', {})
        consecutive_data = [
            {'指标': '连续亏损序列数', '数值': consecutive.get('total_loss_streaks', 0)},
            {'指标': '平均连续亏损次数', '数值': f"{consecutive.get('avg_consecutive_losses', 0):.2f}"},
            {'指标': '最大连续亏损次数', '数值': consecutive.get('max_consecutive_losses', 0)},
            {'指标': '中位数连续亏损次数', '数值': f"{consecutive.get('median_consecutive_losses', 0):.2f}"}
        ]
        
        # 连续亏损分布
        distribution = consecutive.get('loss_streak_distribution', {})
        for streak_count, frequency in distribution.items():
            consecutive_data.append({
                '指标': f'{streak_count}次连续亏损',
                '数值': f'{frequency}次'
            })
        
        analysis_sheets['连续亏损分析'] = pd.DataFrame(consecutive_data)
        
        # 2. 假信号原因分析
        false_signals = analysis_results.get('false_signal_causes', {})
        false_signal_data = [
            {'指标': '总亏损交易数', '数值': false_signals.get('total_losing_trades', 0)},
            {'指标': '做多亏损交易数', '数值': false_signals.get('long_losing_trades', 0)},
            {'指标': '做空亏损交易数', '数值': false_signals.get('short_losing_trades', 0)},
        ]
        
        # 添加平均距离信息
        long_distances = false_signals.get('long_avg_distances', {})
        short_distances = false_signals.get('short_avg_distances', {})
        
        false_signal_data.extend([
            {'指标': '做多平均DEMA144距离%', '数值': f"{long_distances.get('dema144', 0):.2f}"},
            {'指标': '做多平均DEMA169距离%', '数值': f"{long_distances.get('dema169', 0):.2f}"},
            {'指标': '做空平均DEMA144距离%', '数值': f"{short_distances.get('dema144', 0):.2f}"},
            {'指标': '做空平均DEMA169距离%', '数值': f"{short_distances.get('dema169', 0):.2f}"}
        ])
        
        # 添加潜在原因
        causes = false_signals.get('potential_causes', [])
        for cause in causes:
            false_signal_data.append({
                '指标': cause['cause'],
                '数值': f"{cause['affected_trades']}笔 ({cause['percentage']:.1f}%)"
            })
        
        analysis_sheets['假信号分析'] = pd.DataFrame(false_signal_data)
        
        # 3. 整体统计对比
        overall = analysis_results.get('overall_stats', {})
        long_stats = overall.get('long_trades_stats', {})
        short_stats = overall.get('short_trades_stats', {})
        
        overall_data = [
            {'类型': '整体', '交易数': overall.get('total_trades', 0), 
             '胜率': f"{overall.get('overall_win_rate', 0):.2%}",
             '总盈亏': f"{overall.get('total_pnl', 0):.2f}",
             '平均盈亏': f"{overall.get('avg_pnl', 0):.2f}"},
            {'类型': '做多', '交易数': long_stats.get('count', 0),
             '胜率': f"{long_stats.get('win_rate', 0):.2%}",
             '总盈亏': f"{long_stats.get('total_pnl', 0):.2f}",
             '平均盈亏': f"{long_stats.get('total_pnl', 0) / long_stats.get('count', 1):.2f}"},
            {'类型': '做空', '交易数': short_stats.get('count', 0),
             '胜率': f"{short_stats.get('win_rate', 0):.2%}",
             '总盈亏': f"{short_stats.get('total_pnl', 0):.2f}",
             '平均盈亏': f"{short_stats.get('total_pnl', 0) / short_stats.get('count', 1):.2f}"}
        ]
        
        analysis_sheets['统计对比'] = pd.DataFrame(overall_data)
        
        return analysis_sheets

    def _create_holding_time_analysis(self):
        """创建持仓时间分析DataFrame"""
        if not self.trades:
            return pd.DataFrame()
            
        # 只处理有持仓时间的交易
        valid_trades = [t for t in self.trades if t.get("持仓时间") is not None]
        if not valid_trades:
            return pd.DataFrame({"分析": ["没有有效的持仓时间数据"]})
            
        # 持仓时间区间（小时）
        holding_time_ranges = [
            (0, 1, "0-1小时"),
            (1, 4, "1-4小时"),
            (4, 8, "4-8小时"),
            (8, 24, "8-24小时"),
            (24, 72, "1-3天"),
            (72, float('inf'), "3天以上")
        ]
        
        analysis_data = []
        
        for min_hours, max_hours, range_name in holding_time_ranges:
            # 找出在该范围内的交易
            range_trades = []
            for trade in valid_trades:
                hours = trade["持仓时间"].total_seconds() / 3600
                if min_hours <= hours < max_hours:
                    range_trades.append(trade)
            
            # 计算统计数据
            if range_trades:
                total_trades = len(range_trades)
                profitable = sum(1 for t in range_trades if t["盈亏"] > 0)
                win_rate = profitable / total_trades
                total_pnl = sum(t["盈亏"] for t in range_trades)
                avg_pnl = total_pnl / total_trades
                
                analysis_data.append({
                    "持仓时间范围": range_name,
                    "交易数量": total_trades,
                    "盈利交易数": profitable,
                    "亏损交易数": total_trades - profitable,
                    "胜率": f"{win_rate:.2%}",
                    "总盈亏": f"{total_pnl:.2f}",
                    "平均盈亏": f"{avg_pnl:.2f}"
                })
        
        return pd.DataFrame(analysis_data)


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
