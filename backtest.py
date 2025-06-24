import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

class BacktestEngine:
    """回测引擎类，负责模拟交易和计算绩效指标"""
    
    def __init__(self, config):
        """
        初始化回测引擎
        
        参数:
            config (dict): 配置信息
        """
        self.config = config
        self.backtest_config = config.get("backtest", {})
        
        # 获取回测参数
        self.initial_capital = self.backtest_config.get("initial_capital", 10000)
        self.leverage = self.backtest_config.get("leverage", 1)
        self.risk_per_trade = self.backtest_config.get("risk_per_trade", 0.02)
        
    def calculate_position_size(self, entry_price, stop_loss, capital):
        """
        计算头寸大小
        
        参数:
            entry_price (float): 入场价格
            stop_loss (float): 止损价格
            capital (float): 可用资金
            
        返回:
            float: 头寸大小(以基础货币计量)
        """
        # 计算愿意承担的风险金额
        risk_amount = capital * self.risk_per_trade
        
        # 计算价格差异(风险)百分比
        price_risk_percent = abs(entry_price - stop_loss) / entry_price
        
        # 根据杠杆调整风险百分比
        effective_risk_percent = price_risk_percent / self.leverage
        
        # 计算头寸大小(以基础货币计量)
        if effective_risk_percent > 0:
            position_size = risk_amount / effective_risk_percent
            # 调整为杠杆后的头寸大小
            position_size = position_size * self.leverage / entry_price
        else:
            position_size = 0
            
        return position_size
    
    def run_backtest(self, signals_df):
        """
        运行回测
        
        参数:
            signals_df (pd.DataFrame): 包含交易信号的数据
            
        返回:
            pd.DataFrame: 添加了交易结果的数据
        """
        with st.spinner("正在运行回测..."):
            # 创建副本避免修改原始数据
            backtest_df = signals_df.copy()
            
            # 添加资金列并初始化为初始资金
            backtest_df["capital"] = self.initial_capital
            
            # 添加头寸大小列
            backtest_df["position_size"] = 0.0
            
            # 添加交易盈亏列
            backtest_df["trade_pnl"] = 0.0
            
            # 添加累计盈亏列
            backtest_df["cumulative_pnl"] = 0.0
            
            # 添加净值列
            backtest_df["equity"] = self.initial_capital
            
            # 添加持仓状态列
            backtest_df["position"] = ""
            
            # 遍历数据计算交易指标
            current_capital = self.initial_capital
            current_position = 0.0
            entry_price = 0.0
            position_type = None  # "long"或"short"
            cumulative_pnl = 0.0
            win_count = 0
            loss_count = 0
            
            for i in range(len(backtest_df)):
                backtest_df.loc[backtest_df.index[i], "capital"] = current_capital
                
                # 检查是否有买入信号
                if backtest_df.iloc[i]["buy_signal"]:
                    # 如果有空头头寸，先平仓
                    if position_type == "short" and current_position > 0:
                        # 计算空头平仓盈亏
                        exit_price = backtest_df.iloc[i]["close"]
                        pnl = current_position * (entry_price - exit_price) * self.leverage
                        current_capital += pnl
                        cumulative_pnl += pnl
                        
                        # 更新统计
                        if pnl > 0:
                            win_count += 1
                        elif pnl < 0:
                            loss_count += 1
                        
                        # 记录平仓结果
                        backtest_df.loc[backtest_df.index[i], "trade_pnl"] = pnl
                    
                    # 计算新的多头头寸大小
                    entry_price = backtest_df.iloc[i]["close"]
                    stop_loss = backtest_df.iloc[i]["stop_loss_buy"]
                    
                    current_position = self.calculate_position_size(
                        entry_price, 
                        stop_loss, 
                        current_capital
                    )
                    
                    # 更新状态
                    position_type = "long"
                    backtest_df.loc[backtest_df.index[i], "position_size"] = current_position
                    backtest_df.loc[backtest_df.index[i], "position"] = "多头"
                    
                # 检查是否有卖出信号
                elif backtest_df.iloc[i]["sell_signal"]:
                    # 如果有多头头寸，先平仓
                    if position_type == "long" and current_position > 0:
                        # 计算多头平仓盈亏
                        exit_price = backtest_df.iloc[i]["close"]
                        pnl = current_position * (exit_price - entry_price) * self.leverage
                        current_capital += pnl
                        cumulative_pnl += pnl
                        
                        # 更新统计
                        if pnl > 0:
                            win_count += 1
                        elif pnl < 0:
                            loss_count += 1
                        
                        # 记录平仓结果
                        backtest_df.loc[backtest_df.index[i], "trade_pnl"] = pnl
                    
                    # 计算新的空头头寸大小
                    entry_price = backtest_df.iloc[i]["close"]
                    stop_loss = backtest_df.iloc[i]["stop_loss_sell"]
                    
                    current_position = self.calculate_position_size(
                        entry_price, 
                        stop_loss, 
                        current_capital
                    )
                    
                    # 更新状态
                    position_type = "short"
                    backtest_df.loc[backtest_df.index[i], "position_size"] = current_position
                    backtest_df.loc[backtest_df.index[i], "position"] = "空头"
                
                # 更新累计盈亏和净值
                backtest_df.loc[backtest_df.index[i], "cumulative_pnl"] = cumulative_pnl
                backtest_df.loc[backtest_df.index[i], "equity"] = self.initial_capital + cumulative_pnl
            
            # 计算汇总数据
            total_trades = win_count + loss_count
            win_rate = win_count / total_trades if total_trades > 0 else 0
            
            # 计算最大回撤
            backtest_df["drawdown"] = 1 - backtest_df["equity"] / backtest_df["equity"].cummax()
            max_drawdown = backtest_df["drawdown"].max()
            
            # 计算净值增长率
            growth_rate = (current_capital - self.initial_capital) / self.initial_capital
            
            # 保存汇总数据
            self.summary = {
                "初始资金": self.initial_capital,
                "当前资金": current_capital,
                "净利润": cumulative_pnl,
                "净利润率": growth_rate,
                "杠杆倍数": self.leverage,
                "交易次数": total_trades,
                "盈利次数": win_count,
                "亏损次数": loss_count,
                "胜率": win_rate,
                "最大回撤": max_drawdown
            }
            
            st.success("回测运行完成")
            return backtest_df
    
    def get_summary(self):
        """
        获取回测汇总结果
        
        返回:
            dict: 回测汇总结果
        """
        if hasattr(self, "summary"):
            return self.summary
        else:
            return {"error": "请先运行回测"}
    
    def export_results(self, backtest_df):
        """
        导出回测结果
        
        参数:
            backtest_df (pd.DataFrame): 回测结果数据
            
        返回:
            str: 导出的文件路径
        """
        # 创建文件名
        symbol = self.config.get("trading", {}).get("symbol", "Unknown").replace("/", "_")
        timeframe = self.config.get("trading", {}).get("timeframe", "Unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_{symbol}_{timeframe}_{timestamp}.csv"
        
        # 导出CSV
        try:
            backtest_df.to_csv(filename)
            return filename
        except Exception as e:
            st.error(f"导出失败: {e}")
            return None

def render_backtest_ui(config, signals_df=None):
    """
    渲染回测UI界面
    
    参数:
        config (dict): 配置信息
        signals_df (pd.DataFrame): 包含信号的数据
        
    返回:
        pd.DataFrame: 回测结果
    """
    st.header("回测分析")
    
    if signals_df is None:
        if "signals_df" in st.session_state:
            signals_df = st.session_state.signals_df
        else:
            st.warning("请先生成交易信号")
            return None
    
    # 创建回测引擎
    backtest_engine = BacktestEngine(config)
    
    # 显示当前配置
    backtest_config = config.get("backtest", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("初始资金", f"{backtest_config.get('initial_capital', 10000):,.0f} USDT")
    col2.metric("杠杆倍数", f"{backtest_config.get('leverage', 1)}倍")
    col3.metric("风险比例", f"{backtest_config.get('risk_per_trade', 0.02):.1%}")
    
    # 运行回测按钮
    if st.button("运行回测"):
        # 运行回测
        backtest_df = backtest_engine.run_backtest(signals_df)
        
        # 保存到session_state
        st.session_state.backtest_df = backtest_df
        
        # 获取汇总结果
        summary = backtest_engine.get_summary()
        
        # 显示回测结果
        st.subheader("回测结果")
        
        # 结果指标
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("净利润", f"{summary['净利润']:,.2f} USDT")
        col2.metric("净利润率", f"{summary['净利润率']:.2%}")
        col3.metric("胜率", f"{summary['胜率']:.2%}")
        col4.metric("最大回撤", f"{summary['最大回撤']:.2%}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("交易次数", summary["交易次数"])
        col2.metric("盈利次数", summary["盈利次数"])
        col3.metric("亏损次数", summary["亏损次数"])
        col4.metric("盈亏比", f"{float(summary['盈利次数'])/float(summary['亏损次数']):.2f}" if int(summary['亏损次数']) > 0 else "")
        
        # 导出按钮
        if st.button("导出回测结果"):
            filename = backtest_engine.export_results(backtest_df)
            if filename:
                st.success(f"回测结果已导出至 {filename}")
        
        return backtest_df
    
    # 如果已经运行过回测
    if "backtest_df" in st.session_state:
        backtest_df = st.session_state.backtest_df
        st.success(f"已加载回测数据 ({len(backtest_df)} 条记录)")
        
        # 创建回测引擎以便获取汇总数据
        backtest_engine = BacktestEngine(config)
        backtest_engine.run_backtest(backtest_df)
        summary = backtest_engine.get_summary()
        
        # 显示汇总按钮
        if st.button("显示回测汇总"):
            # 显示回测结果
            st.subheader("回测结果")
            
            # 结果指标
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("净利润", f"{summary['净利润']:,.2f} USDT")
            col2.metric("净利润率", f"{summary['净利润率']:.2%}")
            col3.metric("胜率", f"{summary['胜率']:.2%}")
            col4.metric("最大回撤", f"{summary['最大回撤']:.2%}")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("交易次数", summary["交易次数"])
            col2.metric("盈利次数", summary["盈利次数"])
            col3.metric("亏损次数", summary["亏损次数"])
            col4.metric("盈亏比", f"{float(summary['盈利次数'])/float(summary['亏损次数']):.2f}" if int(summary['亏损次数']) > 0 else "")
        
        return backtest_df
    
    return None

if __name__ == "__main__":
    # 测试回测UI
    import yaml
    from signals import SignalGenerator
    from indicators import IndicatorCalculator
    
    st.set_page_config(page_title="阿翔趋势交易系统 - 回测分析", layout="wide")
    
    # 加载配置
    with open("config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    
    # 创建测试数据
    if "signals_df" not in st.session_state:
        # 创建模拟数据
        date_range = pd.date_range(start="2023-01-01", periods=200, freq="H")
        data = {
            "open": np.random.normal(10000, 500, len(date_range)),
            "high": np.random.normal(10100, 500, len(date_range)),
            "low": np.random.normal(9900, 500, len(date_range)),
            "close": np.random.normal(10050, 500, len(date_range)),
            "volume": np.random.normal(100, 20, len(date_range))
        }
        df = pd.DataFrame(data, index=date_range)
        
        # 计算指标
        indicator_calculator = IndicatorCalculator(config)
        indicators_df = indicator_calculator.calculate_all_indicators(df)
        
        # 生成信号
        signal_generator = SignalGenerator(config)
        signals_df = signal_generator.generate_signals(indicators_df)
        signals_df = signal_generator.calculate_risk_reward(signals_df)
        
        st.session_state.signals_df = signals_df
    
    render_backtest_ui(config)
