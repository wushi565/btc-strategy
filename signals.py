import pandas as pd
import numpy as np
import streamlit as st
from indicators import IndicatorCalculator

class SignalGenerator:
    """信号生成器类，负责生成交易信号"""
    
    def __init__(self, config):
        """
        初始化信号生成器
        
        参数:
            config (dict): 配置信息
        """
        self.config = config
        self.signal_config = config.get("signals", {})
        
        # 默认风险收益比
        self.risk_reward_ratio = self.signal_config.get("risk_reward_ratio", 3.0)
        
    def generate_signals(self, df):
        """
        生成交易信号
        
        参数:
            df (pd.DataFrame): 包含技术指标的数据
            
        返回:
            pd.DataFrame: 添加了交易信号的数据
        """
        with st.spinner("正在生成交易信号..."):
            # 创建副本避免修改原始数据
            result_df = df.copy()
            
            # 初始化买卖信号列
            result_df['buy_signal'] = False
            result_df['sell_signal'] = False
            
            # 计算买入信号
            for i in range(1, len(result_df)):
                # 买入条件: 趋势转为上升且价格在均线上方
                result_df.loc[result_df.index[i], 'buy_signal'] = (
                    result_df['trend'].iloc[i] == 1 and 
                    result_df['trend'].iloc[i-1] == -1 and
                    result_df['close'].iloc[i] > result_df['dema_144'].iloc[i] and 
                    result_df['close'].iloc[i] > result_df['dema_169'].iloc[i]
                )
                
                # 卖出条件: 趋势转为下降且价格在均线下方
                result_df.loc[result_df.index[i], 'sell_signal'] = (
                    result_df['trend'].iloc[i] == -1 and 
                    result_df['trend'].iloc[i-1] == 1 and
                    result_df['close'].iloc[i] < result_df['dema_144'].iloc[i] and 
                    result_df['close'].iloc[i] < result_df['dema_169'].iloc[i]
                )
            
            st.success("交易信号生成完成")
            return result_df
    
    def calculate_risk_reward(self, df):
        """
        计算风险收益比和止损止盈价格
        
        参数:
            df (pd.DataFrame): 包含交易信号的数据
            
        返回:
            pd.DataFrame: 添加了风险收益计算的数据
        """
        with st.spinner("正在计算风险收益比..."):
            # 创建副本避免修改原始数据
            result_df = df.copy()
            
            # 初始化止损止盈列
            result_df['stop_loss_buy'] = np.nan
            result_df['take_profit_buy'] = np.nan
            result_df['stop_loss_sell'] = np.nan
            result_df['take_profit_sell'] = np.nan
            
            # 计算买入信号的止损止盈
            buy_signals = result_df[result_df['buy_signal']]
            for idx in buy_signals.index:
                # 买入信号的止损设为DEMA169
                result_df.loc[idx, 'stop_loss_buy'] = result_df.loc[idx, 'dema_169']
                
                # 计算买入风险
                entry_price = result_df.loc[idx, 'close']
                stop_loss = result_df.loc[idx, 'stop_loss_buy']
                risk = entry_price - stop_loss
                
                # 计算止盈价格 (风险收益比)
                take_profit = entry_price + (risk * self.risk_reward_ratio)
                result_df.loc[idx, 'take_profit_buy'] = take_profit
            
            # 计算卖出信号的止损止盈
            sell_signals = result_df[result_df['sell_signal']]
            for idx in sell_signals.index:
                # 卖出信号的止损设为DEMA144
                result_df.loc[idx, 'stop_loss_sell'] = result_df.loc[idx, 'dema_144']
                
                # 计算卖出风险
                entry_price = result_df.loc[idx, 'close']
                stop_loss = result_df.loc[idx, 'stop_loss_sell']
                risk = stop_loss - entry_price
                
                # 计算止盈价格 (风险收益比)
                take_profit = entry_price - (risk * self.risk_reward_ratio)
                result_df.loc[idx, 'take_profit_sell'] = take_profit
                
            st.success("风险收益比计算完成")
            return result_df
    
    def summarize_signals(self, df):
        """
        统计交易信号
        
        参数:
            df (pd.DataFrame): 包含交易信号的数据
            
        返回:
            dict: 信号统计结果
        """
        # 提取买卖信号
        buy_signals = df[df['buy_signal']]
        sell_signals = df[df['sell_signal']]
        
        # 统计信号数量
        total_signals = len(buy_signals) + len(sell_signals)
        
        # 创建汇总信息
        summary = {
            "总信号数": total_signals,
            "买入信号数": len(buy_signals),
            "卖出信号数": len(sell_signals),
            "最新趋势": "上升" if df['trend'].iloc[-1] == 1 else "下降",
            "买入信号": [],
            "卖出信号": []
        }
        
        # 获取最近的买入信号
        recent_buy_signals = buy_signals.tail(5)
        for idx, row in recent_buy_signals.iterrows():
            signal_info = {
                "时间": idx.strftime('%Y-%m-%d %H:%M:%S'),
                "价格": row['close'],
                "止损": row['stop_loss_buy'],
                "止盈": row['take_profit_buy'],
                "风险比": f"1:{self.risk_reward_ratio:.1f}"
            }
            summary["买入信号"].append(signal_info)
        
        # 获取最近的卖出信号
        recent_sell_signals = sell_signals.tail(5)
        for idx, row in recent_sell_signals.iterrows():
            signal_info = {
                "时间": idx.strftime('%Y-%m-%d %H:%M:%S'),
                "价格": row['close'],
                "止损": row['stop_loss_sell'],
                "止盈": row['take_profit_sell'],
                "风险比": f"1:{self.risk_reward_ratio:.1f}"
            }
            summary["卖出信号"].append(signal_info)
        
        return summary

def render_signal_ui(config, indicators_df=None):
    """
    渲染信号分析UI界面
    
    参数:
        config (dict): 配置信息
        indicators_df (pd.DataFrame): 包含指标的数据
        
    返回:
        pd.DataFrame: 带有信号的DataFrame
    """
    st.header("信号分析")
    
    if indicators_df is None:
        if 'indicators_df' in st.session_state:
            indicators_df = st.session_state.indicators_df
        else:
            st.warning("请先计算技术指标")
            return None
    
    # 创建信号生成器
    signal_generator = SignalGenerator(config)
    
    # 显示当前配置
    signal_config = config.get("signals", {})
    st.info(f"风险收益比: 1:{signal_config.get('risk_reward_ratio', 3.0)}")
    
    # 生成信号按钮
    if st.button("生成交易信号"):
        # 生成信号
        signals_df = signal_generator.generate_signals(indicators_df)
        
        # 计算风险收益比
        signals_df = signal_generator.calculate_risk_reward(signals_df)
        
        # 保存到session_state
        st.session_state.signals_df = signals_df
        
        # 统计信号
        signal_summary = signal_generator.summarize_signals(signals_df)
        
        # 显示信号汇总
        st.subheader("信号统计")
        col1, col2, col3 = st.columns(3)
        col1.metric("总信号数", signal_summary["总信号数"])
        col2.metric("买入信号", signal_summary["买入信号数"])
        col3.metric("卖出信号", signal_summary["卖出信号数"])
        
        # 显示最新趋势
        st.info(f"当前趋势: {signal_summary['最新趋势']}")
        
        # 显示信号表格
        if signal_summary["买入信号"]:
            st.subheader("最近买入信号")
            buy_df = pd.DataFrame(signal_summary["买入信号"])
            st.dataframe(buy_df)
        
        if signal_summary["卖出信号"]:
            st.subheader("最近卖出信号")
            sell_df = pd.DataFrame(signal_summary["卖出信号"])
            st.dataframe(sell_df)
        
        return signals_df
    
    # 如果已经生成过信号
    if 'signals_df' in st.session_state:
        signals_df = st.session_state.signals_df
        st.success(f"已加载信号数据 ({len(signals_df)} 条记录)")
        
        # 显示信号统计按钮
        if st.button("显示信号统计"):
            # 统计信号
            signal_summary = signal_generator.summarize_signals(signals_df)
            
            # 显示信号汇总
            st.subheader("信号统计")
            col1, col2, col3 = st.columns(3)
            col1.metric("总信号数", signal_summary["总信号数"])
            col2.metric("买入信号", signal_summary["买入信号数"])
            col3.metric("卖出信号", signal_summary["卖出信号数"])
            
            # 显示最新趋势
            st.info(f"当前趋势: {signal_summary['最新趋势']}")
            
            # 显示信号表格
            if signal_summary["买入信号"]:
                st.subheader("最近买入信号")
                buy_df = pd.DataFrame(signal_summary["买入信号"])
                st.dataframe(buy_df)
            
            if signal_summary["卖出信号"]:
                st.subheader("最近卖出信号")
                sell_df = pd.DataFrame(signal_summary["卖出信号"])
                st.dataframe(sell_df)
        
        return signals_df
    
    return None

if __name__ == "__main__":
    # 测试信号分析UI
    import yaml
    
    st.set_page_config(page_title="阿翔趋势交易系统 - 信号分析", layout="wide")
    
    # 加载配置
    with open("config.yaml", 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    # 创建测试数据
    if 'indicators_df' not in st.session_state:
        # 创建模拟数据
        date_range = pd.date_range(start='2023-01-01', periods=200, freq='H')
        data = {
            'open': np.random.normal(10000, 500, len(date_range)),
            'high': np.random.normal(10100, 500, len(date_range)),
            'low': np.random.normal(9900, 500, len(date_range)),
            'close': np.random.normal(10050, 500, len(date_range)),
            'volume': np.random.normal(100, 20, len(date_range))
        }
        df = pd.DataFrame(data, index=date_range)
        
        # 计算指标
        try:
            indicator_calculator = IndicatorCalculator(config)
            indicators_df = indicator_calculator.calculate_all_indicators(df)
            st.session_state.indicators_df = indicators_df
        except Exception as e:
            st.error(f"计算指标错误: {e}")
            indicators_df = None
    
    render_signal_ui(config)
