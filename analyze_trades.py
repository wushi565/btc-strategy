import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import datetime
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
matplotlib.rcParams['axes.unicode_minus'] = False

def analyze_trade_signals(excel_file):
    """
    分析Excel中的交易记录，统计真假信号情况
    """
    print(f"分析交易记录: {excel_file}")
    
    try:
        # 读取交易记录sheet
        trade_df = pd.read_excel(excel_file, sheet_name='交易记录')
        
        if trade_df.empty:
            print("交易记录为空")
            return
        
        # 筛选平仓记录
        close_trades = trade_df[trade_df['操作类型'].str.contains('平')]
        
        if close_trades.empty:
            print("没有找到平仓记录")
            return
        
        print(f"共找到 {len(close_trades)} 条平仓记录")
        
        # 按平仓原因分类
        take_profit_trades = close_trades[close_trades['操作类型'].str.contains('止盈')]
        stop_loss_trades = close_trades[close_trades['操作类型'].str.contains('止损')]
        signal_close_trades = close_trades[close_trades['操作类型'].str.contains('信号')]
        
        # 计算真假信号数量
        true_signals = len(take_profit_trades)
        false_signals = len(stop_loss_trades)
        neutral_signals = len(signal_close_trades)
        
        print(f"\n===== 信号分析 =====")
        print(f"真信号(止盈平仓): {true_signals} ({true_signals/len(close_trades)*100:.1f}%)")
        print(f"假信号(止损平仓): {false_signals} ({false_signals/len(close_trades)*100:.1f}%)")
        print(f"中性信号(信号平仓): {neutral_signals} ({neutral_signals/len(close_trades)*100:.1f}%)")
        
        # 计算连续信号统计
        # 创建交易结果序列: 1表示真信号(盈利), 0表示假信号(亏损)
        results = []
        for _, trade in close_trades.iterrows():
            if '止盈' in trade['操作类型']:
                results.append(1)  # 真信号
            elif '止损' in trade['操作类型']:
                results.append(0)  # 假信号
            else:
                # 对于信号平仓，检查盈亏
                if trade['盈亏'] > 0:
                    results.append(1)  # 算作真信号
                else:
                    results.append(0)  # 算作假信号
        
        if not results:
            print("无法分析连续信号")
            return
        
        # 分析连续假信号
        false_streaks = get_consecutive_streaks(results, 0)
        if false_streaks:
            print(f"\n===== 连续假信号分析 =====")
            print(f"最大连续假信号次数: {max(false_streaks)}")
            print(f"平均连续假信号次数: {np.mean(false_streaks):.2f}")
            print(f"连续假信号分布: {dict(pd.Series(false_streaks).value_counts().sort_index())}")
        
        # 分析连续真信号
        true_streaks = get_consecutive_streaks(results, 1)
        if true_streaks:
            print(f"\n===== 连续真信号分析 =====")
            print(f"最大连续真信号次数: {max(true_streaks)}")
            print(f"平均连续真信号次数: {np.mean(true_streaks):.2f}")
            print(f"连续真信号分布: {dict(pd.Series(true_streaks).value_counts().sort_index())}")
        
        # 分析多空方向的真假信号
        long_trades = close_trades[close_trades['操作类型'].str.contains('平多')]
        short_trades = close_trades[close_trades['操作类型'].str.contains('平空')]
        
        # 多头统计
        if not long_trades.empty:
            long_true = len(long_trades[long_trades['操作类型'].str.contains('止盈')])
            long_false = len(long_trades[long_trades['操作类型'].str.contains('止损')])
            print(f"\n===== 多头信号分析 =====")
            print(f"多头总交易: {len(long_trades)}")
            print(f"多头真信号: {long_true} ({long_true/len(long_trades)*100:.1f}%)")
            print(f"多头假信号: {long_false} ({long_false/len(long_trades)*100:.1f}%)")
        
        # 空头统计
        if not short_trades.empty:
            short_true = len(short_trades[short_trades['操作类型'].str.contains('止盈')])
            short_false = len(short_trades[short_trades['操作类型'].str.contains('止损')])
            print(f"\n===== 空头信号分析 =====")
            print(f"空头总交易: {len(short_trades)}")
            print(f"空头真信号: {short_true} ({short_true/len(short_trades)*100:.1f}%)")
            print(f"空头假信号: {short_false} ({short_false/len(short_trades)*100:.1f}%)")
        
        # 分析假信号原因
        if not stop_loss_trades.empty:
            print(f"\n===== 假信号原因分析 =====")
            analyze_false_signals(stop_loss_trades, excel_file)
        
        # 可视化
        create_signal_charts(true_signals, false_signals, neutral_signals, false_streaks, true_streaks)
        
        return {
            "true_signals": true_signals,
            "false_signals": false_signals,
            "neutral_signals": neutral_signals,
            "false_streaks": false_streaks,
            "true_streaks": true_streaks,
            "long_true": long_true if not long_trades.empty else 0,
            "long_false": long_false if not long_trades.empty else 0,
            "short_true": short_true if not short_trades.empty else 0,
            "short_false": short_false if not short_trades.empty else 0
        }
        
    except Exception as e:
        print(f"分析过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_consecutive_streaks(results, value):
    """
    计算列表中连续出现value值的序列长度
    """
    streaks = []
    current_streak = 0
    
    for item in results:
        if item == value:
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append(current_streak)
                current_streak = 0
    
    # 添加最后一个连续序列
    if current_streak > 0:
        streaks.append(current_streak)
    
    return streaks

def analyze_false_signals(false_trades, excel_file):
    """
    分析假信号的可能原因
    """
    # 统计各种原因
    reasons = []
    
    # 检查是否有距离数据
    has_distance_data = False
    try:
        # 尝试读取距离分析数据
        dist_df144 = pd.read_excel(excel_file, sheet_name='距离分析144')
        dist_df169 = pd.read_excel(excel_file, sheet_name='距离分析169')
        has_distance_data = True
    except:
        has_distance_data = False
    
    # 1. 检查假信号与价格距离DEMA线的关系
    if has_distance_data:
        print("距离分析:")
        # DEMA144距离与盈亏关系
        print("\nDEMA144距离与平均盈亏关系:")
        for _, row in dist_df144.iterrows():
            print(f"距离 {row['距离组144']} - 平均盈亏: {row['平均盈亏']:.2f}")
        
        # DEMA169距离与盈亏关系
        print("\nDEMA169距离与平均盈亏关系:")
        for _, row in dist_df169.iterrows():
            print(f"距离 {row['距离组169']} - 平均盈亏: {row['平均盈亏']:.2f}")
        
        # 分析结论
        poor_dist144 = dist_df144[dist_df144['平均盈亏'] < 0]
        if not poor_dist144.empty:
            min_dist = poor_dist144['距离组144'].min()
            reasons.append(f"当价格距离DEMA144小于{min_dist}时，容易出现假信号")
        
        poor_dist169 = dist_df169[dist_df169['平均盈亏'] < 0]
        if not poor_dist169.empty:
            min_dist = poor_dist169['距离组169'].min()
            reasons.append(f"当价格距离DEMA169小于{min_dist}时，容易出现假信号")
    
    # 2. 分析是否有市场环境因素
    # 按交易时间排序
    false_trades_sorted = false_trades.sort_values(by='时间')
    
    # 检查止损间隔
    times = false_trades_sorted['时间'].tolist()
    if len(times) > 1:
        intervals = [(times[i+1] - times[i]).total_seconds()/3600 for i in range(len(times)-1)]
        short_intervals = [i for i in intervals if i < 24]  # 小于24小时的间隔
        
        if short_intervals and len(short_intervals) > 0.3 * len(intervals):
            reasons.append(f"有{len(short_intervals)}次假信号在24小时内连续出现，可能是市场波动性增加")
    
    # 3. 检查持仓时间与假信号的关系
    if 'holding_time' in false_trades.columns or '持仓时间(小时)' in false_trades.columns:
        holding_key = '持仓时间(小时)' if '持仓时间(小时)' in false_trades.columns else 'holding_time'
        avg_holding = false_trades[holding_key].mean()
        
        if avg_holding < 12:
            reasons.append(f"假信号平均持仓时间较短({avg_holding:.2f}小时)，可能是短期市场噪音导致")
        elif avg_holding > 72:
            reasons.append(f"假信号平均持仓时间较长({avg_holding:.2f}小时)，可能是大趋势反转")
    
    # 4. 多空方向分析
    long_false = false_trades[false_trades['操作类型'].str.contains('平多')].shape[0]
    short_false = false_trades[false_trades['操作类型'].str.contains('平空')].shape[0]
    
    if long_false > 1.5 * short_false:
        reasons.append(f"多头假信号明显多于空头({long_false}:{short_false})，可能做多条件过于宽松")
    elif short_false > 1.5 * long_false:
        reasons.append(f"空头假信号明显多于多头({short_false}:{long_false})，可能做空条件过于宽松")
    
    # 5. 根据时间分布分析
    if '时间' in false_trades.columns and len(false_trades) > 5:
        false_trades['year'] = false_trades['时间'].dt.year
        false_trades['month'] = false_trades['时间'].dt.month
        
        year_counts = false_trades.groupby('year').size()
        month_counts = false_trades.groupby('month').size()
        
        # 检查是否某些时间段假信号特别多
        if year_counts.max() > year_counts.mean() * 1.5:
            worst_year = year_counts.idxmax()
            reasons.append(f"{worst_year}年假信号特别多，可能是特殊市场环境")
        
        if month_counts.max() > month_counts.mean() * 1.5:
            worst_month = month_counts.idxmax()
            month_names = {1:'一月', 2:'二月', 3:'三月', 4:'四月', 5:'五月', 6:'六月',
                           7:'七月', 8:'八月', 9:'九月', 10:'十月', 11:'十一月', 12:'十二月'}
            reasons.append(f"{month_names[worst_month]}的假信号特别多，可能存在季节性因素")
    
    # 输出原因分析
    if reasons:
        print("\n可能的假信号原因:")
        for i, reason in enumerate(reasons, 1):
            print(f"{i}. {reason}")
    else:
        print("没有找到明显的假信号原因模式")
    
    return reasons

def create_signal_charts(true_signals, false_signals, neutral_signals, false_streaks, true_streaks):
    """
    创建信号分析图表
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('交易信号分析图表', fontsize=16)
    
    # 1. 信号成功率饼图
    labels = ['真信号(止盈)', '假信号(止损)', '中性信号(信号平仓)']
    sizes = [true_signals, false_signals, neutral_signals]
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    
    axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('信号成功率分布')
    
    # 2. 连续假信号分布
    if false_streaks:
        loss_dist = pd.Series(false_streaks).value_counts().sort_index()
        axes[0, 1].bar(loss_dist.index, loss_dist.values, color='#e74c3c', alpha=0.7)
        axes[0, 1].set_title('连续假信号次数分布')
        axes[0, 1].set_xlabel('连续假信号次数')
        axes[0, 1].set_ylabel('发生频率')
    else:
        axes[0, 1].text(0.5, 0.5, '无假信号数据', ha='center', va='center')
    
    # 3. 连续真信号分布
    if true_streaks:
        win_dist = pd.Series(true_streaks).value_counts().sort_index()
        axes[1, 0].bar(win_dist.index, win_dist.values, color='#2ecc71', alpha=0.7)
        axes[1, 0].set_title('连续真信号次数分布')
        axes[1, 0].set_xlabel('连续真信号次数')
        axes[1, 0].set_ylabel('发生频率')
    else:
        axes[1, 0].text(0.5, 0.5, '无真信号数据', ha='center', va='center')
    
    # 4. 真假信号比较条形图
    axes[1, 1].bar(['真信号', '假信号'], [true_signals, false_signals], color=['#2ecc71', '#e74c3c'])
    axes[1, 1].set_title('真假信号数量对比')
    axes[1, 1].set_ylabel('信号数量')
    
    plt.tight_layout()
    plt.savefig('signal_analysis_chart.png', dpi=300, bbox_inches='tight')
    print("分析图表已保存为: signal_analysis_chart.png")
    plt.show()

def get_latest_trade_file():
    """获取最新的交易记录文件"""
    trade_files = []
    for file in os.listdir('trades'):
        if file.endswith('.xlsx'):
            full_path = os.path.join('trades', file)
            mod_time = os.path.getmtime(full_path)
            trade_files.append((full_path, mod_time))
    
    # 按修改时间排序，最新的在前
    trade_files.sort(key=lambda x: x[1], reverse=True)
    
    if trade_files:
        return trade_files[0][0]  # 返回最新文件的路径
    else:
        return None

def main():
    """主函数"""
    print("\n===== 交易信号分析工具 =====")
    
    # 自动获取最新的交易记录文件
    latest_file = get_latest_trade_file()
    
    if latest_file:
        print(f"自动选择最新的交易记录文件: {latest_file}")
        analyze_trade_signals(latest_file)
    else:
        print("没有找到交易记录文件")

if __name__ == "__main__":
    main() 