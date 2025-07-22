import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from main import run_strategy
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
matplotlib.rcParams['axes.unicode_minus'] = False

def analyze_signals_performance():
    """
    分析交易信号的表现，统计假信号和真信号
    """
    print("开始运行策略回测...")
    
    # 运行策略，使用本地数据
    data_file = "data/BTC_USDT_1h_20190101_to_20250627.csv"
    signals_df = run_strategy(data_source=data_file, visualize=False)
    
    if signals_df is None:
        print("无法获取回测数据")
        return
    
    print("分析交易信号表现...")
    
    # 从signals_df中提取交易记录
    trade_actions = signals_df[signals_df['trade_action'].notna()].copy()
    
    # 分析交易结果
    results = analyze_trade_results(trade_actions)
    
    # 生成报告
    generate_signal_analysis_report(results)
    
    return results

def analyze_trade_results(trade_actions):
    """
    分析交易结果，统计假信号和真信号
    """
    results = {
        'total_trades': 0,
        'profitable_trades': 0,  # 真信号（止盈平仓）
        'losing_trades': 0,      # 假信号（止损平仓）
        'signal_flat_trades': 0, # 信号平仓
        'consecutive_losses': [], # 连续假信号记录
        'consecutive_wins': [],   # 连续真信号记录
        'trade_details': [],      # 详细交易记录
        'false_signal_reasons': []  # 假信号原因分析
    }
    
    current_streak = 0  # 当前连续状态
    current_streak_type = None  # 'win' or 'loss'
    
    # 遍历交易动作
    i = 0
    while i < len(trade_actions):
        action = trade_actions.iloc[i]
        
        # 如果是开仓动作，寻找对应的平仓动作
        if action['trade_action'] in ['开多', '开空']:
            # 查找对应的平仓动作
            close_action = None
            for j in range(i + 1, len(trade_actions)):
                next_action = trade_actions.iloc[j]
                if next_action['trade_action'] in ['止盈平多', '止损平多', '信号平多', '止盈平空', '止损平空', '信号平空']:
                    close_action = next_action
                    break
            
            if close_action is not None:
                results['total_trades'] += 1
                
                # 分析交易结果
                trade_detail = {
                    'entry_time': action.name,
                    'exit_time': close_action.name,
                    'entry_action': action['trade_action'],
                    'exit_action': close_action['trade_action'],
                    'entry_price': action['close'],
                    'exit_price': close_action['close'],
                    'pnl': close_action['trade_pnl'],
                    'entry_dema144_dist': abs(action['close'] - action['dema144']) if 'dema144' in action else 0,
                    'entry_dema169_dist': abs(action['close'] - action['dema169']) if 'dema169' in action else 0,
                    'supertrend_strength': action.get('supertrend_upper', 0) - action.get('supertrend_lower', 0) if 'supertrend_upper' in action else 0
                }
                
                results['trade_details'].append(trade_detail)
                
                # 判断交易类型
                if '止盈' in close_action['trade_action']:
                    results['profitable_trades'] += 1
                    trade_type = 'win'
                elif '止损' in close_action['trade_action']:
                    results['losing_trades'] += 1
                    trade_type = 'loss'
                    
                    # 分析假信号原因
                    reason = analyze_false_signal_reason(action, close_action, trade_actions)
                    results['false_signal_reasons'].append(reason)
                else:
                    results['signal_flat_trades'] += 1
                    trade_type = 'neutral'
                
                # 计算连续统计
                if trade_type == current_streak_type:
                    current_streak += 1
                else:
                    # 保存之前的连续记录
                    if current_streak > 0:
                        if current_streak_type == 'win':
                            results['consecutive_wins'].append(current_streak)
                        elif current_streak_type == 'loss':
                            results['consecutive_losses'].append(current_streak)
                    
                    # 开始新的连续记录
                    current_streak = 1
                    current_streak_type = trade_type
        
        i += 1
    
    # 保存最后的连续记录
    if current_streak > 0:
        if current_streak_type == 'win':
            results['consecutive_wins'].append(current_streak)
        elif current_streak_type == 'loss':
            results['consecutive_losses'].append(current_streak)
    
    return results

def analyze_false_signal_reason(entry_action, exit_action, trade_actions):
    """
    分析假信号的原因
    """
    reason = {
        'entry_time': entry_action.name,
        'entry_action': entry_action['trade_action'],
        'exit_action': exit_action['trade_action'],
        'potential_causes': []
    }
    
    # 1. 分析DEMA距离 - 如果开仓时距离DEMA线太近，可能是假突破
    if 'dema144' in entry_action and 'dema169' in entry_action:
        dema144_dist = abs(entry_action['close'] - entry_action['dema144'])
        dema169_dist = abs(entry_action['close'] - entry_action['dema169'])
        
        if dema144_dist < entry_action['close'] * 0.005:  # 距离小于0.5%
            reason['potential_causes'].append('距离DEMA144太近，可能是假突破')
        if dema169_dist < entry_action['close'] * 0.005:  # 距离小于0.5%
            reason['potential_causes'].append('距离DEMA169太近，可能是假突破')
    
    # 2. 分析市场波动性
    if 'high' in entry_action and 'low' in entry_action:
        volatility = (entry_action['high'] - entry_action['low']) / entry_action['close']
        if volatility > 0.03:  # 波动超过3%
            reason['potential_causes'].append(f'市场波动性过高({volatility:.2%})，容易触发止损')
    
    # 3. 分析Supertrend强度
    if 'supertrend_upper' in entry_action and 'supertrend_lower' in entry_action:
        supertrend_range = entry_action['supertrend_upper'] - entry_action['supertrend_lower']
        if supertrend_range > entry_action['close'] * 0.1:  # 范围超过10%
            reason['potential_causes'].append('Supertrend通道过宽，趋势不够明确')
    
    # 4. 如果没有找到明显原因，标记为市场噪音
    if not reason['potential_causes']:
        reason['potential_causes'].append('市场短期噪音或趋势反转')
    
    return reason

def generate_signal_analysis_report(results):
    """
    生成信号分析报告
    """
    print("\n" + "="*60)
    print("           交易信号分析报告")
    print("="*60)
    
    # 基本统计
    total = results['total_trades']
    profitable = results['profitable_trades']
    losing = results['losing_trades']
    signal_flat = results['signal_flat_trades']
    
    print(f"\n📊 基本统计:")
    print(f"总交易次数: {total}")
    print(f"真信号(止盈): {profitable} ({profitable/total*100:.1f}%)")
    print(f"假信号(止损): {losing} ({losing/total*100:.1f}%)")
    print(f"信号平仓: {signal_flat} ({signal_flat/total*100:.1f}%)")
    
    # 连续假信号分析
    consecutive_losses = results['consecutive_losses']
    if consecutive_losses:
        print(f"\n🔴 连续假信号分析:")
        print(f"最大连续假信号次数: {max(consecutive_losses)}")
        print(f"平均连续假信号次数: {np.mean(consecutive_losses):.1f}")
        print(f"连续假信号分布: {dict(pd.Series(consecutive_losses).value_counts().sort_index())}")
    
    # 连续真信号分析
    consecutive_wins = results['consecutive_wins']
    if consecutive_wins:
        print(f"\n🟢 连续真信号分析:")
        print(f"最大连续真信号次数: {max(consecutive_wins)}")
        print(f"平均连续真信号次数: {np.mean(consecutive_wins):.1f}")
        print(f"连续真信号分布: {dict(pd.Series(consecutive_wins).value_counts().sort_index())}")
    
    # 假信号原因分析
    print(f"\n🔍 假信号原因分析:")
    all_causes = []
    for reason in results['false_signal_reasons']:
        all_causes.extend(reason['potential_causes'])
    
    cause_counts = pd.Series(all_causes).value_counts()
    for cause, count in cause_counts.items():
        print(f"  • {cause}: {count}次 ({count/losing*100:.1f}%)")
    
    # 距离分析
    if results['trade_details']:
        print(f"\n📏 开仓位置分析:")
        trade_df = pd.DataFrame(results['trade_details'])
        
        # 按盈利情况分组分析
        profitable_trades = trade_df[trade_df['pnl'] > 0]
        losing_trades = trade_df[trade_df['pnl'] <= 0]
        
        if len(profitable_trades) > 0:
            print(f"盈利交易平均距DEMA144: {profitable_trades['entry_dema144_dist'].mean():.2f}")
            print(f"盈利交易平均距DEMA169: {profitable_trades['entry_dema169_dist'].mean():.2f}")
        
        if len(losing_trades) > 0:
            print(f"亏损交易平均距DEMA144: {losing_trades['entry_dema144_dist'].mean():.2f}")
            print(f"亏损交易平均距DEMA169: {losing_trades['entry_dema169_dist'].mean():.2f}")
    
    print("\n" + "="*60)
    
    # 生成可视化图表
    create_analysis_charts(results)

def create_analysis_charts(results):
    """
    创建分析图表
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('交易信号分析图表', fontsize=16)
    
    # 1. 信号成功率饼图
    labels = ['真信号(止盈)', '假信号(止损)', '信号平仓']
    sizes = [results['profitable_trades'], results['losing_trades'], results['signal_flat_trades']]
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    
    axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('信号成功率分布')
    
    # 2. 连续假信号分布
    if results['consecutive_losses']:
        loss_dist = pd.Series(results['consecutive_losses']).value_counts().sort_index()
        axes[0, 1].bar(loss_dist.index, loss_dist.values, color='#e74c3c', alpha=0.7)
        axes[0, 1].set_title('连续假信号次数分布')
        axes[0, 1].set_xlabel('连续假信号次数')
        axes[0, 1].set_ylabel('发生频率')
    
    # 3. 连续真信号分布
    if results['consecutive_wins']:
        win_dist = pd.Series(results['consecutive_wins']).value_counts().sort_index()
        axes[1, 0].bar(win_dist.index, win_dist.values, color='#2ecc71', alpha=0.7)
        axes[1, 0].set_title('连续真信号次数分布')
        axes[1, 0].set_xlabel('连续真信号次数')
        axes[1, 0].set_ylabel('发生频率')
    
    # 4. 假信号原因分析
    all_causes = []
    for reason in results['false_signal_reasons']:
        all_causes.extend(reason['potential_causes'])
    
    if all_causes:
        cause_counts = pd.Series(all_causes).value_counts().head(5)  # 显示前5个原因
        axes[1, 1].barh(range(len(cause_counts)), cause_counts.values, color='#f39c12', alpha=0.7)
        axes[1, 1].set_yticks(range(len(cause_counts)))
        axes[1, 1].set_yticklabels([cause[:20] + '...' if len(cause) > 20 else cause 
                                   for cause in cause_counts.index])
        axes[1, 1].set_title('假信号主要原因')
        axes[1, 1].set_xlabel('发生次数')
    
    plt.tight_layout()
    plt.savefig('signal_analysis_charts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"图表已保存为: signal_analysis_charts.png")

if __name__ == "__main__":
    results = analyze_signals_performance()
