import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
matplotlib.rcParams['axes.unicode_minus'] = False

class SignalAnalyzer:
    """信号分析器，用于分析交易信号的质量和特征"""
    
    def __init__(self):
        """初始化信号分析器"""
        self.analysis_results = {}
    
    def analyze_signals(self, close_trades):
        """
        分析交易信号质量
        
        参数:
            close_trades (pd.DataFrame): 平仓交易记录
            
        返回:
            dict: 信号分析结果
        """
        if close_trades.empty:
            print("没有找到平仓记录")
            return {}
        
        print(f"分析 {len(close_trades)} 条平仓记录")
        
        # 按平仓原因分类
        take_profit_trades = close_trades[close_trades['操作类型'].str.contains('止盈')]
        stop_loss_trades = close_trades[close_trades['操作类型'].str.contains('止损')]
        signal_close_trades = close_trades[close_trades['操作类型'].str.contains('信号')]
        
        # 计算真假信号数量
        true_signals = len(take_profit_trades)
        false_signals = len(stop_loss_trades)
        neutral_signals = len(signal_close_trades)
        
        # 创建交易结果序列: 1表示真信号(盈利), 0表示假信号(亏损)
        results = []
        for _, trade in close_trades.iterrows():
            if '止盈' in trade['操作类型']:
                results.append(1)  # 真信号
            elif '止损' in trade['操作类型']:
                results.append(0)  # 假信号
            else:
                # 对于信号平仓，检查盈亏
                if '盈亏' in trade and trade['盈亏'] > 0:
                    results.append(1)  # 算作真信号
                else:
                    results.append(0)  # 算作假信号
        
        # 分析连续假信号
        false_streaks = self._get_consecutive_streaks(results, 0)
        
        # 分析连续真信号
        true_streaks = self._get_consecutive_streaks(results, 1)
        
        # 分析多空方向的真假信号
        long_trades = close_trades[close_trades['操作类型'].str.contains('平多')]
        short_trades = close_trades[close_trades['操作类型'].str.contains('平空')]
        
        # 多头统计
        long_true = len(long_trades[long_trades['操作类型'].str.contains('止盈')]) if not long_trades.empty else 0
        long_false = len(long_trades[long_trades['操作类型'].str.contains('止损')]) if not long_trades.empty else 0
        long_signal = len(long_trades[long_trades['操作类型'].str.contains('信号')]) if not long_trades.empty else 0
        
        # 空头统计
        short_true = len(short_trades[short_trades['操作类型'].str.contains('止盈')]) if not short_trades.empty else 0
        short_false = len(short_trades[short_trades['操作类型'].str.contains('止损')]) if not short_trades.empty else 0
        short_signal = len(short_trades[short_trades['操作类型'].str.contains('信号')]) if not short_trades.empty else 0
        
        # 假信号原因分析
        false_signal_reasons = self._analyze_false_signal_reasons(stop_loss_trades) if not stop_loss_trades.empty else []
        
        # 收集结果
        self.analysis_results = {
            # 整体信号统计
            "total_trades": len(close_trades),
            "true_signals": true_signals,
            "false_signals": false_signals,
            "neutral_signals": neutral_signals,
            "true_signal_rate": true_signals / len(close_trades) if len(close_trades) > 0 else 0,
            "false_signal_rate": false_signals / len(close_trades) if len(close_trades) > 0 else 0,
            
            # 连续信号统计
            "max_false_streak": max(false_streaks) if false_streaks else 0,
            "avg_false_streak": np.mean(false_streaks) if false_streaks else 0,
            "false_streak_dist": dict(pd.Series(false_streaks).value_counts().sort_index()) if false_streaks else {},
            
            "max_true_streak": max(true_streaks) if true_streaks else 0,
            "avg_true_streak": np.mean(true_streaks) if true_streaks else 0,
            "true_streak_dist": dict(pd.Series(true_streaks).value_counts().sort_index()) if true_streaks else {},
            
            # 多空方向统计
            "long_trades": len(long_trades),
            "long_true": long_true,
            "long_false": long_false,
            "long_signal": long_signal,
            "long_true_rate": long_true / len(long_trades) if len(long_trades) > 0 else 0,
            "long_false_rate": long_false / len(long_trades) if len(long_trades) > 0 else 0,
            
            "short_trades": len(short_trades),
            "short_true": short_true,
            "short_false": short_false,
            "short_signal": short_signal,
            "short_true_rate": short_true / len(short_trades) if len(short_trades) > 0 else 0,
            "short_false_rate": short_false / len(short_trades) if len(short_trades) > 0 else 0,
            
            # 假信号原因
            "false_signal_reasons": false_signal_reasons
        }
        
        return self.analysis_results
    
    def _get_consecutive_streaks(self, results, value):
        """
        计算列表中连续出现value值的序列长度
        
        参数:
            results (list): 结果列表
            value (int): 要搜索的值
            
        返回:
            list: 连续序列长度列表
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
    
    def _analyze_false_signal_reasons(self, false_trades):
        """
        分析假信号的可能原因
        
        参数:
            false_trades (pd.DataFrame): 假信号交易记录
            
        返回:
            list: 假信号原因列表
        """
        reasons = []
        
        # 1. 多空方向分析
        long_false = false_trades[false_trades['操作类型'].str.contains('平多')].shape[0]
        short_false = false_trades[false_trades['操作类型'].str.contains('平空')].shape[0]
        
        if long_false > 1.5 * short_false and long_false > 10:
            reasons.append(f"多头假信号明显多于空头({long_false}:{short_false})，可能做多条件过于宽松")
        elif short_false > 1.5 * long_false and short_false > 10:
            reasons.append(f"空头假信号明显多于多头({short_false}:{long_false})，可能做空条件过于宽松")
        
        # 2. 分析持仓时间
        if '持仓时间(小时)' in false_trades.columns:
            avg_holding = false_trades['持仓时间(小时)'].mean()
            
            if avg_holding < 12 and len(false_trades) > 10:
                reasons.append(f"假信号平均持仓时间较短({avg_holding:.2f}小时)，可能是短期市场噪音导致")
            elif avg_holding > 72 and len(false_trades) > 10:
                reasons.append(f"假信号平均持仓时间较长({avg_holding:.2f}小时)，可能是大趋势反转")
        
        # 3. 检查止损间隔
        if '时间' in false_trades.columns and len(false_trades) > 10:
            # 按交易时间排序
            false_trades_sorted = false_trades.sort_values(by='时间')
            times = false_trades_sorted['时间'].tolist()
            
            if len(times) > 1:
                intervals = [(times[i+1] - times[i]).total_seconds()/3600 for i in range(len(times)-1)]
                short_intervals = [i for i in intervals if i < 24]  # 小于24小时的间隔
                
                if short_intervals and len(short_intervals) > 0.3 * len(intervals):
                    reasons.append(f"有{len(short_intervals)}次假信号在24小时内连续出现，可能是市场波动性增加")
        
        # 4. 根据时间分布分析
        if '时间' in false_trades.columns and len(false_trades) > 10:
            false_trades = false_trades.copy()  # 创建副本以避免设置副本警告
            false_trades['year'] = false_trades['时间'].dt.year
            false_trades['month'] = false_trades['时间'].dt.month
            
            year_counts = false_trades.groupby('year').size()
            month_counts = false_trades.groupby('month').size()
            
            # 检查是否某些时间段假信号特别多
            if len(year_counts) > 1 and year_counts.max() > year_counts.mean() * 1.5:
                worst_year = year_counts.idxmax()
                reasons.append(f"{worst_year}年假信号特别多，可能是特殊市场环境")
            
            if len(month_counts) > 2 and month_counts.max() > month_counts.mean() * 1.5:
                worst_month = month_counts.idxmax()
                month_names = {1:'一月', 2:'二月', 3:'三月', 4:'四月', 5:'五月', 6:'六月',
                               7:'七月', 8:'八月', 9:'九月', 10:'十月', 11:'十一月', 12:'十二月'}
                reasons.append(f"{month_names[worst_month]}的假信号特别多，可能存在季节性因素")
        
        # 如果没有找到明显原因，给出通用解释
        if not reasons:
            reasons.append("没有发现明显的假信号模式，可能是多种因素综合导致")
        
        return reasons
    
    def generate_report(self):
        """
        生成人类可读的报告
        
        返回:
            str: 文本形式的分析报告
        """
        if not self.analysis_results:
            return "尚未进行信号分析"
        
        # 构建报告文本
        report = []
        report.append("===== 交易信号分析报告 =====\n")
        
        # 整体信号统计
        report.append("【信号统计】")
        report.append(f"总交易次数: {self.analysis_results['total_trades']}")
        report.append(f"真信号(止盈平仓): {self.analysis_results['true_signals']} ({self.analysis_results['true_signal_rate']*100:.1f}%)")
        report.append(f"假信号(止损平仓): {self.analysis_results['false_signals']} ({self.analysis_results['false_signal_rate']*100:.1f}%)")
        report.append(f"中性信号(信号平仓): {self.analysis_results['neutral_signals']} ({(1-self.analysis_results['true_signal_rate']-self.analysis_results['false_signal_rate'])*100:.1f}%)")
        
        # 连续信号统计
        report.append("\n【连续假信号分析】")
        report.append(f"最大连续假信号次数: {self.analysis_results['max_false_streak']}")
        report.append(f"平均连续假信号次数: {self.analysis_results['avg_false_streak']:.2f}")
        report.append(f"连续假信号分布: {self.analysis_results['false_streak_dist']}")
        
        report.append("\n【连续真信号分析】")
        report.append(f"最大连续真信号次数: {self.analysis_results['max_true_streak']}")
        report.append(f"平均连续真信号次数: {self.analysis_results['avg_true_streak']:.2f}")
        report.append(f"连续真信号分布: {self.analysis_results['true_streak_dist']}")
        
        # 多空方向统计
        report.append("\n【多头信号分析】")
        report.append(f"多头总交易: {self.analysis_results['long_trades']}")
        report.append(f"多头真信号: {self.analysis_results['long_true']} ({self.analysis_results['long_true_rate']*100:.1f}%)")
        report.append(f"多头假信号: {self.analysis_results['long_false']} ({self.analysis_results['long_false_rate']*100:.1f}%)")
        
        report.append("\n【空头信号分析】")
        report.append(f"空头总交易: {self.analysis_results['short_trades']}")
        report.append(f"空头真信号: {self.analysis_results['short_true']} ({self.analysis_results['short_true_rate']*100:.1f}%)")
        report.append(f"空头假信号: {self.analysis_results['short_false']} ({self.analysis_results['short_false_rate']*100:.1f}%)")
        
        # 假信号原因
        report.append("\n【假信号原因分析】")
        for i, reason in enumerate(self.analysis_results['false_signal_reasons'], 1):
            report.append(f"{i}. {reason}")
        
        return "\n".join(report)
    
    def create_analysis_charts(self, output_file=None):
        """
        创建信号分析图表
        
        参数:
            output_file (str, optional): 输出文件路径
            
        返回:
            bool: 是否成功创建图表
        """
        if not self.analysis_results:
            print("尚未进行信号分析，无法创建图表")
            return False
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('交易信号分析图表', fontsize=16)
        
        # 1. 信号成功率饼图
        labels = ['真信号(止盈)', '假信号(止损)', '中性信号(信号平仓)']
        sizes = [
            self.analysis_results['true_signals'], 
            self.analysis_results['false_signals'], 
            self.analysis_results['neutral_signals']
        ]
        colors = ['#2ecc71', '#e74c3c', '#f39c12']
        
        axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('信号成功率分布')
        
        # 2. 连续假信号分布
        if self.analysis_results['false_streak_dist']:
            dist = pd.Series(self.analysis_results['false_streak_dist']).sort_index()
            axes[0, 1].bar(dist.index, dist.values, color='#e74c3c', alpha=0.7)
            axes[0, 1].set_title('连续假信号次数分布')
            axes[0, 1].set_xlabel('连续假信号次数')
            axes[0, 1].set_ylabel('发生频率')
        else:
            axes[0, 1].text(0.5, 0.5, '无假信号数据', ha='center', va='center')
        
        # 3. 连续真信号分布
        if self.analysis_results['true_streak_dist']:
            dist = pd.Series(self.analysis_results['true_streak_dist']).sort_index()
            axes[1, 0].bar(dist.index, dist.values, color='#2ecc71', alpha=0.7)
            axes[1, 0].set_title('连续真信号次数分布')
            axes[1, 0].set_xlabel('连续真信号次数')
            axes[1, 0].set_ylabel('发生频率')
        else:
            axes[1, 0].text(0.5, 0.5, '无真信号数据', ha='center', va='center')
        
        # 4. 多空信号对比条形图
        labels = ['多头', '空头']
        true_vals = [self.analysis_results['long_true'], self.analysis_results['short_true']]
        false_vals = [self.analysis_results['long_false'], self.analysis_results['short_false']]
        
        x = np.arange(len(labels))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, true_vals, width, label='真信号', color='#2ecc71')
        axes[1, 1].bar(x + width/2, false_vals, width, label='假信号', color='#e74c3c')
        
        axes[1, 1].set_title('多空真假信号对比')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(labels)
        axes[1, 1].set_ylabel('信号数量')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"分析图表已保存为: {output_file}")
        
        plt.close()
        return True
    
    def get_excel_data(self):
        """
        获取用于Excel报表的数据
        
        返回:
            tuple: (信号统计数据, 连续信号分布数据, 假信号原因数据)
        """
        if not self.analysis_results:
            return [], [], []
        
        # 1. 信号统计数据
        signal_stats = [
            ["总交易次数", self.analysis_results['total_trades']],
            ["真信号数量", self.analysis_results['true_signals']],
            ["假信号数量", self.analysis_results['false_signals']],
            ["中性信号数量", self.analysis_results['neutral_signals']],
            ["真信号比例", f"{self.analysis_results['true_signal_rate']*100:.2f}%"],
            ["假信号比例", f"{self.analysis_results['false_signal_rate']*100:.2f}%"],
            ["最大连续假信号", self.analysis_results['max_false_streak']],
            ["平均连续假信号", f"{self.analysis_results['avg_false_streak']:.2f}"],
            ["最大连续真信号", self.analysis_results['max_true_streak']],
            ["平均连续真信号", f"{self.analysis_results['avg_true_streak']:.2f}"],
            ["多头总交易", self.analysis_results['long_trades']],
            ["多头真信号数", self.analysis_results['long_true']],
            ["多头假信号数", self.analysis_results['long_false']],
            ["多头真信号比例", f"{self.analysis_results['long_true_rate']*100:.2f}%"],
            ["多头假信号比例", f"{self.analysis_results['long_false_rate']*100:.2f}%"],
            ["空头总交易", self.analysis_results['short_trades']],
            ["空头真信号数", self.analysis_results['short_true']],
            ["空头假信号数", self.analysis_results['short_false']],
            ["空头真信号比例", f"{self.analysis_results['short_true_rate']*100:.2f}%"],
            ["空头假信号比例", f"{self.analysis_results['short_false_rate']*100:.2f}%"]
        ]
        
        # 2. 连续信号分布
        streak_dist = []
        
        # 添加连续假信号分布
        if self.analysis_results['false_streak_dist']:
            streak_dist.append(["连续假信号次数", "发生频率"])
            for streak, count in sorted(self.analysis_results['false_streak_dist'].items()):
                streak_dist.append([streak, count])
            
            # 添加空行分隔
            streak_dist.append(["", ""])
        
        # 添加连续真信号分布
        if self.analysis_results['true_streak_dist']:
            streak_dist.append(["连续真信号次数", "发生频率"])
            for streak, count in sorted(self.analysis_results['true_streak_dist'].items()):
                streak_dist.append([streak, count])
        
        # 3. 假信号原因
        reasons_data = []
        for i, reason in enumerate(self.analysis_results['false_signal_reasons'], 1):
            reasons_data.append([i, reason])
        
        return signal_stats, streak_dist, reasons_data 