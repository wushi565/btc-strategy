# =============================================================================
# 信号模式分析器
# 功能：分析假信号和真信号的模式，统计连续亏损情况
# 版本：1.0
# =============================================================================

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

class SignalPatternAnalyzer:
    """信号模式分析器，用于分析假信号和真信号的模式"""
    
    def __init__(self):
        """初始化分析器"""
        self.analysis_results = {}
        
    def analyze_signal_patterns(self, trades):
        """
        分析信号模式
        
        参数:
            trades: 交易记录列表
            
        返回:
            dict: 分析结果
        """
        if not trades:
            return {}
            
        # 按时间排序交易
        sorted_trades = sorted(trades, key=lambda x: x.get('开仓时间', datetime.now()))
        
        # 分析连续亏损模式
        consecutive_analysis = self._analyze_consecutive_losses(sorted_trades)
        
        # 分析距离与盈利关系
        distance_analysis = self._analyze_distance_profitability(sorted_trades)
        
        # 分析假信号原因
        false_signal_analysis = self._analyze_false_signal_causes(sorted_trades)
        
        # 计算整体统计
        overall_stats = self._calculate_overall_stats(sorted_trades)
        
        self.analysis_results = {
            'consecutive_losses': consecutive_analysis,
            'distance_profitability': distance_analysis,
            'false_signal_causes': false_signal_analysis,
            'overall_stats': overall_stats,
            'total_trades': len(sorted_trades)
        }
        
        return self.analysis_results
    
    def _analyze_consecutive_losses(self, trades):
        """分析连续亏损模式"""
        consecutive_losses = []
        current_loss_streak = 0
        loss_streaks = []
        
        for trade in trades:
            pnl = trade.get('盈亏', 0)
            
            if pnl < 0:  # 亏损交易
                current_loss_streak += 1
            else:  # 盈利交易
                if current_loss_streak > 0:
                    loss_streaks.append(current_loss_streak)
                current_loss_streak = 0
        
        # 处理最后的连续亏损
        if current_loss_streak > 0:
            loss_streaks.append(current_loss_streak)
        
        if loss_streaks:
            analysis = {
                'total_loss_streaks': len(loss_streaks),
                'avg_consecutive_losses': np.mean(loss_streaks),
                'max_consecutive_losses': max(loss_streaks),
                'min_consecutive_losses': min(loss_streaks),
                'median_consecutive_losses': np.median(loss_streaks),
                'loss_streak_distribution': self._get_streak_distribution(loss_streaks)
            }
        else:
            analysis = {
                'total_loss_streaks': 0,
                'avg_consecutive_losses': 0,
                'max_consecutive_losses': 0,
                'min_consecutive_losses': 0,
                'median_consecutive_losses': 0,
                'loss_streak_distribution': {}
            }
        
        return analysis
    
    def _get_streak_distribution(self, streaks):
        """获取连续亏损次数分布"""
        distribution = {}
        for streak in streaks:
            if streak not in distribution:
                distribution[streak] = 0
            distribution[streak] += 1
        return distribution
    
    def _analyze_distance_profitability(self, trades):
        """分析DEMA距离与盈利能力的关系"""
        distance_ranges = [
            (0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, float('inf'))
        ]
        
        analysis = {
            'dema144_analysis': self._analyze_distance_for_dema(trades, 'DEMA144距离%', distance_ranges),
            'dema169_analysis': self._analyze_distance_for_dema(trades, 'DEMA169距离%', distance_ranges),
            'optimal_distance_ranges': {}
        }
        
        # 找出最优距离范围
        for dema_type in ['dema144_analysis', 'dema169_analysis']:
            best_range = None
            best_win_rate = 0
            
            for range_key, stats in analysis[dema_type].items():
                if stats['win_rate'] > best_win_rate and stats['trade_count'] >= 3:  # 至少3次交易
                    best_win_rate = stats['win_rate']
                    best_range = range_key
            
            analysis['optimal_distance_ranges'][dema_type] = {
                'range': best_range,
                'win_rate': best_win_rate
            }
        
        return analysis
    
    def _analyze_distance_for_dema(self, trades, distance_field, ranges):
        """分析特定DEMA距离的盈利情况"""
        analysis = {}
        
        for min_dist, max_dist in ranges:
            range_key = f"{min_dist}-{max_dist}%" if max_dist != float('inf') else f"{min_dist}%+"
            
            # 筛选在该距离范围内的交易
            range_trades = []
            for trade in trades:
                dist = abs(trade.get(distance_field, 0))
                if min_dist <= dist < max_dist:
                    range_trades.append(trade)
            
            if range_trades:
                profitable_trades = sum(1 for t in range_trades if t.get('盈亏', 0) > 0)
                total_pnl = sum(t.get('盈亏', 0) for t in range_trades)
                avg_pnl = total_pnl / len(range_trades)
                win_rate = profitable_trades / len(range_trades)
                
                analysis[range_key] = {
                    'trade_count': len(range_trades),
                    'profitable_trades': profitable_trades,
                    'total_pnl': total_pnl,
                    'avg_pnl': avg_pnl,
                    'win_rate': win_rate
                }
            else:
                analysis[range_key] = {
                    'trade_count': 0,
                    'profitable_trades': 0,
                    'total_pnl': 0,
                    'avg_pnl': 0,
                    'win_rate': 0
                }
        
        return analysis
    
    def _analyze_false_signal_causes(self, trades):
        """分析假信号（亏损交易）的原因"""
        losing_trades = [t for t in trades if t.get('盈亏', 0) < 0]
        
        if not losing_trades:
            return {'total_losing_trades': 0}
        
        # 按交易方向分析
        long_losses = [t for t in losing_trades if t.get('交易方向', '') == 'long']
        short_losses = [t for t in losing_trades if t.get('交易方向', '') == 'short']
        
        # 分析距离特征
        long_avg_dist144 = np.mean([abs(t.get('DEMA144距离%', 0)) for t in long_losses]) if long_losses else 0
        long_avg_dist169 = np.mean([abs(t.get('DEMA169距离%', 0)) for t in long_losses]) if long_losses else 0
        
        short_avg_dist144 = np.mean([abs(t.get('DEMA144距离%', 0)) for t in short_losses]) if short_losses else 0
        short_avg_dist169 = np.mean([abs(t.get('DEMA169距离%', 0)) for t in short_losses]) if short_losses else 0
        
        analysis = {
            'total_losing_trades': len(losing_trades),
            'long_losing_trades': len(long_losses),
            'short_losing_trades': len(short_losses),
            'long_avg_distances': {
                'dema144': long_avg_dist144,
                'dema169': long_avg_dist169
            },
            'short_avg_distances': {
                'dema144': short_avg_dist144,
                'dema169': short_avg_dist169
            },
            'potential_causes': self._identify_potential_causes(losing_trades)
        }
        
        return analysis
    
    def _identify_potential_causes(self, losing_trades):
        """识别潜在的假信号原因"""
        causes = []
        
        # 检查是否距离DEMA太近
        close_entries = sum(1 for t in losing_trades 
                          if abs(t.get('DEMA144距离%', 0)) < 0.5 or abs(t.get('DEMA169距离%', 0)) < 0.5)
        
        if close_entries > len(losing_trades) * 0.3:  # 超过30%
            causes.append({
                'cause': '距离DEMA线太近',
                'affected_trades': close_entries,
                'percentage': close_entries / len(losing_trades) * 100
            })
        
        # 检查是否距离DEMA太远
        far_entries = sum(1 for t in losing_trades 
                         if abs(t.get('DEMA144距离%', 0)) > 5 or abs(t.get('DEMA169距离%', 0)) > 5)
        
        if far_entries > len(losing_trades) * 0.3:
            causes.append({
                'cause': '距离DEMA线太远',
                'affected_trades': far_entries,
                'percentage': far_entries / len(losing_trades) * 100
            })
        
        # 检查亏损幅度
        large_losses = sum(1 for t in losing_trades if t.get('盈亏率', 0) < -0.05)  # 亏损超过5%
        
        if large_losses > 0:
            causes.append({
                'cause': '大额亏损（>5%）',
                'affected_trades': large_losses,
                'percentage': large_losses / len(losing_trades) * 100
            })
        
        return causes
    
    def _calculate_overall_stats(self, trades):
        """计算整体统计数据"""
        if not trades:
            return {}
        
        total_trades = len(trades)
        profitable_trades = sum(1 for t in trades if t.get('盈亏', 0) > 0)
        losing_trades = total_trades - profitable_trades
        
        total_pnl = sum(t.get('盈亏', 0) for t in trades)
        avg_pnl = total_pnl / total_trades
        
        # 分别计算做多和做空统计
        long_trades = [t for t in trades if t.get('交易方向', '') == 'long']
        short_trades = [t for t in trades if t.get('交易方向', '') == 'short']
        
        long_profitable = sum(1 for t in long_trades if t.get('盈亏', 0) > 0) if long_trades else 0
        short_profitable = sum(1 for t in short_trades if t.get('盈亏', 0) > 0) if short_trades else 0
        
        stats = {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'losing_trades': losing_trades,
            'overall_win_rate': profitable_trades / total_trades if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'long_trades_stats': {
                'count': len(long_trades),
                'profitable': long_profitable,
                'win_rate': long_profitable / len(long_trades) if long_trades else 0,
                'total_pnl': sum(t.get('盈亏', 0) for t in long_trades)
            },
            'short_trades_stats': {
                'count': len(short_trades),
                'profitable': short_profitable,
                'win_rate': short_profitable / len(short_trades) if short_trades else 0,
                'total_pnl': sum(t.get('盈亏', 0) for t in short_trades)
            }
        }
        
        return stats
    
    def get_analysis_summary(self):
        """获取分析摘要报告"""
        if not self.analysis_results:
            return "尚未进行分析，请先调用 analyze_signal_patterns() 方法"
        
        summary = []
        summary.append("=== 信号模式分析报告 ===\n")
        
        # 整体统计
        overall = self.analysis_results.get('overall_stats', {})
        summary.append(f"总交易次数: {overall.get('total_trades', 0)}")
        summary.append(f"整体胜率: {overall.get('overall_win_rate', 0):.2%}")
        summary.append(f"总盈亏: {overall.get('total_pnl', 0):.2f}")
        summary.append(f"平均盈亏: {overall.get('avg_pnl', 0):.2f}\n")
        
        # 连续亏损分析
        consecutive = self.analysis_results.get('consecutive_losses', {})
        summary.append("=== 连续亏损分析 ===")
        summary.append(f"连续亏损序列数: {consecutive.get('total_loss_streaks', 0)}")
        summary.append(f"平均连续亏损次数: {consecutive.get('avg_consecutive_losses', 0):.2f}")
        summary.append(f"最大连续亏损次数: {consecutive.get('max_consecutive_losses', 0)}")
        summary.append(f"连续亏损分布: {consecutive.get('loss_streak_distribution', {})}\n")
        
        # 最优距离范围
        optimal_ranges = self.analysis_results.get('distance_profitability', {}).get('optimal_distance_ranges', {})
        summary.append("=== 最优DEMA距离范围 ===")
        for dema_type, range_info in optimal_ranges.items():
            dema_name = "DEMA144" if "144" in dema_type else "DEMA169"
            summary.append(f"{dema_name} 最优距离: {range_info.get('range', 'N/A')} (胜率: {range_info.get('win_rate', 0):.2%})")
        
        return "\n".join(summary)