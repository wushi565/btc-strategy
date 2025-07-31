"""
信号质量检测和验证工具
用于评估交易信号的质量和有效性
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class SignalQualityValidator:
    """信号质量验证器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.quality_metrics = {}
        
    def validate_signal_distribution(self, signals: pd.Series, signal_name: str = "signal") -> Dict[str, float]:
        """验证信号分布质量"""
        metrics = {}
        
        # 基础统计
        signal_ratio = signals.mean()
        signal_count = len(signals)
        positive_signals = signals.sum()
        
        metrics.update({
            f"{signal_name}_ratio": signal_ratio,
            f"{signal_name}_count": signal_count,
            f"{signal_name}_positive_count": positive_signals,
            f"{signal_name}_negative_count": signal_count - positive_signals
        })
        
        # 分布质量评估
        metrics[f"{signal_name}_distribution_score"] = self._calculate_distribution_score(
            signal_ratio
        )
        
        # 信号密度检查
        metrics[f"{signal_name}_density_score"] = self._calculate_density_score(signals)
        
        # 连续性检查
        metrics[f"{signal_name}_continuity_score"] = self._calculate_continuity_score(signals)
        
        return metrics
    
    def validate_signal_effectiveness(self, signals: pd.Series, returns: pd.Series, 
                                    signal_name: str = "signal") -> Dict[str, float]:
        """验证信号有效性"""
        metrics = {}
        
        # 对齐数据
        aligned_data = pd.DataFrame({
            'signal': signals,
            'return': returns
        }).dropna()
        
        if len(aligned_data) == 0:
            return {f"{signal_name}_effectiveness_score": 0.0}
        
        # 信号预测准确性
        positive_signals = aligned_data[aligned_data['signal'] == 1]
        negative_signals = aligned_data[aligned_data['signal'] == 0]
        
        if len(positive_signals) > 0:
            positive_accuracy = (positive_signals['return'] > 0).mean()
            positive_avg_return = positive_signals['return'].mean()
        else:
            positive_accuracy = 0.0 
            positive_avg_return = 0.0
            
        if len(negative_signals) > 0:
            negative_accuracy = (negative_signals['return'] <= 0).mean()
            negative_avg_return = negative_signals['return'].mean()
        else:
            negative_accuracy = 0.0
            negative_avg_return = 0.0
        
        metrics.update({
            f"{signal_name}_positive_accuracy": positive_accuracy,
            f"{signal_name}_negative_accuracy": negative_accuracy,
            f"{signal_name}_positive_avg_return": positive_avg_return,
            f"{signal_name}_negative_avg_return": negative_avg_return,
            f"{signal_name}_effectiveness_score": (positive_accuracy + negative_accuracy) / 2
        })
        
        # 信息比率
        if aligned_data['return'].std() > 0:
            information_ratio = aligned_data['return'].mean() / aligned_data['return'].std()
            metrics[f"{signal_name}_information_ratio"] = information_ratio
        
        # 夏普比率 (简化版)
        if aligned_data['return'].std() > 0:
            sharpe_ratio = aligned_data['return'].mean() / aligned_data['return'].std()
            metrics[f"{signal_name}_sharpe_ratio"] = sharpe_ratio
        
        return metrics
    
    def validate_multiple_periods(self, df: pd.DataFrame, 
                                periods: List[int] = [1, 3, 5]) -> Dict[str, Any]:
        """验证多个时间周期的信号质量"""
        validation_results = {
            'summary': {},
            'detailed': {},
            'recommendations': []
        }
        
        for period in periods:
            signal_col = f'target_{period}'
            if signal_col not in df.columns:
                continue
                
            # 计算对应的收益率
            if 'close' in df.columns:
                returns = df['close'].shift(-period) / df['close'] - 1
            else:
                continue
            
            # 分布验证
            dist_metrics = self.validate_signal_distribution(
                df[signal_col], f"target_{period}"
            )
            
            # 有效性验证
            eff_metrics = self.validate_signal_effectiveness(
                df[signal_col], returns, f"target_{period}"
            )
            
            # 合并结果
            period_metrics = {**dist_metrics, **eff_metrics}
            validation_results['detailed'][f'period_{period}'] = period_metrics
            
            # 总结关键指标
            validation_results['summary'][f'period_{period}'] = {
                'signal_ratio': period_metrics.get(f'target_{period}_ratio', 0),
                'distribution_score': period_metrics.get(f'target_{period}_distribution_score', 0),
                'effectiveness_score': period_metrics.get(f'target_{period}_effectiveness_score', 0),
                'positive_accuracy': period_metrics.get(f'target_{period}_positive_accuracy', 0)
            }
        
        # 生成建议
        validation_results['recommendations'] = self._generate_recommendations(
            validation_results['summary']
        )
        
        return validation_results
    
    def _calculate_distribution_score(self, signal_ratio: float) -> float:
        """计算分布质量评分 (0-1)"""
        # 理想的信号比例在20%-80%之间
        if 0.2 <= signal_ratio <= 0.8:
            # 在理想范围内，越接近50%评分越高
            distance_from_optimal = abs(signal_ratio - 0.5)
            score = 1.0 - (distance_from_optimal / 0.3) * 0.5
        elif 0.1 <= signal_ratio < 0.2 or 0.8 < signal_ratio <= 0.9:
            # 在可接受范围内
            score = 0.5
        else:
            # 分布过于极端
            score = 0.1
            
        return max(0.0, min(1.0, score))
    
    def _calculate_density_score(self, signals: pd.Series) -> float:
        """计算信号密度评分"""
        if len(signals) < 10:
            return 0.0
            
        # 计算信号变化频率
        signal_changes = (signals.diff() != 0).sum()
        total_periods = len(signals)
        change_ratio = signal_changes / total_periods
        
        # 理想的变化比例在5%-50%之间
        if 0.05 <= change_ratio <= 0.5:
            score = 1.0 - abs(change_ratio - 0.25) / 0.25 * 0.5
        elif change_ratio < 0.05:
            score = 0.2  # 信号变化太少
        else:
            score = 0.3  # 信号变化太频繁
            
        return max(0.0, min(1.0, score))
    
    def _calculate_continuity_score(self, signals: pd.Series) -> float:
        """计算信号连续性评分"""
        if len(signals) < 5:
            return 0.0
        
        # 计算连续信号长度
        signal_groups = signals.groupby((signals != signals.shift()).cumsum())
        group_lengths = signal_groups.size()
        
        # 理想的连续长度在2-10之间
        avg_length = group_lengths.mean()
        if 2 <= avg_length <= 10:
            score = 1.0 - abs(avg_length - 5) / 5 * 0.3
        elif avg_length < 2:
            score = 0.4  # 信号过于频繁切换
        else:
            score = 0.6  # 信号过于持续
            
        return max(0.0, min(1.0, score))
    
    def _generate_recommendations(self, summary: Dict[str, Dict]) -> List[str]:
        """基于验证结果生成改进建议"""
        recommendations = []
        
        for period, metrics in summary.items():
            signal_ratio = metrics.get('signal_ratio', 0)
            dist_score = metrics.get('distribution_score', 0)
            eff_score = metrics.get('effectiveness_score', 0)
            
            # 信号分布建议
            if signal_ratio < 0.1:
                recommendations.append(
                    f"{period}: 信号比例过低({signal_ratio:.1%})，建议降低阈值或调整策略"
                )
            elif signal_ratio > 0.9:
                recommendations.append(
                    f"{period}: 信号比例过高({signal_ratio:.1%})，建议提高阈值或增加过滤条件"
                )
            
            # 分布质量建议
            if dist_score < 0.5:
                recommendations.append(
                    f"{period}: 分布质量较差(评分:{dist_score:.2f})，建议启用自适应阈值"
                )
            
            # 有效性建议
            if eff_score < 0.5:
                recommendations.append(
                    f"{period}: 信号有效性较低(评分:{eff_score:.2f})，建议增加技术指标确认"
                )
        
        if not recommendations:
            recommendations.append("所有时间周期的信号质量都在可接受范围内")
            
        return recommendations
    
    def print_validation_report(self, validation_results: Dict[str, Any]):
        """打印验证报告"""
        print("\n" + "="*60)
        print("📊 信号质量验证报告")
        print("="*60)
        
        # 总结报告
        print("\n🎯 各周期信号质量总结:")
        print("-" * 40)
        for period, metrics in validation_results['summary'].items():
            print(f"{period}:")
            print(f"  信号比例: {metrics['signal_ratio']:.1%}")
            print(f"  分布评分: {metrics['distribution_score']:.2f}/1.0")
            print(f"  有效性评分: {metrics['effectiveness_score']:.2f}/1.0")
            print(f"  预测准确性: {metrics['positive_accuracy']:.1%}")
            print()
        
        # 改进建议
        print("💡 改进建议:")
        print("-" * 40)
        for i, rec in enumerate(validation_results['recommendations'], 1):
            print(f"{i}. {rec}")
        
        print("\n" + "="*60)


def test_signal_quality():
    """测试信号质量验证器"""
    # 创建测试数据
    np.random.seed(42)
    n_samples = 1000
    
    # 模拟价格数据
    price_changes = np.random.normal(0, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(price_changes))
    
    # 创建测试DataFrame
    df = pd.DataFrame({
        'close': prices,
        'target_1': np.random.binomial(1, 0.3, n_samples),  # 30%买入信号
        'target_3': np.random.binomial(1, 0.4, n_samples),  # 40%买入信号  
        'target_5': np.random.binomial(1, 0.5, n_samples)   # 50%买入信号
    })
    
    # 创建验证器
    validator = SignalQualityValidator({})
    
    # 运行验证
    results = validator.validate_multiple_periods(df, [1, 3, 5])
    
    # 打印报告  
    validator.print_validation_report(results)
    
    return results


if __name__ == "__main__":
    test_signal_quality()