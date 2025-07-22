#!/usr/bin/env python3
"""
性能基准测试脚本
用于对比优化前后的性能差异
"""

import time
import os
import sys
import pandas as pd
import numpy as np
import psutil
from datetime import datetime
import matplotlib.pyplot as plt
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

def get_memory_usage():
    """获取当前内存使用量(MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def time_function(func, *args, **kwargs):
    """测量函数执行时间"""
    start_memory = get_memory_usage()
    start_time = time.time()
    
    result = func(*args, **kwargs)
    
    end_time = time.time()
    end_memory = get_memory_usage()
    
    return {
        'result': result,
        'time': end_time - start_time,
        'memory_used': end_memory - start_memory,
        'peak_memory': end_memory
    }

def load_test_data(config):
    """加载测试数据"""
    from data import DataManager
    
    data_manager = DataManager(config)
    local_files = data_manager.list_local_data()
    
    if not local_files:
        print("❌ 未找到本地数据文件，请先运行数据获取")
        return None
    
    # 使用第一个可用的数据文件
    data_file = local_files[0]["file_path"]
    print(f"📁 使用数据文件: {data_file}")
    
    data_df = data_manager.fetch_klines(cache_file=data_file)
    if data_df is None:
        print("❌ 数据加载失败")
        return None
    
    return data_df

def benchmark_indicators(config, data_df, sample_sizes=[500, 1000, 2000]):
    """测试指标计算性能"""
    print("\n🔧 指标计算性能测试")
    print("=" * 50)
    
    results = {}
    
    for size in sample_sizes:
        test_data = data_df.tail(size).copy()
        print(f"\n📊 测试数据大小: {size} 行")
        
        # 测试原版本
        try:
            from indicators import IndicatorCalculator
            
            calc_original = IndicatorCalculator(config)
            original_result = time_function(
                calc_original.calculate_all_indicators, 
                test_data
            )
            print(f"📈 原版本: {original_result['time']:.2f}秒, 内存: {original_result['memory_used']:.1f}MB")
            
        except ImportError:
            print("⚠️ 原版本指标计算器不可用")
            original_result = None
        
        # 测试优化版本
        try:
            from indicators_optimized import OptimizedIndicatorCalculator
            
            calc_optimized = OptimizedIndicatorCalculator(config)
            optimized_result = time_function(
                calc_optimized.calculate_all_indicators_optimized,
                test_data
            )
            print(f"🚀 优化版本: {optimized_result['time']:.2f}秒, 内存: {optimized_result['memory_used']:.1f}MB")
            
        except ImportError:
            print("⚠️ 优化版本指标计算器不可用")
            optimized_result = None
        
        # 计算性能提升
        if original_result and optimized_result:
            time_improvement = (original_result['time'] - optimized_result['time']) / original_result['time'] * 100
            memory_improvement = (original_result['memory_used'] - optimized_result['memory_used']) / max(original_result['memory_used'], 1) * 100
            
            print(f"⚡ 时间提升: {time_improvement:.1f}%")
            print(f"💾 内存优化: {memory_improvement:.1f}%")
            
            results[size] = {
                'original': original_result,
                'optimized': optimized_result,
                'time_improvement': time_improvement,
                'memory_improvement': memory_improvement
            }
    
    return results

def benchmark_trading(config, data_df, sample_sizes=[500, 1000]):
    """测试交易处理性能"""
    print("\n📈 交易处理性能测试")
    print("=" * 50)
    
    results = {}
    
    for size in sample_sizes:
        test_data = data_df.tail(size).copy()
        print(f"\n📊 测试数据大小: {size} 行")
        
        # 先计算指标和信号（使用优化版本）
        from indicators_optimized import OptimizedIndicatorCalculator
        from signals import SignalGenerator
        
        calc = OptimizedIndicatorCalculator(config)
        indicators_df = calc.calculate_all_indicators_optimized(test_data)
        
        signal_gen = SignalGenerator(config)
        signals_df = signal_gen.generate_signals(indicators_df)
        signals_df = signal_gen.calculate_risk_reward(signals_df)
        
        # 测试原版本交易处理
        try:
            from trade_recorder import TradeRecorder
            
            recorder_original = TradeRecorder(config)
            original_result = time_function(
                recorder_original.process_signals,
                signals_df
            )
            print(f"📈 原版本: {original_result['time']:.2f}秒, 内存: {original_result['memory_used']:.1f}MB")
            
        except ImportError:
            print("⚠️ 原版本交易记录器不可用")
            original_result = None
        
        # 测试优化版本交易处理
        try:
            from trade_recorder_optimized import OptimizedTradeRecorder
            
            recorder_optimized = OptimizedTradeRecorder(config)
            optimized_result = time_function(
                recorder_optimized.process_signals_vectorized,
                signals_df
            )
            print(f"🚀 优化版本: {optimized_result['time']:.2f}秒, 内存: {optimized_result['memory_used']:.1f}MB")
            
        except ImportError:
            print("⚠️ 优化版本交易记录器不可用")
            optimized_result = None
        
        # 计算性能提升
        if original_result and optimized_result:
            time_improvement = (original_result['time'] - optimized_result['time']) / original_result['time'] * 100
            memory_improvement = (original_result['memory_used'] - optimized_result['memory_used']) / max(original_result['memory_used'], 1) * 100
            
            print(f"⚡ 时间提升: {time_improvement:.1f}%")
            print(f"💾 内存优化: {memory_improvement:.1f}%")
            
            results[size] = {
                'original': original_result,
                'optimized': optimized_result,
                'time_improvement': time_improvement,
                'memory_improvement': memory_improvement
            }
    
    return results

def benchmark_full_strategy(config, data_df, size=1000):
    """测试完整策略性能"""
    print("\n🎯 完整策略性能测试")
    print("=" * 50)
    
    test_data = data_df.tail(size).copy()
    print(f"📊 测试数据大小: {size} 行")
    
    # 测试原版本完整流程
    try:
        from main import run_strategy
        
        print("\n📈 测试原版本完整流程...")
        original_result = time_function(
            run_strategy,
            data_source=test_data,
            visualize=False,
            custom_config=config,
            analyze_signals=False,
            export_excel=False
        )
        print(f"原版本总耗时: {original_result['time']:.2f}秒")
        print(f"内存使用: {original_result['memory_used']:.1f}MB")
        
    except ImportError:
        print("⚠️ 原版本策略不可用")
        original_result = None
    
    # 测试优化版本完整流程
    try:
        from main_optimized import run_strategy_optimized
        
        print("\n🚀 测试优化版本完整流程...")
        optimized_result = time_function(
            run_strategy_optimized,
            data_source=test_data,
            visualize=False,
            custom_config=config,
            export_excel=False
        )
        print(f"优化版本总耗时: {optimized_result['time']:.2f}秒")
        print(f"内存使用: {optimized_result['memory_used']:.1f}MB")
        
    except ImportError:
        print("⚠️ 优化版本策略不可用")
        optimized_result = None
    
    # 计算性能提升
    if original_result and optimized_result:
        time_improvement = (original_result['time'] - optimized_result['time']) / original_result['time'] * 100
        memory_improvement = (original_result['memory_used'] - optimized_result['memory_used']) / max(original_result['memory_used'], 1) * 100
        
        print(f"\n🎉 整体性能提升:")
        print(f"⚡ 时间提升: {time_improvement:.1f}%")
        print(f"💾 内存优化: {memory_improvement:.1f}%")
        print(f"🕒 时间节省: {original_result['time'] - optimized_result['time']:.2f}秒")
        
        return {
            'original': original_result,
            'optimized': optimized_result,
            'time_improvement': time_improvement,
            'memory_improvement': memory_improvement
        }
    
    return None

def create_performance_chart(indicator_results, trading_results, full_results):
    """创建性能对比图表"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('性能优化对比结果', fontsize=16)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 指标计算时间对比
        ax1 = axes[0, 0]
        if indicator_results:
            sizes = list(indicator_results.keys())
            original_times = [indicator_results[s]['original']['time'] for s in sizes]
            optimized_times = [indicator_results[s]['optimized']['time'] for s in sizes]
            
            x = np.arange(len(sizes))
            width = 0.35
            
            ax1.bar(x - width/2, original_times, width, label='原版本', color='skyblue')
            ax1.bar(x + width/2, optimized_times, width, label='优化版本', color='lightgreen')
            
            ax1.set_xlabel('数据大小')
            ax1.set_ylabel('时间 (秒)')
            ax1.set_title('指标计算时间对比')
            ax1.set_xticks(x)
            ax1.set_xticklabels([f'{s}行' for s in sizes])
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. 内存使用对比
        ax2 = axes[0, 1]
        if indicator_results:
            original_memory = [indicator_results[s]['original']['memory_used'] for s in sizes]
            optimized_memory = [indicator_results[s]['optimized']['memory_used'] for s in sizes]
            
            ax2.bar(x - width/2, original_memory, width, label='原版本', color='salmon')
            ax2.bar(x + width/2, optimized_memory, width, label='优化版本', color='lightcoral')
            
            ax2.set_xlabel('数据大小')
            ax2.set_ylabel('内存 (MB)')
            ax2.set_title('内存使用对比')
            ax2.set_xticks(x)
            ax2.set_xticklabels([f'{s}行' for s in sizes])
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. 性能提升百分比
        ax3 = axes[1, 0]
        if indicator_results and trading_results:
            categories = ['指标计算', '交易处理']
            time_improvements = [
                np.mean([indicator_results[s]['time_improvement'] for s in indicator_results.keys()]),
                np.mean([trading_results[s]['time_improvement'] for s in trading_results.keys()])
            ]
            
            bars = ax3.bar(categories, time_improvements, color=['lightblue', 'lightgreen'])
            ax3.set_ylabel('时间提升 (%)')
            ax3.set_title('各模块时间提升')
            ax3.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, time_improvements):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.1f}%',
                        ha='center', va='bottom')
        
        # 4. 整体性能对比
        ax4 = axes[1, 1]
        if full_results:
            categories = ['时间 (秒)', '内存 (MB)']
            original_values = [full_results['original']['time'], full_results['original']['memory_used']]
            optimized_values = [full_results['optimized']['time'], full_results['optimized']['memory_used']]
            
            x = np.arange(len(categories))
            width = 0.35
            
            ax4.bar(x - width/2, original_values, width, label='原版本', color='orange')
            ax4.bar(x + width/2, optimized_values, width, label='优化版本', color='green')
            
            ax4.set_ylabel('数值')
            ax4.set_title('整体性能对比')
            ax4.set_xticks(x)
            ax4.set_xticklabels(categories)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        chart_filename = f"performance_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
        print(f"📊 性能对比图表已保存为: {chart_filename}")
        plt.show()
        
    except Exception as e:
        print(f"❌ 创建图表时出错: {e}")

def main():
    """主函数"""
    print("🚀 Bitcoin Trading Strategy Performance Benchmark")
    print("=" * 60)
    
    # 加载配置
    try:
        from main_optimized import load_config
        config = load_config()
    except:
        try:
            from main import load_config
            config = load_config()
        except:
            print("❌ 无法加载配置文件")
            return
    
    print(f"⚙️ 配置加载成功")
    print(f"💼 初始资金: ${config.get('backtest', {}).get('initial_capital', 10000):,}")
    print(f"⚡ 杠杆: {config.get('backtest', {}).get('leverage', 1.0)}x")
    print(f"📈 策略: {config.get('signals', {}).get('strategy', '默认')}")
    
    # 加载测试数据
    data_df = load_test_data(config)
    if data_df is None:
        return
    
    print(f"📊 数据加载成功: {len(data_df)} 行数据")
    print(f"📅 数据时间范围: {data_df.index.min()} 至 {data_df.index.max()}")
    
    # 运行基准测试
    print(f"🧪 开始性能基准测试...")
    print(f"🖥️ 系统信息: {psutil.cpu_count()} CPU核心, {psutil.virtual_memory().total / 1024**3:.1f}GB 内存")
    
    # 1. 指标计算基准测试
    indicator_results = benchmark_indicators(config, data_df)
    
    # 2. 交易处理基准测试  
    trading_results = benchmark_trading(config, data_df)
    
    # 3. 完整策略基准测试
    full_results = benchmark_full_strategy(config, data_df)
    
    # 4. 创建性能图表
    if indicator_results or trading_results or full_results:
        create_performance_chart(indicator_results, trading_results, full_results)
    
    # 5. 输出总结
    print("\n" + "=" * 60)
    print("📋 基准测试总结")
    print("=" * 60)
    
    if full_results:
        print(f"🎯 整体性能提升: {full_results['time_improvement']:.1f}%")
        print(f"⚡ 时间节省: {full_results['original']['time'] - full_results['optimized']['time']:.2f}秒")
        print(f"💾 内存优化: {full_results['memory_improvement']:.1f}%")
    
    if indicator_results:
        avg_indicator_improvement = np.mean([indicator_results[s]['time_improvement'] for s in indicator_results.keys()])
        print(f"🔧 指标计算平均提升: {avg_indicator_improvement:.1f}%")
    
    if trading_results:
        avg_trading_improvement = np.mean([trading_results[s]['time_improvement'] for s in trading_results.keys()])
        print(f"📈 交易处理平均提升: {avg_trading_improvement:.1f}%")
    
    print("\n✅ 基准测试完成!")

if __name__ == "__main__":
    main()
