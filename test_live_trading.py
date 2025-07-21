#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时交易系统测试脚本
用于验证各个组件是否正常工作
"""

import yaml
import logging
from datetime import datetime
import traceback

# 导入项目模块
from data import DataManager
from indicators import IndicatorCalculator
from signals import SignalGenerator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_live_trading')

def test_config_loading():
    """测试配置文件加载"""
    print("="*50)
    print("测试配置文件加载...")
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("✓ 配置文件加载成功")
        print(f"  交易对: {config.get('trading', {}).get('symbol', 'N/A')}")
        print(f"  时间周期: {config.get('trading', {}).get('timeframe', 'N/A')}")
        print(f"  是否期货: {config.get('trading', {}).get('is_futures', 'N/A')}")
        return config
    except Exception as e:
        print(f"✗ 配置文件加载失败: {e}")
        return None

def test_api_config():
    """测试API配置"""
    print("="*50)
    print("测试API配置...")
    try:
        with open('api_config.yaml', 'r', encoding='utf-8') as f:
            api_config = yaml.safe_load(f)
        
        api_key = api_config.get('api_key', '')
        api_secret = api_config.get('api_secret', '')
        
        if api_key and api_key != 'your_api_key_here':
            print("✓ API配置已设置")
            print(f"  API密钥: {api_key[:8]}...")
        else:
            print("⚠ API配置未设置，请编辑 api_config.yaml 文件")
        
        return api_config
    except Exception as e:
        print(f"✗ API配置加载失败: {e}")
        return None

def test_data_manager(config):
    """测试数据管理器"""
    print("="*50)
    print("测试数据管理器...")
    try:
        data_manager = DataManager(config)
        print("✓ 数据管理器初始化成功")
        
        # 测试连接
        print("  测试交易所连接...")
        if data_manager.test_connection():
            print("✓ 交易所连接成功")
        else:
            print("✗ 交易所连接失败")
        
        return data_manager
    except Exception as e:
        print(f"✗ 数据管理器初始化失败: {e}")
        return None

def test_indicator_calculator(config):
    """测试指标计算器"""
    print("="*50)
    print("测试指标计算器...")
    try:
        calculator = IndicatorCalculator(config)
        print("✓ 指标计算器初始化成功")
        print(f"  DEMA144周期: {calculator.dema144_len}")
        print(f"  DEMA169周期: {calculator.dema169_len}")
        print(f"  ATR周期: {calculator.atr_period}")
        print(f"  ADX周期: {calculator.adx_period}")
        return calculator
    except Exception as e:
        print(f"✗ 指标计算器初始化失败: {e}")
        return None

def test_signal_generator(config):
    """测试信号生成器"""
    print("="*50)
    print("测试信号生成器...")
    try:
        generator = SignalGenerator(config)
        print("✓ 信号生成器初始化成功")
        print(f"  策略: {generator.strategy_name}")
        print(f"  风险回报比: {generator.risk_reward_ratio}")
        return generator
    except Exception as e:
        print(f"✗ 信号生成器初始化失败: {e}")
        return None

def test_market_data_simulation():
    """测试市场数据模拟"""
    print("="*50)
    print("测试市场数据模拟...")
    try:
        import pandas as pd
        import numpy as np
        
        # 创建模拟数据
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        np.random.seed(42)
        
        # 生成模拟价格数据
        base_price = 50000
        returns = np.random.normal(0, 0.02, 100)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # 创建OHLCV数据
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = price * (1 + np.random.normal(0, 0.005))
            volume = np.random.randint(100, 1000)
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        print("✓ 模拟数据生成成功")
        print(f"  数据范围: {df.index[0]} 到 {df.index[-1]}")
        print(f"  数据行数: {len(df)}")
        
        return df
    except Exception as e:
        print(f"✗ 模拟数据生成失败: {e}")
        return None

def test_full_pipeline(config):
    """测试完整的数据处理流程"""
    print("="*50)
    print("测试完整的数据处理流程...")
    try:
        # 生成模拟数据
        df = test_market_data_simulation()
        if df is None:
            return False
        
        # 计算指标
        calculator = IndicatorCalculator(config)
        df_with_indicators = calculator.calculate_all_indicators(df)
        
        # 生成信号
        generator = SignalGenerator(config)
        df_with_signals = generator.generate_signals(df_with_indicators)
        
        print("✓ 完整流程测试成功")
        print(f"  最终数据列数: {len(df_with_signals.columns)}")
        print(f"  信号列: {[col for col in df_with_signals.columns if 'signal' in col or 'position' in col]}")
        
        # 检查是否有信号
        buy_signals = df_with_signals['buy_signal'].sum()
        sell_signals = df_with_signals['sell_signal'].sum()
        print(f"  买入信号数量: {buy_signals}")
        print(f"  卖出信号数量: {sell_signals}")
        
        return True
    except Exception as e:
        print(f"✗ 完整流程测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("比特币实时交易系统 - 组件测试")
    print("="*50)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 测试配置文件
    config = test_config_loading()
    if config is None:
        print("配置加载失败，测试终止")
        return
    
    # 测试API配置
    api_config = test_api_config()
    
    # 测试数据管理器
    data_manager = test_data_manager(config)
    
    # 测试指标计算器
    calculator = test_indicator_calculator(config)
    
    # 测试信号生成器
    generator = test_signal_generator(config)
    
    # 测试完整流程
    pipeline_success = test_full_pipeline(config)
    
    # 总结
    print("="*50)
    print("测试总结:")
    print(f"  配置文件: {'✓' if config else '✗'}")
    print(f"  API配置: {'✓' if api_config else '✗'}")
    print(f"  数据管理器: {'✓' if data_manager else '✗'}")
    print(f"  指标计算器: {'✓' if calculator else '✗'}")
    print(f"  信号生成器: {'✓' if generator else '✗'}")
    print(f"  完整流程: {'✓' if pipeline_success else '✗'}")
    
    if all([config, data_manager, calculator, generator, pipeline_success]):
        print("\n🎉 所有测试通过！系统可以正常运行。")
        print("\n下一步:")
        print("1. 如果API配置未设置，请编辑 api_config.yaml")
        print("2. 运行 python live_trading.py 开始实时交易")
    else:
        print("\n⚠ 部分测试失败，请检查相关配置和依赖。")

if __name__ == '__main__':
    main() 