import yaml
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import mplfinance as mpf

# 导入关键模块
from data import DataManager
from indicators import IndicatorCalculator
from signals import SupertrendDEMAStrategy, SupertrendDEMAAdxStrategy, SignalGenerator
from trade_recorder import TradeRecorder

def load_config():
    """加载配置文件"""
    try:
        if os.path.exists("config.yaml"):
            with open("config.yaml", "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
                return config
        else:
            # 返回默认配置
            return {
                "trading": {
                    "symbol": "BTC/USDT",
                    "timeframe": "1h",
                    "is_futures": True
                },
                "network": {
                    "http_proxy": "http://127.0.0.1:7890",
                    "https_proxy": "http://127.0.0.1:7890"
                },
                "indicators": {
                    "dema144_len": 144,
                    "dema169_len": 169,
                    "atr_period": 34,
                    "atr_multiplier": 3.0
                },
                "signals": {
                    "risk_reward_ratio": 3.0
                },
                "backtest": {
                    "initial_capital": 10000,
                    "leverage": 1.0,
                    "risk_per_trade": 0.02
                }
            }
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return {}

def fetch_data(config, start_time=None, end_time=None, use_local_data=None, cache_file=None):
    """获取K线数据"""
    # 初始化数据管理器
    print("初始化数据管理器...")
    data_manager = DataManager(config)
    
    # 如果指定了本地数据文件，直接加载
    if cache_file and os.path.exists(cache_file):
        print(f"使用指定的本地数据文件: {cache_file}")
        data_df = data_manager.fetch_klines(cache_file=cache_file)
        if data_df is None or data_df.empty:
            print("从指定文件加载数据失败，请检查文件路径。")
            return None
        print(f"成功加载 {len(data_df)} 条K线数据")
        return data_df
        
    # 设置默认时间范围
    if start_time is None:
        start_time = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if end_time is None:
        end_time = datetime.now().strftime("%Y-%m-%d")
    
    print(f"获取数据时间范围: {start_time} 到 {end_time}")
    
    # 检查本地是否有数据
    has_local_data, cache_path = data_manager.check_local_data(start_time, end_time)
    
    if use_local_data is None and has_local_data:
        response = input("发现本地缓存数据，是否使用? (y/n): ").lower()
        if response in ['y', 'yes']:
            use_local_data = True
        elif response in ['n', 'no']:
            use_local_data = False
    
    # 如果选择不使用本地数据，需要先测试连接
    if use_local_data is False:
        connection_success = data_manager.test_connection()
        if not connection_success:
            print("交易所连接失败，请检查网络和代理设置。")
            return None
    
    # 获取数据
    data_df = data_manager.fetch_klines(start_time, end_time, use_local_data)
    
    if data_df is None or data_df.empty:
        print("未获取到数据，请检查时间范围和网络设置。")
        return None
    
    print(f"成功获取 {len(data_df)} 条K线数据")
    return data_df

def calculate_indicators(config, data_df):
    """计算技术指标"""
    if data_df is None or data_df.empty:
        print("没有数据，无法计算指标")
        return None
    
    print("计算技术指标...")
    indicator_calculator = IndicatorCalculator(config)
    indicators_df = indicator_calculator.calculate_all_indicators(data_df)
    return indicators_df

def generate_signals(config, indicators_df):
    """生成交易信号"""
    if indicators_df is None or indicators_df.empty:
        print("没有指标数据，无法生成信号")
        return None
    
    # 显示使用的策略名称
    strategy_name = config.get("signals", {}).get("strategy", "Supertrend和DEMA策略")
    print(f"使用策略: {strategy_name}")
    
    print("生成交易信号...")
    # 使用SignalGenerator来根据配置选择合适的策略
    signal_generator = SignalGenerator(config)
    signals_df = signal_generator.generate_signals(indicators_df)
    signals_df = signal_generator.calculate_risk_reward(signals_df)
    
    # 在DataFrame的属性中保存策略名称，用于可视化
    signals_df.attrs["strategy_name"] = strategy_name
    
    return signals_df

def process_trades(config, signals_df):
    """处理交易并记录结果"""
    if signals_df is None or signals_df.empty:
        print("没有信号数据，无法处理交易")
        return None, None
    
    print("处理交易信号...")
    trade_recorder = TradeRecorder(config)
    backtest_df = trade_recorder.process_signals(signals_df)
    
    # 获取交易统计
    summary = trade_recorder.get_trade_summary()
    print("\n===== 交易统计 =====")
    for key, value in summary.items():
        if isinstance(value, float):
            if key in ['胜率', '最大回撤', '净利润率']:
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # 导出到Excel
    output_file = trade_recorder.export_to_excel(signals_df)
    if output_file:
        print(f"交易记录已导出至: {output_file}")
    
    return backtest_df, trade_recorder

def visualize_strategy(signals_df, title=None):
    """可视化策略"""
    if signals_df is None or signals_df.empty:
        print("没有信号数据，无法可视化")
        return
    
    print("生成策略可视化图表...")
    # 准备数据
    plot_data = signals_df.copy()
    
    # 如果没有提供标题，尝试从数据中获取策略名称
    if title is None:
        if "strategy_name" in signals_df.attrs:
            title = signals_df.attrs["strategy_name"]
        else:
            title = "交易策略"
    
    # 创建自定义样式
    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350',
                              wick='inherit', edge='inherit', 
                              volume='inherit')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', y_on_right=True)
    
    # 定义额外添加的图表
    add_plots = [
        mpf.make_addplot(plot_data['dema144'], color='blue', width=1),
        mpf.make_addplot(plot_data['dema169'], color='purple', width=1),
        mpf.make_addplot(plot_data['supertrend_upper'], color='red', width=1),
        mpf.make_addplot(plot_data['supertrend_lower'], color='green', width=1),
    ]
    
    # 添加买卖信号标记
    buy_signals = plot_data[plot_data['buy_signal']]
    sell_signals = plot_data[plot_data['sell_signal']]
    
    if not buy_signals.empty:
        add_plots.append(
            mpf.make_addplot(buy_signals['low'] * 0.99, scatter=True, 
                           marker='^', color='green', markersize=100)
        )
    
    if not sell_signals.empty:
        add_plots.append(
            mpf.make_addplot(sell_signals['high'] * 1.01, scatter=True, 
                           marker='v', color='red', markersize=100)
        )
    
    # 创建图表
    fig, axes = mpf.plot(
        plot_data,
        type='candle',
        style=s,
        title=title,
        ylabel='价格',
        volume=True,
        figsize=(20, 10),
        addplot=add_plots,
        returnfig=True
    )
    
    # 添加图例
    axes[0].legend(['DEMA144', 'DEMA169', 'Supertrend上轨', 'Supertrend下轨'])
    
    # 显示图表
    plt.show()
    
    # 返回统计信息
    buy_count = len(buy_signals)
    sell_count = len(sell_signals)
    print(f"买入信号数量: {buy_count}")
    print(f"卖出信号数量: {sell_count}")
    print(f"总信号数量: {buy_count + sell_count}")

def run_strategy(data_source=None, start_time=None, end_time=None, visualize=False, custom_config=None):
    """运行策略"""
    # 加载配置
    config = custom_config if custom_config else load_config()
    
    # 打印当前使用的策略配置
    strategy_name = config.get("signals", {}).get("strategy", "默认策略")
    print(f"运行策略函数使用的策略: {strategy_name}")
    
    # 获取数据
    if isinstance(data_source, pd.DataFrame):
        data_df = data_source
    else:
        data_df = fetch_data(config, start_time, end_time, None, data_source)
    
    if data_df is None:
        return None
    
    # 计算指标
    indicators_df = calculate_indicators(config, data_df)
    
    if indicators_df is None:
        return None
    
    # 生成信号
    signals_df = generate_signals(config, indicators_df)
    
    if signals_df is None:
        return None
    
    # 处理交易
    backtest_df, trade_recorder = process_trades(config, signals_df)
    
    # 可视化
    if visualize:
        visualize_strategy(signals_df)
    
    return signals_df

def main():
    """主函数"""
    print("\n===== 比特币交易策略回测系统 =====")
    
    # 加载配置
    config = load_config()
    
    # 获取可用策略列表
    from signals import StrategyFactory
    available_strategies = StrategyFactory.get_strategy_list()
    
    # 选择策略
    print("\n可用策略:")
    for i, strategy in enumerate(available_strategies):
        print(f"{i+1}. {strategy}")
    
    # 显示当前配置中的策略
    current_strategy = config.get("signals", {}).get("strategy", "Supertrend和DEMA策略")
    print(f"当前配置中的策略: {current_strategy}")
    
    strategy_choice = input(f"请选择策略 (1-{len(available_strategies)}): ")
    try:
        strategy_index = int(strategy_choice) - 1
        if 0 <= strategy_index < len(available_strategies):
            selected_strategy = available_strategies[strategy_index]
            # 更新配置中的策略
            if "signals" not in config:
                config["signals"] = {}
            config["signals"]["strategy"] = selected_strategy
            print(f"已选择策略: {selected_strategy}")
        else:
            print("无效的选择，使用默认策略")
    except ValueError:
        print("无效的输入，使用默认策略")
    
    print("\n选择数据来源:")
    print("1. 使用本地数据文件")
    print("2. 按日期范围获取数据")
    
    choice = input("请选择数据来源 (1-2): ")
    
    # 询问是否进行可视化
    vis_choice = input("是否需要可视化结果? (y/n): ").lower()
    enable_visualization = vis_choice in ['y', 'yes']
    
    if choice == "1":
        # 使用本地数据文件
        # 初始化数据管理器
        config = load_config()
        data_manager = DataManager(config)
        
        # 列出本地数据文件
        local_files = data_manager.list_local_data()
        if not local_files:
            print("未找到本地数据文件，请选择其他方式获取数据")
            return
        
        print("找到{}个本地数据文件:".format(len(local_files)))
        for i, file_info in enumerate(local_files):
            file_path = file_info.get("file_path", "")
            symbol = file_info.get("symbol", "未知")
            timeframe = file_info.get("timeframe", "未知")
            start_date = file_info.get("start_date", "未知")
            end_date = file_info.get("end_date", "未知")
            row_count = file_info.get("row_count", 0)
            print(f"{i+1}. {symbol}/{timeframe} - {start_date} 至 {end_date}, 共{row_count}行")
        
        file_choice = input(f"请选择数据文件 (1-{len(local_files)}): ")
        try:
            file_index = int(file_choice) - 1
            if 0 <= file_index < len(local_files):
                selected_file = local_files[file_index]["file_path"]
                print(f"选择的文件: {selected_file}")
                
                # 运行策略
                signals_df = run_strategy(data_source=selected_file, visualize=enable_visualization, custom_config=config)
                
                if signals_df is not None:
                    print("策略运行完成")
                else:
                    print("无效的选择!")
        except ValueError:
            print("请输入有效的数字!")
            
    elif choice == "2":
        # 按日期范围获取数据
        print("\n设置数据范围 (留空使用默认值)")
        default_start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        default_end = datetime.now().strftime("%Y-%m-%d")
        
        start_input = input(f"开始日期 (默认: {default_start}): ")
        end_input = input(f"结束日期 (默认: {default_end}): ")
        
        start_time = start_input if start_input else default_start
        end_time = end_input if end_input else default_end
        
        # 运行策略
        signals_df = run_strategy(start_time=start_time, end_time=end_time, visualize=enable_visualization, custom_config=config)
        
        if signals_df is not None:
            print("策略运行完成")
            
        else:
            print("无效的选择!")

if __name__ == "__main__":
    main()
