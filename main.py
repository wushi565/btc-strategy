# =============================================================================
# 比特币交易策略回测系统 - 优化版本
# 功能：基于Supertrend和DEMA指标的交易策略回测、参数优化和性能分析
# 作者：AI助手优化版本
# 版本：2.0 (优化版)
# =============================================================================

# 标准库导入
import yaml          # YAML配置文件解析
import os            # 操作系统接口
import pandas as pd  # 数据处理和分析
import numpy as np   # 数值计算
from datetime import datetime, timedelta  # 日期时间处理
import matplotlib.pyplot as plt           # 绘图基础库
import mplfinance as mpf                  # 金融数据可视化
import warnings                           # 警告控制

# 自定义模块导入 - 使用优化版本
from data import DataManager                    # 数据管理器 - 负责数据获取和缓存
from indicators import OptimizedIndicatorCalculator  # 优化指标计算器 - 使用向量化操作
from signals import SignalGenerator                  # 信号生成器 - 基于指标生成交易信号
from trade_recorder import OptimizedTradeRecorder    # 优化交易记录器 - 使用向量化回测

# 性能优化设置
# 忽略pandas性能警告和用户警告以提高运行速度
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def load_config():
    """
    加载系统配置文件
    
    功能说明:
    - 尝试从config.yaml文件加载配置
    - 如果文件不存在，返回默认配置
    - 包含交易设置、网络代理、技术指标参数等
    
    返回:
        dict: 配置字典，包含所有系统参数
    """
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
                    "enable_proxy": False,
                    "http_proxy": "http://127.0.0.1:7890",
                    "https_proxy": "http://127.0.0.1:7890"
                },
                "indicators": {
                    "dema144_len": 144,
                    "dema169_len": 169,
                    "atr_period": 34,
                    "atr_multiplier": 3.0,
                    "adx_period": 14,
                    "adx_threshold": 20
                },
                "signals": {
                    "risk_reward_ratio": 3.0,
                    "strategy": "Supertrend和DEMA策略"
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

def fetch_data_optimized(config, start_time=None, end_time=None, use_local_data=None, cache_file=None):
    """
    优化的数据获取函数
    
    功能说明:
    - 优先使用本地缓存文件提高加载速度
    - 如果没有缓存，从交易所API获取数据
    - 支持灵活的时间范围设置
    
    参数:
        config (dict): 系统配置
        start_time (str): 开始时间 (YYYY-MM-DD)
        end_time (str): 结束时间 (YYYY-MM-DD)
        use_local_data (bool): 是否使用本地数据
        cache_file (str): 指定的缓存文件路径
    
    返回:
        pd.DataFrame: K线数据，包含 OHLCV 列
    """
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
    
    # 其他数据获取逻辑保持不变
    return data_manager.fetch_klines(start_time, end_time, use_local_data)

def calculate_indicators_optimized(config, data_df):
    """优化的指标计算"""
    if data_df is None or data_df.empty:
        print("没有数据，无法计算指标")
        return None
    
    print("使用优化版本计算技术指标...")
    # 使用优化的指标计算器
    indicator_calculator = OptimizedIndicatorCalculator(config)
    indicators_df = indicator_calculator.calculate_all_indicators_optimized(data_df)
    return indicators_df

def generate_signals_optimized(config, indicators_df):
    """优化的信号生成"""
    if indicators_df is None or indicators_df.empty:
        print("没有指标数据，无法生成信号")
        return None
    
    strategy_name = config.get("signals", {}).get("strategy", "Supertrend和DEMA策略")
    print(f"使用策略: {strategy_name}")
    
    print("生成交易信号...")
    signal_generator = SignalGenerator(config)
    signals_df = signal_generator.generate_signals(indicators_df)
    signals_df = signal_generator.calculate_risk_reward(signals_df)
    
    # 在DataFrame的属性中保存策略名称
    signals_df.attrs["strategy_name"] = strategy_name
    
    return signals_df

def process_trades_optimized(config, signals_df):
    """优化的交易处理"""
    if signals_df is None or signals_df.empty:
        print("没有信号数据，无法处理交易")
        return None, None
    
    print("使用优化版本处理交易信号...")
    # 使用优化的交易记录器
    trade_recorder = OptimizedTradeRecorder(config)
    backtest_df = trade_recorder.process_signals_vectorized(signals_df)
    
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
    
    return backtest_df, trade_recorder

def visualize_strategy_optimized(signals_df, title=None):
    """优化的策略可视化"""
    if signals_df is None or signals_df.empty:
        print("没有信号数据，无法可视化")
        return
    
    print("生成策略可视化图表...")
    
    # 准备数据 - 只使用必要的列以提高性能
    required_cols = ['open', 'high', 'low', 'close', 'volume', 
                    'dema144', 'dema169', 'supertrend_upper', 'supertrend_lower',
                    'buy_signal', 'sell_signal']
    
    plot_data = signals_df[required_cols].copy()
    
    if title is None:
        title = signals_df.attrs.get("strategy_name", "交易策略")
    
    try:
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
        
        # 添加买卖信号标记（仅显示有信号的点）
        buy_signals = plot_data[plot_data['buy_signal']]
        sell_signals = plot_data[plot_data['sell_signal']]
        
        if not buy_signals.empty:
            add_plots.append(
                mpf.make_addplot(buy_signals['low'] * 0.99, scatter=True, 
                               marker='^', color='green', markersize=80)
            )
        
        if not sell_signals.empty:
            add_plots.append(
                mpf.make_addplot(sell_signals['high'] * 1.01, scatter=True, 
                               marker='v', color='red', markersize=80)
            )
        
        # 创建图表
        fig, axes = mpf.plot(
            plot_data,
            type='candle',
            style=s,
            title=title,
            ylabel='价格',
            volume=True,
            figsize=(18, 10),
            addplot=add_plots,
            returnfig=True
        )
        
        # 显示图表
        plt.show()
        
        # 返回统计信息
        buy_count = len(buy_signals)
        sell_count = len(sell_signals)
        print(f"买入信号数量: {buy_count}")
        print(f"卖出信号数量: {sell_count}")
        print(f"总信号数量: {buy_count + sell_count}")
        
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        print("建议检查数据格式或减少数据量")

def run_strategy_optimized(data_source=None, start_time=None, end_time=None, 
                          visualize=False, custom_config=None, export_excel=True):
    """
    优化版本的策略运行函数
    
    参数:
        data_source (str or pd.DataFrame): 数据源
        start_time (str): 开始时间
        end_time (str): 结束时间
        visualize (bool): 是否可视化
        custom_config (dict): 自定义配置
        export_excel (bool): 是否导出Excel
        
    返回:
        tuple: (DataFrame, TradeRecorder对象)
    """
    print("=== 运行优化版本的策略 ===")
    
    # 加载配置
    config = custom_config if custom_config else load_config()
    
    # 获取数据
    if isinstance(data_source, pd.DataFrame):
        data_df = data_source
    else:
        data_df = fetch_data_optimized(config, start_time, end_time, None, data_source)
    
    if data_df is None:
        return None, None
    
    # 计算指标
    indicators_df = calculate_indicators_optimized(config, data_df)
    
    if indicators_df is None:
        return None, None
    
    # 生成信号
    signals_df = generate_signals_optimized(config, indicators_df)
    
    if signals_df is None:
        return None, None
    
    # 处理交易
    backtest_df, trade_recorder = process_trades_optimized(config, signals_df)
    
    # 确保返回的DataFrame包含处理后的交易数据
    if backtest_df is not None:
        signals_df = backtest_df
    
    # 可视化
    if visualize:
        visualize_strategy_optimized(signals_df)
    
    # 导出到Excel (使用简化版本)
    if export_excel and trade_recorder is not None:
        output_file = trade_recorder.export_to_excel_streamlined(signals_df)
        if output_file:
            print(f"交易记录已导出至: {output_file}")
    
    return signals_df, trade_recorder

def performance_comparison():
    """性能对比测试"""
    print("\n=== 性能对比测试 ===")
    
    # 加载配置和数据
    config = load_config()
    data_manager = DataManager(config)
    local_files = data_manager.list_local_data()
    
    if not local_files:
        print("未找到本地数据文件，无法进行性能对比")
        return
    
    # 选择第一个可用的数据文件
    data_file = local_files[0]["file_path"]
    print(f"使用数据文件: {data_file}")
    
    # 加载数据
    data_df = fetch_data_optimized(config, cache_file=data_file)
    if data_df is None:
        print("数据加载失败")
        return
    
    # 限制数据量进行测试
    test_data = data_df.tail(1000).copy()  # 使用最后1000行数据
    print(f"测试数据大小: {len(test_data)} 行")
    
    import time
    
    # 测试原版本
    try:
        from indicators import IndicatorCalculator
        from trade_recorder import TradeRecorder
        
        print("\n--- 原版本测试 ---")
        start_time = time.time()
        
        # 指标计算
        indicator_calc = IndicatorCalculator(config)
        indicators_df1 = indicator_calc.calculate_all_indicators(test_data)
        
        # 信号生成
        signal_generator = SignalGenerator(config)
        signals_df1 = signal_generator.generate_signals(indicators_df1)
        
        # 交易处理
        trade_recorder1 = TradeRecorder(config)
        backtest_df1 = trade_recorder1.process_signals(signals_df1)
        
        time1 = time.time() - start_time
        print(f"原版本总耗时: {time1:.2f}秒")
        
    except ImportError:
        print("原版本文件不存在，跳过对比")
        time1 = None
    
    # 测试优化版本
    print("\n--- 优化版本测试 ---")
    start_time = time.time()
    
    # 指标计算
    indicator_calc_opt = OptimizedIndicatorCalculator(config)
    indicators_df2 = indicator_calc_opt.calculate_all_indicators_optimized(test_data)
    
    # 信号生成
    signal_generator = SignalGenerator(config)
    signals_df2 = signal_generator.generate_signals(indicators_df2)
    
    # 交易处理
    trade_recorder2 = OptimizedTradeRecorder(config)
    backtest_df2 = trade_recorder2.process_signals_vectorized(signals_df2)
    
    time2 = time.time() - start_time
    print(f"优化版本总耗时: {time2:.2f}秒")
    
    # 性能对比
    if time1 is not None:
        improvement = (time1 - time2) / time1 * 100
        print(f"\n=== 性能提升 ===")
        print(f"速度提升: {improvement:.1f}%")
        print(f"时间节省: {time1 - time2:.2f}秒")
    
    # 内存占用对比
    import psutil
    import os
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"当前内存占用: {memory_mb:.1f} MB")

def main():
    """
    系统主入口函数 - 优化版本
    
    功能说明:
    - 提供交互式菜单选择交易策略
    - 支持回测、性能比较、杠杆优化功能
    - 使用优化的算法和向量化操作提高性能
    """
    print("\n===== 比特币交易策略回测系统 (优化版本) =====")
    
    # 加载配置
    config = load_config()
    
    # 获取可用策略列表
    from signals import StrategyFactory
    available_strategies = StrategyFactory.get_strategy_list()
    
    # 选择策略
    print("\n可用策略:")
    for i, strategy in enumerate(available_strategies):
        print(f"{i+1}. {strategy}")
    
    current_strategy = config.get("signals", {}).get("strategy", "Supertrend和DEMA策略")
    print(f"当前配置中的策略: {current_strategy}")
    
    strategy_choice = input(f"请选择策略 (1-{len(available_strategies)}): ")
    try:
        strategy_index = int(strategy_choice) - 1
        if 0 <= strategy_index < len(available_strategies):
            config["signals"]["strategy"] = available_strategies[strategy_index]
            print(f"已选择策略: {config['signals']['strategy']}")
        else:
            print("无效的选择，使用默认策略")
    except ValueError:
        print("无效的输入，使用默认策略")
    
    # 选择功能
    print("\n选择功能:")
    print("1. 回测策略 (优化版本)")
    print("2. 性能对比测试")
    print("3. 基本杠杆优化")
    print("4. 动态杠杆优化 (自适应搜索)")
    
    function_choice = input("请选择功能 (1-4): ")
    
    if function_choice == "1":
        # 回测策略
        print("\n===== 回测策略 (优化版本) =====")
        
        # 搜索本地数据文件
        data_manager = DataManager(config)
        print("正在搜索本地数据文件...")
        local_files = data_manager.list_local_data()
        
        if local_files:
            print(f"找到{len(local_files)}个本地数据文件:")
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
                    
                    # 是否开启可视化
                    vis_choice = input("是否显示策略可视化图表? (y/n): ").lower()
                    enable_visualization = vis_choice in ['y', 'yes']
                    
                    # 运行优化版本策略
                    signals_df, trade_recorder = run_strategy_optimized(
                        data_source=selected_file, 
                        visualize=enable_visualization, 
                        custom_config=config
                    )
                    
                    if signals_df is not None:
                        print("策略回测完成!")
                    else:
                        print("回测失败!")
                else:
                    print("无效的选择!")
            except ValueError:
                print("请输入有效的数字")
        else:
            print("未找到本地数据文件")
    
    elif function_choice == "2":
        # 性能对比测试
        performance_comparison()
    
    elif function_choice == "3":
        # 基本杠杆优化
        print("\n===== 基本杠杆优化 =====")
        try:
            from leverage_optimizer import main as run_leverage_optimizer
            run_leverage_optimizer()
        except ImportError:
            print("杠杆优化模块未找到")
        except Exception as e:
            print(f"运行杠杆优化时出错: {e}")
    
    elif function_choice == "4":
        # 动态杠杆优化
        print("\n===== 动态杠杆优化 =====")
        try:
            from dynamic_leverage_optimizer import main as run_dynamic_optimizer
            run_dynamic_optimizer()
        except ImportError:
            print("动态杠杆优化模块未找到")
        except Exception as e:
            print(f"运行动态杠杆优化时出错: {e}")
    
    else:
        print("无效的选择!")

if __name__ == "__main__":
    main()
