# =============================================================================
# 比特币交易策略回测系统 - 优化整合版本
# 功能：整合传统技术指标交易系统与机器学习增强系统
# 版本：3.0 (整合版)
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

# 导入整合系统
from trading_system_integrator import TradingSystemIntegrator, create_trading_system

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
        elif os.path.exists("config_default.yaml"):
            # 如果主配置不存在，尝试使用默认配置
            with open("config_default.yaml", "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
                return config
        else:
            # 返回内置默认配置
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
                },
                "enable_ml": False
            }
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return {}

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
    
    # 确保所有必要列都存在
    for col in required_cols:
        if col not in signals_df.columns:
            print(f"缺少必要的列: {col}，无法可视化")
            return
    
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

def show_ml_performance(system):
    """显示机器学习模型性能"""
    if not system.has_ml_support or system.ml_enhancer is None:
        print("机器学习功能未启用")
        return
    
    try:
        ml_status = system.get_system_status().get("ml_status", {})
        
        if not ml_status:
            print("未获取到机器学习状态信息")
            return
        
        print("\n===== 机器学习模型性能 =====")
        
        # 显示基本信息
        print(f"模型训练状态: {'已训练' if ml_status.get('trained', False) else '未训练'}")
        print(f"最后更新时间: {ml_status.get('last_update', '未知')}")
        print(f"特征数量: {ml_status.get('feature_count', 0)}")
        print(f"置信度阈值: {ml_status.get('confidence_threshold', 0.6)}")
        
        # 显示指标
        metrics = ml_status.get("metrics", {})
        if metrics:
            print("\n模型性能指标:")
            for model_name, model_metrics in metrics.items():
                print(f"  {model_name}:")
                for metric_name, value in model_metrics.items():
                    if isinstance(value, float):
                        print(f"    {metric_name}: {value:.4f}")
                    else:
                        print(f"    {metric_name}: {value}")
    except Exception as e:
        print(f"获取机器学习性能信息失败: {e}")

def run_backtest_menu(system):
    """运行回测菜单"""
    print("\n===== 回测系统 =====")
    
    # 检查本地数据文件
    if hasattr(system, 'data_manager') and hasattr(system.data_manager, 'list_local_data'):
        local_files = system.data_manager.list_local_data()
        
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
                    
                    # 选择是否使用ML
                    use_ml = False
                    if system.has_ml_support:
                        ml_choice = input("是否使用机器学习增强信号? (y/n): ").lower()
                        use_ml = ml_choice in ['y', 'yes']
                    
                    # 是否开启可视化
                    vis_choice = input("是否显示策略可视化图表? (y/n): ").lower()
                    enable_visualization = vis_choice in ['y', 'yes']
                    
                    # 运行回测
                    print(f"开始回测 {selected_file}...")
                    backtest_result = system.run_backtest(
                        data_source=selected_file, 
                        use_ml=use_ml
                    )
                    
                    if backtest_result["success"]:
                        signals_df = backtest_result["signals_df"]
                        trade_recorder = backtest_result["trade_recorder"]
                        summary = backtest_result["summary"]
                        
                        # 显示回测结果
                        print("\n===== 回测结果 =====")
                        for key, value in summary.items():
                            if isinstance(value, float):
                                if key in ['胜率', '最大回撤', '净利润率']:
                                    print(f"{key}: {value:.2%}")
                                else:
                                    print(f"{key}: {value:.2f}")
                            else:
                                print(f"{key}: {value}")
                        
                        # 显示可视化
                        if enable_visualization:
                            visualize_strategy_optimized(signals_df)
                        
                        # 如果使用了ML，显示ML性能
                        if use_ml:
                            show_ml_performance(system)
                        
                    else:
                        print(f"回测失败: {backtest_result.get('error', '未知错误')}")
                    
                else:
                    print("无效的选择!")
            except ValueError:
                print("请输入有效的数字")
        else:
            print("未找到本地数据文件")
    else:
        print("数据管理器初始化失败，无法列出本地文件")

def train_ml_models_menu(system):
    """训练机器学习模型菜单"""
    if not system.has_ml_support or system.ml_enhancer is None:
        print("机器学习功能未启用，请在配置文件中设置 enable_ml: true")
        return
    
    print("\n===== 训练机器学习模型 =====")
    
    # 检查本地数据文件
    if hasattr(system, 'data_manager') and hasattr(system.data_manager, 'list_local_data'):
        local_files = system.data_manager.list_local_data()
        
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
            
            file_choice = input(f"请选择训练数据文件 (1-{len(local_files)}): ")
            try:
                file_index = int(file_choice) - 1
                if 0 <= file_index < len(local_files):
                    selected_file = local_files[file_index]["file_path"]
                    
                    # 加载数据
                    print(f"加载训练数据: {selected_file}...")
                    data_df = system.data_manager.fetch_klines(cache_file=selected_file)
                    
                    if data_df is not None and not data_df.empty:
                        # 训练模型
                        print("开始训练机器学习模型...")
                        training_result = system.train_ml_models(data_df)
                        
                        if training_result.get("success", False):
                            # 详细的训练结果已经在ModelTrainer中显示了
                            # 这里只显示模型性能对比表
                            metrics = training_result.get("metrics", {})
                            if metrics:
                                print("\n📈 模型性能对比:")
                                print(f"{'模型名称':<15} {'准确率':<8} {'AUC':<8} {'精确率':<8} {'召回率':<8}")
                                print("-" * 55)
                                for model_name, model_metrics in metrics.items():
                                    accuracy = model_metrics.get('accuracy', 0)
                                    auc = model_metrics.get('auc', 0)
                                    precision = model_metrics.get('precision', 0)
                                    recall = model_metrics.get('recall', 0)
                                    
                                    # 标记最佳模型
                                    marker = "⭐" if model_name == training_result.get('best_model') else "  "
                                    print(f"{marker} {model_name:<13} {accuracy:.1%}    {auc:.3f}    {precision:.1%}    {recall:.1%}")
                        else:
                            print(f"❌ 模型训练失败: {training_result.get('error', '未知错误')}")
                    else:
                        print("数据加载失败")
                else:
                    print("无效的选择!")
            except ValueError:
                print("请输入有效的数字")
        else:
            print("未找到本地数据文件")
    else:
        print("数据管理器初始化失败，无法列出本地文件")

def update_ml_models_menu(system):
    """更新机器学习模型菜单"""
    if not system.has_ml_support or system.ml_enhancer is None:
        print("机器学习功能未启用")
        return
    
    print("\n===== 更新机器学习模型 =====")
    
    # 检查ML状态
    ml_status = system.get_system_status().get("ml_status", {})
    if not ml_status.get("trained", False):
        print("模型尚未训练，请先训练模型")
        return
    
    # 检查本地数据文件
    if hasattr(system, 'data_manager') and hasattr(system.data_manager, 'list_local_data'):
        local_files = system.data_manager.list_local_data()
        
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
            
            file_choice = input(f"请选择更新数据文件 (1-{len(local_files)}): ")
            try:
                file_index = int(file_choice) - 1
                if 0 <= file_index < len(local_files):
                    selected_file = local_files[file_index]["file_path"]
                    
                    # 加载数据
                    print(f"加载更新数据: {selected_file}...")
                    data_df = system.data_manager.fetch_klines(cache_file=selected_file)
                    
                    if data_df is not None and not data_df.empty:
                        # 更新模型
                        print("开始更新机器学习模型...")
                        update_result = system.update_ml_models(data_df)
                        
                        if update_result.get("success", False):
                            if update_result.get("updated", False):
                                print("模型更新成功!")
                                print(f"性能提升: {update_result.get('performance_improvement', 0):.2f}%")
                                
                                # 显示新性能指标
                                metrics = update_result.get("metrics", {})
                                if metrics:
                                    print("\n更新后的模型性能指标:")
                                    for model_name, model_metrics in metrics.items():
                                        print(f"  {model_name}:")
                                        for metric_name, value in model_metrics.items():
                                            if isinstance(value, float):
                                                print(f"    {metric_name}: {value:.4f}")
                                            else:
                                                print(f"    {metric_name}: {value}")
                            else:
                                print(f"模型未更新: {update_result.get('reason', '性能未提升')}")
                        else:
                            print(f"模型更新失败: {update_result.get('error', '未知错误')}")
                    else:
                        print("数据加载失败")
                else:
                    print("无效的选择!")
            except ValueError:
                print("请输入有效的数字")
        else:
            print("未找到本地数据文件")
    else:
        print("数据管理器初始化失败，无法列出本地文件")

def live_signal_menu(system):
    """实时信号生成菜单"""
    print("\n===== 生成实时交易信号 =====")
    
    # 检查本地数据文件
    if hasattr(system, 'data_manager') and hasattr(system.data_manager, 'list_local_data'):
        local_files = system.data_manager.list_local_data()
        
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
                    
                    # 选择是否使用ML
                    use_ml = False
                    if system.has_ml_support:
                        ml_choice = input("是否使用机器学习增强信号? (y/n): ").lower()
                        use_ml = ml_choice in ['y', 'yes']
                        
                        # 如果使用ML但模型未训练，提示训练
                        if use_ml:
                            ml_status = system.get_system_status().get("ml_status", {})
                            if not ml_status.get("trained", False):
                                print("模型尚未训练，请先训练模型")
                                return
                    
                    # 加载数据
                    print(f"加载市场数据: {selected_file}...")
                    data_df = system.data_manager.fetch_klines(cache_file=selected_file)
                    
                    if data_df is not None and not data_df.empty:
                        # 选择预测点
                        latest_date = data_df.index[-1]
                        print(f"最新数据日期: {latest_date}")
                        days_back = input("想查看多少天前的信号? (0表示最新): ")
                        
                        try:
                            days = int(days_back)
                            if days > 0:
                                if days >= len(data_df):
                                    days = len(data_df) - 1
                                signal_data = data_df.iloc[:-days]
                            else:
                                signal_data = data_df
                            
                            # 生成信号
                            print("生成交易信号...")
                            signal_result = system.generate_live_signal(
                                signal_data,
                                use_ml=use_ml
                            )
                            
                            if signal_result.get("success", False):
                                signal = signal_result.get("signal", {})
                                
                                print("\n===== 交易信号 =====")
                                print(f"时间: {signal.get('timestamp', '')}")
                                
                                direction_map = {1: "做多", -1: "做空", 0: "中性"}
                                direction = direction_map.get(signal.get('final_signal', 0), "未知")
                                
                                print(f"信号方向: {direction}")
                                print(f"信号置信度: {signal.get('confidence', 0):.2f}")
                                
                                # 如果有ML信号，显示ML信息
                                if signal.get('ml_signal') is not None:
                                    ml_direction = direction_map.get(signal.get('ml_signal', 0), "未知")
                                    print(f"\nML信号: {ml_direction}")
                                    print(f"ML置信度: {signal.get('ml_confidence', 0):.2f}")
                                    print(f"技术信号: {direction_map.get(signal.get('technical_signal', 0), '未知')}")
                                
                                # 如果有方向，显示风险管理信息
                                if signal.get('direction'):
                                    print(f"\n方向: {signal.get('direction')}")
                                    if signal.get('stop_loss'):
                                        print(f"止损价: {signal.get('stop_loss'):.2f}")
                                    if signal.get('target'):
                                        print(f"止盈价: {signal.get('target'):.2f}")
                            else:
                                print(f"信号生成失败: {signal_result.get('error', '未知错误')}")
                        except ValueError:
                            print("请输入有效的数字")
                    else:
                        print("数据加载失败")
                else:
                    print("无效的选择!")
            except ValueError:
                print("请输入有效的数字")
        else:
            print("未找到本地数据文件")
    else:
        print("数据管理器初始化失败，无法列出本地文件")

def main():
    """
    系统主入口函数 - 整合版本
    
    功能说明:
    - 提供交互式菜单选择交易策略
    - 支持传统交易系统和机器学习增强系统
    - 支持回测、性能比较、杠杆优化功能
    - 使用优化的算法和向量化操作提高性能
    """
    print("\n===== 比特币交易策略回测系统 (整合优化版本) =====")
    
    # 加载配置
    config = load_config()
    
    # 创建整合系统
    print("初始化交易系统...")
    system = create_trading_system()
    
    # 获取系统状态
    system_status = system.get_system_status()
    
    # 显示系统状态
    print(f"系统初始化状态: {'成功' if system_status['is_initialized'] else '失败'}")
    print(f"机器学习支持: {'已启用' if system_status['has_ml_support'] else '未启用'}")
    print(f"当前策略: {system_status['current_strategy']}")
    
    # 获取可用策略列表
    available_strategies = system.get_available_strategies()
    
    # 选择功能菜单
    while True:
        print("\n选择功能:")
        print("1. 回测策略")
        print("2. 生成实时交易信号")
        print("3. 训练机器学习模型")
        print("4. 更新机器学习模型")
        print("5. 杠杆优化工具")
        print("6. 显示系统状态")
        print("0. 退出")
        
        function_choice = input("请选择功能 (0-6): ")
        
        if function_choice == "1":
            # 回测策略
            run_backtest_menu(system)
            
        elif function_choice == "2":
            # 生成实时信号
            live_signal_menu(system)
            
        elif function_choice == "3":
            # 训练ML模型
            train_ml_models_menu(system)
            
        elif function_choice == "4":
            # 更新ML模型
            update_ml_models_menu(system)
            
        elif function_choice == "5":
            # 杠杆优化
            print("\n===== 杠杆优化工具 =====")
            try:
                from leverage_optimizer import main as run_leverage_optimizer
                run_leverage_optimizer()
            except ImportError:
                print("杠杆优化模块未找到")
            except Exception as e:
                print(f"运行杠杆优化时出错: {e}")
                
        elif function_choice == "6":
            # 显示系统状态
            print("\n===== 系统状态 =====")
            status = system.get_system_status()
            
            print(f"系统初始化: {'成功' if status['is_initialized'] else '失败'}")
            print(f"机器学习支持: {'已启用' if status['has_ml_support'] else '未启用'}")
            print(f"当前策略: {status['current_strategy']}")
            print(f"数据准备: {'完成' if status['data_ready'] else '未完成'}")
            
            # 如果有ML状态，显示ML信息
            if status.get('ml_status'):
                show_ml_performance(system)
                
        elif function_choice == "0":
            # 退出
            print("系统退出，谢谢使用!")
            break
            
        else:
            print("无效的选择!")

if __name__ == "__main__":
    main()
