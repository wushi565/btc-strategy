import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from datetime import datetime
import os

# 导入主模块中的函数
from main import run_strategy, load_config

class LeverageOptimizer:
    """杠杆优化器，用于找出最佳杠杆值"""
    
    def __init__(self, data_source, config=None, min_leverage=1.0, max_leverage=10.0, step=0.5):
        """
        初始化杠杆优化器
        
        参数:
            data_source (str or pd.DataFrame): 数据源，可以是文件路径或DataFrame
            config (dict, optional): 配置信息，如果未提供则加载默认配置
            min_leverage (float): 最小杠杆值
            max_leverage (float): 最大杠杆值
            step (float): 杠杆值递增步长
        """
        self.data_source = data_source
        self.config = config if config else load_config()
        self.min_leverage = min_leverage
        self.max_leverage = max_leverage
        self.step = step
        self.results = {}
        self.best_leverage = None
    
    def run_optimization(self, criterion='total_return'):
        """
        运行杠杆优化
        
        参数:
            criterion (str): 优化标准，可选值: 'total_return', 'sharpe_ratio', 'calmar_ratio', 'profit_factor'
            
        返回:
            dict: 优化结果
        """
        # 生成要测试的杠杆值列表
        leverage_values = np.arange(self.min_leverage, self.max_leverage + self.step, self.step)
        
        print(f"开始杠杆优化，测试范围: {self.min_leverage} - {self.max_leverage}，步长: {self.step}")
        
        # 存储每个杠杆值的回测结果
        for leverage in leverage_values:
            print(f"\n===== 测试杠杆值: {leverage} =====")
            
            # 复制配置并修改杠杆值
            config_copy = copy.deepcopy(self.config)
            if "backtest" not in config_copy:
                config_copy["backtest"] = {}
            config_copy["backtest"]["leverage"] = leverage
            
            # 运行回测，但不进行可视化和信号分析，也不导出Excel
            signals_df, trade_recorder = run_strategy(
                data_source=self.data_source, 
                custom_config=config_copy,
                visualize=False,
                analyze_signals=False,
                export_excel=False
            )
            
            # 如果回测成功，计算关键指标
            if signals_df is not None and not signals_df.empty and trade_recorder is not None:
                # 获取交易统计摘要
                trade_summary = trade_recorder.get_trade_summary()
                
                # 提取资金曲线
                equity_curve = signals_df["equity"]
                
                # 计算关键指标
                initial_capital = config_copy["backtest"].get("initial_capital", 10000)
                final_equity = equity_curve.iloc[-1]
                total_return = (final_equity - initial_capital) / initial_capital
                
                # 计算最大回撤
                max_drawdown = trade_summary.get("最大回撤", 0)
                
                # 计算日收益率序列
                daily_returns = equity_curve.pct_change().dropna()
                
                # 计算夏普比率
                risk_free_rate = 0.02 / 252  # 假设无风险年化收益率为2%
                excess_returns = daily_returns - risk_free_rate
                sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
                
                # 计算卡尔玛比率
                annual_return = (1 + total_return) ** (252 / len(equity_curve)) - 1
                calmar_ratio = annual_return / max_drawdown if max_drawdown != 0 else float('inf')
                
                # 从交易记录中提取盈利因子
                profit_factor = trade_summary.get("盈亏比", 0)
                
                # 存储结果
                self.results[leverage] = {
                    "total_return": total_return,
                    "max_drawdown": max_drawdown,
                    "sharpe_ratio": sharpe_ratio,
                    "calmar_ratio": calmar_ratio,
                    "profit_factor": profit_factor,
                    "final_equity": final_equity,
                    "win_rate": trade_summary.get("胜率", 0),
                    "trade_count": trade_summary.get("交易次数", 0)
                }
                
                print(f"总回报率: {total_return:.2%}")
                print(f"最大回撤: {max_drawdown:.2%}")
                print(f"夏普比率: {sharpe_ratio:.2f}")
                print(f"卡尔玛比率: {calmar_ratio:.2f}")
                print(f"盈利因子: {profit_factor:.2f}")
                print(f"胜率: {self.results[leverage]['win_rate']:.2%}")
                print(f"交易次数: {self.results[leverage]['trade_count']}")
            else:
                print("回测失败，跳过此杠杆值")
        
        # 根据指定标准找出最佳杠杆值
        if self.results:
            if criterion == 'total_return':
                self.best_leverage = max(self.results.items(), key=lambda x: x[1]["total_return"])[0]
            elif criterion == 'sharpe_ratio':
                self.best_leverage = max(self.results.items(), key=lambda x: x[1]["sharpe_ratio"])[0]
            elif criterion == 'calmar_ratio':
                self.best_leverage = max(self.results.items(), key=lambda x: x[1]["calmar_ratio"])[0]
            elif criterion == 'profit_factor':
                self.best_leverage = max(self.results.items(), key=lambda x: x[1]["profit_factor"])[0]
            
            print(f"\n最佳杠杆值 (基于 {criterion}): {self.best_leverage}")
            print("最佳杠杆值的表现:")
            for key, value in self.results[self.best_leverage].items():
                if isinstance(value, float):
                    if key in ['total_return', 'max_drawdown', 'win_rate']:
                        print(f"{key}: {value:.2%}")
                    else:
                        print(f"{key}: {value:.2f}")
                else:
                    print(f"{key}: {value}")
        
        return self.results
    
    def visualize_results(self, save_path=None):
        """
        可视化优化结果
        
        参数:
            save_path (str, optional): 图表保存路径
        """
        if not self.results:
            print("没有可视化的结果")
            return
        
        # 设置中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('杠杆优化结果', fontsize=16)
        
        # 排序杠杆值
        leverages = sorted(self.results.keys())
        
        # 1. 总回报率
        total_returns = [self.results[lev]["total_return"] for lev in leverages]
        axes[0, 0].plot(leverages, total_returns, marker='o', linestyle='-')
        axes[0, 0].set_title('杠杆 vs 总回报率')
        axes[0, 0].set_xlabel('杠杆')
        axes[0, 0].set_ylabel('总回报率')
        axes[0, 0].grid(True)
        
        # 标出最佳值
        best_return = max(total_returns)
        best_leverage_return = leverages[total_returns.index(best_return)]
        axes[0, 0].plot(best_leverage_return, best_return, 'r*', markersize=10)
        axes[0, 0].text(best_leverage_return, best_return, f" 最佳: {best_leverage_return}", verticalalignment='bottom')
        
        # 2. 夏普比率
        sharpe_ratios = [self.results[lev]["sharpe_ratio"] for lev in leverages]
        axes[0, 1].plot(leverages, sharpe_ratios, marker='o', linestyle='-')
        axes[0, 1].set_title('杠杆 vs 夏普比率')
        axes[0, 1].set_xlabel('杠杆')
        axes[0, 1].set_ylabel('夏普比率')
        axes[0, 1].grid(True)
        
        # 标出最佳值
        best_sharpe = max(sharpe_ratios)
        best_leverage_sharpe = leverages[sharpe_ratios.index(best_sharpe)]
        axes[0, 1].plot(best_leverage_sharpe, best_sharpe, 'r*', markersize=10)
        axes[0, 1].text(best_leverage_sharpe, best_sharpe, f" 最佳: {best_leverage_sharpe}", verticalalignment='bottom')
        
        # 3. 卡尔玛比率
        calmar_ratios = [self.results[lev]["calmar_ratio"] for lev in leverages]
        axes[1, 0].plot(leverages, calmar_ratios, marker='o', linestyle='-')
        axes[1, 0].set_title('杠杆 vs 卡尔玛比率')
        axes[1, 0].set_xlabel('杠杆')
        axes[1, 0].set_ylabel('卡尔玛比率')
        axes[1, 0].grid(True)
        
        # 标出最佳值
        best_calmar = max(calmar_ratios)
        best_leverage_calmar = leverages[calmar_ratios.index(best_calmar)]
        axes[1, 0].plot(best_leverage_calmar, best_calmar, 'r*', markersize=10)
        axes[1, 0].text(best_leverage_calmar, best_calmar, f" 最佳: {best_leverage_calmar}", verticalalignment='bottom')
        
        # 4. 最大回撤
        max_drawdowns = [self.results[lev]["max_drawdown"] for lev in leverages]
        axes[1, 1].plot(leverages, max_drawdowns, marker='o', linestyle='-')
        axes[1, 1].set_title('杠杆 vs 最大回撤')
        axes[1, 1].set_xlabel('杠杆')
        axes[1, 1].set_ylabel('最大回撤')
        axes[1, 1].grid(True)
        
        # 标出最小值
        best_drawdown = min(max_drawdowns)
        best_leverage_drawdown = leverages[max_drawdowns.index(best_drawdown)]
        axes[1, 1].plot(best_leverage_drawdown, best_drawdown, 'r*', markersize=10)
        axes[1, 1].text(best_leverage_drawdown, best_drawdown, f" 最佳: {best_leverage_drawdown}", verticalalignment='bottom')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"优化结果图表已保存为: {save_path}")
        
        plt.show()
    
    def export_results_to_excel(self, filename=None):
        """
        将优化结果导出到Excel
        
        参数:
            filename (str, optional): Excel文件名，不提供则自动生成
            
        返回:
            str: 导出的文件路径
        """
        if not self.results:
            print("没有结果可以导出")
            return None
        
        # 生成文件名
        if filename is None:
            strategy_name = self.config.get("signals", {}).get("strategy", "Supertrend和DEMA策略")
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"leverage_optimization_{strategy_name}_{now}.xlsx"
        
        # 准备数据
        data = []
        for leverage, metrics in self.results.items():
            row = {"杠杆": leverage}
            row.update({k: v for k, v in metrics.items()})
            data.append(row)
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 设置列顺序
        columns = ["杠杆", "total_return", "max_drawdown", "sharpe_ratio", "calmar_ratio", "profit_factor", "final_equity", "win_rate", "trade_count"]
        df = df[columns]
        
        # 重命名列
        df.columns = ["杠杆", "总回报率", "最大回撤", "夏普比率", "卡尔玛比率", "盈利因子", "最终净值", "胜率", "交易次数"]
        
        # 导出到Excel
        try:
            df.to_excel(filename, index=False)
            print(f"优化结果已导出到: {filename}")
            return filename
        except Exception as e:
            print(f"导出到Excel失败: {e}")
            return None

def run_leverage_optimization(data_source, min_leverage=1.0, max_leverage=10.0, step=0.5, criterion='sharpe_ratio'):
    """
    运行杠杆优化并可视化结果
    
    参数:
        data_source (str or pd.DataFrame): 数据源，可以是文件路径或DataFrame
        min_leverage (float): 最小杠杆值
        max_leverage (float): 最大杠杆值
        step (float): 杠杆值递增步长
        criterion (str): 优化标准
        
    返回:
        LeverageOptimizer: 优化器实例
    """
    # 加载配置
    config = load_config()
    
    # 创建优化器
    optimizer = LeverageOptimizer(
        data_source=data_source,
        config=config,
        min_leverage=min_leverage,
        max_leverage=max_leverage,
        step=step
    )
    
    # 运行优化
    optimizer.run_optimization(criterion=criterion)
    
    # 可视化结果
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_path = f"leverage_optimization_chart_{now}.png"
    optimizer.visualize_results(save_path=chart_path)
    
    # 导出结果到Excel
    optimizer.export_results_to_excel()
    
    return optimizer

def main():
    """主函数"""
    print("\n===== 杠杆优化工具 =====")
    
    # 加载配置
    config = load_config()
    
    # 获取数据源
    from main import DataManager
    data_manager = DataManager(config)
    
    # 检查本地数据文件
    print("正在搜索本地数据文件...")
    local_files = data_manager.list_local_data()
    if not local_files:
        print("未找到本地数据文件")
        return
    
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
            data_source = local_files[file_index]["file_path"]
        else:
            print("无效的选择")
            return
    except ValueError:
        print("请输入有效的数字")
        return
    
    # 设置优化参数
    print("\n设置杠杆优化参数:")
    try:
        min_leverage = float(input("最小杠杆值 (默认: 1.0): ") or 1.0)
        max_leverage = float(input("最大杠杆值 (默认: 10.0): ") or 10.0)
        step = float(input("杠杆步长 (默认: 0.5): ") or 0.5)
    except ValueError:
        print("请输入有效的数字")
        return
    
    # 选择优化标准
    print("\n选择优化标准:")
    print("1. 总回报率 (最大化总盈利)")
    print("2. 夏普比率 (风险调整回报)")
    print("3. 卡尔玛比率 (回报/最大回撤)")
    print("4. 盈利因子 (总盈利/总亏损)")
    
    criterion_choice = input("请选择优化标准 (1-4): ")
    criterion = 'total_return'
    if criterion_choice == '2':
        criterion = 'sharpe_ratio'
    elif criterion_choice == '3':
        criterion = 'calmar_ratio'
    elif criterion_choice == '4':
        criterion = 'profit_factor'
    
    # 运行优化
    optimizer = run_leverage_optimization(
        data_source=data_source,
        min_leverage=min_leverage,
        max_leverage=max_leverage,
        step=step,
        criterion=criterion
    )
    
    # 显示最优结果
    print(f"\n基于{criterion}的最佳杠杆值: {optimizer.best_leverage}")
    
if __name__ == "__main__":
    main() 