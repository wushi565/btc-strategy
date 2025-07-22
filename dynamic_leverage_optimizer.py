# =============================================================================
# 动态杠杆优化器
# 功能：自动回测系统，使用自适应搜索算法寻找最佳杠杆值
# 版本：1.0
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from datetime import datetime
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# 导入主模块中的函数
from main import run_strategy, load_config
from leverage_optimizer import LeverageOptimizer

class DynamicLeverageOptimizer:
    """动态杠杆优化器，使用自适应搜索算法寻找最佳杠杆值"""
    
    def __init__(self, data_source, config=None, 
                 min_leverage=1.0, max_leverage=20.0, 
                 initial_step=1.0, min_step=0.1,
                 parallel_jobs=None, criterion='sharpe_ratio'):
        """
        初始化动态杠杆优化器
        
        参数:
            data_source (str or pd.DataFrame): 数据源，可以是文件路径或DataFrame
            config (dict, optional): 配置信息，如果未提供则加载默认配置
            min_leverage (float): 初始最小杠杆值
            max_leverage (float): 初始最大杠杆值
            initial_step (float): 初始搜索步长
            min_step (float): 最小搜索步长（精度）
            parallel_jobs (int): 并行任务数，默认为None（自动选择）
            criterion (str): 优化标准，可选值: 'total_return', 'sharpe_ratio', 'calmar_ratio', 'profit_factor'
        """
        self.data_source = data_source
        self.config = config if config else load_config()
        self.min_leverage = min_leverage
        self.max_leverage = max_leverage
        self.initial_step = initial_step
        self.min_step = min_step
        self.parallel_jobs = parallel_jobs
        self.criterion = criterion
        
        # 结果存储
        self.results = {}
        self.best_leverage = None
        self.search_history = []
        
        # 确保配置中包含backtest部分
        if "backtest" not in self.config:
            self.config["backtest"] = {}
    
    def run_optimization(self, max_iterations=20, refinement_threshold=3):
        """
        运行动态杠杆优化
        
        参数:
            max_iterations (int): 最大搜索迭代次数
            refinement_threshold (int): 在缩小搜索范围前的迭代次数
            
        返回:
            dict: 优化结果
        """
        print(f"\n===== 开始动态杠杆优化 =====")
        print(f"优化标准: {self.criterion}")
        print(f"初始搜索范围: {self.min_leverage} - {self.max_leverage}, 步长: {self.initial_step}")
        
        current_min = self.min_leverage
        current_max = self.max_leverage
        current_step = self.initial_step
        iteration = 0
        
        while iteration < max_iterations and current_step >= self.min_step:
            iteration += 1
            print(f"\n--- 迭代 {iteration}/{max_iterations} ---")
            print(f"当前搜索范围: {current_min:.2f} - {current_max:.2f}, 步长: {current_step:.2f}")
            
            # 生成本次迭代要测试的杠杆值
            leverage_values = np.arange(current_min, current_max + current_step/2, current_step)
            leverage_values = np.round(leverage_values, 2)  # 保留两位小数
            
            print(f"将测试 {len(leverage_values)} 个杠杆值")
            
            # 运行批量回测
            iteration_results = self._run_batch_backtest(leverage_values)
            
            # 更新整体结果
            self.results.update(iteration_results)
            
            # 记录搜索历史
            self.search_history.append({
                'iteration': iteration,
                'min_leverage': current_min,
                'max_leverage': current_max,
                'step': current_step,
                'values_tested': list(leverage_values),
                'best_in_iteration': self._find_best_value(iteration_results)
            })
            
            # 更新最佳杠杆值
            self.best_leverage = self._find_best_value(self.results)
            
            print(f"当前最佳杠杆值: {self.best_leverage:.2f}")
            print(f"最佳性能 ({self.criterion}): {self.results[self.best_leverage][self.criterion]:.4f}")
            
            # 根据结果缩小搜索范围
            if iteration >= refinement_threshold:
                # 找到性能最好的区域
                sorted_results = sorted(
                    [(lev, res[self.criterion]) for lev, res in iteration_results.items()],
                    key=lambda x: x[1], reverse=True
                )
                
                if sorted_results:
                    # 选取前50%的结果来缩小范围
                    top_leverages = [lev for lev, _ in sorted_results[:max(1, len(sorted_results)//2)]]
                    
                    if top_leverages:
                        # 缩小搜索范围
                        new_min = max(current_min, min(top_leverages) - current_step)
                        new_max = min(current_max, max(top_leverages) + current_step)
                        
                        # 如果范围足够小，减小步长
                        if new_max - new_min < current_step * 5:
                            current_step = max(self.min_step, current_step / 2)
                            
                        current_min = new_min
                        current_max = new_max
                        
                        print(f"缩小搜索范围至: {current_min:.2f} - {current_max:.2f}, 新步长: {current_step:.2f}")
            
            # 如果找到的最佳值在边界上，扩大搜索范围
            if self.best_leverage == current_min:
                current_min = max(0.1, current_min - current_step * 2)
                print(f"最佳值在下边界，扩展范围至: {current_min:.2f} - {current_max:.2f}")
            elif self.best_leverage == current_max:
                current_max = current_max + current_step * 2
                print(f"最佳值在上边界，扩展范围至: {current_min:.2f} - {current_max:.2f}")
        
        print("\n===== 动态杠杆优化完成 =====")
        print(f"最佳杠杆值: {self.best_leverage:.2f}")
        print(f"最佳{self.criterion}: {self.results[self.best_leverage][self.criterion]:.4f}")
        
        # 输出其他重要指标
        best_result = self.results[self.best_leverage]
        print(f"总回报率: {best_result['total_return']:.2%}")
        print(f"最大回撤: {best_result['max_drawdown']:.2%}")
        print(f"夏普比率: {best_result['sharpe_ratio']:.2f}")
        print(f"卡尔玛比率: {best_result['calmar_ratio']:.2f}")
        print(f"胜率: {best_result['win_rate']:.2%}")
        print(f"交易次数: {best_result['trade_count']}")
        
        return self.results
    
    def _run_batch_backtest(self, leverage_values):
        """
        批量运行回测
        
        参数:
            leverage_values (list): 要测试的杠杆值列表
            
        返回:
            dict: 回测结果
        """
        start_time = time.time()
        results = {}
        
        # 决定是否使用并行处理
        if self.parallel_jobs is not None and self.parallel_jobs > 1:
            print(f"使用并行处理，{self.parallel_jobs}个并行任务")
            results = self._run_parallel_backtest(leverage_values, self.parallel_jobs)
        else:
            print("使用顺序处理")
            # 顺序处理
            for leverage in leverage_values:
                print(f"\n测试杠杆值: {leverage}")
                result = self._run_single_backtest(leverage)
                if result:
                    results[leverage] = result
        
        duration = time.time() - start_time
        print(f"\n完成批量回测，耗时: {duration:.2f}秒")
        print(f"成功测试: {len(results)}/{len(leverage_values)} 个杠杆值")
        
        return results
    
    def _run_parallel_backtest(self, leverage_values, num_workers):
        """
        并行运行回测
        
        参数:
            leverage_values (list): 要测试的杠杆值列表
            num_workers (int): 并行工作进程数
            
        返回:
            dict: 回测结果
        """
        results = {}
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            futures = {executor.submit(self._run_single_backtest, leverage): leverage 
                      for leverage in leverage_values}
            
            # 收集结果
            for future in as_completed(futures):
                leverage = futures[future]
                try:
                    result = future.result()
                    if result:
                        results[leverage] = result
                        print(f"杠杆值 {leverage} 测试完成，{self.criterion}: {result.get(self.criterion, 'N/A')}")
                except Exception as exc:
                    print(f"杠杆值 {leverage} 处理出错: {exc}")
        
        return results
    
    def _run_single_backtest(self, leverage):
        """
        运行单个杠杆值的回测
        
        参数:
            leverage (float): 要测试的杠杆值
            
        返回:
            dict: 回测结果指标
        """
        try:
            # 复制配置并设置杠杆值
            config_copy = copy.deepcopy(self.config)
            config_copy["backtest"]["leverage"] = leverage
            
            # 运行回测
            signals_df, trade_recorder = run_strategy(
                data_source=self.data_source, 
                custom_config=config_copy,
                visualize=False,
                analyze_signals=False,
                export_excel=False
            )
            
            # 如果回测成功，计算并返回指标
            if signals_df is not None and not signals_df.empty and trade_recorder is not None:
                # 获取交易统计摘要
                trade_summary = trade_recorder.get_trade_summary()
                
                # 提取资金曲线
                equity_curve = signals_df["equity"]
                
                # 计算关键指标
                initial_capital = config_copy["backtest"].get("initial_capital", 10000)
                final_equity = equity_curve.iloc[-1]
                total_return = (final_equity - initial_capital) / initial_capital
                
                # 获取最大回撤
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
                
                # 获取其他统计数据
                win_rate = trade_summary.get("胜率", 0)
                trade_count = trade_summary.get("交易次数", 0)
                avg_profit = trade_summary.get("平均盈利", 0)
                avg_loss = trade_summary.get("平均亏损", 0)
                
                # 获取持仓时间相关指标
                avg_holding_time = trade_summary.get("平均持仓时间", None)
                profit_holding_time = trade_summary.get("盈利交易平均持仓时间", None)
                loss_holding_time = trade_summary.get("亏损交易平均持仓时间", None)
                
                # 构建结果字典
                result = {
                    "leverage": leverage,
                    "total_return": total_return,
                    "max_drawdown": max_drawdown,
                    "sharpe_ratio": sharpe_ratio,
                    "calmar_ratio": calmar_ratio,
                    "profit_factor": profit_factor,
                    "final_equity": final_equity,
                    "win_rate": win_rate,
                    "trade_count": trade_count,
                    "avg_profit": avg_profit,
                    "avg_loss": avg_loss,
                    "avg_holding_time": avg_holding_time,
                    "profit_holding_time": profit_holding_time,
                    "loss_holding_time": loss_holding_time
                }
                
                return result
            
            return None
        
        except Exception as e:
            print(f"杠杆值 {leverage} 回测失败: {e}")
            return None
    
    def _find_best_value(self, results):
        """
        根据指定标准找出最佳杠杆值
        
        参数:
            results (dict): 回测结果字典
            
        返回:
            float: 最佳杠杆值
        """
        if not results:
            return None
            
        # 根据优化标准排序
        sorted_results = sorted(
            [(lev, res[self.criterion]) for lev, res in results.items()],
            key=lambda x: x[1], reverse=True
        )
        
        # 返回最佳杠杆值
        return sorted_results[0][0] if sorted_results else None
    
    def visualize_results(self, include_search_path=True, save_path=None):
        """
        可视化优化结果
        
        参数:
            include_search_path (bool): 是否包含搜索路径
            save_path (str): 保存路径
        """
        if not self.results:
            print("没有结果可视化")
            return
            
        # 设置中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文
        plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
        
        # 创建网格图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('动态杠杆优化结果', fontsize=16)
        
        # 获取所有测试过的杠杆值
        leverage_values = sorted(self.results.keys())
        
        # 提取各个指标
        total_returns = [self.results[lev]["total_return"] for lev in leverage_values]
        sharpe_ratios = [self.results[lev]["sharpe_ratio"] for lev in leverage_values]
        calmar_ratios = [self.results[lev]["calmar_ratio"] for lev in leverage_values]
        max_drawdowns = [self.results[lev]["max_drawdown"] for lev in leverage_values]
        
        # 1. 绘制总回报率
        axes[0, 0].plot(leverage_values, total_returns, 'o-', color='blue')
        axes[0, 0].set_title('杠杆 vs 总回报率')
        axes[0, 0].set_xlabel('杠杆')
        axes[0, 0].set_ylabel('总回报率')
        axes[0, 0].grid(True)
        
        # 标出最佳值
        if 'total_return' == self.criterion:
            best_idx = total_returns.index(max(total_returns))
            best_lev = leverage_values[best_idx]
            best_val = total_returns[best_idx]
            axes[0, 0].plot(best_lev, best_val, 'r*', markersize=12)
            axes[0, 0].annotate(f'最佳: {best_lev}', 
                              xy=(best_lev, best_val),
                              xytext=(best_lev, best_val * 0.9),
                              arrowprops=dict(arrowstyle='->'))
        
        # 2. 绘制夏普比率
        axes[0, 1].plot(leverage_values, sharpe_ratios, 'o-', color='green')
        axes[0, 1].set_title('杠杆 vs 夏普比率')
        axes[0, 1].set_xlabel('杠杆')
        axes[0, 1].set_ylabel('夏普比率')
        axes[0, 1].grid(True)
        
        # 标出最佳值
        if 'sharpe_ratio' == self.criterion:
            best_idx = sharpe_ratios.index(max(sharpe_ratios))
            best_lev = leverage_values[best_idx]
            best_val = sharpe_ratios[best_idx]
            axes[0, 1].plot(best_lev, best_val, 'r*', markersize=12)
            axes[0, 1].annotate(f'最佳: {best_lev}', 
                              xy=(best_lev, best_val),
                              xytext=(best_lev, best_val * 0.9),
                              arrowprops=dict(arrowstyle='->'))
        
        # 3. 绘制卡尔玛比率
        axes[1, 0].plot(leverage_values, calmar_ratios, 'o-', color='purple')
        axes[1, 0].set_title('杠杆 vs 卡尔玛比率')
        axes[1, 0].set_xlabel('杠杆')
        axes[1, 0].set_ylabel('卡尔玛比率')
        axes[1, 0].grid(True)
        
        # 标出最佳值
        if 'calmar_ratio' == self.criterion:
            best_idx = calmar_ratios.index(max(calmar_ratios))
            best_lev = leverage_values[best_idx]
            best_val = calmar_ratios[best_idx]
            axes[1, 0].plot(best_lev, best_val, 'r*', markersize=12)
            axes[1, 0].annotate(f'最佳: {best_lev}', 
                              xy=(best_lev, best_val),
                              xytext=(best_lev, best_val * 0.9),
                              arrowprops=dict(arrowstyle='->'))
        
        # 4. 绘制最大回撤
        axes[1, 1].plot(leverage_values, max_drawdowns, 'o-', color='red')
        axes[1, 1].set_title('杠杆 vs 最大回撤')
        axes[1, 1].set_xlabel('杠杆')
        axes[1, 1].set_ylabel('最大回撤')
        axes[1, 1].grid(True)
        
        # 如果需要，绘制搜索路径
        if include_search_path and self.search_history:
            search_ax = fig.add_subplot(3, 1, 3)
            
            # 绘制每次迭代的搜索范围
            iterations = [h['iteration'] for h in self.search_history]
            min_ranges = [h['min_leverage'] for h in self.search_history]
            max_ranges = [h['max_leverage'] for h in self.search_history]
            best_in_iter = [h['best_in_iteration'] for h in self.search_history]
            
            search_ax.plot(iterations, min_ranges, 'b-', label='最小杠杆')
            search_ax.plot(iterations, max_ranges, 'r-', label='最大杠杆')
            search_ax.plot(iterations, best_in_iter, 'g*-', label='每轮最佳')
            
            search_ax.set_title('搜索路径')
            search_ax.set_xlabel('迭代次数')
            search_ax.set_ylabel('杠杆值')
            search_ax.grid(True)
            search_ax.legend()
        
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
            filename (str): 导出文件名
            
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
            filename = f"dynamic_leverage_optimization_{strategy_name}_{now}.xlsx"
        
        try:
            # 创建Excel写入器
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # 1. 优化结果
                data = []
                for leverage, metrics in self.results.items():
                    row = {"杠杆": leverage}
                    
                    # 添加数值指标
                    for key, value in metrics.items():
                        if key != "leverage":
                            row[key] = value
                            
                    data.append(row)
                
                # 创建DataFrame
                results_df = pd.DataFrame(data)
                
                # 排序和设置列顺序
                results_df = results_df.sort_values('杠杆')
                
                # 重命名列
                column_map = {
                    "total_return": "总回报率",
                    "max_drawdown": "最大回撤",
                    "sharpe_ratio": "夏普比率",
                    "calmar_ratio": "卡尔玛比率",
                    "profit_factor": "盈利因子",
                    "final_equity": "最终净值",
                    "win_rate": "胜率",
                    "trade_count": "交易次数",
                    "avg_profit": "平均盈利",
                    "avg_loss": "平均亏损"
                }
                
                results_df = results_df.rename(columns=column_map)
                results_df.to_excel(writer, sheet_name='优化结果', index=False)
                
                # 2. 搜索历史
                if self.search_history:
                    history_data = []
                    for h in self.search_history:
                        history_data.append({
                            "迭代次数": h['iteration'],
                            "最小杠杆": h['min_leverage'],
                            "最大杠杆": h['max_leverage'],
                            "步长": h['step'],
                            "测试值数量": len(h['values_tested']),
                            "本轮最佳杠杆": h['best_in_iteration']
                        })
                    
                    history_df = pd.DataFrame(history_data)
                    history_df.to_excel(writer, sheet_name='搜索历史', index=False)
                
                # 3. 最佳杠杆详细结果
                if self.best_leverage in self.results:
                    best_result = self.results[self.best_leverage]
                    best_data = []
                    
                    for key, value in best_result.items():
                        metric_name = column_map.get(key, key)
                        if isinstance(value, float):
                            if key in ['total_return', 'max_drawdown', 'win_rate']:
                                formatted_value = f"{value:.2%}"
                            else:
                                formatted_value = f"{value:.4f}"
                        else:
                            formatted_value = str(value)
                            
                        best_data.append({
                            "指标": metric_name,
                            "数值": formatted_value
                        })
                    
                    best_df = pd.DataFrame(best_data)
                    best_df.to_excel(writer, sheet_name='最佳杠杆详情', index=False)
            
            print(f"优化结果已导出到: {filename}")
            return filename
            
        except Exception as e:
            print(f"导出到Excel失败: {e}")
            return None

def run_dynamic_optimization(data_source, min_leverage=1.0, max_leverage=20.0, 
                            criterion='sharpe_ratio', parallel_jobs=None):
    """
    运行动态杠杆优化并可视化结果
    
    参数:
        data_source (str or pd.DataFrame): 数据源
        min_leverage (float): 最小杠杆值
        max_leverage (float): 最大杠杆值
        criterion (str): 优化标准
        parallel_jobs (int): 并行任务数
        
    返回:
        DynamicLeverageOptimizer: 优化器实例
    """
    # 加载配置
    config = load_config()
    
    # 创建动态优化器
    optimizer = DynamicLeverageOptimizer(
        data_source=data_source,
        config=config,
        min_leverage=min_leverage,
        max_leverage=max_leverage,
        initial_step=1.0,
        min_step=0.1,
        parallel_jobs=parallel_jobs,
        criterion=criterion
    )
    
    # 运行优化
    optimizer.run_optimization(max_iterations=10)
    
    # 可视化结果
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_path = f"dynamic_leverage_optimization_chart_{now}.png"
    optimizer.visualize_results(include_search_path=True, save_path=chart_path)
    
    # 导出结果到Excel
    optimizer.export_results_to_excel()
    
    return optimizer

def main():
    """主函数"""
    print("\n===== 动态杠杆优化工具 =====")
    
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
    print("\n设置动态杠杆优化参数:")
    try:
        min_leverage = float(input("初始最小杠杆值 (默认: 1.0): ") or 1.0)
        max_leverage = float(input("初始最大杠杆值 (默认: 20.0): ") or 20.0)
        
        # 选择并行处理
        use_parallel = input("是否使用并行处理加速优化? (y/n, 默认: n): ").lower()
        parallel_jobs = None
        if use_parallel.startswith('y'):
            import multiprocessing
            max_cpus = multiprocessing.cpu_count()
            suggested = max(1, max_cpus - 1)  # 留一个核心给系统
            parallel_jobs = int(input(f"并行任务数 (建议: {suggested}, 最大: {max_cpus}): ") or suggested)
    except ValueError:
        print("请输入有效的数字")
        return
    
    # 选择优化标准
    print("\n选择优化标准:")
    print("1. 总回报率 (最大化总盈利)")
    print("2. 夏普比率 (风险调整回报)")
    print("3. 卡尔玛比率 (回报/最大回撤)")
    print("4. 盈利因子 (总盈利/总亏损)")
    
    criterion_choice = input("请选择优化标准 (1-4, 默认: 2): ") or '2'
    criterion = 'sharpe_ratio'  # 默认
    if criterion_choice == '1':
        criterion = 'total_return'
    elif criterion_choice == '3':
        criterion = 'calmar_ratio'
    elif criterion_choice == '4':
        criterion = 'profit_factor'
    
    # 运行优化
    run_dynamic_optimization(
        data_source=data_source,
        min_leverage=min_leverage,
        max_leverage=max_leverage,
        criterion=criterion,
        parallel_jobs=parallel_jobs
    )

if __name__ == "__main__":
    main() 