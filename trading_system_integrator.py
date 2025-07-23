"""
比特币交易系统集成器

该模块整合了传统技术指标交易系统与机器学习增强系统，提供统一的接口。
可以选择使用传统策略、ML策略或两者结合进行交易和回测。
"""

import os
import yaml
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

# 导入传统交易系统组件
from data import DataManager
from indicators import OptimizedIndicatorCalculator
from signals import SignalGenerator, StrategyFactory
from trade_recorder import OptimizedTradeRecorder

# 导入ML系统组件
from ml_engine import MLSignalEnhancer

class TradingSystemIntegrator:
    """
    交易系统集成器
    
    整合传统技术分析交易系统和机器学习增强系统
    提供统一的接口进行回测和实时交易
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化系统集成器
        
        参数:
            config_path (str): 配置文件路径
        """
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("TradingSystemIntegrator")
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 系统组件
        self.data_manager = None
        self.indicator_calculator = None
        self.signal_generator = None
        self.trade_recorder = None
        self.ml_enhancer = None
        
        # 系统状态
        self.is_initialized = False
        self.has_ml_support = self.config.get("enable_ml", False)
        
        # 初始化系统
        self._initialize_system()
        
        self.logger.info(f"交易系统集成器初始化完成，ML支持: {self.has_ml_support}")
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
                self.logger.info(f"成功加载配置: {config_path}")
                return config
        except FileNotFoundError:
            self.logger.warning(f"配置文件未找到: {config_path}，使用默认配置")
            return self._load_default_config()
        except Exception as e:
            self.logger.error(f"加载配置失败: {e}")
            return self._load_default_config()
    
    def _load_default_config(self) -> Dict:
        """加载默认配置"""
        default_config_path = "config_default.yaml"
        if os.path.exists(default_config_path):
            with open(default_config_path, "r", encoding="utf-8") as file:
                return yaml.safe_load(file)
        
        # 内置默认配置
        return {
            "trading": {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "is_futures": True
            },
            "indicators": {
                "dema144_len": 144,
                "dema169_len": 169,
                "atr_period": 34,
                "atr_multiplier": 3.0
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
            "enable_ml": False,
            "ml_integration": {
                "confidence_threshold": 0.6,
                "signal_weight_technical": 0.6,
                "signal_weight_ml": 0.4
            }
        }
    
    def _initialize_system(self):
        """初始化系统组件"""
        try:
            # 初始化传统系统组件
            self.logger.info("初始化传统交易系统组件...")
            self.data_manager = DataManager(self.config)
            self.indicator_calculator = OptimizedIndicatorCalculator(self.config)
            self.signal_generator = SignalGenerator(self.config)
            self.trade_recorder = OptimizedTradeRecorder(self.config)
            
            # 如果启用ML，初始化ML组件
            if self.has_ml_support:
                self.logger.info("初始化机器学习增强组件...")
                self.ml_enhancer = MLSignalEnhancer(self.config)
            
            self.is_initialized = True
            
        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}")
            self.is_initialized = False
    
    def run_backtest(self, 
                    data_source: Any = None, 
                    start_time: str = None, 
                    end_time: str = None,
                    use_ml: Optional[bool] = None) -> Dict:
        """
        运行回测
        
        参数:
            data_source (Any): 数据源，可以是文件路径或DataFrame
            start_time (str): 开始时间，如 '2023-01-01'
            end_time (str): 结束时间，如 '2023-12-31'
            use_ml (bool): 是否使用ML增强，None表示使用配置默认值
            
        返回:
            Dict: 回测结果
        """
        if not self.is_initialized:
            return {"success": False, "error": "系统未正确初始化"}
        
        # 确定是否使用ML
        use_ml_signals = use_ml if use_ml is not None else self.has_ml_support
        
        try:
            # 1. 获取数据
            self.logger.info("获取回测数据...")
            df = self._get_data(data_source, start_time, end_time)
            if df is None or df.empty:
                return {"success": False, "error": "无法获取回测数据"}
            
            # 2. 计算指标
            self.logger.info("计算技术指标...")
            indicators_df = self.indicator_calculator.calculate_all_indicators_optimized(df)
            
            # 3. 生成传统信号
            self.logger.info("生成传统交易信号...")
            signals_df = self.signal_generator.generate_signals(indicators_df)
            signals_df = self.signal_generator.calculate_risk_reward(signals_df)
            
            # 4. 如果启用ML，增强信号
            if use_ml_signals and self.ml_enhancer is not None:
                self.logger.info("使用机器学习增强信号...")
                # 首先训练模型（如果尚未训练）
                if not getattr(self.ml_enhancer, 'models_trained', False):
                    self.logger.info("训练ML模型...")
                    self.ml_enhancer.train_models(df)
                
                # 为每个时间点生成ML增强信号
                ml_signals = pd.Series(index=signals_df.index, dtype=float)
                ml_confidence = pd.Series(index=signals_df.index, dtype=float)
                
                # 使用滚动窗口进行预测以避免前瞻偏差
                for i in range(100, len(signals_df)):
                    history_data = df.iloc[:i]
                    current_data = df.iloc[i:i+1]
                    
                    # 获取传统信号
                    tech_signal = signals_df["position"].iloc[i] if "position" in signals_df.columns else 0
                    
                    # 生成ML预测
                    prediction = self.ml_enhancer.predict_signals(
                        current_data, 
                        pd.Series([tech_signal])
                    )
                    
                    if prediction.get("success", False):
                        ml_signals.iloc[i] = prediction.get("signal", 0)
                        ml_confidence.iloc[i] = prediction.get("confidence_score", 0.5)
                
                # 合并信号
                signals_df = self._integrate_signals(
                    signals_df, 
                    ml_signals, 
                    ml_confidence
                )
            
            # 5. 处理交易
            self.logger.info("处理交易信号...")
            backtest_df, trade_recorder = self._process_trades(signals_df)
            
            # 6. 计算回测结果
            self.logger.info("计算回测结果...")
            summary = trade_recorder.get_trade_summary()
            
            return {
                "success": True,
                "signals_df": signals_df,
                "backtest_df": backtest_df,
                "trade_recorder": trade_recorder,
                "summary": summary,
                "used_ml": use_ml_signals
            }
            
        except Exception as e:
            self.logger.error(f"回测失败: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def generate_live_signal(self, 
                           market_data: pd.DataFrame,
                           use_ml: Optional[bool] = None) -> Dict:
        """
        生成实时交易信号
        
        参数:
            market_data (pd.DataFrame): 最新市场数据
            use_ml (bool): 是否使用ML增强
            
        返回:
            Dict: 交易信号
        """
        if not self.is_initialized:
            return {"success": False, "error": "系统未正确初始化"}
        
        # 确定是否使用ML
        use_ml_signals = use_ml if use_ml is not None else self.has_ml_support
        
        try:
            # 1. 计算指标
            indicators_df = self.indicator_calculator.calculate_all_indicators_optimized(market_data)
            
            # 2. 生成传统信号
            signals_df = self.signal_generator.generate_signals(indicators_df)
            signals_df = self.signal_generator.calculate_risk_reward(signals_df)
            
            # 获取最新信号
            latest_idx = signals_df.index[-1]
            tech_signal = signals_df.loc[latest_idx, "position"] if "position" in signals_df.columns else 0
            
            # 初始化结果
            signal_result = {
                "timestamp": latest_idx,
                "technical_signal": tech_signal,
                "final_signal": tech_signal,
                "confidence": 0.7,  # 默认技术信号置信度
                "ml_signal": None,
                "ml_confidence": None
            }
            
            # 3. 如果启用ML，增强信号
            if use_ml_signals and self.ml_enhancer is not None:
                # 生成ML预测
                ml_prediction = self.ml_enhancer.predict_signals(
                    market_data.tail(1),
                    pd.Series([tech_signal])
                )
                
                if ml_prediction.get("success", False):
                    ml_signal = ml_prediction.get("signal", 0)
                    ml_confidence = ml_prediction.get("confidence_score", 0.5)
                    
                    # 更新结果
                    signal_result["ml_signal"] = ml_signal
                    signal_result["ml_confidence"] = ml_confidence
                    
                    # 整合信号
                    ml_integration = self.config.get("ml_integration", {})
                    tech_weight = ml_integration.get("signal_weight_technical", 0.6)
                    ml_weight = ml_integration.get("signal_weight_ml", 0.4)
                    
                    # 根据置信度调整权重
                    if ml_confidence > 0.8:
                        ml_weight *= 1.2
                        tech_weight = 1.0 - ml_weight
                    elif ml_confidence < 0.5:
                        ml_weight *= 0.5
                        tech_weight = 1.0 - ml_weight
                    
                    # 计算加权信号
                    weighted_signal = (tech_signal * tech_weight) + (ml_signal * ml_weight)
                    final_signal = int(np.sign(weighted_signal))
                    
                    # 更新结果
                    signal_result["final_signal"] = final_signal
                    signal_result["confidence"] = max(ml_confidence, 0.7)
            
            # 添加风险管理信息
            if tech_signal != 0:
                direction = "多" if tech_signal > 0 else "空"
                signal_result["direction"] = direction
                
                # 从signals_df获取止损和目标价格
                if tech_signal > 0 and "stop_loss_buy" in signals_df.columns:
                    signal_result["stop_loss"] = signals_df.loc[latest_idx, "stop_loss_buy"]
                    signal_result["target"] = signals_df.loc[latest_idx, "target_price_buy"] \
                        if "target_price_buy" in signals_df.columns else None
                elif tech_signal < 0 and "stop_loss_sell" in signals_df.columns:
                    signal_result["stop_loss"] = signals_df.loc[latest_idx, "stop_loss_sell"]
                    signal_result["target"] = signals_df.loc[latest_idx, "target_price_sell"] \
                        if "target_price_sell" in signals_df.columns else None
            
            return {
                "success": True,
                "signal": signal_result,
                "used_ml": use_ml_signals and self.ml_enhancer is not None
            }
            
        except Exception as e:
            self.logger.error(f"信号生成失败: {e}")
            return {"success": False, "error": str(e)}
    
    def train_ml_models(self, training_data: pd.DataFrame) -> Dict:
        """
        训练ML模型
        
        参数:
            training_data (pd.DataFrame): 训练数据
            
        返回:
            Dict: 训练结果
        """
        if not self.has_ml_support or self.ml_enhancer is None:
            return {"success": False, "error": "机器学习增强功能未启用"}
        
        try:
            result = self.ml_enhancer.train_models(training_data)
            return result
        except Exception as e:
            self.logger.error(f"模型训练失败: {e}")
            return {"success": False, "error": str(e)}
    
    def update_ml_models(self, new_data: pd.DataFrame) -> Dict:
        """
        更新ML模型
        
        参数:
            new_data (pd.DataFrame): 新数据
            
        返回:
            Dict: 更新结果
        """
        if not self.has_ml_support or self.ml_enhancer is None:
            return {"success": False, "error": "机器学习增强功能未启用"}
        
        try:
            result = self.ml_enhancer.update_models(new_data)
            return result
        except Exception as e:
            self.logger.error(f"模型更新失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_data(self, data_source, start_time, end_time) -> pd.DataFrame:
        """获取数据"""
        if isinstance(data_source, pd.DataFrame):
            return data_source
        else:
            return self.data_manager.fetch_klines(
                start_time=start_time,
                end_time=end_time,
                cache_file=data_source
            )
    
    def _process_trades(self, signals_df) -> Tuple[pd.DataFrame, OptimizedTradeRecorder]:
        """处理交易"""
        backtest_df = self.trade_recorder.process_signals_vectorized(signals_df)
        return backtest_df, self.trade_recorder
    
    def _integrate_signals(self, signals_df, ml_signals, ml_confidence) -> pd.DataFrame:
        """整合传统信号和ML信号"""
        # 复制DataFrame以避免修改原始数据
        integrated_df = signals_df.copy()
        
        # 获取配置中的权重
        ml_integration = self.config.get("ml_integration", {})
        tech_weight = ml_integration.get("signal_weight_technical", 0.6)
        ml_weight = ml_integration.get("signal_weight_ml", 0.4)
        
        # 添加ML信号和置信度列
        integrated_df["ml_signal"] = ml_signals
        integrated_df["ml_confidence"] = ml_confidence
        
        # 计算整合后的信号
        for i in range(len(integrated_df)):
            if pd.isna(ml_signals.iloc[i]) or pd.isna(ml_confidence.iloc[i]):
                continue
                
            # 获取两个信号
            tech_signal = integrated_df["position"].iloc[i] if "position" in integrated_df.columns else 0
            ml_signal = ml_signals.iloc[i]
            confidence = ml_confidence.iloc[i]
            
            # 根据置信度调整权重
            adjusted_ml_weight = ml_weight
            adjusted_tech_weight = tech_weight
            
            if confidence > 0.8:
                adjusted_ml_weight = ml_weight * 1.2
                adjusted_tech_weight = 1.0 - adjusted_ml_weight
            elif confidence < 0.5:
                adjusted_ml_weight = ml_weight * 0.5
                adjusted_tech_weight = 1.0 - adjusted_ml_weight
            
            # 计算加权信号
            weighted_signal = (tech_signal * adjusted_tech_weight) + (ml_signal * adjusted_ml_weight)
            integrated_df.loc[integrated_df.index[i], "integrated_position"] = int(np.sign(weighted_signal))
        
        # 使用整合信号替换原始信号
        if "integrated_position" in integrated_df.columns:
            integrated_df["original_position"] = integrated_df["position"]
            integrated_df["position"] = integrated_df["integrated_position"]
        
        return integrated_df
    
    def get_available_strategies(self) -> list:
        """获取可用的交易策略列表"""
        return StrategyFactory.get_strategy_list()
    
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        status = {
            "is_initialized": self.is_initialized,
            "has_ml_support": self.has_ml_support,
            "current_strategy": self.config.get("signals", {}).get("strategy", "未设置"),
            "data_ready": self.data_manager is not None
        }
        
        # 如果启用了ML，添加ML状态
        if self.has_ml_support and self.ml_enhancer is not None:
            ml_status = {}
            if hasattr(self.ml_enhancer, "get_model_performance"):
                ml_status = self.ml_enhancer.get_model_performance()
            status["ml_status"] = ml_status
        
        return status


# 便捷函数
def create_trading_system(config_path: str = "config.yaml") -> TradingSystemIntegrator:
    """
    创建交易系统实例
    
    参数:
        config_path (str): 配置文件路径
        
    返回:
        TradingSystemIntegrator: 交易系统实例
    """
    return TradingSystemIntegrator(config_path)


if __name__ == "__main__":
    # 简单的测试代码
    system = create_trading_system()
    status = system.get_system_status()
    print(f"系统状态: {status}")
    
    strategies = system.get_available_strategies()
    print(f"可用策略: {strategies}") 