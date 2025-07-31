"""
智能ML训练管理器

提供便捷的训练管理界面，支持：
- 启动时自动检测未完成训练
- 交互式菜单操作
- 训练进度监控
- 检查点管理
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time
from colorama import init, Fore, Back, Style

# 确保能导入本地模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ml_engine.training.model_trainer import ModelTrainer
    from ml_engine.features.feature_engineering import FeatureEngineering
    from ml_engine.utils.model_version_manager import ModelVersionManager
except ImportError as e:
    print(f"⚠️  导入模块失败: {e}")
    print("请确保ML引擎模块在正确的路径下")

# 初始化colorama以支持Windows彩色输出
init(autoreset=True)

class TrainingManager:
    """智能ML训练管理器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化训练管理器
        
        参数:
            config_path (str): 配置文件路径
        """
        self.config_path = config_path
        self.config = None
        self.trainer = None
        self.feature_engineer = None
        self.version_manager = None
        
        # 训练数据
        self.X_train = None
        self.y_train = None
        self.data_loaded = False
        
        # 日志设置
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 界面配置
        self.colors = {
            'header': Fore.CYAN + Style.BRIGHT,
            'success': Fore.GREEN + Style.BRIGHT,
            'warning': Fore.YELLOW + Style.BRIGHT,
            'error': Fore.RED + Style.BRIGHT,
            'info': Fore.BLUE,
            'menu': Fore.MAGENTA,
            'reset': Style.RESET_ALL
        }
        
        print(f"{self.colors['header']}🚀 ML训练管理器启动中...{self.colors['reset']}")
        
        # 初始化组件
        self.initialize_components()
    
    def setup_logging(self):
        """设置日志配置"""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{log_dir}/training_manager.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def initialize_components(self):
        """初始化所有组件"""
        try:
            # 加载配置
            self.load_config()
            
            # 初始化训练器
            self.trainer = ModelTrainer(self.config)
            
            # 初始化特征工程器
            if 'FeatureEngineering' in globals():
                self.feature_engineer = FeatureEngineering(self.config)
            
            # 初始化版本管理器
            if 'ModelVersionManager' in globals():
                self.version_manager = ModelVersionManager(self.config)
            
            print(f"{self.colors['success']}✅ 组件初始化完成{self.colors['reset']}")
            
        except Exception as e:
            print(f"{self.colors['error']}❌ 组件初始化失败: {str(e)}{self.colors['reset']}")
            self.logger.error(f"组件初始化失败: {str(e)}")
            raise e
    
    def load_config(self):
        """加载配置文件"""
        try:
            if not os.path.exists(self.config_path):
                print(f"{self.colors['warning']}⚠️  配置文件不存在，使用默认配置{self.colors['reset']}")
                self.config = self.get_default_config()
                return
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            print(f"{self.colors['success']}✅ 配置文件加载成功{self.colors['reset']}")
            
        except Exception as e:
            print(f"{self.colors['error']}❌ 配置文件加载失败: {str(e)}{self.colors['reset']}")
            print(f"{self.colors['info']}使用默认配置继续...{self.colors['reset']}")
            self.config = self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "ml_signal_enhancer": {
                "model_trainer": {
                    "enable_checkpoints": True,
                    "model_types": ["random_forest", "xgboost", "lightgbm"],
                    "test_size": 0.2,
                    "enable_hyperopt": True,
                    "optuna_trials": 50
                },
                "training_checkpoint": {
                    "checkpoint_dir": "ml_engine/checkpoints/",
                    "auto_save_interval": 1,
                    "max_checkpoints": 5,
                    "enable_data_checkpoints": True
                },
                "data_source": {
                    "file_path": "data/BTC_USDT_1h_20190101_to_20250627.csv",
                    "target_column": "signal",
                    "feature_columns": "auto"
                }
            }
        }
    
    def startup_check(self):
        """启动检测未完成的训练"""
        print(f"\n{self.colors['header']}🔍 检测未完成的训练会话...{self.colors['reset']}")
        
        try:
            # 获取检查点列表
            checkpoints = self.trainer.list_training_checkpoints()
            
            if not checkpoints:
                print(f"{self.colors['info']}📭 没有找到历史训练记录{self.colors['reset']}")
                return False
            
            # 查找可恢复的训练
            resumable_checkpoints = [cp for cp in checkpoints if cp.get('can_resume', False)]
            
            if not resumable_checkpoints:
                print(f"{self.colors['info']}📋 找到 {len(checkpoints)} 个已完成的训练记录{self.colors['reset']}")
                return False
            
            # 显示可恢复的训练
            print(f"{self.colors['warning']}⚡ 发现 {len(resumable_checkpoints)} 个未完成的训练会话:{self.colors['reset']}")
            
            for i, checkpoint in enumerate(resumable_checkpoints[:3]):  # 只显示前3个
                progress = f"{checkpoint['completed_models']}/{checkpoint['total_models']}"
                status_color = self.colors['warning'] if checkpoint['status'] == 'running' else self.colors['error']
                
                print(f"   {i+1}. {status_color}会话: {checkpoint['session_id'][:16]}...{self.colors['reset']}")
                print(f"      进度: {progress} 模型 | 状态: {checkpoint['status']}")
                print(f"      时间: {checkpoint['start_time']}")
                print()
            
            # 询问用户是否恢复
            response = self.get_user_choice(
                f"{self.colors['menu']}🤔 是否恢复最新的未完成训练？{self.colors['reset']}", 
                ['y', 'n', 'yes', 'no'], 
                'y'
            )
            
            if response.lower() in ['y', 'yes']:
                # 恢复最新的训练
                latest_checkpoint = resumable_checkpoints[0]
                return self.resume_specific_training(latest_checkpoint['session_id'])
            
            return False
            
        except Exception as e:
            print(f"{self.colors['error']}❌ 启动检测失败: {str(e)}{self.colors['reset']}")
            self.logger.error(f"启动检测失败: {str(e)}")
            return False
    
    def show_main_menu(self):
        """显示主菜单"""
        while True:
            print(f"\n{self.colors['header']}{'='*60}{self.colors['reset']}")
            print(f"{self.colors['header']}🎯 ML训练管理器 - 主菜单{self.colors['reset']}")
            print(f"{self.colors['header']}{'='*60}{self.colors['reset']}")
            
            # 显示状态信息
            self.show_status_info()
            
            print(f"\n{self.colors['menu']}请选择操作:{self.colors['reset']}")
            print(f"  {self.colors['info']}1.{self.colors['reset']} 🚀 开始新训练")
            print(f"  {self.colors['info']}2.{self.colors['reset']} 🔄 恢复训练")
            print(f"  {self.colors['info']}3.{self.colors['reset']} 📊 查看训练历史")
            print(f"  {self.colors['info']}4.{self.colors['reset']} 🔧 管理检查点")
            print(f"  {self.colors['info']}5.{self.colors['reset']} 📁 加载训练数据")
            print(f"  {self.colors['info']}6.{self.colors['reset']} ⚙️  查看配置")
            print(f"  {self.colors['info']}7.{self.colors['reset']} 📈 查看训练状态")
            print(f"  {self.colors['info']}8.{self.colors['reset']} 🧹 清理检查点")
            print(f"  {self.colors['info']}0.{self.colors['reset']} 🚪 退出")
            
            choice = input(f"\n{self.colors['menu']}请输入选择 (0-8): {self.colors['reset']}").strip()
            
            try:
                if choice == '1':
                    self.start_new_training()
                elif choice == '2':
                    self.resume_training_menu()
                elif choice == '3':
                    self.view_training_history()
                elif choice == '4':
                    self.manage_checkpoints()
                elif choice == '5':
                    self.load_training_data()
                elif choice == '6':
                    self.view_configuration()
                elif choice == '7':
                    self.view_training_status()
                elif choice == '8':
                    self.cleanup_checkpoints()
                elif choice == '0':
                    self.exit_manager()
                    break
                else:
                    print(f"{self.colors['warning']}⚠️  无效选择，请输入 0-8{self.colors['reset']}")
            
            except KeyboardInterrupt:
                print(f"\n{self.colors['warning']}⚠️  操作被用户取消{self.colors['reset']}")
            except Exception as e:
                print(f"{self.colors['error']}❌ 操作失败: {str(e)}{self.colors['reset']}")
                self.logger.error(f"菜单操作失败: {str(e)}")
    
    def show_status_info(self):
        """显示状态信息"""
        print(f"\n{self.colors['info']}📊 当前状态:{self.colors['reset']}")
        
        # 数据状态
        data_status = f"{self.colors['success']}✅ 已加载{self.colors['reset']}" if self.data_loaded else f"{self.colors['warning']}⚠️  未加载{self.colors['reset']}"
        print(f"   数据状态: {data_status}")
        
        if self.data_loaded:
            print(f"   数据规模: {len(self.X_train)} 样本, {self.X_train.shape[1]} 特征")
        
        # 检查点状态
        try:
            stats = self.trainer.get_checkpoint_stats()
            print(f"   检查点: {stats['total_checkpoints']} 个 (可恢复: {stats['resumable_sessions']})")
        except:
            print(f"   检查点: {self.colors['error']}获取失败{self.colors['reset']}")
    
    def start_new_training(self):
        """开始新训练"""
        print(f"\n{self.colors['header']}🚀 开始新训练{self.colors['reset']}")
        
        # 检查数据是否已加载
        if not self.data_loaded:
            print(f"{self.colors['warning']}⚠️  训练数据未加载{self.colors['reset']}")
            
            load_choice = self.get_user_choice(
                "是否现在加载数据？", 
                ['y', 'n'], 
                'y'
            )
            
            if load_choice.lower() == 'y':
                if not self.load_training_data():
                    print(f"{self.colors['error']}❌ 数据加载失败，无法开始训练{self.colors['reset']}")
                    return
            else:
                print(f"{self.colors['info']}训练已取消{self.colors['reset']}")
                return
        
        # 显示训练配置
        self.show_training_config()
        
        # 确认开始训练
        confirm = self.get_user_choice(
            f"{self.colors['menu']}确认开始训练？{self.colors['reset']}", 
            ['y', 'n'], 
            'y'
        )
        
        if confirm.lower() != 'y':
            print(f"{self.colors['info']}训练已取消{self.colors['reset']}")
            return
        
        # 开始训练
        try:
            print(f"\n{self.colors['success']}🔥 启动训练流程...{self.colors['reset']}")
            results = self.trainer.train_models_with_checkpoint(self.X_train, self.y_train)
            
            if results['success']:
                self.show_training_results(results)
            else:
                print(f"{self.colors['error']}❌ 训练失败: {results['error']}{self.colors['reset']}")
                
        except KeyboardInterrupt:
            print(f"\n{self.colors['warning']}⚡ 训练被用户中断{self.colors['reset']}")
            print(f"{self.colors['info']}💡 下次启动时可以选择恢复训练{self.colors['reset']}")
        except Exception as e:
            print(f"{self.colors['error']}❌ 训练异常: {str(e)}{self.colors['reset']}")
            self.logger.error(f"训练异常: {str(e)}")
    
    def resume_training_menu(self):
        """恢复训练菜单"""
        print(f"\n{self.colors['header']}🔄 恢复训练{self.colors['reset']}")
        
        try:
            checkpoints = self.trainer.list_training_checkpoints()
            resumable_checkpoints = [cp for cp in checkpoints if cp.get('can_resume', False)]
            
            if not resumable_checkpoints:
                print(f"{self.colors['warning']}⚠️  没有找到可恢复的训练会话{self.colors['reset']}")
                return
            
            print(f"{self.colors['info']}📋 可恢复的训练会话:{self.colors['reset']}")
            
            for i, checkpoint in enumerate(resumable_checkpoints):
                progress = f"{checkpoint['completed_models']}/{checkpoint['total_models']}"
                print(f"  {i+1}. 会话: {checkpoint['session_id'][:16]}...")
                print(f"     进度: {progress} | 状态: {checkpoint['status']}")
                print(f"     时间: {checkpoint['start_time']}")
                print()
            
            choice = input(f"{self.colors['menu']}选择要恢复的会话 (1-{len(resumable_checkpoints)}, 0=取消): {self.colors['reset']}")
            
            if choice == '0':
                return
            
            try:
                index = int(choice) - 1
                if 0 <= index < len(resumable_checkpoints):
                    session_id = resumable_checkpoints[index]['session_id']
                    self.resume_specific_training(session_id)
                else:
                    print(f"{self.colors['warning']}⚠️  无效选择{self.colors['reset']}")
            except ValueError:
                print(f"{self.colors['warning']}⚠️  请输入有效数字{self.colors['reset']}")
                
        except Exception as e:
            print(f"{self.colors['error']}❌ 获取训练会话失败: {str(e)}{self.colors['reset']}")
    
    def resume_specific_training(self, session_id: str) -> bool:
        """恢复指定的训练会话"""
        try:
            print(f"\n{self.colors['success']}🔄 恢复训练会话: {session_id[:16]}...{self.colors['reset']}")
            
            results = self.trainer.resume_training_from_checkpoint(session_id)
            
            if results['success']:
                self.show_training_results(results)
                return True
            else:
                print(f"{self.colors['error']}❌ 恢复训练失败: {results['error']}{self.colors['reset']}")
                return False
                
        except KeyboardInterrupt:
            print(f"\n{self.colors['warning']}⚡ 恢复训练被用户中断{self.colors['reset']}")
            return False
        except Exception as e:
            print(f"{self.colors['error']}❌ 恢复训练异常: {str(e)}{self.colors['reset']}")
            return False
    
    def view_training_history(self):
        """查看训练历史"""
        print(f"\n{self.colors['header']}📊 训练历史{self.colors['reset']}")
        
        try:
            checkpoints = self.trainer.list_training_checkpoints()
            
            if not checkpoints:
                print(f"{self.colors['info']}📭 没有找到训练历史{self.colors['reset']}")
                return
            
            print(f"{self.colors['success']}找到 {len(checkpoints)} 个训练记录:{self.colors['reset']}")
            print()
            
            for i, checkpoint in enumerate(checkpoints):
                status_color = self.colors['success'] if checkpoint['status'] == 'completed' else self.colors['warning']
                resumable = "可恢复" if checkpoint.get('can_resume', False) else "已完成"
                
                print(f"  {i+1}. {status_color}{checkpoint['session_id'][:20]}...{self.colors['reset']}")
                print(f"     状态: {checkpoint['status']} ({resumable})")
                print(f"     进度: {checkpoint['completed_models']}/{checkpoint['total_models']} 模型")
                print(f"     时间: {checkpoint['start_time']}")
                
                if checkpoint.get('avg_accuracy'):
                    print(f"     平均准确率: {checkpoint['avg_accuracy']:.2%}")
                
                print()
            
            # 询问是否查看详细信息
            if len(checkpoints) > 0:
                choice = input(f"{self.colors['menu']}查看详细信息？输入序号 (1-{len(checkpoints)}, 0=返回): {self.colors['reset']}")
                
                if choice != '0':
                    try:
                        index = int(choice) - 1
                        if 0 <= index < len(checkpoints):
                            self.show_checkpoint_details(checkpoints[index])
                    except ValueError:
                        print(f"{self.colors['warning']}⚠️  无效输入{self.colors['reset']}")
                        
        except Exception as e:
            print(f"{self.colors['error']}❌ 获取训练历史失败: {str(e)}{self.colors['reset']}")
    
    def show_checkpoint_details(self, checkpoint: Dict):
        """显示检查点详细信息"""
        print(f"\n{self.colors['header']}📋 检查点详情{self.colors['reset']}")
        print(f"会话ID: {checkpoint['session_id']}")
        print(f"状态: {checkpoint['status']}")
        print(f"创建时间: {checkpoint['start_time']}")
        print(f"模型进度: {checkpoint['completed_models']}/{checkpoint['total_models']}")
        
        if checkpoint.get('data_shape'):
            data_shape = checkpoint['data_shape']
            print(f"数据规模: {data_shape.get('n_samples', '未知')} 样本, {data_shape.get('n_features', '未知')} 特征")
        
        print(f"可恢复: {'是' if checkpoint.get('can_resume') else '否'}")
    
    def load_training_data(self) -> bool:
        """加载并处理训练数据"""
        print(f"\n{self.colors['header']}📁 加载训练数据{self.colors['reset']}")
        
        try:
            # 从配置获取数据源
            data_config = self.config.get('ml_signal_enhancer', {}).get('data_source', {})
            file_path = data_config.get('file_path', 'data/BTC_USDT_1h_20190101_to_20250627.csv')
            
            print(f"{self.colors['info']}数据文件: {file_path}{self.colors['reset']}")
            
            if not os.path.exists(file_path):
                print(f"{self.colors['warning']}⚠️  数据文件不存在: {file_path}{self.colors['reset']}")
                
                # 询问用户输入新路径
                new_path = input(f"{self.colors['menu']}请输入数据文件路径 (空白=取消): {self.colors['reset']}").strip()
                
                if not new_path:
                    return False
                
                if not os.path.exists(new_path):
                    print(f"{self.colors['error']}❌ 文件不存在: {new_path}{self.colors['reset']}")
                    return False
                
                file_path = new_path
            
            print(f"{self.colors['info']}📖 正在加載原始数据...{self.colors['reset']}")
            
            # 加载原始数据
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.pkl'):
                df = pd.read_pickle(file_path)
            else:
                print(f"{self.colors['error']}❌ 不支持的文件格式{self.colors['reset']}")
                return False
            
            print(f"{self.colors['success']}✅ 原始数据加载成功{self.colors['reset']}")
            print(f"   数据规模: {len(df)} 行 × {len(df.columns)} 列")
            print(f"   可用列: {list(df.columns)}")
            
            # 检查是否包含基本的OHLCV数据
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"{self.colors['error']}❌ 缺少必要的价格数据列: {missing_columns}{self.colors['reset']}")
                return False
            
            print(f"{self.colors['info']}🔧 执行特征工程和信号生成...{self.colors['reset']}")
            
            # 使用特征工程器处理数据
            if self.feature_engineer:
                try:
                    features_df = self.feature_engineer.create_features(df)
                    print(f"{self.colors['success']}✅ 特征工程完成{self.colors['reset']}")
                    print(f"   生成特征数: {features_df.shape[1]} 个")
                    
                    # 准备训练数据
                    X, y = self._prepare_training_data_from_features(features_df)
                    
                except Exception as e:
                    print(f"{self.colors['warning']}⚠️  特征工程失败: {str(e)}{self.colors['reset']}")
                    print(f"{self.colors['info']}📊 使用基础技术指标生成特征...{self.colors['reset']}")
                    X, y = self._create_basic_features_and_signals(df)
            else:
                print(f"{self.colors['info']}📊 使用基础技术指标生成特征...{self.colors['reset']}")
                X, y = self._create_basic_features_and_signals(df)
            
            # 检查数据质量
            if len(X) == 0:
                print(f"{self.colors['error']}❌ 没有有效的特征数据{self.colors['reset']}")
                return False
            
            if len(y.unique()) < 2:
                print(f"{self.colors['error']}❌ 目标变量没有变化{self.colors['reset']}")
                return False
            
            self.X_train = X
            self.y_train = y
            self.data_loaded = True
            
            print(f"{self.colors['success']}✅ 训练数据准备完成{self.colors['reset']}")
            print(f"   特征数量: {X.shape[1]}")
            print(f"   样本数量: {len(X)}")
            print(f"   信号分布: {dict(y.value_counts())}")
            
            # 显示特征名称（前10个）
            feature_names = list(X.columns)[:10]
            print(f"   主要特征: {feature_names}...")
            
            return True
            
        except Exception as e:
            print(f"{self.colors['error']}❌ 数据加载失败: {str(e)}{self.colors['reset']}")
            self.logger.error(f"数据加载失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _prepare_training_data_from_features(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """从特征工程结果准备训练数据"""
        # 查找信号列
        signal_columns = [col for col in features_df.columns if 'target' in col.lower()]
        
        if signal_columns:
            # 检查目标变量的变化
            for target_col in signal_columns:
                target_values = features_df[target_col].dropna()
                unique_values = target_values.unique()
                
                print(f"   检查目标变量 {target_col}: 唯一值 = {unique_values}")
                
                if len(unique_values) >= 2:
                    # 找到有变化的目标变量
                    y = features_df[target_col]
                    X = features_df.drop(columns=signal_columns)
                    print(f"   ✅ 使用目标变量: {target_col}")
                    break
            else:
                # 所有生成的目标变量都没有变化，使用自定义信号生成
                print(f"   ⚠️  所有目标变量都没有变化，使用自定义信号生成...")
                X = features_df.drop(columns=signal_columns).select_dtypes(include=['number'])
                y = self._generate_enhanced_signals(features_df)
        else:
            # 如果没有信号列，生成基于价格变化的信号
            X = features_df.select_dtypes(include=['number'])
            y = self._generate_enhanced_signals(features_df)
        
        # 数据清理
        X = X.fillna(X.mean())
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
        
        return X, y
    
    def _generate_enhanced_signals(self, df: pd.DataFrame) -> pd.Series:
        """增强的信号生成逻辑"""
        print(f"   🔧 使用增强信号生成逻辑...")
        
        if 'close' not in df.columns:
            # 如果没有价格数据，创建平衡的随机信号
            signals = pd.Series(np.random.choice([0, 1], size=len(df), p=[0.6, 0.4]), index=df.index)
            print(f"   📊 生成随机信号 (买入比例: 40%)")
            return signals
        
        # 计算价格变化率
        returns = df['close'].pct_change()
        
        # 方法1: 基于收益率阈值
        buy_threshold = returns.quantile(0.7)  # 70分位数作为买入阈值
        sell_threshold = returns.quantile(0.3)  # 30分位数作为卖出阈值
        
        signals = pd.Series(0, index=df.index)  # 默认为持有
        signals[returns > buy_threshold] = 1    # 买入信号
        
        # 方法2: 如果方法1没有足够变化，使用波动率基础信号
        if len(signals.unique()) < 2:
            print(f"   🔄 收益率信号无变化，使用波动率信号...")
            
            # 计算滚动波动率
            volatility = returns.rolling(window=20).std()
            volatility_ma = volatility.rolling(window=10).mean()
            
            # 当波动率低于平均值时，可能是积累期，给买入信号
            signals = (volatility < volatility_ma).astype(int)
        
        # 方法3: 如果还是没有变化，使用价格位置信号
        if len(signals.unique()) < 2:
            print(f"   🔄 波动率信号无变化，使用价格位置信号...")
            
            # 计算价格在最近高低点的位置
            high_20 = df['high'].rolling(window=20).max() if 'high' in df.columns else df['close'].rolling(window=20).max()
            low_20 = df['low'].rolling(window=20).min() if 'low' in df.columns else df['close'].rolling(window=20).min()
            
            price_position = (df['close'] - low_20) / (high_20 - low_20)
            
            # 价格在中低位时给买入信号
            signals = (price_position < 0.6).astype(int)
        
        # 方法4: 最后的保险措施 - 基于时间的周期性信号
        if len(signals.unique()) < 2:
            print(f"   🔄 价格位置信号无变化，使用周期性信号...")
            
            # 创建基于索引的周期性信号
            signals = pd.Series(index=df.index, dtype=int)
            
            # 每5个时间点中有2个买入信号
            pattern = [0, 0, 1, 0, 1]  # 40%的买入比例
            for i in range(len(signals)):
                signals.iloc[i] = pattern[i % len(pattern)]
        
        # 最终检查
        unique_signals = signals.unique()
        signal_counts = signals.value_counts()
        
        print(f"   ✅ 信号生成完成:")
        print(f"   - 唯一信号值: {unique_signals}")
        print(f"   - 信号分布: {dict(signal_counts)}")
        print(f"   - 买入信号比例: {signals.mean():.2%}")
        
        return signals
    
    def _create_basic_features_and_signals(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """创建基础技术指标特征和交易信号"""
        print(f"   {self.colors['info']}计算技术指标...{self.colors['reset']}")
        
        # 确保数据按时间排序
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        # 基础价格特征
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # 移动平均线
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'close_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
        
        # 波动率指标
        df['volatility_10'] = df['returns'].rolling(window=10).std()
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # 成交量指标（如果有成交量数据）
        if 'volume' in df.columns:
            df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_10']
            df['price_volume'] = df['close'] * df['volume']
        
        print(f"   {self.colors['info']}生成交易信号...{self.colors['reset']}")
        
        # 使用更稳健的信号生成逻辑
        signals = self._generate_enhanced_signals(df)
        
        # 准备特征矩阵
        feature_columns = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        X = df[feature_columns].select_dtypes(include=['number'])
        X = X.fillna(X.mean())
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
        
        # 移除前面的NaN行，确保X和y长度一致
        valid_idx = X.dropna().index
        X = X.loc[valid_idx]
        
        # 确保signals和X的索引对齐
        signals = signals.reindex(valid_idx, fill_value=0)
        
        print(f"   {self.colors['success']}✅ 生成了 {len(feature_columns)} 个技术指标特征{self.colors['reset']}")
        
        return X, signals
    
    def _generate_price_signals(self, df: pd.DataFrame) -> pd.Series:
        """基于价格变化生成简单的交易信号"""
        if 'close' in df.columns:
            # 基于未来收益生成信号
            future_returns = df['close'].shift(-1) / df['close'] - 1
            # 如果未来1期收益 > 0.1% 则为买入信号
            signals = (future_returns > 0.001).astype(int)
        else:
            # 如果没有价格数据，生成随机信号用于测试
            signals = pd.Series(np.random.choice([0, 1], size=len(df)), index=df.index)
        
        return signals
    
    def show_training_config(self):
        """显示训练配置"""
        print(f"\n{self.colors['info']}⚙️  训练配置:{self.colors['reset']}")
        
        trainer_config = self.config.get('ml_signal_enhancer', {}).get('model_trainer', {})
        
        print(f"   模型类型: {trainer_config.get('model_types', [])}")
        print(f"   测试集比例: {trainer_config.get('test_size', 0.2)}")
        print(f"   超参数优化: {'开启' if trainer_config.get('enable_hyperopt', True) else '关闭'}")
        print(f"   优化试验次数: {trainer_config.get('optuna_trials', 100)}")
        print(f"   检查点功能: {'开启' if trainer_config.get('enable_checkpoints', True) else '关闭'}")
    
    def show_training_results(self, results: Dict):
        """显示训练结果"""
        print(f"\n{self.colors['success']}🎉 训练完成!{self.colors['reset']}")
        print(f"   会话ID: {results.get('session_id', '未知')}")
        print(f"   完成模型: {len(results.get('models', {}))} 个")
        print(f"   最佳模型: {results.get('best_model', '未知')}")
        print(f"   训练样本: {results.get('training_samples', 0)}")
        print(f"   测试样本: {results.get('test_samples', 0)}")
        
        if results.get('resumed'):
            print(f"   {self.colors['info']}🔄 这是一次恢复训练{self.colors['reset']}")
        
        # 显示模型性能
        metrics = results.get('metrics', {})
        if metrics:
            print(f"\n{self.colors['info']}📊 模型性能:{self.colors['reset']}")
            for model_name, model_metrics in metrics.items():
                accuracy = model_metrics.get('accuracy', 0)
                f1_score = model_metrics.get('f1_score', 0)
                print(f"   {model_name}: 准确率={accuracy:.2%}, F1={f1_score:.3f}")
    
    def manage_checkpoints(self):
        """管理检查点"""
        print(f"\n{self.colors['header']}🔧 检查点管理{self.colors['reset']}")
        
        try:
            stats = self.trainer.get_checkpoint_stats()
            
            print(f"{self.colors['info']}📊 检查点统计:{self.colors['reset']}")
            print(f"   总数量: {stats['total_checkpoints']}")
            print(f"   可恢复: {stats['resumable_sessions']}")
            print(f"   已完成: {stats['completed_sessions']}")
            print(f"   存储目录: {stats['checkpoint_dir']}")
            print(f"   自动保存间隔: 每 {stats['auto_save_interval']} 个模型")
            print(f"   最大保留数: {stats['max_checkpoints']}")
            
        except Exception as e:
            print(f"{self.colors['error']}❌ 获取检查点信息失败: {str(e)}{self.colors['reset']}")
    
    def view_configuration(self):
        """查看配置"""
        print(f"\n{self.colors['header']}⚙️  系统配置{self.colors['reset']}")
        
        print(f"{self.colors['info']}配置文件: {self.config_path}{self.colors['reset']}")
        
        ml_config = self.config.get('ml_signal_enhancer', {})
        
        # 训练配置
        trainer_config = ml_config.get('model_trainer', {})
        print(f"\n{self.colors['info']}🤖 训练配置:{self.colors['reset']}")
        for key, value in trainer_config.items():
            print(f"   {key}: {value}")
        
        # 检查点配置
        checkpoint_config = ml_config.get('training_checkpoint', {})
        print(f"\n{self.colors['info']}💾 检查点配置:{self.colors['reset']}")
        for key, value in checkpoint_config.items():
            print(f"   {key}: {value}")
    
    def view_training_status(self):
        """查看训练状态"""
        print(f"\n{self.colors['header']}📈 训练状态{self.colors['reset']}")
        
        try:
            stats = self.trainer.get_checkpoint_stats()
            
            if stats.get('current_session'):
                print(f"{self.colors['warning']}⚡ 有正在进行的训练会话{self.colors['reset']}")
                print(f"   会话ID: {stats['current_session']}")
            else:
                print(f"{self.colors['info']}💤 当前没有活跃的训练会话{self.colors['reset']}")
            
            checkpoints = self.trainer.list_training_checkpoints()
            
            if checkpoints:
                latest = checkpoints[0]
                print(f"\n{self.colors['info']}📋 最近的训练记录:{self.colors['reset']}")
                print(f"   会话: {latest['session_id'][:20]}...")
                print(f"   状态: {latest['status']}")
                print(f"   进度: {latest['completed_models']}/{latest['total_models']}")
                print(f"   时间: {latest['start_time']}")
                
        except Exception as e:
            print(f"{self.colors['error']}❌ 获取训练状态失败: {str(e)}{self.colors['reset']}")
    
    def cleanup_checkpoints(self):
        """清理检查点"""
        print(f"\n{self.colors['header']}🧹 清理检查点{self.colors['reset']}")
        
        try:
            checkpoints = self.trainer.list_training_checkpoints()
            
            if not checkpoints:
                print(f"{self.colors['info']}✅ 没有需要清理的检查点{self.colors['reset']}")
                return
            
            completed_checkpoints = [cp for cp in checkpoints if cp['status'] == 'completed']
            
            if not completed_checkpoints:
                print(f"{self.colors['info']}✅ 没有已完成的检查点需要清理{self.colors['reset']}")
                return
            
            print(f"{self.colors['info']}找到 {len(completed_checkpoints)} 个已完成的检查点{self.colors['reset']}")
            
            confirm = self.get_user_choice(
                f"{self.colors['warning']}确认删除所有已完成的检查点？{self.colors['reset']}", 
                ['y', 'n'], 
                'n'
            )
            
            if confirm.lower() == 'y':
                deleted_count = 0
                for checkpoint in completed_checkpoints:
                    if self.trainer.delete_training_checkpoint(checkpoint['session_id']):
                        deleted_count += 1
                        print(f"   ✅ 删除: {checkpoint['session_id'][:20]}...")
                    else:
                        print(f"   ❌ 删除失败: {checkpoint['session_id'][:20]}...")
                
                print(f"\n{self.colors['success']}🧹 清理完成，删除了 {deleted_count} 个检查点{self.colors['reset']}")
            else:
                print(f"{self.colors['info']}清理已取消{self.colors['reset']}")
                
        except Exception as e:
            print(f"{self.colors['error']}❌ 清理检查点失败: {str(e)}{self.colors['reset']}")
    
    def get_user_choice(self, prompt: str, valid_choices: List[str], default: str = None) -> str:
        """获取用户选择"""
        while True:
            if default:
                user_input = input(f"{prompt} (默认: {default}): ").strip()
                if not user_input:
                    return default
            else:
                user_input = input(f"{prompt}: ").strip()
            
            if user_input.lower() in [choice.lower() for choice in valid_choices]:
                return user_input
            
            print(f"{self.colors['warning']}⚠️  请输入有效选择: {'/'.join(valid_choices)}{self.colors['reset']}")
    
    def exit_manager(self):
        """退出管理器"""
        print(f"\n{self.colors['success']}👋 感谢使用ML训练管理器！{self.colors['reset']}")
        
        # 检查是否有活跃训练
        try:
            stats = self.trainer.get_checkpoint_stats()
            if stats.get('current_session'):
                print(f"{self.colors['warning']}⚠️  提醒: 有正在进行的训练会话{self.colors['reset']}")
                print(f"{self.colors['info']}💡 下次启动时可以选择恢复训练{self.colors['reset']}")
        except:
            pass
        
        print(f"{self.colors['header']}🚀 再见！{self.colors['reset']}")
    
    def run(self):
        """运行训练管理器"""
        try:
            print(f"{self.colors['success']}✅ 训练管理器启动成功{self.colors['reset']}")
            
            # 启动检测
            startup_resumed = self.startup_check()
            
            if not startup_resumed:
                # 如果没有恢复训练，显示主菜单
                self.show_main_menu()
            
        except KeyboardInterrupt:
            print(f"\n{self.colors['warning']}⚡ 程序被用户中断{self.colors['reset']}")
        except Exception as e:
            print(f"{self.colors['error']}❌ 运行异常: {str(e)}{self.colors['reset']}")
            self.logger.error(f"运行异常: {str(e)}")
        finally:
            print(f"{self.colors['header']}👋 训练管理器退出{self.colors['reset']}")

def main():
    """主函数"""
    try:
        manager = TrainingManager()
        manager.run()
    except Exception as e:
        print(f"❌ 启动失败: {str(e)}")

if __name__ == "__main__":
    main()