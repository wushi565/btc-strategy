"""
模型训练模块

实现需求1中的模型训练功能：
- 多种机器学习模型训练
- 超参数优化
- 交叉验证评估
- 模型性能分析
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
import optuna
from datetime import datetime
import os
from tqdm import tqdm
import time
from ..utils.training_checkpoint import TrainingCheckpoint

warnings.filterwarnings('ignore')

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, config: Dict):
        """
        初始化模型训练器
        
        参数:
            config (Dict): 配置信息
        """
        self.config = config
        self.ml_config = config.get("ml_signal_enhancer", {})
        self.trainer_config = self.ml_config.get("model_trainer", {})
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 训练配置
        self.test_size = self.trainer_config.get("test_size", 0.2)
        self.validation_size = self.trainer_config.get("validation_size", 0.2)
        self.random_state = self.trainer_config.get("random_state", 42)
        self.cv_folds = self.trainer_config.get("cv_folds", 5)
        self.enable_hyperopt = self.trainer_config.get("enable_hyperopt", True)
        self.n_trials = self.trainer_config.get("optuna_trials", 100)
        
        # 模型保存路径
        self.model_save_path = self.trainer_config.get("model_save_path", "models/")
        os.makedirs(self.model_save_path, exist_ok=True)
        
        # 支持的模型类型
        self.model_types = self.trainer_config.get("model_types", [
            "random_forest", "xgboost", "lightgbm", "logistic_regression"
        ])
        
        # 已训练的模型
        self.trained_models = {}
        self.model_metrics = {}
        
        # 初始化检查点管理器
        self.enable_checkpoints = self.trainer_config.get("enable_checkpoints", True)
        if self.enable_checkpoints:
            self.checkpoint_manager = TrainingCheckpoint(config)
        else:
            self.checkpoint_manager = None
        
        self.logger.info("模型训练器初始化完成")
    
    def train_models_with_checkpoint(self, X: pd.DataFrame, y: pd.Series, 
                                    resume_session_id: Optional[str] = None) -> Dict:
        """
        训练多个机器学习模型（支持中断恢复）
        
        参数:
            X (pd.DataFrame): 特征数据
            y (pd.Series): 目标变量
            resume_session_id (str, optional): 要恢复的会话ID
            
        返回:
            Dict: 训练结果
        """
        if not self.enable_checkpoints:
            self.logger.warning("检查点功能未启用，使用常规训练模式")
            return self.train_models(X, y)
        
        try:
            # 尝试恢复训练
            resume_data = None
            if resume_session_id or self.checkpoint_manager:
                self.logger.info("检查是否有可恢复的训练会话...")
                resume_result = self.checkpoint_manager.resume_training(resume_session_id)
                
                if resume_result['success']:
                    resume_data = resume_result
                    self.logger.info(f"恢复训练会话: {resume_data['session_id']}")
                    self.logger.info(f"已完成 {len(resume_data['completed_models'])} 个模型，还需训练 {len(resume_data['remaining_models'])} 个")
                    
                    # 使用保存的训练数据（如果可用）
                    if resume_data['training_data']:
                        X = resume_data['training_data']['X_train']
                        y = resume_data['training_data']['y_train']
                        self.logger.info("使用保存的训练数据")
                else:
                    self.logger.info("没有可恢复的训练会话，开始新训练")
            
            # 数据划分
            print("📊 准备训练数据...")
            X_train, X_test, y_train, y_test = self._split_data(X, y)
            print(f"   • 训练集: {len(X_train)} 样本")
            print(f"   • 测试集: {len(X_test)} 样本")
            
            # 确定要训练的模型
            if resume_data:
                # 恢复模式：只训练剩余的模型
                models_to_train = resume_data['remaining_models']
                completed_models = resume_data['completed_models']
                completed_metrics = resume_data['model_metrics']
                session_id = resume_data['session_id']
            else:
                # 新训练模式：训练所有模型
                models_to_train = self.model_types.copy()
                completed_models = {}
                completed_metrics = {}
                
                # 创建新的训练会话
                training_config = {
                    'model_types': self.model_types.copy(),
                    'test_size': self.test_size,
                    'enable_hyperopt': self.enable_hyperopt,
                    'n_trials': self.n_trials
                }
                session_id = self.checkpoint_manager.start_training_session(training_config, X_train, y_train)
            
            print(f"\n🤖 {'继续' if resume_data else '开始'}训练 {len(models_to_train)} 个模型")
            
            # 使用进度条显示训练进程
            with tqdm(total=len(models_to_train), desc="🔥 模型训练进度", 
                     bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                     ncols=100) as pbar:
                
                for model_type in models_to_train:
                    model_start = time.time()
                    pbar.set_description(f"🔥 训练{model_type}")
                    
                    try:
                        self.logger.info(f"训练 {model_type} 模型...")
                        
                        # 获取模型和参数
                        model, param_grid = self._get_model_and_params(model_type)
                        
                        # 超参数优化
                        if self.enable_hyperopt:
                            best_model = self._optimize_hyperparameters(
                                model, X_train, y_train, model_type
                            )
                        else:
                            best_model = model.fit(X_train, y_train)
                        
                        # 评估模型
                        model_metrics = self._evaluate_model(
                            best_model, X_test, y_test, model_type
                        )
                        
                        # 保存模型到检查点
                        self.checkpoint_manager.save_model_progress(
                            model_type, best_model, model_metrics
                        )
                        
                        # 更新已完成的模型
                        completed_models[model_type] = best_model
                        completed_metrics[model_type] = model_metrics
                        
                        # 更新进度条
                        model_time = time.time() - model_start
                        accuracy = model_metrics['accuracy']
                        pbar.set_postfix({"准确率": f"{accuracy:.1%}", "耗时": f"{model_time:.1f}s"})
                        pbar.update(1)
                        
                        self.logger.info(f"{model_type} 训练完成，准确率: {accuracy:.4f}")
                        
                    except Exception as e:
                        self.logger.error(f"{model_type} 训练失败: {str(e)}")
                        # 继续训练其他模型
                        pbar.update(1)
                        continue
            
            print(f"\n💾 保存训练结果...")
            
            # 保存最终模型
            self._save_trained_models(completed_models)
            
            # 更新内部状态
            self.trained_models = completed_models
            self.model_metrics = completed_metrics
            
            # 完成训练会话
            final_results = {
                'models': completed_models,
                'metrics': completed_metrics,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': X.shape[1]
            }
            
            if self.checkpoint_manager:
                self.checkpoint_manager.complete_training_session(final_results)
            
            # 训练总结
            total_completed = len(completed_models)
            best_model_name = self._select_best_model(completed_metrics)
            best_accuracy = completed_metrics[best_model_name]['accuracy'] if best_model_name else 0
            
            print(f"\n🎉 模型训练完成!")
            print(f"   • 完成模型: {total_completed} 个")
            print(f"   • 最佳模型: {best_model_name} (准确率: {best_accuracy:.1%})" if best_model_name else "")
            print(f"   • 训练样本: {len(X_train)} 个")
            print(f"   • 测试样本: {len(X_test)} 个")
            
            results = {
                'success': True,
                'models': completed_models,
                'metrics': completed_metrics,
                'best_model': best_model_name,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': X.shape[1],
                'session_id': session_id,
                'resumed': resume_data is not None
            }
            
            self.logger.info("模型训练完成")
            return results
            
        except Exception as e:
            self.logger.error(f"带检查点的模型训练失败: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def list_training_checkpoints(self) -> List[Dict]:
        """
        列出可用的训练检查点
        
        返回:
            List[Dict]: 检查点列表
        """
        if not self.checkpoint_manager:
            return []
        
        return self.checkpoint_manager.list_checkpoints()
    
    def resume_training_from_checkpoint(self, session_id: str, 
                                      X: Optional[pd.DataFrame] = None, 
                                      y: Optional[pd.Series] = None) -> Dict:
        """
        从指定检查点恢复训练
        
        参数:
            session_id (str): 会话ID
            X (pd.DataFrame, optional): 新的训练数据（如果不提供，将使用保存的数据）
            y (pd.Series, optional): 新的训练标签
            
        返回:
            Dict: 训练结果
        """
        if not self.checkpoint_manager:
            return {'success': False, 'error': '检查点功能未启用'}
        
        if X is not None and y is not None:
            return self.train_models_with_checkpoint(X, y, session_id)
        else:
            # 使用保存的数据恢复训练
            resume_result = self.checkpoint_manager.resume_training(session_id)
            if not resume_result['success']:
                return resume_result
            
            if not resume_result['training_data']:
                return {
                    'success': False, 
                    'error': '检查点中没有保存训练数据，请提供新的训练数据'
                }
            
            saved_data = resume_result['training_data']
            return self.train_models_with_checkpoint(saved_data['X_train'], saved_data['y_train'], session_id)
    
    def delete_training_checkpoint(self, session_id: str) -> bool:
        """
        删除指定的训练检查点
        
        参数:
            session_id (str): 会话ID
            
        返回:
            bool: 是否删除成功
        """
        if not self.checkpoint_manager:
            return False
        
        return self.checkpoint_manager.delete_checkpoint(session_id)
    
    def get_checkpoint_stats(self) -> Dict:
        """
        获取检查点统计信息
        
        返回:
            Dict: 统计信息
        """
        if not self.checkpoint_manager:
            return {'checkpoints_enabled': False}
        
        stats = self.checkpoint_manager.get_checkpoint_stats()
        stats['checkpoints_enabled'] = True
        return stats
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        训练多个机器学习模型
        
        参数:
            X (pd.DataFrame): 特征数据
            y (pd.Series): 目标变量
            
        返回:
            Dict: 训练结果
        """
        self.logger.info("开始训练多个ML模型...")
        start_time = time.time()
        
        try:
            # 数据划分
            print("📊 准备训练数据...")
            X_train, X_test, y_train, y_test = self._split_data(X, y)
            print(f"   • 训练集: {len(X_train)} 样本")
            print(f"   • 测试集: {len(X_test)} 样本")
            
            models = {}
            metrics = {}
            
            # 使用进度条显示模型训练进程
            print(f"\n🤖 开始训练 {len(self.model_types)} 个模型")
            with tqdm(total=len(self.model_types), desc="🔥 模型训练进度", 
                     bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                     ncols=100) as pbar:
                
                for model_type in self.model_types:
                    model_start = time.time()
                    pbar.set_description(f"🔥 训练{model_type}")
                    
                    self.logger.info(f"训练 {model_type} 模型...")
                    
                    # 获取模型和参数
                    model, param_grid = self._get_model_and_params(model_type)
                    
                    # 超参数优化
                    if self.enable_hyperopt:
                        best_model = self._optimize_hyperparameters(
                            model, X_train, y_train, model_type
                        )
                    else:
                        best_model = model.fit(X_train, y_train)
                    
                    # 评估模型
                    model_metrics = self._evaluate_model(
                        best_model, X_test, y_test, model_type
                    )
                    
                    models[model_type] = best_model
                    metrics[model_type] = model_metrics
                    
                    # 更新进度条
                    model_time = time.time() - model_start
                    accuracy = model_metrics['accuracy']
                    pbar.set_postfix({"准确率": f"{accuracy:.1%}", "耗时": f"{model_time:.1f}s"})
                    pbar.update(1)
                    
                    self.logger.info(f"{model_type} 训练完成，准确率: {accuracy:.4f}")
            
            print(f"\n💾 保存模型...")
            # 保存模型
            self._save_trained_models(models)
            
            # 更新内部状态
            self.trained_models = models
            self.model_metrics = metrics
            
            # 训练总结
            total_time = time.time() - start_time
            best_model_name = self._select_best_model(metrics)
            best_accuracy = metrics[best_model_name]['accuracy']
            
            print(f"\n🎉 模型训练完成!")
            print(f"   • 总耗时: {total_time:.1f}秒")
            print(f"   • 最佳模型: {best_model_name} (准确率: {best_accuracy:.1%})")
            print(f"   • 训练样本: {len(X_train)} 个")
            print(f"   • 测试样本: {len(X_test)} 个")
            
            results = {
                'success': True,
                'models': models,
                'metrics': metrics,
                'best_model': best_model_name,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'total_time': total_time,
                'feature_count': X.shape[1]
            }
            
            self.logger.info("所有模型训练完成")
            return results
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def update_models(self, X_new: pd.DataFrame, y_new: pd.Series) -> Dict:
        """
        增量更新已训练的模型
        
        参数:
            X_new (pd.DataFrame): 新的特征数据
            y_new (pd.Series): 新的目标变量
            
        返回:
            Dict: 更新结果
        """
        if not self.trained_models:
            self.logger.warning("没有已训练的模型可以更新")
            return {'success': False, 'error': '没有已训练的模型'}
        
        try:
            self.logger.info("开始增量更新模型...")
            
            updated_models = {}
            updated_metrics = {}
            
            # 准备验证数据
            X_train, X_val, y_train, y_val = train_test_split(
                X_new, y_new, test_size=0.3, random_state=self.random_state
            )
            
            for model_type, model in self.trained_models.items():
                self.logger.info(f"更新 {model_type} 模型...")
                
                # 根据模型类型选择更新策略
                if hasattr(model, 'partial_fit'):
                    # 支持增量学习的模型
                    updated_model = model.partial_fit(X_train, y_train)
                elif model_type in ['xgboost', 'lightgbm']:
                    # 基于树的模型重新训练
                    updated_model = self._retrain_tree_model(
                        model, X_train, y_train, model_type
                    )
                else:
                    # 其他模型重新训练
                    updated_model = model.fit(X_train, y_train)
                
                # 评估更新后的模型
                new_metrics = self._evaluate_model(
                    updated_model, X_val, y_val, model_type
                )
                
                updated_models[model_type] = updated_model
                updated_metrics[model_type] = new_metrics
                
                self.logger.info(f"{model_type} 更新完成，新准确率: {new_metrics['accuracy']:.4f}")
            
            results = {
                'success': True,
                'updated_models': updated_models,
                'metrics': updated_metrics,
                'improvement_summary': self._compare_performance(updated_metrics)
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"模型更新失败: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """划分训练和测试数据"""
        # 使用时间序列划分避免数据泄漏
        split_idx = int(len(X) * (1 - self.test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def _get_model_and_params(self, model_type: str) -> Tuple[Any, Dict]:
        """获取模型实例和超参数网格"""
        if model_type == "random_forest":
            model = RandomForestClassifier(random_state=self.random_state)
            params = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_type == "xgboost":
            model = xgb.XGBClassifier(random_state=self.random_state)
            params = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        elif model_type == "lightgbm":
            model = lgb.LGBMClassifier(random_state=self.random_state, verbose=-1)
            params = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100]
            }
        elif model_type == "logistic_regression":
            model = LogisticRegression(random_state=self.random_state, max_iter=1000)
            params = {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        return model, params
    
    def _optimize_hyperparameters(self, model: Any, X_train: pd.DataFrame, 
                                y_train: pd.Series, model_type: str) -> Any:
        """使用Optuna进行超参数优化"""
        def objective(trial):
            if model_type == "random_forest":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                }
                clf = RandomForestClassifier(random_state=self.random_state, **params)
            elif model_type == "xgboost":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0)
                }
                clf = xgb.XGBClassifier(random_state=self.random_state, **params)
            elif model_type == "lightgbm":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100)
                }
                clf = lgb.LGBMClassifier(random_state=self.random_state, verbose=-1, **params)
            elif model_type == "logistic_regression":
                params = {
                    'C': trial.suggest_float('C', 0.001, 10, log=True),
                    'penalty': trial.suggest_categorical('penalty', ['l1', 'l2'])
                }
                if params['penalty'] == 'l1':
                    params['solver'] = 'liblinear'
                else:
                    params['solver'] = 'lbfgs'
                clf = LogisticRegression(random_state=self.random_state, max_iter=1000, **params)
            
            # 时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=3)
            scores = cross_val_score(clf, X_train, y_train, cv=tscv, scoring='accuracy')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        # 使用最优参数训练模型
        best_params = study.best_params
        if model_type == "random_forest":
            best_model = RandomForestClassifier(random_state=self.random_state, **best_params)
        elif model_type == "xgboost":
            best_model = xgb.XGBClassifier(random_state=self.random_state, **best_params)
        elif model_type == "lightgbm":
            best_model = lgb.LGBMClassifier(random_state=self.random_state, verbose=-1, **best_params)
        elif model_type == "logistic_regression":
            best_model = LogisticRegression(random_state=self.random_state, max_iter=1000, **best_params)
        
        return best_model.fit(X_train, y_train)
    
    def _evaluate_model(self, model: Any, X_test: pd.DataFrame, 
                       y_test: pd.Series, model_type: str) -> Dict:
        """评估模型性能"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        if y_pred_proba is not None:
            metrics['auc_roc'] = roc_auc_score(y_test, y_pred_proba)
        
        return metrics
    
    def _select_best_model(self, metrics: Dict) -> str:
        """选择最佳模型"""
        best_model = None
        best_score = 0
        
        for model_type, model_metrics in metrics.items():
            # 综合评分（可以调整权重）
            score = (model_metrics['accuracy'] * 0.4 + 
                    model_metrics['f1_score'] * 0.3 + 
                    model_metrics.get('auc_roc', 0) * 0.3)
            
            if score > best_score:
                best_score = score
                best_model = model_type
        
        return best_model
    
    def _save_trained_models(self, models: Dict):
        """保存训练好的模型"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_type, model in models.items():
            filename = f"{model_type}_{timestamp}.joblib"
            filepath = os.path.join(self.model_save_path, filename)
            joblib.dump(model, filepath)
            self.logger.info(f"模型已保存: {filepath}")
    
    def _retrain_tree_model(self, model: Any, X_new: pd.DataFrame, 
                          y_new: pd.Series, model_type: str) -> Any:
        """重新训练基于树的模型"""
        # 获取原始参数
        params = model.get_params()
        
        # 创建新模型实例
        if model_type == "xgboost":
            new_model = xgb.XGBClassifier(**params)
        elif model_type == "lightgbm":
            new_model = lgb.LGBMClassifier(**params)
        else:
            new_model = model.__class__(**params)
        
        return new_model.fit(X_new, y_new)
    
    def _compare_performance(self, new_metrics: Dict) -> Dict:
        """比较新旧模型性能"""
        improvements = {}
        
        for model_type, new_metric in new_metrics.items():
            if model_type in self.model_metrics:
                old_metric = self.model_metrics[model_type]
                
                accuracy_change = new_metric['accuracy'] - old_metric['accuracy']
                f1_change = new_metric['f1_score'] - old_metric['f1_score']
                
                improvements[model_type] = {
                    'accuracy_change': accuracy_change,
                    'f1_change': f1_change,
                    'improved': accuracy_change > 0 and f1_change > 0
                }
        
        return improvements
    
    def get_model_info(self) -> Dict:
        """获取已训练模型信息"""
        return {
            'trained_models': list(self.trained_models.keys()),
            'model_metrics': self.model_metrics,
            'best_model': self._select_best_model(self.model_metrics) if self.model_metrics else None
        }