"""
模型集成模块

实现需求1中的模型集成功能：
- 多模型投票决策
- 权重优化
- 元学习器集成
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings('ignore')

class ModelEnsemble:
    """模型集成器"""
    
    def __init__(self, config: Dict):
        """
        初始化模型集成器
        
        参数:
            config (Dict): 配置信息
        """
        self.config = config
        self.ml_config = config.get("ml_signal_enhancer", {})
        self.ensemble_config = self.ml_config.get("model_ensemble", {})
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 集成配置
        self.ensemble_method = self.ensemble_config.get("method", "voting")  # voting, stacking, weighted
        self.voting_type = self.ensemble_config.get("voting_type", "soft")  # hard, soft
        self.meta_learner = self.ensemble_config.get("meta_learner", "logistic_regression")
        
        # 集成模型
        self.ensemble_model = None
        self.model_weights = {}
        self.performance_history = {}
        
        self.logger.info("模型集成器初始化完成")
    
    def create_ensemble(self, models: Dict[str, Any], 
                       X_train: Optional[pd.DataFrame] = None,
                       y_train: Optional[pd.Series] = None) -> Dict:
        """
        创建集成模型
        
        参数:
            models (Dict[str, Any]): 基础模型字典
            X_train (pd.DataFrame, optional): 训练数据
            y_train (pd.Series, optional): 训练标签
            
        返回:
            Dict: 集成结果
        """
        if not models:
            raise ValueError("没有提供基础模型")
        
        try:
            self.logger.info(f"创建 {self.ensemble_method} 集成模型...")
            
            # 准备模型列表
            model_list = [(name, model) for name, model in models.items()]
            
            if self.ensemble_method == "voting":
                self.ensemble_model = self._create_voting_ensemble(model_list)
            elif self.ensemble_method == "stacking":
                if X_train is None or y_train is None:
                    raise ValueError("Stacking需要训练数据")
                self.ensemble_model = self._create_stacking_ensemble(model_list, X_train, y_train)
            elif self.ensemble_method == "weighted":
                self.ensemble_model = self._create_weighted_ensemble(models)
            else:
                raise ValueError(f"不支持的集成方法: {self.ensemble_method}")
            
            # 评估集成性能
            ensemble_score = 0
            if X_train is not None and y_train is not None:
                ensemble_score = self._evaluate_ensemble(X_train, y_train)
            
            results = {
                'success': True,
                'ensemble_model': self.ensemble_model,
                'ensemble_method': self.ensemble_method,
                'ensemble_score': ensemble_score,
                'base_models_count': len(models)
            }
            
            self.logger.info(f"集成模型创建完成，评分: {ensemble_score:.4f}")
            return results
            
        except Exception as e:
            self.logger.error(f"集成模型创建失败: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, features: pd.DataFrame) -> Dict:
        """
        使用集成模型进行预测
        
        参数:
            features (pd.DataFrame): 特征数据
            
        返回:
            Dict: 预测结果
        """
        if self.ensemble_model is None:
            raise ValueError("集成模型尚未创建")
        
        try:
            # 预测
            prediction = self.ensemble_model.predict(features)[0]
            
            result = {
                'signal': int(prediction),
                'ensemble_method': self.ensemble_method
            }
            
            # 概率预测
            if hasattr(self.ensemble_model, 'predict_proba'):
                probabilities = self.ensemble_model.predict_proba(features)[0]
                result['probabilities'] = probabilities.tolist()
                result['probability'] = float(probabilities[1])
                
                # 信号强度
                result['signal_strength'] = abs(probabilities[1] - 0.5) * 2
            else:
                result['probability'] = 0.5
                result['signal_strength'] = 0.5
            
            # 集成置信度
            result['confidence'] = self._calculate_ensemble_confidence(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"集成预测失败: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def update_ensemble(self, updated_models: Dict[str, Any],
                       X_train: Optional[pd.DataFrame] = None,
                       y_train: Optional[pd.Series] = None) -> Dict:
        """
        更新集成模型
        
        参数:
            updated_models (Dict[str, Any]): 更新后的模型
            X_train (pd.DataFrame, optional): 训练数据
            y_train (pd.Series, optional): 训练标签
            
        返回:
            Dict: 更新结果
        """
        try:
            self.logger.info("更新集成模型...")
            
            # 重新创建集成
            result = self.create_ensemble(updated_models, X_train, y_train)
            
            if result['success']:
                self.logger.info("集成模型更新完成")
                return {
                    'success': True,
                    'updated_ensemble': result['ensemble_model'],
                    'new_score': result['ensemble_score']
                }
            else:
                return result
                
        except Exception as e:
            self.logger.error(f"集成模型更新失败: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_voting_ensemble(self, model_list: List[tuple]) -> VotingClassifier:
        """创建投票集成"""
        return VotingClassifier(
            estimators=model_list,
            voting=self.voting_type
        )
    
    def _create_stacking_ensemble(self, model_list: List[tuple],
                                 X_train: pd.DataFrame, y_train: pd.Series) -> StackingClassifier:
        """创建堆叠集成"""
        # 元学习器
        if self.meta_learner == "logistic_regression":
            meta_clf = LogisticRegression(random_state=42)
        else:
            meta_clf = LogisticRegression(random_state=42)  # 默认
        
        stacking_clf = StackingClassifier(
            estimators=model_list,
            final_estimator=meta_clf,
            cv=3
        )
        
        # 训练堆叠模型
        stacking_clf.fit(X_train, y_train)
        return stacking_clf
    
    def _create_weighted_ensemble(self, models: Dict[str, Any]) -> 'WeightedEnsemble':
        """创建加权集成"""
        return WeightedEnsemble(models, self.model_weights)
    
    def _evaluate_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series) -> float:
        """评估集成性能"""
        try:
            if hasattr(self.ensemble_model, 'fit'):
                # 如果模型还未训练，先训练
                if self.ensemble_method != "stacking":  # stacking已经在创建时训练了
                    self.ensemble_model.fit(X_train, y_train)
            
            # 交叉验证评分
            scores = cross_val_score(self.ensemble_model, X_train, y_train, cv=3, scoring='accuracy')
            return scores.mean()
        except Exception as e:
            self.logger.warning(f"集成评估失败: {str(e)}")
            return 0.0
    
    def _calculate_ensemble_confidence(self, prediction_result: Dict) -> float:
        """计算集成置信度"""
        probability = prediction_result.get('probability', 0.5)
        signal_strength = prediction_result.get('signal_strength', 0.5)
        
        # 基于概率偏离度计算置信度
        prob_deviation = abs(probability - 0.5) * 2
        
        # 集成模型的置信度通常更高
        base_confidence = (prob_deviation * 0.6 + signal_strength * 0.4)
        
        # 集成加成
        ensemble_bonus = 0.1 if self.ensemble_method in ['stacking', 'weighted'] else 0.05
        
        confidence = min(base_confidence + ensemble_bonus, 1.0)
        return max(confidence, 0.0)
    
    def optimize_weights(self, models: Dict[str, Any], 
                        X_val: pd.DataFrame, y_val: pd.Series,
                        performance_metrics: Dict) -> Dict:
        """
        优化模型权重
        
        参数:
            models (Dict[str, Any]): 模型字典
            X_val (pd.DataFrame): 验证数据
            y_val (pd.Series): 验证标签
            performance_metrics (Dict): 性能指标
            
        返回:
            Dict: 优化结果
        """
        try:
            # 基于性能指标计算初始权重
            initial_weights = {}
            total_score = 0
            
            for model_name, metrics in performance_metrics.items():
                if model_name in models:
                    # 综合评分
                    score = (metrics.get('accuracy', 0) * 0.4 + 
                            metrics.get('f1_score', 0) * 0.3 + 
                            metrics.get('auc_roc', 0) * 0.3)
                    initial_weights[model_name] = max(score, 0.1)  # 最小权重0.1
                    total_score += initial_weights[model_name]
            
            # 归一化权重
            if total_score > 0:
                for model_name in initial_weights.keys():
                    initial_weights[model_name] /= total_score
            
            self.model_weights = initial_weights
            
            # 简单的网格搜索优化（实际应用中可以使用更复杂的优化方法）
            best_weights = initial_weights.copy()
            best_score = self._evaluate_weighted_ensemble(models, best_weights, X_val, y_val)
            
            # 尝试调整权重
            for _ in range(10):  # 简单的迭代优化
                test_weights = best_weights.copy()
                
                # 随机调整一个模型的权重
                model_to_adjust = np.random.choice(list(test_weights.keys()))
                adjustment = np.random.uniform(-0.1, 0.1)
                test_weights[model_to_adjust] = max(0.05, min(0.95, 
                    test_weights[model_to_adjust] + adjustment))
                
                # 重新归一化
                total_weight = sum(test_weights.values())
                for name in test_weights.keys():
                    test_weights[name] /= total_weight
                
                # 评估
                score = self._evaluate_weighted_ensemble(models, test_weights, X_val, y_val)
                
                if score > best_score:
                    best_score = score
                    best_weights = test_weights.copy()
            
            self.model_weights = best_weights
            
            return {
                'success': True,
                'optimized_weights': best_weights,
                'best_score': best_score,
                'improvement': best_score - self._evaluate_weighted_ensemble(
                    models, initial_weights, X_val, y_val)
            }
            
        except Exception as e:
            self.logger.error(f"权重优化失败: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _evaluate_weighted_ensemble(self, models: Dict[str, Any], 
                                  weights: Dict[str, float],
                                  X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """评估加权集成性能"""
        try:
            # 获取每个模型的预测
            predictions = []
            for model_name, model in models.items():
                if model_name in weights:
                    pred_proba = model.predict_proba(X_val)[:, 1]
                    predictions.append(pred_proba * weights[model_name])
            
            if not predictions:
                return 0.0
            
            # 加权平均
            ensemble_proba = np.sum(predictions, axis=0)
            ensemble_pred = (ensemble_proba > 0.5).astype(int)
            
            # 计算准确率
            accuracy = (ensemble_pred == y_val).mean()
            return accuracy
            
        except Exception as e:
            self.logger.warning(f"加权集成评估失败: {str(e)}")
            return 0.0
    
    def get_ensemble_info(self) -> Dict:
        """获取集成模型信息"""
        return {
            'ensemble_method': self.ensemble_method,
            'has_ensemble_model': self.ensemble_model is not None,
            'model_weights': self.model_weights.copy(),
            'voting_type': self.voting_type if self.ensemble_method == 'voting' else None,
            'meta_learner': self.meta_learner if self.ensemble_method == 'stacking' else None
        }


class WeightedEnsemble:
    """加权集成预测器"""
    
    def __init__(self, models: Dict[str, Any], weights: Dict[str, float]):
        self.models = models
        self.weights = weights
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        predictions = []
        total_weight = 0
        
        for model_name, model in self.models.items():
            if model_name in self.weights:
                weight = self.weights[model_name]
                pred = model.predict(X)
                predictions.append(pred * weight)
                total_weight += weight
        
        if total_weight == 0:
            return np.zeros(len(X))
        
        weighted_pred = np.sum(predictions, axis=0) / total_weight
        return (weighted_pred > 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """概率预测"""
        predictions = []
        total_weight = 0
        
        for model_name, model in self.models.items():
            if model_name in self.weights and hasattr(model, 'predict_proba'):
                weight = self.weights[model_name]
                pred_proba = model.predict_proba(X)
                predictions.append(pred_proba * weight)
                total_weight += weight
        
        if not predictions:
            # 返回均匀概率
            return np.full((len(X), 2), 0.5)
        
        weighted_proba = np.sum(predictions, axis=0) / total_weight
        return weighted_proba