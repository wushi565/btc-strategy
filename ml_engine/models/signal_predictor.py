"""
信号预测器模块

实现需求1中的信号预测功能：
- 使用训练好的模型进行预测
- 提供概率和置信度评分
- 支持多模型预测集成
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
import warnings

warnings.filterwarnings('ignore')

class SignalPredictor:
    """信号预测器"""
    
    def __init__(self, config: Dict):
        """
        初始化信号预测器
        
        参数:
            config (Dict): 配置信息
        """
        self.config = config
        self.ml_config = config.get("ml_signal_enhancer", {})
        self.predictor_config = self.ml_config.get("signal_predictor", {})
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 预测配置
        self.confidence_threshold = self.predictor_config.get("confidence_threshold", 0.6)
        self.probability_threshold = self.predictor_config.get("probability_threshold", 0.55)
        
        # 已加载的模型
        self.models = {}
        self.model_weights = {}
        
        self.logger.info("信号预测器初始化完成")
    
    def load_models(self, models: Dict[str, Any], weights: Optional[Dict[str, float]] = None):
        """
        加载训练好的模型
        
        参数:
            models (Dict[str, Any]): 模型字典
            weights (Dict[str, float], optional): 模型权重
        """
        self.models = models
        
        # 设置模型权重
        if weights is None:
            # 均等权重
            self.model_weights = {name: 1.0/len(models) for name in models.keys()}
        else:
            self.model_weights = weights
        
        self.logger.info(f"已加载 {len(models)} 个模型")
    
    def predict(self, features: pd.DataFrame, 
                return_probabilities: bool = True,
                return_confidence: bool = True) -> Dict:
        """
        使用所有模型进行预测
        
        参数:
            features (pd.DataFrame): 特征数据
            return_probabilities (bool): 是否返回概率
            return_confidence (bool): 是否返回置信度
            
        返回:
            Dict: 预测结果
        """
        if not self.models:
            raise ValueError("没有加载任何模型")
        
        try:
            predictions = {}
            
            for model_name, model in self.models.items():
                # 单个模型预测
                pred_result = self._predict_single_model(
                    model, features, model_name, 
                    return_probabilities, return_confidence
                )
                predictions[model_name] = pred_result
            
            # 计算集成预测
            ensemble_result = self._ensemble_predictions(predictions)
            
            return {
                'individual_predictions': predictions,
                'ensemble': ensemble_result,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"预测失败: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def predict_single(self, model_name: str, features: pd.DataFrame,
                      return_probabilities: bool = True) -> Dict:
        """
        使用单个模型进行预测
        
        参数:
            model_name (str): 模型名称
            features (pd.DataFrame): 特征数据
            return_probabilities (bool): 是否返回概率
            
        返回:
            Dict: 预测结果
        """
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 未加载")
        
        model = self.models[model_name]
        return self._predict_single_model(model, features, model_name, return_probabilities)
    
    def _predict_single_model(self, model: Any, features: pd.DataFrame,
                             model_name: str, return_probabilities: bool = True,
                             return_confidence: bool = True) -> Dict:
        """单个模型预测"""
        # 基本预测
        prediction = model.predict(features)[0]  # 假设预测单个样本
        
        result = {
            'signal': int(prediction),
            'model_name': model_name
        }
        
        # 概率预测
        if return_probabilities and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            result['probabilities'] = probabilities.tolist()
            result['probability'] = float(probabilities[1])  # 正类概率
            
            # 基于概率的信号强度
            prob_strength = abs(probabilities[1] - 0.5) * 2  # 归一化到0-1
            result['signal_strength'] = prob_strength
        else:
            result['probability'] = 0.5
            result['signal_strength'] = 0.5
        
        # 置信度计算
        if return_confidence:
            confidence = self._calculate_model_confidence(result)
            result['confidence'] = confidence
        
        return result
    
    def _ensemble_predictions(self, predictions: Dict) -> Dict:
        """集成多个模型的预测结果"""
        # 加权投票
        weighted_signals = []
        weighted_probs = []
        confidences = []
        
        for model_name, pred in predictions.items():
            weight = self.model_weights.get(model_name, 1.0)
            
            weighted_signals.append(pred['signal'] * weight)
            weighted_probs.append(pred.get('probability', 0.5) * weight)
            confidences.append(pred.get('confidence', 0.5))
        
        # 计算集成结果
        ensemble_signal_score = sum(weighted_signals) / sum(self.model_weights.values())
        ensemble_probability = sum(weighted_probs) / sum(self.model_weights.values())
        
        # 最终信号决策
        if ensemble_signal_score > 0.5:
            final_signal = 1
        elif ensemble_signal_score < -0.5:
            final_signal = -1
        else:
            final_signal = 0
        
        # 集成置信度
        ensemble_confidence = np.mean(confidences)
        
        # 信号强度
        signal_strength = abs(ensemble_probability - 0.5) * 2
        
        return {
            'signal': final_signal,
            'probability': ensemble_probability,
            'confidence': ensemble_confidence,
            'signal_strength': signal_strength,
            'should_trade': ensemble_confidence >= self.confidence_threshold
        }
    
    def _calculate_model_confidence(self, prediction_result: Dict) -> float:
        """计算单个模型的置信度"""
        probability = prediction_result.get('probability', 0.5)
        signal_strength = prediction_result.get('signal_strength', 0.5)
        
        # 基于概率偏离度和信号强度计算置信度
        prob_deviation = abs(probability - 0.5) * 2  # 归一化到0-1
        
        # 综合置信度
        confidence = (prob_deviation * 0.7 + signal_strength * 0.3)
        
        return min(max(confidence, 0.0), 1.0)
    
    def update_model_weights(self, performance_metrics: Dict):
        """
        根据模型性能更新权重
        
        参数:
            performance_metrics (Dict): 模型性能指标
        """
        total_score = 0
        scores = {}
        
        # 计算每个模型的综合得分
        for model_name in self.models.keys():
            if model_name in performance_metrics:
                metrics = performance_metrics[model_name]
                # 综合评分
                score = (metrics.get('accuracy', 0) * 0.4 + 
                        metrics.get('f1_score', 0) * 0.3 + 
                        metrics.get('auc_roc', 0) * 0.3)
                scores[model_name] = score
                total_score += score
        
        # 更新权重
        if total_score > 0:
            for model_name in scores.keys():
                self.model_weights[model_name] = scores[model_name] / total_score
        
        self.logger.info("模型权重已更新")
    
    def get_prediction_summary(self, predictions: Dict) -> Dict:
        """获取预测结果摘要"""
        if 'ensemble' not in predictions:
            return {}
        
        ensemble = predictions['ensemble']
        individual = predictions['individual_predictions']
        
        # 模型一致性
        signals = [pred['signal'] for pred in individual.values()]
        signal_consistency = len(set(signals)) == 1  # 所有模型信号一致
        
        # 平均置信度
        confidences = [pred.get('confidence', 0.5) for pred in individual.values()]
        avg_confidence = np.mean(confidences)
        
        return {
            'final_signal': ensemble['signal'],
            'final_probability': ensemble['probability'],
            'final_confidence': ensemble['confidence'],
            'signal_consistency': signal_consistency,
            'average_confidence': avg_confidence,
            'should_trade': ensemble['should_trade'],
            'participating_models': len(individual)
        }
    
    def validate_prediction_quality(self, features: pd.DataFrame) -> Dict:
        """验证预测质量"""
        # 检查特征质量
        feature_quality = {
            'has_missing_values': features.isnull().any().any(),
            'missing_percentage': features.isnull().sum().sum() / (len(features) * len(features.columns)),
            'feature_count': len(features.columns),
            'sample_count': len(features)
        }
        
        # 特征范围检查
        numeric_features = features.select_dtypes(include=[np.number])
        if len(numeric_features.columns) > 0:
            feature_quality['extreme_values'] = (
                (numeric_features > 10).any().any() or 
                (numeric_features < -10).any().any()
            )
        
        return {
            'feature_quality': feature_quality,
            'prediction_viable': (
                not feature_quality['has_missing_values'] and 
                feature_quality['feature_count'] > 0
            )
        }