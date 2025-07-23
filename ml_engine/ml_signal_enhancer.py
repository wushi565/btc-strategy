"""
机器学习信号增强器主控制器

实现需求1的核心功能：
- 集成多种机器学习模型
- 特征工程和信号生成
- 置信度评分系统
- 在线学习和模型更新
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings

from .features.feature_engineering import FeatureEngineering
from .training.model_trainer import ModelTrainer
from .models.signal_predictor import SignalPredictor
from .models.model_ensemble import ModelEnsemble
from .utils.model_version_manager import ModelVersionManager

# 忽略模型训练过程中的警告
warnings.filterwarnings('ignore', category=UserWarning)

class MLSignalEnhancer:
    """
    机器学习信号增强器
    
    根据需求1的验收标准实现：
    1. 机器学习模块对历史价格模式和指标进行特征工程和训练
    2. 训练完成后，对新市场数据进行预测并提供置信度分数
    3. 提供二次确认，增强交易决策
    4. 根据置信度大小决定是否执行交易
    5. 根据新数据不断更新和优化
    """
    
    def __init__(self, config: Dict):
        """
        初始化ML信号增强器
        
        参数:
            config (Dict): 配置信息
        """
        self.config = config
        self.ml_config = config.get("ml_signal_enhancer", {})
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 初始化核心组件
        self.feature_engineering = FeatureEngineering(config)
        self.model_trainer = ModelTrainer(config)
        self.signal_predictor = SignalPredictor(config)
        self.model_ensemble = ModelEnsemble(config)
        self.version_manager = ModelVersionManager(config)
        
        # 模型状态
        self.models_trained = False
        self.last_update_time = None
        self.performance_metrics = {}
        
        # 配置参数
        self.confidence_threshold = self.ml_config.get("confidence_threshold", 0.6)
        self.update_frequency = self.ml_config.get("update_frequency", 24)  # 小时
        self.enable_online_learning = self.ml_config.get("enable_online_learning", True)
        
        self.logger.info("ML信号增强器初始化完成")
    
    def train_models(self, df: pd.DataFrame) -> Dict:
        """
        训练机器学习模型
        
        实现验收标准1: 对历史价格模式和指标进行特征工程和训练
        
        参数:
            df (pd.DataFrame): 历史数据
            
        返回:
            Dict: 训练结果和性能指标
        """
        self.logger.info("开始训练ML模型...")
        
        try:
            # 1. 特征工程
            self.logger.info("执行特征工程...")
            features_df = self.feature_engineering.create_features(df)
            
            # 2. 准备训练数据
            X, y = self._prepare_training_data(features_df)
            
            # 3. 训练多个模型
            self.logger.info("训练多个ML模型...")
            training_results = self.model_trainer.train_models(X, y)
            
            # 4. 创建集成模型
            self.logger.info("创建模型集成...")
            ensemble_results = self.model_ensemble.create_ensemble(
                training_results['models'], X, y
            )
            
            # 5. 保存模型版本
            model_version = self.version_manager.save_models(
                training_results['models'],
                ensemble_results['ensemble_model'],
                training_results['metrics']
            )
            
            # 6. 加载模型到信号预测器
            self.signal_predictor.load_models(training_results['models'])
            
            self.models_trained = True
            self.last_update_time = datetime.now()
            self.performance_metrics = training_results['metrics']
            
            results = {
                'success': True,
                'model_version': model_version,
                'metrics': training_results['metrics'],
                'ensemble_score': ensemble_results['ensemble_score'],
                'feature_count': len(self.feature_engineering.feature_names),
                'training_samples': len(X)
            }
            
            self.logger.info(f"模型训练完成，版本: {model_version}")
            return results
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def predict_signals(self, df: pd.DataFrame, 
                       technical_signals: Optional[pd.Series] = None) -> Dict:
        """
        生成ML增强的交易信号
        
        实现验收标准2,3,4: 对新数据进行预测，提供置信度分数，进行二次确认
        
        参数:
            df (pd.DataFrame): 市场数据
            technical_signals (pd.Series, optional): 技术指标信号
            
        返回:
            Dict: 预测结果包含信号、概率和置信度
        """
        if not self.models_trained:
            self.logger.warning("模型尚未训练，无法生成信号")
            return {'success': False, 'error': '模型尚未训练'}
        
        try:
            # 1. 特征工程
            features_df = self.feature_engineering.create_features(df)
            
            # 2. 获取最新数据点的特征
            latest_features = features_df.iloc[-1:].drop(['target'], axis=1, errors='ignore')
            
            # 3. 模型预测
            prediction_results = self.signal_predictor.predict(
                latest_features,
                return_probabilities=True
            )
            
            # 4. 集成预测
            ensemble_prediction = self.model_ensemble.predict(latest_features)
            
            # 5. 计算置信度分数
            confidence_score = self._calculate_confidence_score(
                prediction_results,
                ensemble_prediction
            )
            
            # 6. 生成最终信号
            final_signal = self._generate_final_signal(
                prediction_results,
                confidence_score,
                technical_signals
            )
            
            results = {
                'success': True,
                'signal': final_signal['signal'],
                'direction_probability': final_signal['direction_prob'],
                'strength_score': final_signal['strength'],
                'confidence_score': confidence_score,
                'individual_predictions': prediction_results,
                'ensemble_prediction': ensemble_prediction,
                'should_trade': confidence_score >= self.confidence_threshold,
                'timestamp': datetime.now()
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"信号预测失败: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def update_models(self, new_data: pd.DataFrame) -> Dict:
        """
        在线学习更新模型
        
        实现验收标准5: 根据新数据不断更新和优化
        
        参数:
            new_data (pd.DataFrame): 新的市场数据
            
        返回:
            Dict: 更新结果
        """
        if not self.enable_online_learning:
            return {'success': False, 'error': '在线学习未启用'}
        
        try:
            # 检查是否需要更新
            if not self._should_update_models():
                return {'success': False, 'error': '尚未达到更新时间'}
            
            self.logger.info("开始在线学习更新...")
            
            # 1. 特征工程
            features_df = self.feature_engineering.create_features(new_data)
            X_new, y_new = self._prepare_training_data(features_df)
            
            # 2. 增量训练
            update_results = self.model_trainer.update_models(X_new, y_new)
            
            # 3. 更新集成模型
            ensemble_update = self.model_ensemble.update_ensemble(
                update_results['updated_models']
            )
            
            # 4. 性能评估
            performance_change = self._evaluate_performance_change(
                update_results['metrics']
            )
            
            # 5. 决定是否采用更新
            if performance_change['improved']:
                # 保存新版本
                new_version = self.version_manager.save_models(
                    update_results['updated_models'],
                    ensemble_update['updated_ensemble'],
                    update_results['metrics']
                )
                
                self.performance_metrics = update_results['metrics']
                self.last_update_time = datetime.now()
                
                results = {
                    'success': True,
                    'updated': True,
                    'new_version': new_version,
                    'performance_improvement': performance_change['improvement_pct'],
                    'metrics': update_results['metrics']
                }
            else:
                # 性能下降，回滚到之前版本
                self.version_manager.rollback_models()
                results = {
                    'success': True,
                    'updated': False,
                    'reason': '性能下降，已回滚',
                    'performance_change': performance_change['improvement_pct']
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"模型更新失败: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_model_performance(self) -> Dict:
        """获取模型性能指标"""
        return {
            'trained': self.models_trained,
            'last_update': self.last_update_time,
            'metrics': self.performance_metrics,
            'feature_count': len(self.feature_engineering.feature_names) if hasattr(self.feature_engineering, 'feature_names') else 0,
            'confidence_threshold': self.confidence_threshold
        }
    
    def _prepare_training_data(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """准备训练数据"""
        self.logger.info(f"开始准备训练数据，原始行数: {len(features_df)}")
        
        # 记录原始列
        orig_columns = features_df.columns
        
        # 1. 检查无穷大值和极值
        inf_mask = np.isinf(features_df.select_dtypes(include=[np.number])).any(axis=1)
        inf_count = inf_mask.sum()
        if inf_count > 0:
            self.logger.warning(f"发现包含无穷大值的行: {inf_count}，正在处理...")
            # 替换每列中的无穷大值为NaN
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
        
        # 2. 检查特征有效性
        num_cols = features_df.select_dtypes(include=[np.number]).columns
        feature_stats = {}
        for col in num_cols:
            if col != 'target':
                feature_stats[col] = {
                    'nan_count': features_df[col].isna().sum(),
                    'min': features_df[col].min() if not features_df[col].isna().all() else None,
                    'max': features_df[col].max() if not features_df[col].isna().all() else None,
                }
                
                # 剔除全为NaN或常数的特征
                if features_df[col].isna().all() or features_df[col].nunique() <= 1:
                    self.logger.warning(f"剔除无效特征: {col}，全为NaN或常数")
                    features_df = features_df.drop(col, axis=1)
                
                # 限制极端值
                if col in features_df.columns and not features_df[col].isna().all():
                    if (features_df[col].abs() > 1e10).any():
                        self.logger.warning(f"特征 {col} 包含极端值，进行截断")
                        features_df[col] = features_df[col].clip(-1e10, 1e10)
        
        # 3. 使用中位数填充剩余的NaN值
        nan_count_before = features_df.isna().sum().sum()
        if nan_count_before > 0:
            self.logger.warning(f"数据中包含 {nan_count_before} 个NaN值，使用中位数填充")
            for col in features_df.select_dtypes(include=[np.number]).columns:
                if col != 'target':
                    median_val = features_df[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    features_df[col] = features_df[col].fillna(median_val)
                    
        # 4. 最终检查
        # 删除任何仍然包含NaN的行
        clean_df = features_df.dropna()
        rows_dropped = len(features_df) - len(clean_df)
        if rows_dropped > 0:
            self.logger.warning(f"删除了 {rows_dropped} 行包含NaN的数据")
        
        # 最后一次检查无穷大值
        if np.isinf(clean_df.select_dtypes(include=[np.number])).any().any():
            self.logger.error("数据清理后仍存在无穷大值，强制替换为0")
            clean_df = clean_df.replace([np.inf, -np.inf], 0)
        
        # 验证所有特征数据都是有限值
        if not np.isfinite(clean_df.select_dtypes(include=[np.number])).all().all():
            self.logger.error("数据中仍存在无限或NaN值，使用0填充")
            clean_df = clean_df.fillna(0).replace([np.inf, -np.inf], 0)
        
        # 分离特征和目标
        if 'target' in clean_df.columns:
            X = clean_df.drop('target', axis=1)
            y = clean_df['target']
        else:
            # 如果没有目标列，创建基于价格变化的目标
            X = clean_df
            price_change = clean_df['close'].pct_change().shift(-1)  # 下一期收益
            y = (price_change > 0).astype(int)  # 1表示上涨，0表示下跌
        
        self.logger.info(f"训练数据准备完成: {len(X)} 行, {X.shape[1]} 个特征")
        return X, y
    
    def _calculate_confidence_score(self, predictions: Dict, 
                                  ensemble_prediction: Dict) -> float:
        """计算置信度分数"""
        # 基于预测一致性和概率值计算置信度
        if not predictions.get('success', False):
            return 0.5
        
        individual_preds = predictions.get('individual_predictions', {})
        individual_probs = [pred.get('probability', 0.5) for pred in individual_preds.values()]
        ensemble_prob = ensemble_prediction.get('probability', 0.5)
        
        # 计算预测一致性
        if len(individual_probs) == 0:
            individual_probs = [0.5]  # 默认值
        consistency = 1.0 - np.std(individual_probs) if len(individual_probs) > 1 else 1.0
        
        # 计算平均概率偏离度
        avg_prob = np.mean(individual_probs)
        prob_deviation = abs(avg_prob - 0.5) * 2  # 归一化到0-1
        
        # 综合置信度
        ensemble_prob = ensemble_prob if ensemble_prob is not None else 0.5
        confidence = (consistency * 0.4 + prob_deviation * 0.4 + 
                     abs(ensemble_prob - 0.5) * 2 * 0.2)
        
        return min(max(confidence, 0.0), 1.0)
    
    def _generate_final_signal(self, predictions: Dict, confidence: float,
                             technical_signals: Optional[pd.Series]) -> Dict:
        """生成最终交易信号"""
        # 集成所有模型的预测
        signals = []
        probs = []
        
        if not predictions.get('success', False):
            return {'signal': 0, 'direction_prob': 0.5, 'strength': 0.0}
            
        individual_preds = predictions.get('individual_predictions', {})
        for model_name, pred in individual_preds.items():
            signals.append(pred.get('signal', 0))
            probs.append(pred.get('probability', 0.5))
        
        # 多数投票决定信号方向
        avg_signal = np.mean(signals)
        direction = 1 if avg_signal > 0 else -1 if avg_signal < 0 else 0
        
        # 计算强度
        strength = abs(avg_signal) * confidence
        
        # 如有技术信号，进行二次确认
        if technical_signals is not None and len(technical_signals) > 0:
            tech_signal = technical_signals.iloc[-1] if not pd.isna(technical_signals.iloc[-1]) else 0
            
            # 信号一致性检查
            if direction != 0 and tech_signal != 0:
                if np.sign(direction) == np.sign(tech_signal):
                    # 信号一致，增强强度
                    strength *= 1.2
                else:
                    # 信号冲突，降低强度
                    strength *= 0.6
        
        return {
            'signal': direction,
            'direction_prob': np.mean(probs),
            'strength': min(strength, 1.0)
        }
    
    def _should_update_models(self) -> bool:
        """检查是否应该更新模型"""
        if self.last_update_time is None:
            return True
        
        hours_since_update = (datetime.now() - self.last_update_time).total_seconds() / 3600
        return hours_since_update >= self.update_frequency
    
    def _evaluate_performance_change(self, new_metrics: Dict) -> Dict:
        """评估性能变化"""
        if not self.performance_metrics:
            return {'improved': True, 'improvement_pct': 0}
        
        # 比较关键指标
        key_metric = 'accuracy'  # 可以根据需要调整
        
        old_score = self.performance_metrics.get(key_metric, 0)
        new_score = new_metrics.get(key_metric, 0)
        
        improvement_pct = ((new_score - old_score) / old_score * 100) if old_score > 0 else 0
        
        return {
            'improved': improvement_pct > 1.0,  # 至少1%的改进
            'improvement_pct': improvement_pct
        }