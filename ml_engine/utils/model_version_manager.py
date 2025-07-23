"""
模型版本管理器

实现需求1中的模型版本管理功能：
- 模型版本控制
- 模型元数据管理
- 模型回滚机制
- 性能跟踪
"""

import pandas as pd
import numpy as np
import logging
import json
import os
import shutil
from typing import Dict, List, Any, Optional
from datetime import datetime
import joblib
import warnings

warnings.filterwarnings('ignore')

class ModelVersionManager:
    """模型版本管理器"""
    
    def __init__(self, config: Dict):
        """
        初始化模型版本管理器
        
        参数:
            config (Dict): 配置信息
        """
        self.config = config
        self.ml_config = config.get("ml_signal_enhancer", {})
        self.version_config = self.ml_config.get("model_version_manager", {})
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 版本管理配置
        self.base_path = self.version_config.get("base_path", "models/versions/")
        self.max_versions = self.version_config.get("max_versions", 10)
        self.auto_cleanup = self.version_config.get("auto_cleanup", True)
        
        # 创建目录
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(os.path.join(self.base_path, "metadata"), exist_ok=True)
        
        # 版本跟踪
        self.current_version = None
        self.version_history = []
        self.version_metadata = {}
        
        # 加载现有版本信息
        self._load_version_history()
        
        self.logger.info("模型版本管理器初始化完成")
    
    def save_models(self, models: Dict[str, Any], 
                   ensemble_model: Any,
                   performance_metrics: Dict,
                   metadata: Optional[Dict] = None) -> str:
        """
        保存模型版本
        
        参数:
            models (Dict[str, Any]): 基础模型字典
            ensemble_model (Any): 集成模型
            performance_metrics (Dict): 性能指标
            metadata (Dict, optional): 额外元数据
            
        返回:
            str: 版本号
        """
        try:
            # 生成版本号
            version_id = self._generate_version_id()
            
            self.logger.info(f"保存模型版本: {version_id}")
            
            # 创建版本目录
            version_path = os.path.join(self.base_path, version_id)
            os.makedirs(version_path, exist_ok=True)
            
            # 保存基础模型
            models_path = os.path.join(version_path, "models")
            os.makedirs(models_path, exist_ok=True)
            
            saved_models = {}
            for model_name, model in models.items():
                model_file = os.path.join(models_path, f"{model_name}.joblib")
                joblib.dump(model, model_file)
                saved_models[model_name] = model_file
                self.logger.debug(f"已保存模型: {model_name}")
            
            # 保存集成模型
            ensemble_file = os.path.join(version_path, "ensemble_model.joblib")
            if ensemble_model is not None:
                joblib.dump(ensemble_model, ensemble_file)
            
            # 保存元数据
            version_metadata = {
                'version_id': version_id,
                'created_at': datetime.now().isoformat(),
                'models': list(models.keys()),
                'performance_metrics': performance_metrics,
                'model_files': saved_models,
                'ensemble_file': ensemble_file if ensemble_model is not None else None,
                'metadata': metadata or {}
            }
            
            metadata_file = os.path.join(self.base_path, "metadata", f"{version_id}.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(version_metadata, f, indent=2, ensure_ascii=False)
            
            # 更新版本历史
            self.version_history.append(version_id)
            self.version_metadata[version_id] = version_metadata
            self.current_version = version_id
            
            # 保存版本历史
            self._save_version_history()
            
            # 自动清理旧版本
            if self.auto_cleanup:
                self._cleanup_old_versions()
            
            self.logger.info(f"模型版本 {version_id} 保存成功")
            return version_id
            
        except Exception as e:
            self.logger.error(f"保存模型版本失败: {str(e)}")
            raise e
    
    def load_models(self, version_id: Optional[str] = None) -> Dict:
        """
        加载模型版本
        
        参数:
            version_id (str, optional): 版本号，默认加载最新版本
            
        返回:
            Dict: 加载的模型和元数据
        """
        try:
            if version_id is None:
                version_id = self.current_version
            
            if version_id is None or version_id not in self.version_metadata:
                raise ValueError(f"版本 {version_id} 不存在")
            
            self.logger.info(f"加载模型版本: {version_id}")
            
            metadata = self.version_metadata[version_id]
            version_path = os.path.join(self.base_path, version_id)
            
            # 加载基础模型
            models = {}
            for model_name, model_file in metadata['model_files'].items():
                if os.path.exists(model_file):
                    models[model_name] = joblib.load(model_file)
                    self.logger.debug(f"已加载模型: {model_name}")
            
            # 加载集成模型
            ensemble_model = None
            ensemble_file = metadata.get('ensemble_file')
            if ensemble_file and os.path.exists(ensemble_file):
                ensemble_model = joblib.load(ensemble_file)
            
            result = {
                'version_id': version_id,
                'models': models,
                'ensemble_model': ensemble_model,
                'metadata': metadata,
                'performance_metrics': metadata['performance_metrics']
            }
            
            self.logger.info(f"模型版本 {version_id} 加载成功")
            return result
            
        except Exception as e:
            self.logger.error(f"加载模型版本失败: {str(e)}")
            raise e
    
    def rollback_models(self, target_version: Optional[str] = None) -> Dict:
        """
        回滚到指定版本
        
        参数:
            target_version (str, optional): 目标版本，默认回滚到上一版本
            
        返回:
            Dict: 回滚结果
        """
        try:
            if target_version is None:
                # 回滚到上一版本
                if len(self.version_history) < 2:
                    raise ValueError("没有可回滚的版本")
                target_version = self.version_history[-2]
            
            if target_version not in self.version_metadata:
                raise ValueError(f"目标版本 {target_version} 不存在")
            
            self.logger.info(f"回滚模型到版本: {target_version}")
            
            # 加载目标版本
            rollback_result = self.load_models(target_version)
            
            # 更新当前版本
            old_version = self.current_version
            self.current_version = target_version
            
            # 保存版本历史
            self._save_version_history()
            
            result = {
                'success': True,
                'old_version': old_version,
                'new_version': target_version,
                'rollback_data': rollback_result
            }
            
            self.logger.info(f"成功回滚到版本 {target_version}")
            return result
            
        except Exception as e:
            self.logger.error(f"模型回滚失败: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def compare_versions(self, version1: str, version2: str) -> Dict:
        """
        比较两个版本的性能
        
        参数:
            version1 (str): 版本1
            version2 (str): 版本2
            
        返回:
            Dict: 比较结果
        """
        try:
            if version1 not in self.version_metadata or version2 not in self.version_metadata:
                raise ValueError("指定的版本不存在")
            
            metadata1 = self.version_metadata[version1]
            metadata2 = self.version_metadata[version2]
            
            metrics1 = metadata1['performance_metrics']
            metrics2 = metadata2['performance_metrics']
            
            comparison = {
                'version1': {
                    'id': version1,
                    'created_at': metadata1['created_at'],
                    'metrics': metrics1
                },
                'version2': {
                    'id': version2,
                    'created_at': metadata2['created_at'],
                    'metrics': metrics2
                },
                'improvements': {}
            }
            
            # 比较每个模型的性能
            for model_name in set(metrics1.keys()) & set(metrics2.keys()):
                if isinstance(metrics1[model_name], dict) and isinstance(metrics2[model_name], dict):
                    model_comparison = {}
                    for metric_name in set(metrics1[model_name].keys()) & set(metrics2[model_name].keys()):
                        value1 = metrics1[model_name].get(metric_name, 0)
                        value2 = metrics2[model_name].get(metric_name, 0)
                        
                        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                            improvement = value2 - value1
                            improvement_pct = (improvement / value1 * 100) if value1 != 0 else 0
                            
                            model_comparison[metric_name] = {
                                'value1': value1,
                                'value2': value2,
                                'improvement': improvement,
                                'improvement_pct': improvement_pct
                            }
                    
                    comparison['improvements'][model_name] = model_comparison
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"版本比较失败: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_version_list(self) -> List[Dict]:
        """获取版本列表"""
        version_list = []
        
        for version_id in self.version_history:
            if version_id in self.version_metadata:
                metadata = self.version_metadata[version_id]
                version_info = {
                    'version_id': version_id,
                    'created_at': metadata['created_at'],
                    'models': metadata['models'],
                    'is_current': version_id == self.current_version
                }
                
                # 添加性能摘要
                metrics = metadata['performance_metrics']
                if metrics:
                    # 计算平均性能
                    accuracy_scores = []
                    for model_metrics in metrics.values():
                        if isinstance(model_metrics, dict) and 'accuracy' in model_metrics:
                            accuracy_scores.append(model_metrics['accuracy'])
                    
                    if accuracy_scores:
                        version_info['avg_accuracy'] = np.mean(accuracy_scores)
                
                version_list.append(version_info)
        
        # 按创建时间倒序排序
        version_list.sort(key=lambda x: x['created_at'], reverse=True)
        return version_list
    
    def delete_version(self, version_id: str) -> Dict:
        """
        删除指定版本
        
        参数:
            version_id (str): 版本号
            
        返回:
            Dict: 删除结果
        """
        try:
            if version_id == self.current_version:
                raise ValueError("不能删除当前版本")
            
            if version_id not in self.version_metadata:
                raise ValueError(f"版本 {version_id} 不存在")
            
            self.logger.info(f"删除模型版本: {version_id}")
            
            # 删除版本文件夹
            version_path = os.path.join(self.base_path, version_id)
            if os.path.exists(version_path):
                shutil.rmtree(version_path)
            
            # 删除元数据文件
            metadata_file = os.path.join(self.base_path, "metadata", f"{version_id}.json")
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
            
            # 更新版本历史
            if version_id in self.version_history:
                self.version_history.remove(version_id)
            
            if version_id in self.version_metadata:
                del self.version_metadata[version_id]
            
            # 保存更新后的版本历史
            self._save_version_history()
            
            self.logger.info(f"版本 {version_id} 删除成功")
            return {'success': True, 'deleted_version': version_id}
            
        except Exception as e:
            self.logger.error(f"删除版本失败: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _generate_version_id(self) -> str:
        """生成版本号"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"v_{timestamp}"
    
    def _load_version_history(self):
        """加载版本历史"""
        try:
            history_file = os.path.join(self.base_path, "version_history.json")
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.version_history = data.get('version_history', [])
                    self.current_version = data.get('current_version')
            
            # 加载元数据
            metadata_dir = os.path.join(self.base_path, "metadata")
            if os.path.exists(metadata_dir):
                for filename in os.listdir(metadata_dir):
                    if filename.endswith('.json'):
                        version_id = filename[:-5]  # 移除.json后缀
                        metadata_file = os.path.join(metadata_dir, filename)
                        
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            self.version_metadata[version_id] = json.load(f)
            
        except Exception as e:
            self.logger.warning(f"加载版本历史失败: {str(e)}")
    
    def _save_version_history(self):
        """保存版本历史"""
        try:
            history_data = {
                'version_history': self.version_history,
                'current_version': self.current_version,
                'last_updated': datetime.now().isoformat()
            }
            
            history_file = os.path.join(self.base_path, "version_history.json")
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"保存版本历史失败: {str(e)}")
    
    def _cleanup_old_versions(self):
        """清理旧版本"""
        if len(self.version_history) <= self.max_versions:
            return
        
        # 保留最新的max_versions个版本
        versions_to_delete = self.version_history[:-self.max_versions]
        
        for version_id in versions_to_delete:
            if version_id != self.current_version:  # 不删除当前版本
                try:
                    self.delete_version(version_id)
                    self.logger.info(f"自动清理版本: {version_id}")
                except Exception as e:
                    self.logger.warning(f"清理版本 {version_id} 失败: {str(e)}")
    
    def get_version_manager_info(self) -> Dict:
        """获取版本管理器信息"""
        return {
            'base_path': self.base_path,
            'max_versions': self.max_versions,
            'current_version': self.current_version,
            'total_versions': len(self.version_history),
            'version_history': self.version_history[-5:],  # 最近5个版本
            'auto_cleanup': self.auto_cleanup
        }