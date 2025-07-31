"""
训练检查点管理器

实现训练中断恢复功能：
- 保存训练进度检查点
- 恢复中断的训练
- 支持增量训练状态管理
"""

import os
import json
import pickle
import logging
import copy
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import joblib
import pandas as pd
import numpy as np

class TrainingCheckpoint:
    """训练检查点管理器"""
    
    def __init__(self, config: Dict):
        """
        初始化训练检查点管理器
        
        参数:
            config (Dict): 配置信息
        """
        self.config = config
        self.ml_config = config.get("ml_signal_enhancer", {})
        self.checkpoint_config = self.ml_config.get("training_checkpoint", {})
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 检查点配置
        self.checkpoint_dir = self.checkpoint_config.get("checkpoint_dir", "ml_engine/checkpoints/")
        self.auto_save_interval = self.checkpoint_config.get("auto_save_interval", 1)  # 每训练N个模型保存一次
        self.max_checkpoints = self.checkpoint_config.get("max_checkpoints", 5)
        self.enable_data_checkpoints = self.checkpoint_config.get("enable_data_checkpoints", True)
        
        # 创建检查点目录
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(self.checkpoint_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(self.checkpoint_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.checkpoint_dir, "metadata"), exist_ok=True)
        
        # 当前训练状态
        self.current_session_id = None
        self.training_progress = {}
        
        self.logger.info("训练检查点管理器初始化完成")
    
    def start_training_session(self, training_config: Dict, 
                              X_train: pd.DataFrame, 
                              y_train: pd.Series) -> str:
        """
        开始新的训练会话
        
        参数:
            training_config (Dict): 训练配置
            X_train (pd.DataFrame): 训练特征
            y_train (pd.Series): 训练标签
            
        返回:
            str: 会话ID
        """
        try:
            # 生成会话ID
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.current_session_id = session_id
            
            self.logger.info(f"开始训练会话: {session_id}")
            
            # 保存训练数据
            if self.enable_data_checkpoints:
                self._save_training_data(session_id, X_train, y_train)
            
            # 初始化训练进度
            self.training_progress = {
                'session_id': session_id,
                'start_time': datetime.now().isoformat(),
                'training_config': training_config,
                'model_types': training_config.get('model_types', []),
                'completed_models': [],
                'failed_models': [],
                'current_model': None,
                'total_models': len(training_config.get('model_types', [])),
                'status': 'running',
                'data_shape': {
                    'n_samples': len(X_train),
                    'n_features': X_train.shape[1]
                }
            }
            
            # 保存初始检查点
            self._save_checkpoint(session_id)
            
            return session_id
            
        except Exception as e:
            self.logger.error(f"启动训练会话失败: {str(e)}")
            raise e
    
    def save_model_progress(self, model_name: str, model: Any, 
                           metrics: Dict, hyperparams: Optional[Dict] = None):
        """
        保存单个模型的训练进度
        
        参数:
            model_name (str): 模型名称
            model (Any): 训练好的模型
            metrics (Dict): 模型性能指标
            hyperparams (Dict, optional): 超参数
        """
        if not self.current_session_id:
            self.logger.warning("没有活跃的训练会话")
            return
        
        try:
            self.logger.info(f"保存模型进度: {model_name}")
            
            # 更新当前模型状态
            self.training_progress['current_model'] = model_name
            
            # 保存模型文件
            model_file = os.path.join(
                self.checkpoint_dir, "models", 
                f"{self.current_session_id}_{model_name}.joblib"
            )
            joblib.dump(model, model_file)
            
            # 保存模型元数据
            model_metadata = {
                'model_name': model_name,
                'saved_at': datetime.now().isoformat(),
                'model_file': model_file,
                'metrics': metrics,
                'hyperparams': hyperparams or {},
                'session_id': self.current_session_id
            }
            
            metadata_file = os.path.join(
                self.checkpoint_dir, "metadata",
                f"{self.current_session_id}_{model_name}.json"
            )
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(model_metadata, f, indent=2, ensure_ascii=False)
            
            # 更新训练进度
            if model_name not in self.training_progress['completed_models']:
                self.training_progress['completed_models'].append(model_name)
            
            # 检查是否需要自动保存检查点
            completed_count = len(self.training_progress['completed_models'])
            if completed_count % self.auto_save_interval == 0:
                self._save_checkpoint(self.current_session_id)
                self.logger.info(f"自动保存检查点 - 已完成 {completed_count} 个模型")
            
        except Exception as e:
            self.logger.error(f"保存模型进度失败: {str(e)}")
            self._record_model_failure(model_name, str(e))
    
    def complete_training_session(self, final_results: Dict):
        """
        完成训练会话
        
        参数:
            final_results (Dict): 最终训练结果
        """
        if not self.current_session_id:
            self.logger.warning("没有活跃的训练会话")
            return
        
        try:
            self.logger.info(f"完成训练会话: {self.current_session_id}")
            
            # 创建可序列化的结果副本
            serializable_results = {}
            
            # 处理模型对象 - 不直接序列化模型对象，而是保存路径或引用
            if 'models' in final_results:
                # 存储模型路径而不是模型对象
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                serializable_results['model_paths'] = {
                    model_name: f"models/{model_name}_{timestamp}.joblib"
                    for model_name in final_results['models'].keys()
                }
                # 不包含模型对象
            
            # 复制其他可序列化的结果
            for key, value in final_results.items():
                if key != 'models':
                    # 对于metrics等其他信息，直接复制
                    serializable_results[key] = value
            
            # 更新最终状态
            self.training_progress['status'] = 'completed'
            self.training_progress['end_time'] = datetime.now().isoformat()
            self.training_progress['final_results'] = serializable_results
            self.training_progress['current_model'] = None
            
            # 保存最终检查点
            self._save_checkpoint(self.current_session_id)
            
            # 清理当前会话
            self.current_session_id = None
            self.training_progress = {}
            
            # 自动清理旧检查点
            self._cleanup_old_checkpoints()
            
        except Exception as e:
            self.logger.error(f"完成训练会话失败: {str(e)}")
    
    def resume_training(self, session_id: Optional[str] = None) -> Dict:
        """
        恢复中断的训练
        
        参数:
            session_id (str, optional): 要恢复的会话ID，默认恢复最新的未完成会话
            
        返回:
            Dict: 恢复的训练状态
        """
        try:
            # 查找要恢复的会话
            if session_id is None:
                session_id = self._find_latest_incomplete_session()
            
            if session_id is None:
                return {
                    'success': False,
                    'error': '没有找到可恢复的训练会话'
                }
            
            self.logger.info(f"恢复训练会话: {session_id}")
            
            # 加载检查点
            checkpoint_data = self._load_checkpoint(session_id)
            if not checkpoint_data:
                return {
                    'success': False,
                    'error': f'无法加载会话 {session_id} 的检查点'
                }
            
            # 恢复训练状态
            self.current_session_id = session_id
            self.training_progress = checkpoint_data['training_progress']
            
            # 加载已完成的模型
            completed_models = {}
            model_metrics = {}
            
            for model_name in self.training_progress['completed_models']:
                model_data = self._load_model_data(session_id, model_name)
                if model_data:
                    completed_models[model_name] = model_data['model']
                    model_metrics[model_name] = model_data['metrics']
            
            # 确定还需要训练的模型
            all_models = set(self.training_progress['model_types'])
            completed_set = set(self.training_progress['completed_models'])
            remaining_models = list(all_models - completed_set)
            
            # 加载训练数据
            training_data = None
            if self.enable_data_checkpoints:
                training_data = self._load_training_data(session_id)
            
            resume_info = {
                'success': True,
                'session_id': session_id,
                'completed_models': completed_models,
                'model_metrics': model_metrics,
                'remaining_models': remaining_models,
                'training_progress': self.training_progress.copy(),
                'training_data': training_data,
                'progress_percentage': len(completed_models) / self.training_progress['total_models'] * 100
            }
            
            self.logger.info(f"成功恢复训练会话，已完成 {len(completed_models)}/{self.training_progress['total_models']} 个模型")
            return resume_info
            
        except Exception as e:
            self.logger.error(f"恢复训练失败: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def list_checkpoints(self) -> List[Dict]:
        """
        列出所有可用的检查点
        
        返回:
            List[Dict]: 检查点列表
        """
        checkpoints = []
        
        try:
            checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) 
                              if f.endswith('_checkpoint.json')]
            
            for checkpoint_file in checkpoint_files:
                session_id = checkpoint_file.replace('_checkpoint.json', '')
                checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_file)
                
                try:
                    with open(checkpoint_path, 'r', encoding='utf-8') as f:
                        checkpoint_data = json.load(f)
                    
                    progress = checkpoint_data.get('training_progress', {})
                    
                    checkpoint_info = {
                        'session_id': session_id,
                        'start_time': progress.get('start_time'),
                        'status': progress.get('status', 'unknown'),
                        'completed_models': len(progress.get('completed_models', [])),
                        'total_models': progress.get('total_models', 0),
                        'current_model': progress.get('current_model'),
                        'data_shape': progress.get('data_shape', {}),
                        'can_resume': progress.get('status') in ['running', 'interrupted']
                    }
                    
                    checkpoints.append(checkpoint_info)
                except json.JSONDecodeError as je:
                    self.logger.error(f"解析检查点文件 {checkpoint_file} 失败: {str(je)}")
                    # 跳过损坏的检查点文件
                    continue
            
            # 按时间倒序排序
            checkpoints.sort(key=lambda x: x['start_time'] or '', reverse=True)
            
        except Exception as e:
            self.logger.error(f"列出检查点失败: {str(e)}")
        
        return checkpoints
    
    def delete_checkpoint(self, session_id: str) -> bool:
        """
        删除指定的检查点
        
        参数:
            session_id (str): 会话ID
            
        返回:
            bool: 是否删除成功
        """
        try:
            self.logger.info(f"删除检查点: {session_id}")
            
            # 删除主检查点文件
            checkpoint_file = os.path.join(self.checkpoint_dir, f"{session_id}_checkpoint.json")
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
            
            # 删除训练数据
            data_file = os.path.join(self.checkpoint_dir, "data", f"{session_id}_data.pkl")
            if os.path.exists(data_file):
                os.remove(data_file)
            
            # 删除模型文件
            model_dir = os.path.join(self.checkpoint_dir, "models")
            for filename in os.listdir(model_dir):
                if filename.startswith(f"{session_id}_"):
                    os.remove(os.path.join(model_dir, filename))
            
            # 删除元数据文件
            metadata_dir = os.path.join(self.checkpoint_dir, "metadata")
            for filename in os.listdir(metadata_dir):
                if filename.startswith(f"{session_id}_"):
                    os.remove(os.path.join(metadata_dir, filename))
            
            self.logger.info(f"检查点 {session_id} 删除成功")
            return True
            
        except Exception as e:
            self.logger.error(f"删除检查点失败: {str(e)}")
            return False
    
    def _save_checkpoint(self, session_id: str):
        """保存检查点"""
        try:
            # 创建一个新的字典，防止修改原始进度数据
            checkpoint_progress = copy.deepcopy(self.training_progress)
            
            # 确保没有不可序列化的对象
            if 'final_results' in checkpoint_progress and 'models' in checkpoint_progress['final_results']:
                del checkpoint_progress['final_results']['models']
            
            checkpoint_data = {
                'session_id': session_id,
                'saved_at': datetime.now().isoformat(),
                'training_progress': checkpoint_progress,
                'checkpoint_version': '1.0'
            }
            
            # 安全序列化为JSON
            checkpoint_file = os.path.join(self.checkpoint_dir, f"{session_id}_checkpoint.json")
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"保存检查点失败: {str(e)}")
    
    def _load_checkpoint(self, session_id: str) -> Optional[Dict]:
        """加载检查点"""
        try:
            checkpoint_file = os.path.join(self.checkpoint_dir, f"{session_id}_checkpoint.json")
            if not os.path.exists(checkpoint_file):
                return None
            
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            self.logger.error(f"加载检查点失败: {str(e)}")
            return None
    
    def _save_training_data(self, session_id: str, X_train: pd.DataFrame, y_train: pd.Series):
        """保存训练数据"""
        try:
            data_file = os.path.join(self.checkpoint_dir, "data", f"{session_id}_data.pkl")
            data = {
                'X_train': X_train,
                'y_train': y_train,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(data_file, 'wb') as f:
                pickle.dump(data, f)
                
        except Exception as e:
            self.logger.error(f"保存训练数据失败: {str(e)}")
    
    def _load_training_data(self, session_id: str) -> Optional[Dict]:
        """加载训练数据"""
        try:
            data_file = os.path.join(self.checkpoint_dir, "data", f"{session_id}_data.pkl")
            if not os.path.exists(data_file):
                return None
            
            with open(data_file, 'rb') as f:
                return pickle.load(f)
                
        except Exception as e:
            self.logger.error(f"加载训练数据失败: {str(e)}")
            return None
    
    def _load_model_data(self, session_id: str, model_name: str) -> Optional[Dict]:
        """加载模型数据"""
        try:
            # 加载模型文件
            model_file = os.path.join(
                self.checkpoint_dir, "models",
                f"{session_id}_{model_name}.joblib"
            )
            if not os.path.exists(model_file):
                return None
            
            model = joblib.load(model_file)
            
            # 加载元数据
            metadata_file = os.path.join(
                self.checkpoint_dir, "metadata",
                f"{session_id}_{model_name}.json"
            )
            
            metadata = {}
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            
            return {
                'model': model,
                'metrics': metadata.get('metrics', {}),
                'hyperparams': metadata.get('hyperparams', {}),
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"加载模型数据失败: {str(e)}")
            return None
    
    def _find_latest_incomplete_session(self) -> Optional[str]:
        """查找最新的未完成会话"""
        try:
            checkpoints = self.list_checkpoints()
            
            for checkpoint in checkpoints:
                if checkpoint['can_resume']:
                    return checkpoint['session_id']
            
            return None
            
        except Exception as e:
            self.logger.error(f"查找未完成会话失败: {str(e)}")
            return None
    
    def _record_model_failure(self, model_name: str, error_msg: str):
        """记录模型训练失败"""
        if model_name not in self.training_progress.get('failed_models', []):
            self.training_progress.setdefault('failed_models', []).append({
                'model_name': model_name,
                'error': error_msg,
                'failed_at': datetime.now().isoformat()
            })
    
    def _cleanup_old_checkpoints(self):
        """清理旧检查点"""
        try:
            checkpoints = self.list_checkpoints()
            
            # 只保留已完成的检查点中最新的几个
            completed_checkpoints = [cp for cp in checkpoints if cp['status'] == 'completed']
            
            if len(completed_checkpoints) > self.max_checkpoints:
                # 删除多余的检查点
                to_delete = completed_checkpoints[self.max_checkpoints:]
                
                for checkpoint in to_delete:
                    self.delete_checkpoint(checkpoint['session_id'])
                    self.logger.info(f"自动清理旧检查点: {checkpoint['session_id']}")
                    
        except Exception as e:
            self.logger.error(f"清理旧检查点失败: {str(e)}")
    
    def get_checkpoint_stats(self) -> Dict:
        """获取检查点统计信息"""
        try:
            checkpoints = self.list_checkpoints()
            
            stats = {
                'total_checkpoints': len(checkpoints),
                'resumable_sessions': len([cp for cp in checkpoints if cp['can_resume']]),
                'completed_sessions': len([cp for cp in checkpoints if cp['status'] == 'completed']),
                'current_session': self.current_session_id,
                'checkpoint_dir': self.checkpoint_dir,
                'auto_save_interval': self.auto_save_interval,
                'max_checkpoints': self.max_checkpoints
            }
            
            if checkpoints:
                stats['latest_checkpoint'] = checkpoints[0]['session_id']
                stats['latest_checkpoint_time'] = checkpoints[0]['start_time']
            
            return stats
            
        except Exception as e:
            self.logger.error(f"获取检查点统计失败: {str(e)}")
            return {}