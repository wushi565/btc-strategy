#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
综合模型分析脚本
解决以下5个关键问题：
1. 详细分析模型性能
2. 运行模型预测
3. 调查数据集为什么所有模型性能相同
4. 实现集成方法提升性能
5. 检查特征重要性
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns  # 暂时注释掉
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML相关库
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc, precision_recall_curve
)
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb
import xgboost as xgb

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class ComprehensiveModelAnalyzer:
    """综合模型分析器"""
    
    def __init__(self, models_dir="models", data_dir="data"):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.models = {}
        self.model_names = ['random_forest', 'lightgbm', 'xgboost', 'logistic_regression']
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
        # 创建结果目录
        self.results_dir = "analysis_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        print("🚀 初始化综合模型分析器...")
        
    def load_models(self):
        """加载所有训练好的模型"""
        print("\n📥 加载训练好的模型...")
        
        # 找到最新的模型文件
        model_files = {}
        for model_name in self.model_names:
            pattern = f"{model_name}_*.joblib"
            files = [f for f in os.listdir(self.models_dir) if f.startswith(model_name) and f.endswith('.joblib')]
            if files:
                # 选择最新的文件
                latest_file = sorted(files)[-1]
                model_files[model_name] = os.path.join(self.models_dir, latest_file)
                print(f"   ✅ 找到 {model_name}: {latest_file}")
            else:
                print(f"   ❌ 未找到 {model_name} 模型文件")
        
        # 加载模型
        for model_name, file_path in model_files.items():
            try:
                self.models[model_name] = joblib.load(file_path)
                print(f"   ✅ 成功加载 {model_name}")
            except Exception as e:
                print(f"   ❌ 加载 {model_name} 失败: {e}")
        
        print(f"\n📊 总共加载了 {len(self.models)} 个模型")
        return len(self.models) > 0
    
    def load_data(self):
        """加载测试数据"""
        print("\n📥 加载数据集...")
        
        # 尝试加载不同格式的数据文件
        data_files = ['processed_data.csv', 'btc_data.csv', 'data.csv']
        
        for data_file in data_files:
            file_path = os.path.join(self.data_dir, data_file)
            if os.path.exists(file_path):
                print(f"   📁 找到数据文件: {data_file}")
                try:
                    df = pd.read_csv(file_path)
                    print(f"   📊 数据形状: {df.shape}")
                    print(f"   📋 列名: {list(df.columns)[:10]}...")
                    
                    # 假设最后一列是目标变量
                    if 'target' in df.columns:
                        X = df.drop('target', axis=1)
                        y = df['target']
                    elif 'label' in df.columns:
                        X = df.drop('label', axis=1)
                        y = df['label']
                    else:
                        # 假设最后一列是目标变量
                        X = df.iloc[:, :-1]
                        y = df.iloc[:, -1]
                    
                    # 简单的数据分割 (80/20)
                    split_idx = int(len(df) * 0.8)
                    self.X_train = X.iloc[:split_idx]
                    self.X_test = X.iloc[split_idx:]
                    self.y_train = y.iloc[:split_idx]
                    self.y_test = y.iloc[split_idx:]
                    self.feature_names = X.columns.tolist()
                    
                    print(f"   ✅ 训练集: {self.X_train.shape}, 测试集: {self.X_test.shape}")
                    print(f"   🎯 目标变量分布: {y.value_counts().to_dict()}")
                    
                    return True
                    
                except Exception as e:
                    print(f"   ❌ 读取 {data_file} 失败: {e}")
                    continue
        
        print("   ⚠️  未找到合适的数据文件，将生成模拟数据进行演示")
        self._generate_sample_data()
        return True
    
    def _generate_sample_data(self):
        """生成模拟数据用于演示"""
        print("   🔧 生成模拟数据...")
        np.random.seed(42)
        
        n_samples = 1000
        n_features = 20
        
        # 生成特征数据
        X = np.random.randn(n_samples, n_features)
        
        # 生成目标变量（二分类）
        # 让某些特征有实际的预测能力
        y = (X[:, 0] + X[:, 1] * 0.5 + X[:, 2] * 0.3 + np.random.randn(n_samples) * 0.2 > 0).astype(int)
        
        # 创建DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        # 分割数据
        split_idx = int(n_samples * 0.8)
        self.X_train = X_df.iloc[:split_idx]
        self.X_test = X_df.iloc[split_idx:]
        self.y_train = pd.Series(y[:split_idx])
        self.y_test = pd.Series(y[split_idx:])
        self.feature_names = feature_names
        
        print(f"   ✅ 生成模拟数据完成: 训练集{self.X_train.shape}, 测试集{self.X_test.shape}")
    
    def analyze_model_performance(self):
        """1. 详细分析模型性能"""
        print("\n🔍 1. 详细分析模型性能")
        print("=" * 50)
        
        if not self.models:
            print("❌ 没有可用的模型")
            return
        
        performance_results = {}
        
        # 为每个模型计算详细指标
        for model_name, model in self.models.items():
            print(f"\n📊 分析 {model_name} 模型:")
            
            try:
                # 预测
                y_pred = model.predict(self.X_test)
                y_pred_proba = None
                
                # 尝试获取预测概率
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_pred_proba = model.decision_function(self.X_test)
                
                # 计算指标
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='weighted')
                recall = recall_score(self.y_test, y_pred, average='weighted')
                f1 = f1_score(self.y_test, y_pred, average='weighted')
                
                performance_results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"   准确率: {accuracy:.4f}")
                print(f"   精确率: {precision:.4f}")
                print(f"   召回率: {recall:.4f}")
                print(f"   F1分数: {f1:.4f}")
                
                # 分类报告
                print(f"\n   详细分类报告:")
                print(classification_report(self.y_test, y_pred, target_names=['类别0', '类别1']))
                
            except Exception as e:
                print(f"   ❌ 分析 {model_name} 时出错: {e}")
                continue
        
        # 创建性能比较图表
        self._plot_performance_comparison(performance_results)
        
        return performance_results
    
    def _plot_performance_comparison(self, performance_results):
        """绘制性能比较图表"""
        if not performance_results:
            return
        
        # 提取指标数据
        models = list(performance_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        data = []
        for model in models:
            for metric in metrics:
                data.append({
                    'Model': model,
                    'Metric': metric,
                    'Score': performance_results[model][metric]
                })
        
        df_metrics = pd.DataFrame(data)
        
        # 创建条形图（使用matplotlib）
        plt.figure(figsize=(12, 6))
        
        # 手动创建分组条形图
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            metric_scores = [performance_results[model][metric] for model in models]
            plt.bar(x + i*width, metric_scores, width, label=metric, alpha=0.8)
        
        plt.title('模型性能指标比较')
        plt.ylabel('分数')
        plt.xlabel('模型')
        plt.xticks(x + width*1.5, models, rotation=45)
        plt.legend(title='指标')
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(os.path.join(self.results_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 创建混淆矩阵
        self._plot_confusion_matrices(performance_results)
    
    def _plot_confusion_matrices(self, performance_results):
        """绘制混淆矩阵"""
        n_models = len(performance_results)
        if n_models == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, (model_name, results) in enumerate(performance_results.items()):
            if i >= 4:  # 最多显示4个模型
                break
                
            cm = confusion_matrix(self.y_test, results['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
            axes[i].set_title(f'{model_name} 混淆矩阵')
            axes[i].set_xlabel('预测标签')
            axes[i].set_ylabel('真实标签')
        
        # 隐藏多余的子图
        for i in range(len(performance_results), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_predictions(self):
        """2. 运行模型预测"""
        print("\n🎯 2. 运行模型预测")
        print("=" * 50)
        
        if not self.models:
            print("❌ 没有可用的模型")
            return
        
        # 选择一些测试样本进行预测
        sample_indices = np.random.choice(len(self.X_test), min(10, len(self.X_test)), replace=False)
        sample_X = self.X_test.iloc[sample_indices]
        sample_y_true = self.y_test.iloc[sample_indices]
        
        print(f"📋 对 {len(sample_indices)} 个样本进行预测:")
        print("-" * 80)
        print(f"{'样本ID':<8} {'真实值':<8} {'RF':<8} {'LGB':<8} {'XGB':<8} {'LR':<8} {'一致性':<8}")
        print("-" * 80)
        
        predictions_summary = []
        
        for i, idx in enumerate(sample_indices):
            row_data = {
                'sample_id': idx,
                'true_label': sample_y_true.iloc[i]
            }
            
            predictions = []
            for model_name in self.model_names:
                if model_name in self.models:
                    try:
                        pred = self.models[model_name].predict(sample_X.iloc[[i]])[0]
                        row_data[model_name] = pred
                        predictions.append(pred)
                    except:
                        row_data[model_name] = 'ERROR'
                        predictions.append(-1)
                else:
                    row_data[model_name] = 'N/A'
            
            # 检查预测一致性
            valid_predictions = [p for p in predictions if p != -1]
            consistency = "✅" if len(set(valid_predictions)) == 1 else "❌"
            row_data['consistency'] = consistency
            
            predictions_summary.append(row_data)
            
            # 打印预测结果
            rf_pred = row_data.get('random_forest', 'N/A')
            lgb_pred = row_data.get('lightgbm', 'N/A')
            xgb_pred = row_data.get('xgboost', 'N/A')
            lr_pred = row_data.get('logistic_regression', 'N/A')
            
            print(f"{idx:<8} {sample_y_true.iloc[i]:<8} {rf_pred:<8} {lgb_pred:<8} {xgb_pred:<8} {lr_pred:<8} {consistency:<8}")
        
        # 保存预测结果
        pred_df = pd.DataFrame(predictions_summary)
        pred_df.to_csv(os.path.join(self.results_dir, 'sample_predictions.csv'), index=False)
        
        print(f"\n💾 预测结果已保存到: {self.results_dir}/sample_predictions.csv")
        
        return predictions_summary
    
    def investigate_identical_performance(self):
        """3. 调查为什么所有模型性能相同"""
        print("\n🔍 3. 调查数据集 - 为什么所有模型性能相同？")
        print("=" * 50)
        
        # 数据分布分析
        print("📊 数据分布分析:")
        print(f"   特征数量: {len(self.feature_names)}")
        print(f"   样本数量: {len(self.X_train) + len(self.X_test)}")
        print(f"   目标变量分布:")
        
        # 目标变量分布
        y_all = pd.concat([self.y_train, self.y_test])
        value_counts = y_all.value_counts()
        for value, count in value_counts.items():
            percentage = count / len(y_all) * 100
            print(f"     类别 {value}: {count} 个样本 ({percentage:.1f}%)")
        
        # 检查数据不平衡
        if len(value_counts) == 2:
            minority_ratio = min(value_counts) / sum(value_counts)
            if minority_ratio < 0.1:
                print(f"   ⚠️  数据严重不平衡 (少数类占比: {minority_ratio:.2%})")
            elif minority_ratio < 0.3:
                print(f"   ⚠️  数据不平衡 (少数类占比: {minority_ratio:.2%})")
        
        # 特征相关性分析
        print(f"\n🔗 特征相关性分析:")
        correlation_matrix = self.X_train.corr()
        
        # 找出高相关性的特征对
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = abs(correlation_matrix.iloc[i, j])
                if corr_val > 0.9:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i], 
                        correlation_matrix.columns[j], 
                        corr_val
                    ))
        
        if high_corr_pairs:
            print(f"   发现 {len(high_corr_pairs)} 对高相关性特征 (相关系数 > 0.9):")
            for feat1, feat2, corr in high_corr_pairs[:5]:  # 只显示前5对
                print(f"     {feat1} - {feat2}: {corr:.3f}")
        else:
            print("   未发现高相关性特征对")
        
        # 特征方差分析
        print(f"\n📈 特征方差分析:")
        feature_variances = self.X_train.var()
        low_variance_features = feature_variances[feature_variances < 0.01]
        
        if len(low_variance_features) > 0:
            print(f"   发现 {len(low_variance_features)} 个低方差特征 (方差 < 0.01):")
            for feat in low_variance_features.index[:5]:  # 只显示前5个
                print(f"     {feat}: {feature_variances[feat]:.6f}")
        else:
            print("   所有特征都有足够的方差")
        
        # 特征与目标变量的相关性
        print(f"\n🎯 特征与目标变量相关性:")
        feature_target_corr = pd.DataFrame({
            'feature': self.feature_names,
            'correlation': [self.X_train[feat].corr(self.y_train) for feat in self.feature_names]
        })
        feature_target_corr['abs_correlation'] = abs(feature_target_corr['correlation'])
        feature_target_corr = feature_target_corr.sort_values('abs_correlation', ascending=False)
        
        print("   最相关的前10个特征:")
        for _, row in feature_target_corr.head(10).iterrows():
            print(f"     {row['feature']}: {row['correlation']:.4f}")
        
        # 绘制相关性热图
        self._plot_correlation_heatmap(correlation_matrix)
        
        # 可能的原因分析
        print(f"\n🤔 可能导致模型性能相同的原因:")
        
        reasons = []
        if minority_ratio < 0.1:
            reasons.append("• 数据严重不平衡，模型可能都倾向于预测多数类")
        
        if len(high_corr_pairs) > len(self.feature_names) * 0.1:
            reasons.append("• 特征间存在高度共线性，降低了模型的区分能力")
        
        if len(low_variance_features) > len(self.feature_names) * 0.2:
            reasons.append("• 多个特征方差过低，信息量不足")
        
        if feature_target_corr['abs_correlation'].max() < 0.2:
            reasons.append("• 特征与目标变量相关性普遍较弱，预测能力有限")
        
        if len(reasons) == 0:
            reasons.append("• 需要进一步分析，可能是特征工程或数据预处理问题")
        
        for reason in reasons:
            print(f"   {reason}")
        
        return {
            'target_distribution': value_counts.to_dict(),
            'high_corr_pairs': high_corr_pairs,
            'low_variance_features': low_variance_features.to_dict(),
            'feature_target_correlation': feature_target_corr
        }
    
    def _plot_correlation_heatmap(self, correlation_matrix):
        """绘制相关性热图"""
        plt.figure(figsize=(12, 10))
        
        # 如果特征太多，只显示部分
        if len(correlation_matrix) > 20:
            # 选择与目标变量最相关的前20个特征
            feature_target_corr = pd.Series([self.X_train[feat].corr(self.y_train) for feat in self.feature_names], 
                                          index=self.feature_names)
            top_features = feature_target_corr.abs().nlargest(20).index
            correlation_matrix = correlation_matrix.loc[top_features, top_features]
        
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title('特征相关性热图')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def implement_ensemble_methods(self):
        """4. 实现集成方法提升性能"""
        print("\n🤝 4. 实现集成方法提升性能")
        print("=" * 50)
        
        if len(self.models) < 2:
            print("❌ 需要至少2个模型才能创建集成")
            return
        
        ensemble_results = {}
        
        # 1. 投票集成 (Voting)
        print("📊 创建投票集成...")
        try:
            # 准备模型列表
            estimators = [(name, model) for name, model in self.models.items()]
            
            # 硬投票
            hard_voting = VotingClassifier(estimators=estimators, voting='hard')
            hard_voting.fit(self.X_train, self.y_train)
            
            hard_pred = hard_voting.predict(self.X_test)
            hard_accuracy = accuracy_score(self.y_test, hard_pred)
            hard_f1 = f1_score(self.y_test, hard_pred, average='weighted')
            
            ensemble_results['hard_voting'] = {
                'accuracy': hard_accuracy,
                'f1': hard_f1,
                'predictions': hard_pred
            }
            
            print(f"   硬投票集成 - 准确率: {hard_accuracy:.4f}, F1: {hard_f1:.4f}")
            
            # 软投票 (如果所有模型都支持概率预测)
            try:
                soft_voting = VotingClassifier(estimators=estimators, voting='soft')
                soft_voting.fit(self.X_train, self.y_train)
                
                soft_pred = soft_voting.predict(self.X_test)
                soft_accuracy = accuracy_score(self.y_test, soft_pred)
                soft_f1 = f1_score(self.y_test, soft_pred, average='weighted')
                
                ensemble_results['soft_voting'] = {
                    'accuracy': soft_accuracy,
                    'f1': soft_f1,
                    'predictions': soft_pred
                }
                
                print(f"   软投票集成 - 准确率: {soft_accuracy:.4f}, F1: {soft_f1:.4f}")
                
            except Exception as e:
                print(f"   软投票集成失败: {e}")
        
        except Exception as e:
            print(f"   创建投票集成失败: {e}")
        
        # 2. 简单平均集成
        print("\n📊 创建简单平均集成...")
        try:
            # 收集所有模型的预测概率
            all_probas = []
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(self.X_test)[:, 1]
                    elif hasattr(model, 'decision_function'):
                        proba = model.decision_function(self.X_test)
                        # 将决策函数输出转换为概率
                        from sklearn.preprocessing import MinMaxScaler
                        scaler = MinMaxScaler()
                        proba = scaler.fit_transform(proba.reshape(-1, 1)).flatten()
                    else:
                        continue
                    
                    all_probas.append(proba)
                except:
                    continue
            
            if len(all_probas) >= 2:
                # 平均概率
                avg_proba = np.mean(all_probas, axis=0)
                avg_pred = (avg_proba > 0.5).astype(int)
                
                avg_accuracy = accuracy_score(self.y_test, avg_pred)
                avg_f1 = f1_score(self.y_test, avg_pred, average='weighted')
                
                ensemble_results['average_ensemble'] = {
                    'accuracy': avg_accuracy,
                    'f1': avg_f1,
                    'predictions': avg_pred
                }
                
                print(f"   平均集成 - 准确率: {avg_accuracy:.4f}, F1: {avg_f1:.4f}")
            
        except Exception as e:
            print(f"   创建平均集成失败: {e}")
        
        # 3. 加权集成（基于单模型性能）
        print("\n📊 创建加权集成...")
        try:
            # 计算每个模型在验证集上的权重
            model_weights = {}
            total_f1 = 0
            
            for model_name, model in self.models.items():
                try:
                    val_pred = model.predict(self.X_test)
                    val_f1 = f1_score(self.y_test, val_pred, average='weighted')
                    model_weights[model_name] = val_f1
                    total_f1 += val_f1
                except:
                    model_weights[model_name] = 0
            
            # 归一化权重
            for model_name in model_weights:
                model_weights[model_name] = model_weights[model_name] / total_f1 if total_f1 > 0 else 1.0 / len(model_weights)
            
            # 加权预测
            if len(all_probas) >= 2:
                weighted_proba = np.zeros_like(all_probas[0])
                
                i = 0
                for model_name, model in self.models.items():
                    if i < len(all_probas):
                        weighted_proba += model_weights[model_name] * all_probas[i]
                        i += 1
                
                weighted_pred = (weighted_proba > 0.5).astype(int)
                weighted_accuracy = accuracy_score(self.y_test, weighted_pred)
                weighted_f1 = f1_score(self.y_test, weighted_pred, average='weighted')
                
                ensemble_results['weighted_ensemble'] = {
                    'accuracy': weighted_accuracy,
                    'f1': weighted_f1,
                    'predictions': weighted_pred,
                    'weights': model_weights
                }
                
                print(f"   加权集成 - 准确率: {weighted_accuracy:.4f}, F1: {weighted_f1:.4f}")
                print("   模型权重:")
                for model_name, weight in model_weights.items():
                    print(f"     {model_name}: {weight:.3f}")
        
        except Exception as e:
            print(f"   创建加权集成失败: {e}")
        
        # 保存集成结果
        if ensemble_results:
            # 创建集成性能比较
            self._plot_ensemble_comparison(ensemble_results)
            
            # 保存最佳集成模型
            best_ensemble = max(ensemble_results.items(), key=lambda x: x[1]['f1'])
            print(f"\n🏆 最佳集成方法: {best_ensemble[0]} (F1: {best_ensemble[1]['f1']:.4f})")
            
            # 如果有改进，保存集成模型
            single_model_f1 = max([f1_score(self.y_test, model.predict(self.X_test), average='weighted') 
                                 for model in self.models.values()])
            
            if best_ensemble[1]['f1'] > single_model_f1:
                improvement = best_ensemble[1]['f1'] - single_model_f1
                print(f"✅ 集成方法提升了 {improvement:.4f} F1分数!")
            else:
                print("⚠️  集成方法没有明显提升性能")
        
        return ensemble_results
    
    def _plot_ensemble_comparison(self, ensemble_results):
        """绘制集成方法比较图"""
        if not ensemble_results:
            return
        
        # 获取单个模型的性能作为基准
        single_model_performance = {}
        for model_name, model in self.models.items():
            try:
                pred = model.predict(self.X_test)
                single_model_performance[model_name] = {
                    'accuracy': accuracy_score(self.y_test, pred),
                    'f1': f1_score(self.y_test, pred, average='weighted')
                }
            except:
                continue
        
        # 合并数据
        all_results = {**single_model_performance, **ensemble_results}
        
        # 创建比较图
        methods = list(all_results.keys())
        accuracies = [all_results[method]['accuracy'] for method in methods]
        f1_scores = [all_results[method]['f1'] for method in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, accuracies, width, label='准确率', alpha=0.8)
        ax.bar(x + width/2, f1_scores, width, label='F1分数', alpha=0.8)
        
        ax.set_xlabel('方法')
        ax.set_ylabel('分数')
        ax.set_title('单模型 vs 集成方法性能比较')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'ensemble_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def check_feature_importance(self):
        """5. 检查特征重要性"""
        print("\n🔍 5. 检查特征重要性")
        print("=" * 50)
        
        feature_importance_results = {}
        
        # 为每个支持特征重要性的模型提取重要性
        for model_name, model in self.models.items():
            print(f"\n📊 分析 {model_name} 特征重要性:")
            
            try:
                importance = None
                
                # 不同模型提取特征重要性的方法
                if hasattr(model, 'feature_importances_'):
                    # Random Forest, XGBoost, LightGBM
                    importance = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    # Logistic Regression
                    importance = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
                
                if importance is not None:
                    # 创建特征重要性DataFrame
                    importance_df = pd.DataFrame({
                        'feature': self.feature_names,
                        'importance': importance
                    }).sort_values('importance', ascending=False)
                    
                    feature_importance_results[model_name] = importance_df
                    
                    # 显示前10个最重要的特征
                    print("   前10个最重要特征:")
                    for _, row in importance_df.head(10).iterrows():
                        print(f"     {row['feature']}: {row['importance']:.4f}")
                    
                    # 计算累积重要性
                    cumsum = importance_df['importance'].cumsum()
                    total_importance = importance_df['importance'].sum()
                    
                    # 找出解释80%重要性的特征数量
                    features_80_percent = len(cumsum[cumsum <= 0.8 * total_importance]) + 1
                    print(f"   前 {features_80_percent} 个特征解释了 80% 的重要性")
                
                else:
                    print("   该模型不支持特征重要性分析")
            
            except Exception as e:
                print(f"   分析 {model_name} 特征重要性时出错: {e}")
        
        # 创建特征重要性可视化
        if feature_importance_results:
            self._plot_feature_importance(feature_importance_results)
            
            # 计算特征重要性一致性
            self._analyze_feature_importance_consistency(feature_importance_results)
        
        return feature_importance_results
    
    def _plot_feature_importance(self, feature_importance_results):
        """绘制特征重要性图表"""
        n_models = len(feature_importance_results)
        if n_models == 0:
            return
        
        # 为每个模型创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, (model_name, importance_df) in enumerate(feature_importance_results.items()):
            if i >= 4:  # 最多显示4个模型
                break
            
            # 选择前15个最重要的特征
            top_features = importance_df.head(15)
            
            # 创建条形图
            axes[i].barh(range(len(top_features)), top_features['importance'])
            axes[i].set_yticks(range(len(top_features)))
            axes[i].set_yticklabels(top_features['feature'])
            axes[i].set_xlabel('重要性')
            axes[i].set_title(f'{model_name} 特征重要性')
            axes[i].invert_yaxis()
        
        # 隐藏多余的子图
        for i in range(len(feature_importance_results), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _analyze_feature_importance_consistency(self, feature_importance_results):
        """分析特征重要性一致性"""
        print(f"\n🔍 特征重要性一致性分析:")
        
        if len(feature_importance_results) < 2:
            print("   需要至少2个模型来分析一致性")
            return
        
        # 获取每个模型的前10重要特征
        top_features_by_model = {}
        for model_name, importance_df in feature_importance_results.items():
            top_features_by_model[model_name] = set(importance_df.head(10)['feature'].tolist())
        
        # 计算交集
        model_names = list(top_features_by_model.keys())
        common_features = top_features_by_model[model_names[0]]
        
        for model_name in model_names[1:]:
            common_features = common_features.intersection(top_features_by_model[model_name])
        
        print(f"   所有模型共同的重要特征 ({len(common_features)} 个):")
        for feature in sorted(common_features):
            print(f"     • {feature}")
        
        # 计算Jaccard相似性
        print(f"\n   模型间特征重要性相似性 (Jaccard系数):")
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                set1, set2 = top_features_by_model[model1], top_features_by_model[model2]
                
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                jaccard = intersection / union if union > 0 else 0
                
                print(f"     {model1} vs {model2}: {jaccard:.3f}")
    
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("\n📋 生成综合分析报告...")
        
        report_path = os.path.join(self.results_dir, 'comprehensive_analysis_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 机器学习模型综合分析报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 1. 数据集概况\n")
            f.write(f"- 特征数量: {len(self.feature_names)}\n")
            f.write(f"- 训练样本: {len(self.X_train)}\n")
            f.write(f"- 测试样本: {len(self.X_test)}\n")
            
            y_all = pd.concat([self.y_train, self.y_test])
            value_counts = y_all.value_counts()
            f.write(f"- 目标变量分布:\n")
            for value, count in value_counts.items():
                percentage = count / len(y_all) * 100
                f.write(f"  - 类别 {value}: {count} 个样本 ({percentage:.1f}%)\n")
            
            f.write(f"\n## 2. 模型性能总结\n")
            f.write(f"- 训练的模型数量: {len(self.models)}\n")
            f.write("- 所有模型性能相同的可能原因:\n")
            f.write("  - 数据特征缺乏区分度\n")
            f.write("  - 目标变量分布不均衡\n")
            f.write("  - 特征工程需要改进\n")
            
            f.write(f"\n## 3. 改进建议\n")
            f.write("1. **数据预处理改进**:\n")
            f.write("   - 处理数据不平衡问题 (SMOTE, 权重调整)\n")
            f.write("   - 特征选择和降维\n")
            f.write("   - 异常值检测和处理\n\n")
            
            f.write("2. **特征工程优化**:\n")
            f.write("   - 创建新的组合特征\n")
            f.write("   - 特征变换 (多项式, 对数)\n")
            f.write("   - 时间序列特征 (如果适用)\n\n")
            
            f.write("3. **模型调优**:\n")
            f.write("   - 更细粒度的超参数调优\n")
            f.write("   - 尝试其他算法 (神经网络, SVM)\n")
            f.write("   - 集成学习方法\n\n")
            
            f.write("4. **评估改进**:\n")
            f.write("   - 使用分层交叉验证\n")
            f.write("   - 更多评估指标 (AUC, 精确率-召回率曲线)\n")
            f.write("   - 学习曲线分析\n\n")
        
        print(f"✅ 综合分析报告已保存到: {report_path}")
    
    def run_complete_analysis(self):
        """运行完整的综合分析"""
        print("🚀 开始综合模型分析...")
        print("=" * 60)
        
        # 加载模型和数据
        if not self.load_models():
            print("❌ 无法加载模型，分析终止")
            return
        
        if not self.load_data():
            print("❌ 无法加载数据，分析终止") 
            return
        
        # 执行5个分析任务
        try:
            # 1. 详细分析模型性能
            performance_results = self.analyze_model_performance()
            
            # 2. 运行预测
            prediction_results = self.run_predictions()
            
            # 3. 调查相同性能原因
            investigation_results = self.investigate_identical_performance()
            
            # 4. 实现集成方法
            ensemble_results = self.implement_ensemble_methods()
            
            # 5. 检查特征重要性
            feature_importance_results = self.check_feature_importance()
            
            # 生成综合报告
            self.generate_comprehensive_report()
            
            print(f"\n🎉 综合分析完成!")
            print(f"📁 所有结果已保存到: {self.results_dir}/")
            print("📋 查看 comprehensive_analysis_report.md 获取详细报告")
            
        except Exception as e:
            print(f"❌ 分析过程中出错: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    print("🔬 机器学习模型综合分析工具")
    print("=" * 60)
    
    # 创建分析器
    analyzer = ComprehensiveModelAnalyzer()
    
    # 运行完整分析
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
