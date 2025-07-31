"""
详细的ML模型性能分析脚本
分析训练结果、特征重要性和模型预测质量
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns  # 可选依赖，如果不可用会跳过
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import json
from datetime import datetime

def analyze_training_results():
    """分析训练结果"""
    print("=== ML Model Training Analysis ===")
    
    # 查找最新的结果文件
    results_files = [f for f in os.listdir("models") if f.startswith("results_") and f.endswith(".json")]
    if not results_files:
        print("No results files found")
        return
    
    latest_results_file = max(results_files)
    
    with open(f"models/{latest_results_file}", 'r') as f:
        results = json.load(f)
    
    print(f"Analysis of results from: {latest_results_file}")
    print(f"Models evaluated: {list(results.keys())}")
    
    # 创建性能对比表
    performance_df = pd.DataFrame(results).T
    performance_df = performance_df.sort_values('accuracy', ascending=False)
    
    print("\\nModel Performance Ranking:")
    print(performance_df.round(4))
    
    # 保存性能对比
    performance_df.to_csv("models/performance_comparison.csv")
    print("\\nPerformance comparison saved to models/performance_comparison.csv")
    
    return results

def detailed_model_evaluation():
    """详细的模型评估"""
    print("\\n=== Detailed Model Evaluation ===")
    
    try:
        # 加载数据
        data_file = "data/BTC_USDT_1h_20190101_to_20250627.csv"
        df = pd.read_csv(data_file)
        
        # 重新创建特征（使用与训练相同的逻辑）
        from test_ml_direct import create_features_and_signals
        X, y = create_features_and_signals(df)
        
        # 使用最后20%的数据进行评估
        split_point = int(len(X) * 0.8)
        X_eval = X.iloc[split_point:]
        y_eval = y.iloc[split_point:]
        
        print(f"Evaluation data: {X_eval.shape[0]} samples")
        print(f"Target distribution: {dict(y_eval.value_counts())}")
        
        # 找到最新的模型文件
        model_files = [f for f in os.listdir("models") if f.endswith('.joblib')]
        timestamps = set()
        for f in model_files:
            parts = f.split('_')
            if len(parts) >= 2:
                timestamp = '_'.join(parts[-2:]).replace('.joblib', '')
                timestamps.add(timestamp)
        
        latest_timestamp = max(timestamps)
        model_types = ['random_forest', 'xgboost', 'lightgbm']
        
        results = {}
        
        for model_type in model_types:
            model_file = f"{model_type}_{latest_timestamp}.joblib"
            model_path = os.path.join("models", model_file)
            
            if os.path.exists(model_path):
                print(f"\\nEvaluating {model_type}...")
                model = joblib.load(model_path)
                
                # 预测
                y_pred = model.predict(X_eval)
                y_pred_proba = model.predict_proba(X_eval)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # 计算指标
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                accuracy = accuracy_score(y_eval, y_pred)
                precision = precision_score(y_eval, y_pred)
                recall = recall_score(y_eval, y_pred)
                f1 = f1_score(y_eval, y_pred)
                
                results[model_type] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"  F1-Score: {f1:.4f}")
                
                # AUC-ROC (如果有概率预测)
                if y_pred_proba is not None:
                    auc = roc_auc_score(y_eval, y_pred_proba)
                    results[model_type]['auc_roc'] = auc
                    print(f"  AUC-ROC: {auc:.4f}")
                
                # 混淆矩阵
                cm = confusion_matrix(y_eval, y_pred)
                print(f"  Confusion Matrix:\\n{cm}")
                
                # 特征重要性 (如果支持)
                if hasattr(model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'feature': X_eval.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    print(f"  Top 10 Important Features:")
                    for idx, row in feature_importance.head(10).iterrows():
                        print(f"    {row['feature']}: {row['importance']:.4f}")
                    
                    # 保存特征重要性
                    feature_importance.to_csv(f"models/{model_type}_feature_importance.csv", index=False)
        
        # 创建详细评估报告
        evaluation_df = pd.DataFrame(results).T
        evaluation_df.to_csv("models/detailed_evaluation.csv")
        print(f"\\nDetailed evaluation saved to models/detailed_evaluation.csv")
        
        return results, X_eval, y_eval
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def analyze_signal_quality():
    """分析交易信号质量"""
    print("\\n=== Trading Signal Quality Analysis ===")
    
    try:
        # 加载数据
        data_file = "data/BTC_USDT_1h_20190101_to_20250627.csv"
        df = pd.read_csv(data_file)
        
        # 重新创建特征和信号
        from test_ml_direct import create_features_and_signals
        X, y = create_features_and_signals(df)
        
        # 分析信号分布
        print(f"Total signals: {len(y)}")
        print(f"Buy signals (1): {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
        print(f"Hold signals (0): {len(y) - sum(y)} ({(len(y) - sum(y))/len(y)*100:.1f}%)")
        
        # 按时间分析信号分布
        df_with_signals = df.copy()
        df_with_signals['signal'] = y.reindex(df.index, fill_value=0)
        
        if 'timestamp' in df.columns:
            df_with_signals['timestamp'] = pd.to_datetime(df_with_signals['timestamp'])
            df_with_signals['year'] = df_with_signals['timestamp'].dt.year
            df_with_signals['month'] = df_with_signals['timestamp'].dt.month
            
            # 按年份分析
            yearly_signals = df_with_signals.groupby('year')['signal'].agg(['count', 'sum', 'mean'])
            yearly_signals.columns = ['total_periods', 'buy_signals', 'buy_rate']
            print("\\nSignal distribution by year:")
            print(yearly_signals.round(3))
        
        # 计算信号后的收益率
        df_with_signals['future_return_1h'] = df_with_signals['close'].shift(-1) / df_with_signals['close'] - 1
        df_with_signals['future_return_24h'] = df_with_signals['close'].shift(-24) / df_with_signals['close'] - 1
        
        # 分析信号有效性
        buy_signals_mask = df_with_signals['signal'] == 1
        hold_signals_mask = df_with_signals['signal'] == 0
        
        if sum(buy_signals_mask) > 0 and sum(hold_signals_mask) > 0:
            buy_return_1h = df_with_signals[buy_signals_mask]['future_return_1h'].mean()
            hold_return_1h = df_with_signals[hold_signals_mask]['future_return_1h'].mean()
            
            buy_return_24h = df_with_signals[buy_signals_mask]['future_return_24h'].mean()
            hold_return_24h = df_with_signals[hold_signals_mask]['future_return_24h'].mean()
            
            print(f"\\nSignal Performance Analysis:")
            print(f"Buy signals - 1h return: {buy_return_1h:.4f} ({buy_return_1h*100:.2f}%)")
            print(f"Hold signals - 1h return: {hold_return_1h:.4f} ({hold_return_1h*100:.2f}%)")
            print(f"Buy signals - 24h return: {buy_return_24h:.4f} ({buy_return_24h*100:.2f}%)")
            print(f"Hold signals - 24h return: {hold_return_24h:.4f} ({hold_return_24h*100:.2f}%)")
            
            signal_advantage_1h = buy_return_1h - hold_return_1h
            signal_advantage_24h = buy_return_24h - hold_return_24h
            
            print(f"Signal advantage (buy vs hold):")
            print(f"  1h: {signal_advantage_1h:.4f} ({signal_advantage_1h*100:.2f}%)")
            print(f"  24h: {signal_advantage_24h:.4f} ({signal_advantage_24h*100:.2f}%)")
        
        # 保存信号分析结果
        signal_analysis = {
            'total_signals': len(y),
            'buy_signals': int(sum(y)),
            'buy_rate': float(sum(y)/len(y)),
            'hold_signals': int(len(y) - sum(y)),
            'hold_rate': float((len(y) - sum(y))/len(y))
        }
        
        with open("models/signal_analysis.json", 'w') as f:
            json.dump(signal_analysis, f, indent=2)
        
        print("\\nSignal analysis saved to models/signal_analysis.json")
        
    except Exception as e:
        print(f"Signal analysis failed: {e}")
        import traceback
        traceback.print_exc()

def create_model_comparison_chart():
    """创建模型对比图表"""
    print("\\n=== Creating Model Comparison Charts ===")
    
    try:
        # 读取评估结果
        if os.path.exists("models/detailed_evaluation.csv"):
            df = pd.read_csv("models/detailed_evaluation.csv", index_col=0)
            
            # 选择要展示的指标
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            available_metrics = [m for m in metrics if m in df.columns]
            
            if available_metrics:
                # 创建对比图
                plt.figure(figsize=(12, 8))
                
                # 子图1：指标对比柱状图
                plt.subplot(2, 2, 1)
                df[available_metrics].plot(kind='bar', ax=plt.gca())
                plt.title('Model Performance Comparison')
                plt.ylabel('Score')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.xticks(rotation=45)
                
                # 子图2：准确率对比
                plt.subplot(2, 2, 2)
                plt.bar(df.index, df['accuracy'])
                plt.title('Model Accuracy Comparison')
                plt.ylabel('Accuracy')
                plt.xticks(rotation=45)
                
                # 如果有AUC数据，添加AUC对比
                if 'auc_roc' in df.columns:
                    plt.subplot(2, 2, 3)
                    plt.bar(df.index, df['auc_roc'])
                    plt.title('Model AUC-ROC Comparison')
                    plt.ylabel('AUC-ROC')
                    plt.xticks(rotation=45)
                
                # 综合评分（F1分数）
                plt.subplot(2, 2, 4)
                plt.bar(df.index, df['f1_score'])
                plt.title('Model F1-Score Comparison')
                plt.ylabel('F1-Score')
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                plt.savefig('models/model_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                print("Model comparison chart saved to models/model_comparison.png")
        
    except Exception as e:
        print(f"Chart creation failed: {e}")

def generate_summary_report():
    """生成汇总报告"""
    print("\\n=== Generating Summary Report ===")
    
    report = []
    report.append("# ML Model Training Summary Report")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # 基本信息
    model_files = [f for f in os.listdir("models") if f.endswith('.joblib')]
    report.append(f"## Training Overview")
    report.append(f"- Models trained: {len(model_files)}")
    report.append(f"- Model types: Random Forest, XGBoost, LightGBM")
    report.append("")
    
    # 性能结果
    if os.path.exists("models/detailed_evaluation.csv"):
        df = pd.read_csv("models/detailed_evaluation.csv", index_col=0)
        
        report.append("## Model Performance")
        report.append("| Model | Accuracy | Precision | Recall | F1-Score |")
        report.append("|-------|----------|-----------|--------|----------|")
        
        for model_name, row in df.iterrows():
            acc = row.get('accuracy', 0)
            prec = row.get('precision', 0)
            rec = row.get('recall', 0)
            f1 = row.get('f1_score', 0)
            report.append(f"| {model_name} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} |")
        
        report.append("")
        
        # 最佳模型
        best_model = df['accuracy'].idxmax()
        best_accuracy = df.loc[best_model, 'accuracy']
        report.append(f"## Best Model")
        report.append(f"- **{best_model}** with accuracy: {best_accuracy:.4f}")
        report.append("")
    
    # 信号分析
    if os.path.exists("models/signal_analysis.json"):
        with open("models/signal_analysis.json", 'r') as f:
            signal_data = json.load(f)
        
        report.append("## Signal Analysis")
        report.append(f"- Total signals generated: {signal_data.get('total_signals', 0):,}")
        report.append(f"- Buy signals: {signal_data.get('buy_signals', 0):,} ({signal_data.get('buy_rate', 0)*100:.1f}%)")
        report.append(f"- Hold signals: {signal_data.get('hold_signals', 0):,} ({signal_data.get('hold_rate', 0)*100:.1f}%)")
        report.append("")
    
    # 文件列表
    report.append("## Generated Files")
    output_files = [
        "models/performance_comparison.csv",
        "models/detailed_evaluation.csv", 
        "models/signal_analysis.json",
        "models/model_comparison.png"
    ]
    
    for file_path in output_files:
        if os.path.exists(file_path):
            report.append(f"- ✅ {file_path}")
        else:
            report.append(f"- ❌ {file_path}")
    
    report.append("")
    report.append("## Next Steps")
    report.append("1. Review model performance metrics")
    report.append("2. Analyze feature importance")
    report.append("3. Test models on live data")
    report.append("4. Integrate best model into trading system")
    
    # 保存报告
    with open("models/training_summary_report.md", 'w', encoding='utf-8') as f:
        f.write("\\n".join(report))
    
    print("Summary report saved to models/training_summary_report.md")

if __name__ == "__main__":
    print("Starting comprehensive ML model analysis...")
    
    # 执行所有分析
    analyze_training_results()
    detailed_model_evaluation()
    analyze_signal_quality()
    create_model_comparison_chart()
    generate_summary_report()
    
    print("\\n=== Analysis Complete ===")
    print("All analysis files have been generated in the models/ directory")
    print("Check training_summary_report.md for a complete overview")