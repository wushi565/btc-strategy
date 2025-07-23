"""
特征工程模块

实现需求1中的特征工程功能：
- 价格模式特征提取
- 技术指标衍生特征
- 交易量分析特征
- 波动率特征
- 时序特征提取
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import List, Dict, Optional
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from tqdm import tqdm
import time
import warnings

warnings.filterwarnings('ignore')

class FeatureEngineering:
    """特征工程类"""
    
    def __init__(self, config: Dict):
        """
        初始化特征工程器
        
        参数:
            config (Dict): 配置信息
        """
        self.config = config
        self.feature_config = config.get("feature_engineering", {})
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 特征配置
        self.lookback_periods = self.feature_config.get("lookback_periods", [5, 10, 20, 50])
        self.ma_periods = self.feature_config.get("ma_periods", [7, 14, 21, 50, 100, 200])
        self.volatility_periods = self.feature_config.get("volatility_periods", [10, 20, 30])
        self.volume_periods = self.feature_config.get("volume_periods", [10, 20])
        
        # 缩放器
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        
        # 特征名称列表
        self.feature_names = []
        
        self.logger.info("特征工程器初始化完成")
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建机器学习特征
        
        参数:
            df (pd.DataFrame): 原始OHLCV数据
            
        返回:
            pd.DataFrame: 包含所有特征的数据框
        """
        self.logger.info("开始创建ML特征...")
        start_time = time.time()
        
        # 创建副本避免修改原数据
        features_df = df.copy()
        
        # 保存原始索引
        original_index = features_df.index.copy()
        
        # 确保索引是datetime类型（在所有特征创建之前）
        if not isinstance(features_df.index, pd.DatetimeIndex):
            if 'timestamp' in features_df.columns:
                features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
                features_df.set_index('timestamp', inplace=True)
            else:
                # 创建虚拟的datetime索引以支持时序分析
                features_df.index = pd.date_range(start='2020-01-01', periods=len(features_df), freq='H')
        
        # 特征创建步骤
        feature_steps = [
            ("价格特征", self._create_price_features),
            ("技术指标特征", self._create_technical_features), 
            ("交易量特征", self._create_volume_features),
            ("波动率特征", self._create_volatility_features),
            ("时序特征", self._create_temporal_features),
            ("动量特征", self._create_momentum_features),
            ("统计特征", self._create_statistical_features),
            ("目标变量", self._create_target_variable)
        ]
        
        # 使用进度条显示特征创建进程
        print(f"🚀 开始特征工程 ({len(features_df)} 行数据)")
        with tqdm(total=len(feature_steps), desc="🔧 特征工程进度", 
                 bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                 ncols=100) as pbar:
            
            for step_name, step_func in feature_steps:
                step_start = time.time()
                pbar.set_description(f"🔧 创建{step_name}")
                
                # 保存当前索引
                current_index = features_df.index.copy()
                
                # 执行特征创建步骤
                try:
                    # 执行特征创建并确保不会改变索引长度
                    result_df = step_func(features_df)
                    
                    # 检查索引是否改变
                    if not result_df.index.equals(current_index):
                        print(f"⚠️  索引在 {step_name} 步骤中发生了变化，正在修复...")
                        result_df = result_df.reindex(current_index)
                    
                    features_df = result_df
                    
                except Exception as e:
                    print(f"❌ 在创建{step_name}时发生错误: {str(e)}")
                    self.logger.error(f"特征创建错误 ({step_name}): {str(e)}")
                
                step_time = time.time() - step_start
                pbar.set_postfix({"当前步骤": f"{step_time:.1f}s"})
                pbar.update(1)
        
        # 确保最终索引与原始索引完全一致
        if not features_df.index.equals(original_index):
            print("🔄 重新对齐最终索引...")
            features_df = features_df.reindex(original_index)
            
        print(f"\n✅ 特征创建完成，共生成 {len([col for col in features_df.columns if col not in ['target']])} 个特征")
        
        # 数据清理但不删除任何行，仅替换值（在缩放之前）
        print("🧹 正在清理数据...")
        
        # 处理无穷大值和异常值但不删除行
        features_df = self._clean_infinite_values_preserve_rows(features_df)
        
        # 特征缩放（在数据清理之后）
        if self.feature_config.get("scale_features", True):
            print("🔄 正在进行特征缩放...")
            features_df = self._scale_features(features_df)

        # 更新特征名称列表
        self.feature_names = [col for col in features_df.columns if col not in ['target']]
        
        # 时间统计和总结
        total_time = time.time() - start_time
        
        print(f"📊 特征工程完成:")
        print(f"   • 数据行数: {len(features_df)} 行")
        print(f"   • 特征数量: {len(self.feature_names)} 个")
        print(f"   • 总耗时: {total_time:.1f}秒")
        
        self.logger.info(f"特征创建完成，共生成 {len(self.feature_names)} 个特征")
        return features_df
    
    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建价格模式特征"""
        # 保存原始索引
        original_index = df.index.copy()
        
        # 基础价格特征
        df['price_range'] = df['high'] - df['low']
        df['price_change'] = df['close'].pct_change()
        
        # 安全的价格位置计算 - 避免除零
        price_range = df['high'] - df['low']
        df['price_position'] = np.where(
            price_range > 0, 
            (df['close'] - df['low']) / price_range, 
            0.5  # 如果高低相等，设为中位
        )
        
        # 价格相对位置 - 添加除零保护
        for period in self.lookback_periods:
            # 安全的比值计算
            max_val = df['high'].rolling(period).max()
            min_val = df['low'].rolling(period).min()
            mean_val = df['close'].rolling(period).mean()
            
            df[f'close_vs_max_{period}'] = np.where(
                (max_val > 0) & np.isfinite(max_val), 
                np.where(np.isfinite(df['close']), df['close'] / max_val, 1.0), 
                1.0
            )
            df[f'close_vs_min_{period}'] = np.where(
                (min_val > 0) & np.isfinite(min_val), 
                np.where(np.isfinite(df['close']), df['close'] / min_val, 1.0), 
                1.0
            )
            df[f'close_vs_mean_{period}'] = np.where(
                (mean_val > 0) & np.isfinite(mean_val), 
                np.where(np.isfinite(df['close']), df['close'] / mean_val, 1.0), 
                1.0
            )
        
        # 价格变化率特征 - 安全的对数计算
        for period in [1, 2, 3, 5, 10]:
            df[f'return_{period}'] = df['close'].pct_change(period)
            
            # 安全的对数收益率计算
            price_ratio = df['close'] / df['close'].shift(period)
            df[f'log_return_{period}'] = np.where(
                (price_ratio > 0) & np.isfinite(price_ratio),
                np.log(price_ratio),
                0.0
            )
        
        # 价格级别特征 - 添加除零保护
        df['hl_ratio'] = np.where(
            (df['low'] > 0) & np.isfinite(df['low']) & np.isfinite(df['high']), 
            df['high'] / df['low'], 
            1.0
        )
        df['oc_ratio'] = np.where(
            (df['close'] > 0) & np.isfinite(df['open']) & np.isfinite(df['close']), 
            df['open'] / df['close'], 
            1.0
        )
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        
        # 确保索引未改变
        if not df.index.equals(original_index):
            df = df.reindex(original_index)
            
        return df
    
    def _create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建技术指标特征"""
        # 创建副本避免修改原数据
        result_df = df.copy()
        
        # 保存原始索引
        original_index = result_df.index.copy()
        
        try:
            # 移动平均线 - 安全计算
            for period in self.ma_periods:
                sma_val = ta.sma(result_df['close'], length=period)
                # 确保索引对齐
                if hasattr(sma_val, 'index') and not sma_val.index.equals(original_index):
                    sma_val = sma_val.reindex(original_index)
                result_df[f'sma_{period}'] = np.where(np.isfinite(sma_val), sma_val, result_df['close'])
                
                ema_val = ta.ema(result_df['close'], length=period)
                # 确保索引对齐
                if hasattr(ema_val, 'index') and not ema_val.index.equals(original_index):
                    ema_val = ema_val.reindex(original_index)
                result_df[f'ema_{period}'] = np.where(np.isfinite(ema_val), ema_val, result_df['close'])
                
                # 安全的比值计算
                sma_val = result_df[f'sma_{period}']
                ema_val = result_df[f'ema_{period}']
                
                result_df[f'close_vs_sma_{period}'] = np.where(
                    (sma_val > 0) & np.isfinite(sma_val) & np.isfinite(result_df['close']), 
                    result_df['close'] / sma_val, 
                    1.0
                )
                result_df[f'close_vs_ema_{period}'] = np.where(
                    (ema_val > 0) & np.isfinite(ema_val) & np.isfinite(result_df['close']), 
                    result_df['close'] / ema_val, 
                    1.0
                )
            
            # RSI指标
            for period in [14, 21, 30]:
                rsi_val = ta.rsi(result_df['close'], length=period)
                # 确保索引对齐
                if hasattr(rsi_val, 'index') and not rsi_val.index.equals(original_index):
                    rsi_val = rsi_val.reindex(original_index)
                result_df[f'rsi_{period}'] = np.where(np.isfinite(rsi_val), rsi_val, 50.0)
            
            # MACD指标
            macd_data = ta.macd(result_df['close'])
            if macd_data is not None and not macd_data.empty:
                # 确保索引对齐
                if not macd_data.index.equals(original_index):
                    macd_data = macd_data.reindex(original_index)
                    
                result_df['macd'] = np.where(np.isfinite(macd_data.iloc[:, 0]), macd_data.iloc[:, 0], 0.0)
                result_df['macd_signal'] = np.where(np.isfinite(macd_data.iloc[:, 1]), macd_data.iloc[:, 1], 0.0)
                result_df['macd_histogram'] = np.where(np.isfinite(macd_data.iloc[:, 2]), macd_data.iloc[:, 2], 0.0)
            
            # 布林带 - 安全计算
            bb_data = ta.bbands(result_df['close'], length=20)
            if bb_data is not None and not bb_data.empty:
                # 确保索引对齐
                if not bb_data.index.equals(original_index):
                    bb_data = bb_data.reindex(original_index)
                    
                result_df['bb_upper'] = np.where(np.isfinite(bb_data.iloc[:, 0]), bb_data.iloc[:, 0], result_df['close'])
                result_df['bb_middle'] = np.where(np.isfinite(bb_data.iloc[:, 1]), bb_data.iloc[:, 1], result_df['close'])
                result_df['bb_lower'] = np.where(np.isfinite(bb_data.iloc[:, 2]), bb_data.iloc[:, 2], result_df['close'])
                
                # 安全的布林带宽度计算
                bb_middle = bb_data.iloc[:, 1]
                result_df['bb_width'] = np.where(
                    (bb_middle > 0) & np.isfinite(bb_middle) & np.isfinite(bb_data.iloc[:, 0]) & np.isfinite(bb_data.iloc[:, 2]),
                    (bb_data.iloc[:, 0] - bb_data.iloc[:, 2]) / bb_middle,
                    0.0
                )
                
                # 安全的布林带位置计算
                bb_range = bb_data.iloc[:, 0] - bb_data.iloc[:, 2]
                result_df['bb_position'] = np.where(
                    (bb_range > 0) & np.isfinite(bb_range) & np.isfinite(result_df['close']) & np.isfinite(bb_data.iloc[:, 2]),
                    (result_df['close'] - bb_data.iloc[:, 2]) / bb_range,
                    0.5
                )
            
            # 随机指标
            stoch_data = ta.stoch(result_df['high'], result_df['low'], result_df['close'])
            if stoch_data is not None and not stoch_data.empty:
                # 确保索引对齐
                if not stoch_data.index.equals(original_index):
                    stoch_data = stoch_data.reindex(original_index)
                    
                result_df['stoch_k'] = np.where(np.isfinite(stoch_data.iloc[:, 0]), stoch_data.iloc[:, 0], 50.0)
                result_df['stoch_d'] = np.where(np.isfinite(stoch_data.iloc[:, 1]), stoch_data.iloc[:, 1], 50.0)
            
            # ATR - 安全的比值计算
            for period in [14, 21, 30]:
                atr_val = ta.atr(result_df['high'], result_df['low'], result_df['close'], length=period)
                # 确保索引对齐
                if hasattr(atr_val, 'index') and not atr_val.index.equals(original_index):
                    atr_val = atr_val.reindex(original_index)
                    
                result_df[f'atr_{period}'] = np.where(np.isfinite(atr_val), atr_val, 0.0)
                # 避免除零
                result_df[f'atr_ratio_{period}'] = np.where(
                    (result_df['close'] > 0) & np.isfinite(result_df['close']) & np.isfinite(atr_val), 
                    atr_val / result_df['close'], 
                    0.0
                )
            
            # ADX
            adx_data = ta.adx(result_df['high'], result_df['low'], result_df['close'])
            if adx_data is not None and not adx_data.empty:
                # 确保索引对齐
                if not adx_data.index.equals(original_index):
                    adx_data = adx_data.reindex(original_index)
                    
                result_df['adx'] = np.where(np.isfinite(adx_data.iloc[:, 0]), adx_data.iloc[:, 0], 20.0)
                result_df['dmp'] = np.where(
                    adx_data.shape[1] > 1 and np.isfinite(adx_data.iloc[:, 1]), 
                    adx_data.iloc[:, 1], 
                    20.0
                ) if adx_data.shape[1] > 1 else None
                result_df['dmn'] = np.where(
                    adx_data.shape[1] > 2 and np.isfinite(adx_data.iloc[:, 2]), 
                    adx_data.iloc[:, 2], 
                    20.0
                ) if adx_data.shape[1] > 2 else None
                
        except Exception as e:
            self.logger.error(f"创建技术指标时出错: {str(e)}")
            # 确保异常不会中断流程
            pass
            
        # 确保返回的DataFrame与输入索引完全一致
        if not result_df.index.equals(original_index):
            self.logger.warning("技术指标索引不匹配，正在修复...")
            result_df = result_df.reindex(original_index)
            
        return result_df
    
    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建交易量特征"""
        # 保存原始索引
        original_index = df.index.copy()
        
        # 基础量价特征
        df['volume_price'] = np.where(
            np.isfinite(df['volume']) & np.isfinite(df['close']), 
            df['volume'] * df['close'], 
            0.0
        )
        df['volume_change'] = df['volume'].pct_change()
        
        # 量能指标 - 安全的比值计算
        for period in self.volume_periods:
            volume_sma = df['volume'].rolling(period).mean()
            df[f'volume_sma_{period}'] = np.where(np.isfinite(volume_sma), volume_sma, 0.0)
            # 避免除零
            df[f'volume_ratio_{period}'] = np.where(
                (volume_sma > 0) & np.isfinite(volume_sma) & np.isfinite(df['volume']), 
                df['volume'] / volume_sma, 
                1.0
            )
            volume_std = df['volume'].rolling(period).std()
            df[f'volume_std_{period}'] = np.where(np.isfinite(volume_std), volume_std, 0.0)
        
        # OBV (On Balance Volume)
        obv_val = ta.obv(df['close'], df['volume'])
        df['obv'] = np.where(np.isfinite(obv_val), obv_val, 0.0)
        
        # 成交量加权平均价格 - 安全计算
        for period in [10, 20]:
            vwap_val = ta.vwap(df['high'], df['low'], df['close'], df['volume'], length=period)
            df[f'vwap_{period}'] = np.where(np.isfinite(vwap_val), vwap_val, df['close'])
            if f'vwap_{period}' in df.columns:
                # 避免除零
                vwap_values = df[f'vwap_{period}']
                df[f'close_vs_vwap_{period}'] = np.where(
                    (vwap_values > 0) & np.isfinite(vwap_values) & np.isfinite(df['close']), 
                    df['close'] / vwap_values, 
                    1.0
                )
        
        # 价量关系
        df['price_volume_corr'] = df['close'].rolling(20).corr(df['volume'])
        
        # 确保索引未改变
        if not df.index.equals(original_index):
            df = df.reindex(original_index)
            
        return df
    
    def _create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建波动率特征"""
        # 保存原始索引
        original_index = df.index.copy()
        
        # 历史波动率 - 安全计算
        for period in self.volatility_periods:
            returns = df['close'].pct_change()
            vol_std = returns.rolling(period).std()
            df[f'volatility_{period}'] = np.where(np.isfinite(vol_std), vol_std * np.sqrt(252), 0.0)
            
            # 安全的波动率比值计算
            vol_mean = df[f'volatility_{period}'].rolling(period*2).mean()
            df[f'volatility_ratio_{period}'] = np.where(
                (vol_mean > 0) & np.isfinite(vol_mean) & np.isfinite(df[f'volatility_{period}']), 
                df[f'volatility_{period}'] / vol_mean, 
                1.0
            )
        
        # 实现波动率 - 安全的对数计算
        high_low_ratio = np.where(
            (df['low'] > 0) & np.isfinite(df['high']) & np.isfinite(df['low']), 
            df['high'] / df['low'], 
            1.0
        )
        close_open_ratio = np.where(
            (df['open'] > 0) & np.isfinite(df['close']) & np.isfinite(df['open']), 
            df['close'] / df['open'], 
            1.0
        )
        
        df['realized_vol'] = np.where(
            (high_low_ratio > 0) & (close_open_ratio > 0) & np.isfinite(high_low_ratio) & np.isfinite(close_open_ratio),
            np.sqrt(np.log(high_low_ratio)**2 + np.log(close_open_ratio)**2),
            0.0
        )
        
        # 波动率锥
        for period in [10, 20, 30]:
            # 只有在对应的波动率列存在时才计算
            vol_col = f'volatility_{period}'
            if vol_col in df.columns:
                vol_rank = df[vol_col].rolling(period*5).rank() / (period*5)
                df[f'vol_rank_{period}'] = np.where(np.isfinite(vol_rank), vol_rank, 0.5)
        
        # 波动率变化（使用第一个可用的波动率列）
        for period in self.volatility_periods:
            vol_col = f'volatility_{period}'
            if vol_col in df.columns:
                vol_change = df[vol_col].pct_change()
                df['vol_change'] = np.where(np.isfinite(vol_change), vol_change, 0.0)
                break  # 只需要一个波动率变化指标
        
        # 确保索引未改变
        if not df.index.equals(original_index):
            df = df.reindex(original_index)
            
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建时序特征"""
        # 保存原始索引
        original_index = df.index.copy()
        
        # 确保索引是datetime类型
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])  # 确保时间戳列为datetime类型
                df.set_index('timestamp', inplace=True)  # 将时间戳列设为索引
            else:
                # 假设数据已按时间排序，创建虚拟的datetime索引
                df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='H')
        
        # 时间特征
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # 周期性特征
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # 是否为特殊时间
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 28).astype(int)
        
        # 确保索引未改变
        if not df.index.equals(original_index):
            df = df.reindex(original_index)
            
        return df
    
    def _create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建动量特征"""
        # 保存原始索引
        original_index = df.index.copy()
        
        # 价格动量 - 安全计算
        for period in [5, 10, 20, 50]:
            past_price = df['close'].shift(period)
            df[f'momentum_{period}'] = np.where(
                (past_price > 0) & np.isfinite(past_price) & np.isfinite(df['close']),
                df['close'] / past_price - 1,
                0.0
            )
            momentum_rank = df[f'momentum_{period}'].rolling(period*2).rank()
            df[f'momentum_rank_{period}'] = np.where(np.isfinite(momentum_rank), momentum_rank, 0.0)
        
        # ROC (Rate of Change)
        for period in [10, 20, 30]:
            roc_val = ta.roc(df['close'], length=period)
            df[f'roc_{period}'] = np.where(np.isfinite(roc_val), roc_val, 0.0)
        
        # Williams %R
        for period in [14, 21]:
            willr_val = ta.willr(df['high'], df['low'], df['close'], length=period)
            df[f'willr_{period}'] = np.where(np.isfinite(willr_val), willr_val, 0.0)
        
        # CCI (Commodity Channel Index)
        cci_val = ta.cci(df['high'], df['low'], df['close'])
        df['cci'] = np.where(np.isfinite(cci_val), cci_val, 0.0)
        
        # 确保索引未改变
        if not df.index.equals(original_index):
            df = df.reindex(original_index)
            
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建统计特征"""
        # 保存原始索引
        original_index = df.index.copy()
        
        # 滚动统计
        for period in [10, 20, 50]:
            skew_val = df['close'].rolling(period).skew()
            df[f'skew_{period}'] = np.where(np.isfinite(skew_val), skew_val, 0.0)
            kurt_val = df['close'].rolling(period).kurt()
            df[f'kurt_{period}'] = np.where(np.isfinite(kurt_val), kurt_val, 0.0)
            std_val = df['close'].rolling(period).std()
            df[f'std_{period}'] = np.where(np.isfinite(std_val), std_val, 0.0)
            q25_val = df['close'].rolling(period).quantile(0.25)
            df[f'quantile_25_{period}'] = np.where(np.isfinite(q25_val), q25_val, df['close'])
            q75_val = df['close'].rolling(period).quantile(0.75)
            df[f'quantile_75_{period}'] = np.where(np.isfinite(q75_val), q75_val, df['close'])
        
        # Z-Score
        for period in [20, 50]:
            mean = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            df[f'zscore_{period}'] = np.where(
                (std > 0) & np.isfinite(std) & np.isfinite(mean) & np.isfinite(df['close']),
                (df['close'] - mean) / std,
                0.0
            )
        
        # 价格分布位置
        for period in [20, 50]:
            rank_val = df['close'].rolling(period).rank()
            df[f'price_percentile_{period}'] = np.where(np.isfinite(rank_val), rank_val / period, 0.5)
        
        # 确保索引未改变
        if not df.index.equals(original_index):
            df = df.reindex(original_index)
            
        return df
    
    def _create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建目标变量"""
        # 保存原始索引
        original_index = df.index.copy()
        
        # 基于未来收益创建目标
        future_return_periods = self.feature_config.get("target_periods", [1, 3, 5])
        
        for period in future_return_periods:
            future_return = df['close'].shift(-period) / df['close'] - 1
            # 将连续目标转换为分类目标
            df[f'target_{period}'] = (future_return > 0).astype(int)
        
        # 使用主要目标
        main_target_period = future_return_periods[0]
        df['target'] = df[f'target_{main_target_period}']
        
        # 确保索引未改变
        if not df.index.equals(original_index):
            df = df.reindex(original_index)
            
        return df
    
    def _clean_infinite_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理无穷大值和异常值"""
        
        self.logger.info("开始清理无穷大值...")
        
        # 检查哪些列包含无穷大值
        inf_cols = []
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32']:
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    inf_cols.append(f"{col}: {inf_count}")
        
        if inf_cols:
            self.logger.warning(f"发现包含无穷大值的列: {inf_cols}")
        
        # 替换无穷大值和NaN值
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # 对于每一列，用中位数替换NaN值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'target':
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        # 处理任何剩余的NaN值（如果中位数也是NaN）
        df.fillna(0, inplace=True)
        
        # 检查是否还有异常值
        for col in numeric_cols:
            if col != 'target':
                # 使用IQR方法处理极端异常值
                Q1 = df[col].quantile(0.01)
                Q99 = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=Q1, upper=Q99)
        
        # 最终检查
        final_inf_check = np.isinf(df.select_dtypes(include=[np.number])).any().any()
        if final_inf_check:
            self.logger.error("清理后仍然存在无穷大值!")
        else:
            self.logger.info("无穷大值清理完成")
        
        return df
    
    def _clean_infinite_values_preserve_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理无穷大值和异常值，但不删除任何行"""
        
        self.logger.info("开始清理无穷大值...")
        
        # 检查哪些列包含无穷大值
        inf_cols = []
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32']:
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    inf_cols.append(f"{col}: {inf_count}")
        
        if inf_cols:
            self.logger.warning(f"发现包含无穷大值的列: {inf_cols}")
        
        # 替换无穷大值为NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # 对于每一列，用中位数替换NaN值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'target':
                # 计算非NaN值的中位数
                median_val = df[col].median()
                # 如果中位数也是NaN，使用0代替
                if pd.isna(median_val):
                    median_val = 0
                df[col].fillna(median_val, inplace=True)
        
        # 处理任何剩余的NaN值（如果有）
        df.fillna(0, inplace=True)
        
        # 处理极端异常值，但保留行
        for col in numeric_cols:
            if col != 'target':
                # 使用分位数方法处理极端异常值
                try:
                    Q1 = df[col].quantile(0.01)
                    Q99 = df[col].quantile(0.99)
                    # 只有当分位数计算有效时才应用
                    if np.isfinite(Q1) and np.isfinite(Q99):
                        df[col] = df[col].clip(lower=Q1, upper=Q99)
                except Exception as e:
                    self.logger.warning(f"处理列 {col} 的极端值时出错: {str(e)}")
        
        # 最终检查
        final_inf_check = np.isinf(df.select_dtypes(include=[np.number])).any().any()
        if final_inf_check:
            self.logger.error("清理后仍然存在无穷大值，尝试强制替换!")
            # 强制替换任何剩余的无穷大值
            for col in numeric_cols:
                if col != 'target':
                    df[col].replace([np.inf, -np.inf], 0, inplace=True)
        else:
            self.logger.info("无穷大值清理完成")
            
        # 确保没有极端大的值导致模型训练失败
        # 限制所有数值在一个合理的范围内
        for col in numeric_cols:
            if col != 'target':
                # 使用一个保守的范围限制
                max_val = 1e10
                min_val = -1e10
                if (df[col] > max_val).any() or (df[col] < min_val).any():
                    self.logger.warning(f"列 {col} 包含超出范围的值，进行限制")
                    df[col] = df[col].clip(lower=min_val, upper=max_val)
        
        return df
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征缩放"""
        
        # 获取数值特征
        numeric_features = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['target'] + [col for col in numeric_features if col.startswith('target_')]
        features_to_scale = [col for col in numeric_features if col not in exclude_cols]
        
        if len(features_to_scale) > 0:
            # 缩放前的最后一次数据验证
            for col in features_to_scale:
                # 检查是否还有inf值
                if np.isinf(df[col]).any():
                    self.logger.warning(f"列 {col} 在缩放前仍然包含inf值，强制替换")
                    df[col].replace([np.inf, -np.inf], 0, inplace=True)
                
                # 检查是否还有NaN值
                if df[col].isna().any():
                    self.logger.warning(f"列 {col} 在缩放前仍然包含NaN值，强制替换")
                    df[col].fillna(0, inplace=True)
                
                # 检查数值范围
                col_max = df[col].max()
                col_min = df[col].min()
                if not (np.isfinite(col_max) and np.isfinite(col_min)):
                    self.logger.warning(f"列 {col} 包含非有限数值，强制限制")
                    df[col] = df[col].clip(-1e10, 1e10)
                    df[col].fillna(0, inplace=True)
            
            try:
                # 使用RobustScaler对异常值不敏感
                scaled_data = self.robust_scaler.fit_transform(df[features_to_scale])
                df[features_to_scale] = scaled_data
                
                # 缩放后再次验证
                if np.isinf(scaled_data).any() or np.isnan(scaled_data).any():
                    self.logger.error("缩放后仍然存在inf或NaN值！")
                    # 强制替换
                    df[features_to_scale] = np.where(
                        np.isfinite(scaled_data), 
                        scaled_data, 
                        0
                    )
                    
            except Exception as e:
                self.logger.error(f"特征缩放失败: {str(e)}，跳过缩放步骤")
                # 如果缩放失败，至少保证数据干净
                for col in features_to_scale:
                    df[col].replace([np.inf, -np.inf], 0, inplace=True)
                    df[col].fillna(0, inplace=True)
        
        return df
    
    def get_feature_importance_names(self) -> List[str]:
        """获取特征名称用于重要性分析"""
        return self.feature_names.copy()
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换新数据（用于预测）"""
        features_df = self.create_features(df)
        return features_df.drop(['target'], axis=1, errors='ignore')