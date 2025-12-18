"""
特征工程模块 - 负责特征生成、转换与选择
原油期货多模型集成投资策略
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import logging
import warnings

from config import FEATURE_CONFIG

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """特征工程类"""
    
    def __init__(self, config: dict = None):
        """
        初始化特征工程器
        
        Args:
            config: 特征配置字典
        """
        self.config = config or FEATURE_CONFIG
        self.scaler = StandardScaler()
        self.selector = None
        self.feature_names_ = None
        self.selected_features_ = None
        self.is_fitted = False
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建滞后特征
        
        Args:
            df: 输入DataFrame
            
        Returns:
            添加滞后特征后的DataFrame
        """
        df = df.copy()
        lag_periods = self.config.get('lag_periods', 60)
        
        # 定义需要创建滞后特征的列
        price_cols = ['open', 'high', 'low', 'close']
        indicator_cols = ['rsi', 'macd', 'atr']
        
        # 价格对数收益率滞后
        for col in price_cols:
            if col in df.columns:
                log_ret = np.log(df[col] / df[col].shift(1))
                for lag in [1, 2, 3, 5, 10, 20]:
                    if lag <= lag_periods:
                        df[f'{col}_log_ret_lag_{lag}'] = log_ret.shift(lag)
        
        # 成交量变化滞后
        if 'volume' in df.columns:
            vol_pct = df['volume'].pct_change()
            for lag in [1, 2, 3, 5, 10]:
                df[f'volume_pct_lag_{lag}'] = vol_pct.shift(lag)
        
        # 技术指标滞后
        for col in indicator_cols:
            if col in df.columns:
                for lag in [1, 3, 5, 10, 20]:
                    if lag <= lag_periods:
                        df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建动量特征
        
        Args:
            df: 输入DataFrame
            
        Returns:
            添加动量特征后的DataFrame
        """
        df = df.copy()
        
        # 滚动对数收益率和
        if 'log_return' in df.columns:
            for window in self.config.get('momentum_windows', [1, 3, 5, 10, 20]):
                df[f'momentum_{window}d'] = df['log_return'].rolling(window=window).sum()
        
        # 滚动波动率
        if 'log_return' in df.columns:
            for window in self.config.get('volatility_windows', [5, 10, 20]):
                df[f'volatility_{window}d'] = df['log_return'].rolling(window=window).std()
        
        # 价格相对位置：当前价格在过去N日区间的百分比位置
        if 'close' in df.columns:
            for window in [10, 20, 50]:
                rolling_high = df['high'].rolling(window=window).max()
                rolling_low = df['low'].rolling(window=window).min()
                df[f'price_position_{window}d'] = (df['close'] - rolling_low) / (rolling_high - rolling_low)
        
        # 均线乖离率
        for sma_period in [20, 50]:
            sma_col = f'sma_{sma_period}'
            if sma_col in df.columns:
                df[f'ma_deviation_{sma_period}'] = (df['close'] - df[sma_col]) / df[sma_col]
        
        # 价格动量
        if 'close' in df.columns:
            for period in [5, 10, 20]:
                df[f'price_roc_{period}'] = df['close'].pct_change(periods=period)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建交互特征
        
        Args:
            df: 输入DataFrame
            
        Returns:
            添加交互特征后的DataFrame
        """
        df = df.copy()
        
        # 量价交互
        if 'log_return' in df.columns and 'volume' in df.columns:
            vol_change = df['volume'].pct_change()
            df['volume_price_interaction'] = df['log_return'] * vol_change
        
        # RSI与价格收益率交互
        if 'rsi' in df.columns and 'log_return' in df.columns:
            df['rsi_return_interaction'] = df['rsi'] * df['log_return']
        
        # MACD与价格收益率交互
        if 'macd' in df.columns and 'log_return' in df.columns:
            df['macd_return_interaction'] = df['macd'] * df['log_return']
        
        # 宏观指标交互
        macro_cols = [col for col in df.columns if col.endswith('_close')]
        if 'log_return' in df.columns:
            for macro_col in macro_cols:
                macro_return = df[macro_col].pct_change()
                interaction_name = f'{macro_col}_interaction'
                df[interaction_name] = df['log_return'] * macro_return
        
        # ATR与价格变动交互
        if 'atr' in df.columns and 'log_return' in df.columns:
            df['atr_return_interaction'] = df['atr'] * df['log_return']
        
        # 布林带位置与RSI交互
        if 'bb_pct' in df.columns and 'rsi' in df.columns:
            df['bb_rsi_interaction'] = df['bb_pct'] * df['rsi']
        
        return df
    
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建目标变量
        
        Args:
            df: 输入DataFrame
            
        Returns:
            添加目标变量后的DataFrame
        """
        df = df.copy()
        horizon = self.config.get('prediction_horizon', 5)
        
        # 二分类目标：未来N天价格上涨为1，否则为0
        df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
        df['target'] = (df['close'].shift(-horizon) > df['close']).astype(int)
        
        return df
    
    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成所有特征
        
        Args:
            df: 输入DataFrame
            
        Returns:
            包含所有特征的DataFrame
        """
        logger.info("开始生成特征...")
        
        # 创建滞后特征
        df = self.create_lag_features(df)
        logger.info(f"滞后特征生成完成，当前列数: {len(df.columns)}")
        
        # 创建动量特征
        df = self.create_momentum_features(df)
        logger.info(f"动量特征生成完成，当前列数: {len(df.columns)}")
        
        # 创建交互特征
        df = self.create_interaction_features(df)
        logger.info(f"交互特征生成完成，当前列数: {len(df.columns)}")
        
        # 创建目标变量
        df = self.create_target(df)
        logger.info(f"目标变量生成完成")
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        获取特征列名（排除非特征列）
        
        Args:
            df: 输入DataFrame
            
        Returns:
            特征列名列表
        """
        # 排除目标变量和非特征列
        exclude_cols = ['target', 'future_return', 'dividends', 'stock_splits']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
    
    def prepare_data(self, df: pd.DataFrame, 
                     dropna: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备训练数据
        
        Args:
            df: 输入DataFrame
            dropna: 是否删除含NaN的行
            
        Returns:
            (特征DataFrame, 目标Series)
        """
        df = df.copy()
        
        # 删除目标变量为NaN的行
        df = df.dropna(subset=['target'])
        
        # 获取特征列
        feature_cols = self.get_feature_columns(df)
        
        X = df[feature_cols]
        y = df['target']
        
        if dropna:
            # 先将无限值转为NaN，再删除含NaN的行，保持X/y索引一致
            X = X.replace([np.inf, -np.inf], np.nan)
            valid_idx = X.dropna().index
            X = X.loc[valid_idx]
            y = y.loc[valid_idx]
        else:
            # 将无限值转NaN再填0，避免极值冲击
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(0)
        
        return X, y
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureEngineer':
        """
        拟合特征工程器（标准化和特征选择）
        
        Args:
            X: 特征DataFrame
            y: 目标Series
            
        Returns:
            self
        """
        logger.info("开始拟合特征工程器...")
        
        self.feature_names_ = X.columns.tolist()
        
        # 标准化（仅在训练集上fit）
        X_scaled = self.scaler.fit_transform(X)
        
        # 特征选择
        n_features = min(self.config.get('n_features', 50), X.shape[1])
        self.selector = SelectKBest(score_func=f_classif, k=n_features)
        self.selector.fit(X_scaled, y)
        
        # 获取选择的特征名
        selected_mask = self.selector.get_support()
        self.selected_features_ = [f for f, s in zip(self.feature_names_, selected_mask) if s]
        
        logger.info(f"特征选择完成，选择了 {len(self.selected_features_)} 个特征")
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        转换特征（应用标准化和特征选择）
        
        Args:
            X: 特征DataFrame
            
        Returns:
            转换后的特征数组
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer尚未拟合，请先调用fit方法")
        
        # 确保特征一致性
        X_aligned = self._align_features(X)

        # 清洗无限值/NaN，测试阶段不丢行，直接填0
        X_aligned = X_aligned.replace([np.inf, -np.inf], np.nan)
        X_aligned = X_aligned.fillna(0)
        
        # 标准化
        X_scaled = self.scaler.transform(X_aligned)
        
        # 特征选择
        X_selected = self.selector.transform(X_scaled)
        
        return X_selected
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """
        拟合并转换特征
        
        Args:
            X: 特征DataFrame
            y: 目标Series
            
        Returns:
            转换后的特征数组
        """
        self.fit(X, y)
        return self.transform(X)
    
    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        对齐特征列（确保与训练时的特征一致）
        
        Args:
            X: 输入特征DataFrame
            
        Returns:
            对齐后的DataFrame
        """
        X = X.copy()
        
        # 添加缺失的特征列（填0）
        for col in self.feature_names_:
            if col not in X.columns:
                X[col] = 0
                logger.warning(f"缺失特征 {col}，已填充0")
        
        # 只保留训练时的特征，按原顺序排列
        X = X[self.feature_names_]
        
        return X
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性（基于SelectKBest的得分）
        
        Returns:
            特征重要性DataFrame
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer尚未拟合")
        
        scores = self.selector.scores_
        importance_df = pd.DataFrame({
            'feature': self.feature_names_,
            'score': scores
        }).sort_values('score', ascending=False)
        
        return importance_df
    
    def get_selected_features(self) -> List[str]:
        """
        获取选择的特征列表
        
        Returns:
            选择的特征名列表
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer尚未拟合")
        
        return self.selected_features_


class FeatureMatrix:
    """特征矩阵类 - 用于管理完整的特征工程流程"""
    
    def __init__(self, config: dict = None):
        """
        初始化特征矩阵
        
        Args:
            config: 特征配置
        """
        self.config = config or FEATURE_CONFIG
        self.engineer = FeatureEngineer(config)
        self.raw_data = None
        self.featured_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_index = None
        self.test_index = None
    
    def fit_transform_pipeline(self, df: pd.DataFrame, 
                                train_size: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        完整的特征工程流程
        
        Args:
            df: 原始数据DataFrame
            train_size: 训练集比例
            
        Returns:
            (X_train, X_test, y_train, y_test)
        """
        self.raw_data = df.copy()
        
        # 生成所有特征
        self.featured_data = self.engineer.generate_all_features(df)
        
        # 准备数据
        X, y = self.engineer.prepare_data(self.featured_data)
        
        # 时间序列划分（不打乱顺序）
        split_idx = int(len(X) * train_size)
        
        X_train_raw = X.iloc[:split_idx]
        X_test_raw = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # 保存索引
        self.train_index = X_train_raw.index
        self.test_index = X_test_raw.index
        
        # 在训练集上拟合，转换训练集和测试集
        X_train = self.engineer.fit_transform(X_train_raw, y_train)
        X_test = self.engineer.transform(X_test_raw)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train.values
        self.y_test = y_test.values
        
        logger.info(f"特征工程完成:")
        logger.info(f"  训练集形状: {X_train.shape}")
        logger.info(f"  测试集形状: {X_test.shape}")
        
        return X_train, X_test, self.y_train, self.y_test
    
    def get_backtest_data(self) -> pd.DataFrame:
        """
        获取回测所需的数据
        
        Returns:
            包含OHLCV和特征的DataFrame
        """
        if self.featured_data is None:
            raise ValueError("请先调用fit_transform_pipeline")
        
        return self.featured_data
    
    def get_test_features_df(self) -> pd.DataFrame:
        """
        获取测试集特征DataFrame（带索引）
        
        Returns:
            测试集特征DataFrame
        """
        if self.X_test is None:
            raise ValueError("请先调用fit_transform_pipeline")
        
        return pd.DataFrame(
            self.X_test, 
            index=self.test_index,
            columns=self.engineer.selected_features_
        )


def main():
    """测试特征工程功能"""
    from data_collector import DataCollector
    
    # 收集数据
    collector = DataCollector()
    data = collector.get_data()
    
    # 特征工程
    feature_matrix = FeatureMatrix()
    X_train, X_test, y_train, y_test = feature_matrix.fit_transform_pipeline(data)
    
    print(f"\n训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")
    print(f"\n选择的特征:\n{feature_matrix.engineer.get_selected_features()}")
    print(f"\n特征重要性Top 10:\n{feature_matrix.engineer.get_feature_importance().head(10)}")


if __name__ == '__main__':
    main()
