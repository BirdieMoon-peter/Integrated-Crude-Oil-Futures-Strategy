"""
推理接口模块 - 用于模拟盘的实时预测
原油期货多模型集成投资策略

提供 Predictor 类，支持：
1. 单日/最近N日数据输入
2. 维护历史数据缓冲区（解决rolling特征需求）
3. 特征工程处理
4. 输出预测信号
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
import joblib
import logging

from config import FEATURE_CONFIG, MODEL_CONFIG
from feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class Predictor:
    """
    实时预测器
    
    用于模拟盘环境，能够接受增量数据并输出预测信号。
    内部维护历史数据缓冲区以支持rolling特征计算。
    """
    
    def __init__(self, 
                 models_dir: str = 'models',
                 buffer_size: int = 100,
                 feature_config: dict = None):
        """
        初始化预测器
        
        Args:
            models_dir: 模型文件目录
            buffer_size: 历史数据缓冲区大小（天数）
            feature_config: 特征工程配置
        """
        self.models_dir = models_dir
        self.buffer_size = buffer_size
        self.feature_config = feature_config or FEATURE_CONFIG
        
        # 模型组件
        self.models = {}
        self.scaler = None
        self.selector = None
        self.feature_names = None
        self.selected_features = None
        self.ensemble_weights = MODEL_CONFIG.get('ensemble_weights', {
            'rf': 0.4, 'xgb': 0.4, 'bagging': 0.2
        })
        
        # 历史数据缓冲区
        self.data_buffer: Optional[pd.DataFrame] = None
        
        # 特征工程器（用于生成特征，不用于fit）
        self.feature_engineer = FeatureEngineer(self.feature_config)
        
        # 加载模型
        self._load_models()
        self._load_feature_params()
        
        self.is_ready = False
    
    def _load_models(self):
        """加载训练好的模型"""
        model_names = ['rf', 'xgb', 'bagging']
        
        for name in model_names:
            filepath = os.path.join(self.models_dir, f'{name}_model.joblib')
            if os.path.exists(filepath):
                self.models[name] = joblib.load(filepath)
                logger.info(f"模型 {name} 已加载")
            else:
                logger.warning(f"模型文件不存在: {filepath}")
        
        if not self.models:
            raise FileNotFoundError("未找到任何模型文件，请先运行 main.py 训练模型")
    
    def _load_feature_params(self):
        """加载特征工程参数（scaler, selector, feature_names）"""
        # 尝试加载保存的特征工程参数
        scaler_path = os.path.join(self.models_dir, 'scaler.joblib')
        selector_path = os.path.join(self.models_dir, 'selector.joblib')
        feature_names_path = os.path.join(self.models_dir, 'feature_names.joblib')
        selected_features_path = os.path.join(self.models_dir, 'selected_features.joblib')
        
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info("Scaler 已加载")
        else:
            logger.warning(f"Scaler文件不存在: {scaler_path}")
            
        if os.path.exists(selector_path):
            self.selector = joblib.load(selector_path)
            logger.info("Selector 已加载")
        else:
            logger.warning(f"Selector文件不存在: {selector_path}")
            
        if os.path.exists(feature_names_path):
            self.feature_names = joblib.load(feature_names_path)
            logger.info(f"Feature names 已加载，共 {len(self.feature_names)} 个特征")
        else:
            logger.warning(f"Feature names文件不存在: {feature_names_path}")
            
        if os.path.exists(selected_features_path):
            self.selected_features = joblib.load(selected_features_path)
            logger.info(f"Selected features 已加载，共 {len(self.selected_features)} 个特征")
    
    def initialize_buffer(self, historical_data: pd.DataFrame):
        """
        初始化历史数据缓冲区
        
        Args:
            historical_data: 历史数据DataFrame，需包含 OHLCV 和技术指标
        """
        # 保留最近buffer_size天的数据
        if len(historical_data) > self.buffer_size:
            self.data_buffer = historical_data.tail(self.buffer_size).copy()
        else:
            self.data_buffer = historical_data.copy()
        
        self.is_ready = True
        logger.info(f"缓冲区初始化完成，当前数据量: {len(self.data_buffer)}")
    
    def append_data(self, new_data: pd.DataFrame):
        """
        追加新数据到缓冲区
        
        Args:
            new_data: 新的一天或多天数据
        """
        if self.data_buffer is None:
            self.data_buffer = new_data.copy()
        else:
            self.data_buffer = pd.concat([self.data_buffer, new_data])
        
        # 保持缓冲区大小不超过限制
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer = self.data_buffer.tail(self.buffer_size)
        
        self.is_ready = True
    
    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        为数据生成特征
        
        Args:
            df: 原始数据
            
        Returns:
            带特征的DataFrame
        """
        return self.feature_engineer.generate_all_features(df)
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        准备用于预测的特征
        
        Args:
            df: 带特征的DataFrame
            
        Returns:
            处理后的特征数组
        """
        # 获取特征列
        feature_cols = self.feature_engineer.get_feature_columns(df)
        X = df[feature_cols].copy()
        
        # 对齐特征（确保与训练时一致）
        if self.feature_names is not None:
            # 添加缺失的特征列
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0
            # 只保留训练时的特征
            X = X[self.feature_names]
        
        # 清洗无限值/NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # 标准化
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # 特征选择
        if self.selector is not None:
            X_selected = self.selector.transform(X_scaled)
        else:
            X_selected = X_scaled
        
        return X_selected
    
    def predict(self, as_of_date: str = None) -> Dict:
        """
        生成预测信号
        
        Args:
            as_of_date: 预测日期（默认为缓冲区最后一天）
            
        Returns:
            预测结果字典，包含:
            - date: 预测日期
            - probability: 上涨概率
            - signal: 交易信号 (1: 买入, 0: 持有, -1: 卖出)
            - individual_proba: 各模型的预测概率
        """
        if not self.is_ready:
            raise ValueError("预测器未就绪，请先调用 initialize_buffer() 初始化")
        
        if self.data_buffer is None or len(self.data_buffer) == 0:
            raise ValueError("缓冲区为空，无法预测")
        
        # 生成特征
        featured_data = self._generate_features(self.data_buffer)
        
        # 只取最后一行进行预测
        last_row = featured_data.tail(1)
        
        # 准备特征
        X = self._prepare_features(last_row)
        
        # 各模型预测
        individual_proba = {}
        weighted_proba = np.zeros(2)
        total_weight = 0
        
        for name, model in self.models.items():
            proba = model.predict_proba(X)[0]
            individual_proba[name] = proba[1]  # 上涨概率
            
            weight = self.ensemble_weights.get(name, 1.0 / len(self.models))
            weighted_proba += weight * proba
            total_weight += weight
        
        weighted_proba /= total_weight
        up_probability = weighted_proba[1]
        
        # 生成信号
        threshold_buy = 0.55
        threshold_sell = 0.45
        
        if up_probability > threshold_buy:
            signal = 1  # 买入
        elif up_probability < threshold_sell:
            signal = -1  # 卖出
        else:
            signal = 0  # 持有
        
        # 获取日期
        if as_of_date is None:
            as_of_date = str(last_row.index[0].date()) if hasattr(last_row.index[0], 'date') else str(last_row.index[0])
        
        return {
            'date': as_of_date,
            'probability': float(up_probability),
            'signal': int(signal),
            'signal_text': {1: '买入', 0: '持有', -1: '卖出'}[signal],
            'individual_proba': individual_proba,
            'close_price': float(last_row['close'].iloc[0]) if 'close' in last_row.columns else None
        }
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        批量预测（用于回测对比）
        
        Args:
            df: 包含历史数据的DataFrame
            
        Returns:
            包含预测结果的DataFrame
        """
        # 生成特征
        featured_data = self._generate_features(df)
        
        # 准备特征
        feature_cols = self.feature_engineer.get_feature_columns(featured_data)
        X = featured_data[feature_cols].copy()
        
        # 对齐特征
        if self.feature_names is not None:
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_names]
        
        # 清洗
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # 标准化
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # 特征选择
        if self.selector is not None:
            X_selected = self.selector.transform(X_scaled)
        else:
            X_selected = X_scaled
        
        # 加权预测
        weighted_proba = np.zeros((len(X_selected), 2))
        total_weight = 0
        
        for name, model in self.models.items():
            proba = model.predict_proba(X_selected)
            weight = self.ensemble_weights.get(name, 1.0 / len(self.models))
            weighted_proba += weight * proba
            total_weight += weight
        
        weighted_proba /= total_weight
        
        result = pd.DataFrame({
            'probability': weighted_proba[:, 1],
            'signal': np.where(weighted_proba[:, 1] > 0.55, 1,
                              np.where(weighted_proba[:, 1] < 0.45, -1, 0))
        }, index=featured_data.index)
        
        return result


def save_feature_params(feature_engineer: FeatureEngineer, models_dir: str = 'models'):
    """
    保存特征工程参数，供 Predictor 加载
    
    Args:
        feature_engineer: 已拟合的 FeatureEngineer 实例
        models_dir: 保存目录
    """
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(feature_engineer.scaler, os.path.join(models_dir, 'scaler.joblib'))
    joblib.dump(feature_engineer.selector, os.path.join(models_dir, 'selector.joblib'))
    joblib.dump(feature_engineer.feature_names_, os.path.join(models_dir, 'feature_names.joblib'))
    joblib.dump(feature_engineer.selected_features_, os.path.join(models_dir, 'selected_features.joblib'))
    
    logger.info(f"特征工程参数已保存到 {models_dir}")


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    from data_collector import DataCollector
    
    # 加载数据
    collector = DataCollector()
    data = collector.get_data()
    
    # 创建预测器
    predictor = Predictor()
    
    # 初始化缓冲区
    predictor.initialize_buffer(data)
    
    # 预测
    result = predictor.predict()
    print(f"\n预测结果:")
    print(f"  日期: {result['date']}")
    print(f"  上涨概率: {result['probability']:.4f}")
    print(f"  信号: {result['signal_text']}")
    print(f"  各模型概率: {result['individual_proba']}")
