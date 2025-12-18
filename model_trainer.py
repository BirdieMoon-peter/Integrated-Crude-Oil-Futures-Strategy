"""
模型训练模块 - 负责模型构建、训练、交叉验证及集成
原油期货多模型集成投资策略
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import xgboost as xgb
import logging
import warnings
import joblib
import os

from config import MODEL_CONFIG

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """模型训练器类"""
    
    def __init__(self, config: dict = None):
        """
        初始化模型训练器
        
        Args:
            config: 模型配置字典
        """
        self.config = config or MODEL_CONFIG
        self.models = {}
        self.is_trained = False
        self.cv_results = {}
        self.test_results = {}
        
        # 初始化模型
        self._init_models()
    
    def _init_models(self):
        """初始化各个模型"""
        # 随机森林
        rf_config = self.config.get('rf', {})
        self.models['rf'] = RandomForestClassifier(
            n_estimators=rf_config.get('n_estimators', 100),
            max_depth=rf_config.get('max_depth', 10),
            random_state=rf_config.get('random_state', 42),
            n_jobs=rf_config.get('n_jobs', -1)
        )
        
        # XGBoost
        xgb_config = self.config.get('xgb', {})
        self.models['xgb'] = xgb.XGBClassifier(
            n_estimators=xgb_config.get('n_estimators', 100),
            max_depth=xgb_config.get('max_depth', 6),
            learning_rate=xgb_config.get('learning_rate', 0.1),
            eval_metric=xgb_config.get('eval_metric', 'logloss'),
            random_state=xgb_config.get('random_state', 42),
            use_label_encoder=xgb_config.get('use_label_encoder', False)
        )
        
        # Bagging with Logistic Regression
        bagging_config = self.config.get('bagging', {})
        base_estimator = LogisticRegression(max_iter=1000, random_state=42)
        self.models['bagging'] = BaggingClassifier(
            estimator=base_estimator,
            n_estimators=bagging_config.get('n_estimators', 50),
            random_state=bagging_config.get('random_state', 42),
            n_jobs=bagging_config.get('n_jobs', -1)
        )
        
        logger.info("模型初始化完成")
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                       n_splits: int = None) -> Dict[str, Dict]:
        """
        使用时间序列交叉验证评估模型
        
        Args:
            X: 特征数组
            y: 目标数组
            n_splits: 交叉验证折数
            
        Returns:
            各模型的交叉验证结果
        """
        if n_splits is None:
            n_splits = self.config.get('cv_splits', 5)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        logger.info(f"开始 {n_splits} 折时间序列交叉验证...")
        
        for name, model in self.models.items():
            logger.info(f"正在验证 {name}...")
            
            scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
            f1_scores = cross_val_score(model, X, y, cv=tscv, scoring='f1')
            
            self.cv_results[name] = {
                'accuracy_mean': scores.mean(),
                'accuracy_std': scores.std(),
                'f1_mean': f1_scores.mean(),
                'f1_std': f1_scores.std(),
                'all_accuracy': scores.tolist(),
                'all_f1': f1_scores.tolist()
            }
            
            logger.info(f"  {name} - Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
            logger.info(f"  {name} - F1 Score: {f1_scores.mean():.4f} (+/- {f1_scores.std():.4f})")
        
        return self.cv_results
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> 'ModelTrainer':
        """
        训练所有模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            
        Returns:
            self
        """
        logger.info("开始训练模型...")
        
        for name, model in self.models.items():
            logger.info(f"训练 {name}...")
            model.fit(X_train, y_train)
            logger.info(f"  {name} 训练完成")
        
        self.is_trained = True
        logger.info("所有模型训练完成")
        
        return self
    
    def predict(self, X: np.ndarray, model_name: str = None) -> np.ndarray:
        """
        使用指定模型进行预测
        
        Args:
            X: 特征数组
            model_name: 模型名称，None则使用集成投票
            
        Returns:
            预测标签
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        if model_name is not None:
            if model_name not in self.models:
                raise ValueError(f"未知模型: {model_name}")
            return self.models[model_name].predict(X)
        
        # 集成投票预测
        return self.voting_predict(X)
    
    def predict_proba(self, X: np.ndarray, model_name: str = None) -> np.ndarray:
        """
        使用指定模型预测概率
        
        Args:
            X: 特征数组
            model_name: 模型名称，None则使用加权平均
            
        Returns:
            预测概率
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        if model_name is not None:
            if model_name not in self.models:
                raise ValueError(f"未知模型: {model_name}")
            return self.models[model_name].predict_proba(X)
        
        # 加权平均概率
        return self.voting_predict_proba(X)
    
    def voting_predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        加权投票预测
        
        Args:
            X: 特征数组
            threshold: 分类阈值
            
        Returns:
            预测标签
        """
        proba = self.voting_predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
    
    def voting_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        加权投票预测概率
        
        Args:
            X: 特征数组
            
        Returns:
            加权平均预测概率
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        weights = self.config.get('ensemble_weights', {
            'rf': 0.4,
            'xgb': 0.4,
            'bagging': 0.2
        })
        
        weighted_proba = np.zeros((X.shape[0], 2))
        total_weight = 0
        
        for name, model in self.models.items():
            weight = weights.get(name, 1.0 / len(self.models))
            proba = model.predict_proba(X)
            weighted_proba += weight * proba
            total_weight += weight
        
        weighted_proba /= total_weight
        
        return weighted_proba
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """
        在测试集上评估所有模型
        
        Args:
            X_test: 测试特征
            y_test: 测试标签
            
        Returns:
            各模型的评估结果
        """
        logger.info("开始评估模型...")
        
        # 评估各个单独模型
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            self.test_results[name] = self._calculate_metrics(y_test, y_pred, y_proba)
            logger.info(f"\n{name} 评估结果:")
            logger.info(f"  Accuracy: {self.test_results[name]['accuracy']:.4f}")
            logger.info(f"  F1 Score: {self.test_results[name]['f1']:.4f}")
            logger.info(f"  AUC-ROC: {self.test_results[name]['auc_roc']:.4f}")
        
        # 评估集成模型
        y_pred_ensemble = self.voting_predict(X_test)
        y_proba_ensemble = self.voting_predict_proba(X_test)[:, 1]
        
        self.test_results['ensemble'] = self._calculate_metrics(y_test, y_pred_ensemble, y_proba_ensemble)
        logger.info(f"\n集成模型 评估结果:")
        logger.info(f"  Accuracy: {self.test_results['ensemble']['accuracy']:.4f}")
        logger.info(f"  F1 Score: {self.test_results['ensemble']['f1']:.4f}")
        logger.info(f"  AUC-ROC: {self.test_results['ensemble']['auc_roc']:.4f}")
        
        return self.test_results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           y_proba: np.ndarray) -> Dict:
        """
        计算评估指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 预测概率
            
        Returns:
            评估指标字典
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_true, y_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
    
    def get_feature_importance(self, feature_names: List[str] = None) -> pd.DataFrame:
        """
        获取随机森林的特征重要性
        
        Args:
            feature_names: 特征名列表
            
        Returns:
            特征重要性DataFrame
        """
        if 'rf' not in self.models or not self.is_trained:
            raise ValueError("随机森林模型尚未训练")
        
        rf_model = self.models['rf']
        importance = rf_model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_models(self, directory: str = 'models'):
        """
        保存所有模型
        
        Args:
            directory: 保存目录
        """
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.models.items():
            filepath = os.path.join(directory, f'{name}_model.joblib')
            joblib.dump(model, filepath)
            logger.info(f"模型 {name} 已保存到 {filepath}")
        
        # 保存配置
        config_path = os.path.join(directory, 'model_config.joblib')
        joblib.dump(self.config, config_path)
        logger.info(f"配置已保存到 {config_path}")
    
    def load_models(self, directory: str = 'models'):
        """
        加载所有模型
        
        Args:
            directory: 模型目录
        """
        for name in self.models.keys():
            filepath = os.path.join(directory, f'{name}_model.joblib')
            if os.path.exists(filepath):
                self.models[name] = joblib.load(filepath)
                logger.info(f"模型 {name} 已从 {filepath} 加载")
        
        self.is_trained = True
    
    def get_results_summary(self) -> pd.DataFrame:
        """
        获取结果汇总
        
        Returns:
            结果汇总DataFrame
        """
        results = []
        
        for name, metrics in self.test_results.items():
            results.append({
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1'],
                'AUC-ROC': metrics['auc_roc']
            })
        
        return pd.DataFrame(results)


class EnsemblePredictor:
    """集成预测器 - 用于回测时的预测"""
    
    def __init__(self, trainer: ModelTrainer, feature_engineer):
        """
        初始化集成预测器
        
        Args:
            trainer: 训练好的ModelTrainer
            feature_engineer: 训练好的FeatureEngineer
        """
        self.trainer = trainer
        self.feature_engineer = feature_engineer
    
    def predict_series(self, df: pd.DataFrame) -> pd.Series:
        """
        对整个时间序列进行预测
        
        Args:
            df: 包含特征的DataFrame
            
        Returns:
            预测概率Series
        """
        # 获取特征列
        feature_cols = self.feature_engineer.get_feature_columns(df)
        X = df[feature_cols].copy()
        
        # 填充NaN
        X = X.fillna(0)
        
        # 对齐和转换特征
        X_transformed = self.feature_engineer.transform(X)
        
        # 预测概率
        proba = self.trainer.voting_predict_proba(X_transformed)
        
        # 返回上涨概率
        return pd.Series(proba[:, 1], index=df.index, name='forecast')


def main():
    """测试模型训练功能"""
    from data_collector import DataCollector
    from feature_engineering import FeatureMatrix
    
    # 收集数据
    collector = DataCollector()
    data = collector.get_data()
    
    # 特征工程
    feature_matrix = FeatureMatrix()
    X_train, X_test, y_train, y_test = feature_matrix.fit_transform_pipeline(data)
    
    # 模型训练
    trainer = ModelTrainer()
    
    # 交叉验证
    cv_results = trainer.cross_validate(X_train, y_train)
    
    # 训练模型
    trainer.train(X_train, y_train)
    
    # 评估模型
    test_results = trainer.evaluate(X_test, y_test)
    
    # 输出结果
    print("\n" + "="*50)
    print("模型性能汇总")
    print("="*50)
    print(trainer.get_results_summary())
    
    # 特征重要性
    importance = trainer.get_feature_importance(feature_matrix.engineer.selected_features_)
    print("\n特征重要性 Top 10:")
    print(importance.head(10))


if __name__ == '__main__':
    main()
