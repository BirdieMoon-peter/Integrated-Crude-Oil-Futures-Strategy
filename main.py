"""
主程序 - 管道调度器，整合全流程
原油期货多模型集成投资策略
"""

import os
import sys
import logging
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# 导入项目模块
from config import (
    DATA_CONFIG, FEATURE_CONFIG, MODEL_CONFIG, 
    STRATEGY_CONFIG, VIS_CONFIG
)
from data_collector import DataCollector
from feature_engineering import FeatureMatrix, FeatureEngineer
from model_trainer import ModelTrainer, EnsemblePredictor
from strategy_backtest import BacktestEngine, compute_forecasts, LongOnlyMLStrategy
from visualization import Visualizer


class TradingPipeline:
    """量化交易管道类"""
    
    def __init__(self, 
                 data_config: dict = None,
                 feature_config: dict = None,
                 model_config: dict = None,
                 strategy_config: dict = None,
                 vis_config: dict = None):
        """
        初始化交易管道
        
        Args:
            data_config: 数据配置
            feature_config: 特征配置
            model_config: 模型配置
            strategy_config: 策略配置
            vis_config: 可视化配置
        """
        self.data_config = data_config or DATA_CONFIG
        self.feature_config = feature_config or FEATURE_CONFIG
        self.model_config = model_config or MODEL_CONFIG
        self.strategy_config = strategy_config or STRATEGY_CONFIG
        self.vis_config = vis_config or VIS_CONFIG
        
        # 组件
        self.data_collector = None
        self.feature_matrix = None
        self.model_trainer = None
        self.backtest_engine = None
        self.visualizer = None
        
        # 数据
        self.raw_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # 结果
        self.cv_results = None
        self.test_results = None
        self.backtest_stats = None
        self.feature_importance = None
        
        # 创建输出目录
        os.makedirs(self.vis_config.get('output_dir', 'output'), exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
    
    def step1_collect_data(self) -> pd.DataFrame:
        """
        步骤1: 数据收集
        
        Returns:
            原始数据DataFrame
        """
        logger.info("="*60)
        logger.info("步骤1: 数据收集")
        logger.info("="*60)
        
        self.data_collector = DataCollector(self.data_config)
        self.raw_data = self.data_collector.get_data()
        
        # 保存原始数据
        self.data_collector.save_data('data/raw_data.csv')
        
        logger.info(f"数据收集完成:")
        logger.info(f"  数据形状: {self.raw_data.shape}")
        logger.info(f"  时间范围: {self.raw_data.index[0]} 至 {self.raw_data.index[-1]}")
        logger.info(f"  特征数量: {len(self.raw_data.columns)}")
        
        return self.raw_data
    
    def step2_feature_engineering(self) -> tuple:
        """
        步骤2: 特征工程
        
        Returns:
            (X_train, X_test, y_train, y_test)
        """
        logger.info("="*60)
        logger.info("步骤2: 特征工程")
        logger.info("="*60)
        
        if self.raw_data is None:
            self.step1_collect_data()
        
        self.feature_matrix = FeatureMatrix(self.feature_config)
        train_size = self.model_config.get('train_size', 0.8)
        
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.feature_matrix.fit_transform_pipeline(self.raw_data, train_size)
        
        logger.info(f"特征工程完成:")
        logger.info(f"  训练集形状: {self.X_train.shape}")
        logger.info(f"  测试集形状: {self.X_test.shape}")
        logger.info(f"  选择的特征数: {len(self.feature_matrix.engineer.selected_features_)}")
        
        # 输出选择的特征
        logger.info(f"  选择的特征: {self.feature_matrix.engineer.selected_features_[:10]}...")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def step3_model_training(self) -> dict:
        """
        步骤3: 模型训练与评估
        
        Returns:
            测试集评估结果
        """
        logger.info("="*60)
        logger.info("步骤3: 模型训练与评估")
        logger.info("="*60)
        
        if self.X_train is None:
            self.step2_feature_engineering()
        
        self.model_trainer = ModelTrainer(self.model_config)
        
        # 交叉验证
        logger.info("执行交叉验证...")
        self.cv_results = self.model_trainer.cross_validate(
            self.X_train, self.y_train,
            n_splits=self.model_config.get('cv_splits', 5)
        )
        
        # 训练模型
        logger.info("训练模型...")
        self.model_trainer.train(self.X_train, self.y_train)
        
        # 评估模型
        logger.info("评估模型...")
        self.test_results = self.model_trainer.evaluate(self.X_test, self.y_test)
        
        # 获取特征重要性
        self.feature_importance = self.model_trainer.get_feature_importance(
            self.feature_matrix.engineer.selected_features_
        )
        
        # 保存模型
        self.model_trainer.save_models('models')
        
        # 打印结果摘要
        logger.info("\n模型性能汇总:")
        print(self.model_trainer.get_results_summary())
        
        return self.test_results
    
    def step4_backtest(self, long_only: bool = True) -> dict:
        """
        步骤4: 策略回测
        
        Args:
            long_only: 是否仅做多
            
        Returns:
            回测统计结果
        """
        logger.info("="*60)
        logger.info("步骤4: 策略回测")
        logger.info("="*60)
        
        if self.model_trainer is None or not self.model_trainer.is_trained:
            self.step3_model_training()
        
        self.backtest_engine = BacktestEngine(self.strategy_config)
        
        # 获取测试集数据
        test_df = self.feature_matrix.featured_data.loc[self.feature_matrix.test_index]
        
        # 计算预测概率
        logger.info("计算预测概率...")
        forecast = compute_forecasts(test_df, self.model_trainer, self.feature_matrix.engineer)
        
        # 运行回测
        logger.info("运行回测...")
        self.backtest_stats = self.backtest_engine.run_backtest(
            test_df, 
            forecast,
            long_only=long_only
        )
        
        # 打印回测摘要
        self.backtest_engine.print_summary()
        
        return self.backtest_stats
    
    def step5_visualization(self, show_plots: bool = True):
        """
        步骤5: 可视化结果
        
        Args:
            show_plots: 是否显示图表
        """
        logger.info("="*60)
        logger.info("步骤5: 生成可视化报告")
        logger.info("="*60)
        
        self.visualizer = Visualizer(self.vis_config)
        
        # 获取预测结果
        y_pred = self.model_trainer.voting_predict(self.X_test)
        y_proba = self.model_trainer.voting_predict_proba(self.X_test)[:, 1]
        
        # 获取交易记录和权益曲线
        trades = None
        equity_curve = None
        
        if self.backtest_engine is not None:
            trades = self.backtest_engine.get_trades()
            equity_curve = self.backtest_engine.get_equity_curve()
        
        # 生成所有图表
        self.visualizer.generate_report(
            model_results=self.test_results,
            feature_importance=self.feature_importance,
            backtest_stats=self.backtest_stats or {},
            equity_curve=equity_curve,
            trades=trades,
            y_true=self.y_test,
            y_pred=y_pred,
            y_proba=y_proba
        )
        
        # 生成价格信号图（如果有回测数据）
        if self.backtest_engine is not None:
            test_df = self.feature_matrix.featured_data.loc[self.feature_matrix.test_index]
            forecast = compute_forecasts(test_df, self.model_trainer, self.feature_matrix.engineer)
            self.visualizer.plot_price_with_signals(
                test_df, forecast,
                threshold_buy=self.strategy_config.get('threshold_buy', 0.55),
                threshold_sell=self.strategy_config.get('threshold_sell', 0.45)
            )
        
        logger.info(f"所有图表已保存到 {self.vis_config.get('output_dir', 'output')} 目录")
        
        if show_plots:
            import matplotlib.pyplot as plt
            plt.show()
    
    def run_full_pipeline(self, show_plots: bool = True) -> dict:
        """
        运行完整管道
        
        Args:
            show_plots: 是否显示图表
            
        Returns:
            包含所有结果的字典
        """
        start_time = datetime.now()
        logger.info("="*60)
        logger.info("开始运行完整量化交易管道")
        logger.info(f"开始时间: {start_time}")
        logger.info("="*60)
        
        try:
            # 步骤1: 数据收集
            self.step1_collect_data()
            
            # 步骤2: 特征工程
            self.step2_feature_engineering()
            
            # 步骤3: 模型训练
            self.step3_model_training()
            
            # 步骤4: 回测
            self.step4_backtest(long_only=True)
            
            # 步骤5: 可视化
            self.step5_visualization(show_plots=show_plots)
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info("="*60)
            logger.info("管道运行完成!")
            logger.info(f"结束时间: {end_time}")
            logger.info(f"总耗时: {duration}")
            logger.info("="*60)
            
            # 返回结果汇总
            return {
                'cv_results': self.cv_results,
                'test_results': self.test_results,
                'backtest_stats': self.backtest_stats,
                'feature_importance': self.feature_importance,
                'selected_features': self.feature_matrix.engineer.selected_features_,
                'duration': str(duration)
            }
            
        except Exception as e:
            logger.error(f"管道运行出错: {e}")
            raise
    
    def quick_test(self):
        """
        快速测试 - 使用较短时间范围
        """
        # 修改配置为较短时间范围
        self.data_config['start_date'] = '2020-01-01'
        self.data_config['end_date'] = '2024-12-31'
        
        logger.info("运行快速测试模式...")
        return self.run_full_pipeline(show_plots=False)


def main():
    """主函数"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║       原油期货多模型集成投资策略 - 量化交易系统               ║
    ║       WTI Crude Oil Futures ML Trading System                ║
    ║                                                              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  模型: Random Forest + XGBoost + Bagging (Logistic)         ║
    ║  数据: Yahoo Finance (CL=F + 宏观指标)                       ║
    ║  框架: backtesting.py                                        ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 创建管道并运行
    pipeline = TradingPipeline()
    
    # 运行完整管道
    results = pipeline.run_full_pipeline(show_plots=True)
    
    # 输出最终结果
    print("\n" + "="*60)
    print("最终结果汇总")
    print("="*60)
    
    if results['backtest_stats']:
        print(f"\n【回测结果】")
        print(f"  总收益率: {results['backtest_stats'].get('total_return', 0):.2f}%")
        print(f"  年化收益率: {results['backtest_stats'].get('annual_return', 0) or 0:.2f}%")
        print(f"  夏普比率: {results['backtest_stats'].get('sharpe_ratio', 0) or 0:.4f}")
        print(f"  最大回撤: {results['backtest_stats'].get('max_drawdown', 0):.2f}%")
        print(f"  胜率: {results['backtest_stats'].get('win_rate', 0):.2f}%")
    
    if results['test_results']:
        print(f"\n【集成模型性能】")
        ensemble = results['test_results'].get('ensemble', {})
        print(f"  准确率: {ensemble.get('accuracy', 0):.4f}")
        print(f"  F1分数: {ensemble.get('f1', 0):.4f}")
        print(f"  AUC-ROC: {ensemble.get('auc_roc', 0):.4f}")
    
    print(f"\n【运行信息】")
    print(f"  总耗时: {results['duration']}")
    print(f"  选择特征数: {len(results['selected_features'])}")
    
    print("="*60)
    
    return results


if __name__ == '__main__':
    main()
