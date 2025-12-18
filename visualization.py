"""
可视化模块 - 生成性能报表与图表
原油期货多模型集成投资策略
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional
import os
import logging
import warnings

from config import VIS_CONFIG

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Visualizer:
    """可视化类"""
    
    def __init__(self, config: dict = None):
        """
        初始化可视化器
        
        Args:
            config: 可视化配置
        """
        self.config = config or VIS_CONFIG
        self.output_dir = self.config.get('output_dir', 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置默认样式
        try:
            plt.style.use(self.config.get('style', 'seaborn-v0_8-whitegrid'))
        except:
            plt.style.use('default')
    
    def plot_model_comparison(self, results: Dict[str, Dict], 
                              save: bool = True,
                              filename: str = 'model_comparison.png') -> plt.Figure:
        """
        绘制模型性能对比图
        
        Args:
            results: 模型评估结果字典
            save: 是否保存图片
            filename: 文件名
            
        Returns:
            Figure对象
        """
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
        
        fig, axes = plt.subplots(1, 2, figsize=self.config.get('figure_size', (14, 6)))
        
        # 柱状图 - 主要指标
        x = np.arange(len(models))
        width = 0.15
        
        for i, (metric, label) in enumerate(zip(metrics[:4], metric_labels[:4])):
            values = [results[m].get(metric, 0) for m in models]
            axes[0].bar(x + i * width, values, width, label=label)
        
        axes[0].set_xlabel('Models')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Model Performance Comparison')
        axes[0].set_xticks(x + width * 1.5)
        axes[0].set_xticklabels(models, rotation=45)
        axes[0].legend(loc='lower right')
        axes[0].set_ylim(0, 1)
        axes[0].grid(axis='y', alpha=0.3)
        
        # AUC-ROC单独展示
        auc_values = [results[m].get('auc_roc', 0) for m in models]
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        bars = axes[1].bar(models, auc_values, color=colors)
        axes[1].set_xlabel('Models')
        axes[1].set_ylabel('AUC-ROC Score')
        axes[1].set_title('AUC-ROC Comparison')
        axes[1].set_ylim(0, 1)
        axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
        axes[1].legend()
        
        # 在柱上显示数值
        for bar, val in zip(bars, auc_values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=self.config.get('dpi', 100), bbox_inches='tight')
            logger.info(f"模型对比图已保存到 {filepath}")
        
        return fig
    
    def plot_feature_importance(self, importance_df: pd.DataFrame,
                                 top_n: int = None,
                                 save: bool = True,
                                 filename: str = 'feature_importance.png') -> plt.Figure:
        """
        绘制特征重要性图
        
        Args:
            importance_df: 特征重要性DataFrame
            top_n: 显示前N个特征
            save: 是否保存
            filename: 文件名
            
        Returns:
            Figure对象
        """
        if top_n is None:
            top_n = self.config.get('top_features', 15)
        
        df = importance_df.head(top_n).copy()
        df = df.iloc[::-1]  # 反转顺序，使最重要的在顶部
        
        fig, ax = plt.subplots(figsize=self.config.get('figure_size', (12, 8)))
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df)))
        bars = ax.barh(df['feature'], df['importance'], color=colors)
        
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Feature')
        ax.set_title(f'Top {top_n} Feature Importance (Random Forest)')
        ax.grid(axis='x', alpha=0.3)
        
        # 在柱上显示数值
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{width:.4f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=self.config.get('dpi', 100), bbox_inches='tight')
            logger.info(f"特征重要性图已保存到 {filepath}")
        
        return fig
    
    def plot_equity_curve(self, equity_curve: pd.Series,
                          benchmark: pd.Series = None,
                          save: bool = True,
                          filename: str = 'equity_curve.png') -> plt.Figure:
        """
        绘制权益曲线
        
        Args:
            equity_curve: 权益曲线Series
            benchmark: 基准曲线（可选）
            save: 是否保存
            filename: 文件名
            
        Returns:
            Figure对象
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), 
                                  gridspec_kw={'height_ratios': [3, 1]})
        
        # 权益曲线
        axes[0].plot(equity_curve.index, equity_curve.values, 
                     label='Strategy', color='blue', linewidth=2)
        
        if benchmark is not None:
            axes[0].plot(benchmark.index, benchmark.values,
                        label='Buy & Hold', color='gray', linewidth=1.5, alpha=0.7)
        
        axes[0].set_title('Equity Curve', fontsize=14)
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
        
        # 回撤曲线
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max * 100
        
        axes[1].fill_between(drawdown.index, drawdown.values, 0, 
                             color='red', alpha=0.3, label='Drawdown')
        axes[1].plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        axes[1].set_title('Drawdown', fontsize=12)
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].set_xlabel('Date')
        axes[1].legend(loc='lower left')
        axes[1].grid(True, alpha=0.3)
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=self.config.get('dpi', 100), bbox_inches='tight')
            logger.info(f"权益曲线图已保存到 {filepath}")
        
        return fig
    
    def plot_returns_distribution(self, trades: pd.DataFrame,
                                   save: bool = True,
                                   filename: str = 'returns_distribution.png') -> plt.Figure:
        """
        绘制收益分布图
        
        Args:
            trades: 交易记录DataFrame
            save: 是否保存
            filename: 文件名
            
        Returns:
            Figure对象
        """
        fig, axes = plt.subplots(1, 2, figsize=self.config.get('figure_size', (14, 6)))
        
        if 'ReturnPct' in trades.columns:
            returns = trades['ReturnPct'].values
        elif 'PnL' in trades.columns:
            returns = trades['PnL'].values
        else:
            logger.warning("交易记录中没有找到收益列")
            return fig
        
        # 收益分布直方图
        colors = ['green' if r > 0 else 'red' for r in returns]
        axes[0].hist(returns, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
        axes[0].axvline(x=0, color='black', linestyle='--', linewidth=2)
        axes[0].axvline(x=np.mean(returns), color='red', linestyle='-', 
                        linewidth=2, label=f'Mean: {np.mean(returns):.2f}%')
        axes[0].set_xlabel('Return (%)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Trade Returns Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 累计收益
        cumulative_returns = (1 + np.array(returns)/100).cumprod() - 1
        trade_nums = range(1, len(returns) + 1)
        
        axes[1].plot(trade_nums, cumulative_returns * 100, 
                     color='blue', linewidth=2, marker='o', markersize=3)
        axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[1].fill_between(trade_nums, cumulative_returns * 100, 0,
                            where=cumulative_returns >= 0, color='green', alpha=0.3)
        axes[1].fill_between(trade_nums, cumulative_returns * 100, 0,
                            where=cumulative_returns < 0, color='red', alpha=0.3)
        axes[1].set_xlabel('Trade Number')
        axes[1].set_ylabel('Cumulative Return (%)')
        axes[1].set_title('Cumulative Returns by Trade')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=self.config.get('dpi', 100), bbox_inches='tight')
            logger.info(f"收益分布图已保存到 {filepath}")
        
        return fig
    
    def plot_prediction_analysis(self, y_true: np.ndarray, 
                                  y_pred: np.ndarray,
                                  y_proba: np.ndarray,
                                  save: bool = True,
                                  filename: str = 'prediction_analysis.png') -> plt.Figure:
        """
        绘制预测分析图
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 预测概率
            save: 是否保存
            filename: 文件名
            
        Returns:
            Figure对象
        """
        from sklearn.metrics import confusion_matrix, roc_curve, auc
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        im = axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        axes[0].set_xticks([0, 1])
        axes[0].set_yticks([0, 1])
        axes[0].set_xticklabels(['Down', 'Up'])
        axes[0].set_yticklabels(['Down', 'Up'])
        
        # 在格子中显示数值
        for i in range(2):
            for j in range(2):
                axes[0].text(j, i, str(cm[i, j]), ha='center', va='center',
                            color='white' if cm[i, j] > cm.max()/2 else 'black',
                            fontsize=14)
        
        # ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                     label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend(loc='lower right')
        axes[1].grid(True, alpha=0.3)
        
        # 预测概率分布
        axes[2].hist(y_proba[y_true == 0], bins=30, alpha=0.5, 
                     label='Actual Down', color='red')
        axes[2].hist(y_proba[y_true == 1], bins=30, alpha=0.5, 
                     label='Actual Up', color='green')
        axes[2].axvline(x=0.5, color='black', linestyle='--', 
                        linewidth=2, label='Threshold (0.5)')
        axes[2].set_xlabel('Predicted Probability')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Prediction Probability Distribution')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=self.config.get('dpi', 100), bbox_inches='tight')
            logger.info(f"预测分析图已保存到 {filepath}")
        
        return fig
    
    def plot_backtest_summary(self, stats: Dict,
                               save: bool = True,
                               filename: str = 'backtest_summary.png') -> plt.Figure:
        """
        绘制回测摘要图
        
        Args:
            stats: 回测统计字典
            save: 是否保存
            filename: 文件名
            
        Returns:
            Figure对象
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 收益指标
        returns_metrics = {
            'Total Return': stats.get('total_return', 0),
            'Annual Return': stats.get('annual_return', 0) or 0,
            'Buy & Hold': stats.get('buy_hold_return', 0),
        }
        
        colors = ['green' if v > 0 else 'red' for v in returns_metrics.values()]
        axes[0, 0].bar(returns_metrics.keys(), returns_metrics.values(), color=colors)
        axes[0, 0].set_ylabel('Return (%)')
        axes[0, 0].set_title('Return Metrics')
        axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        for i, (k, v) in enumerate(returns_metrics.items()):
            axes[0, 0].text(i, v + (2 if v > 0 else -5), f'{v:.1f}%', 
                           ha='center', fontsize=10)
        
        # 风险指标
        risk_metrics = {
            'Max Drawdown': abs(stats.get('max_drawdown', 0)),
            'Sharpe Ratio': stats.get('sharpe_ratio', 0) or 0,
            'Win Rate': stats.get('win_rate', 0),
        }
        
        ax2 = axes[0, 1]
        x = range(len(risk_metrics))
        bars = ax2.bar(x, risk_metrics.values(), color=['red', 'blue', 'green'])
        ax2.set_xticks(x)
        ax2.set_xticklabels(risk_metrics.keys())
        ax2.set_title('Risk & Performance Metrics')
        for bar, (k, v) in zip(bars, risk_metrics.items()):
            label = f'{v:.2f}' if k == 'Sharpe Ratio' else f'{v:.1f}%'
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    label, ha='center', fontsize=10)
        
        # 交易统计
        trade_stats = {
            'Total Trades': stats.get('total_trades', 0),
            'Best Trade (%)': stats.get('best_trade', 0),
            'Worst Trade (%)': stats.get('worst_trade', 0),
            'Avg Trade (%)': stats.get('avg_trade', 0) or 0,
        }
        
        ax3 = axes[1, 0]
        y_pos = range(len(trade_stats))
        values = list(trade_stats.values())
        colors = ['blue', 'green', 'red', 'orange']
        ax3.barh(y_pos, values, color=colors)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(trade_stats.keys())
        ax3.set_title('Trade Statistics')
        for i, v in enumerate(values):
            ax3.text(v + (0.5 if v >= 0 else -2), i, f'{v:.1f}', va='center', fontsize=10)
        
        # 资金状况
        capital_stats = {
            'Initial': stats.get('initial_capital', 100000),
            'Final': stats.get('final_equity', 0),
            'Peak': stats.get('equity_peak', 0),
        }
        
        ax4 = axes[1, 1]
        bars = ax4.bar(capital_stats.keys(), 
                       [v/1000 for v in capital_stats.values()],  # 转换为千元
                       color=['gray', 'green', 'blue'])
        ax4.set_ylabel('Capital ($K)')
        ax4.set_title('Capital Status')
        for bar, (k, v) in zip(bars, capital_stats.items()):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'${v:,.0f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=self.config.get('dpi', 100), bbox_inches='tight')
            logger.info(f"回测摘要图已保存到 {filepath}")
        
        return fig
    
    def plot_price_with_signals(self, df: pd.DataFrame,
                                 forecast: pd.Series,
                                 threshold_buy: float = 0.55,
                                 threshold_sell: float = 0.45,
                                 save: bool = True,
                                 filename: str = 'price_signals.png') -> plt.Figure:
        """
        绘制价格走势与交易信号
        
        Args:
            df: OHLCV数据
            forecast: 预测概率
            threshold_buy: 买入阈值
            threshold_sell: 卖出阈值
            save: 是否保存
            filename: 文件名
            
        Returns:
            Figure对象
        """
        fig, axes = plt.subplots(3, 1, figsize=(16, 12),
                                  gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # 价格走势
        close_col = 'close' if 'close' in df.columns else 'Close'
        axes[0].plot(df.index, df[close_col], label='Close Price', 
                     color='black', linewidth=1)
        
        # 标记信号
        buy_signals = forecast[forecast > threshold_buy].index
        sell_signals = forecast[forecast < threshold_sell].index
        
        axes[0].scatter(buy_signals, df.loc[buy_signals, close_col],
                       marker='^', color='green', s=50, label='Buy Signal', alpha=0.7)
        axes[0].scatter(sell_signals, df.loc[sell_signals, close_col],
                       marker='v', color='red', s=50, label='Sell Signal', alpha=0.7)
        
        axes[0].set_title('Price Chart with Trading Signals', fontsize=14)
        axes[0].set_ylabel('Price ($)')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # 预测概率
        axes[1].plot(forecast.index, forecast.values, 
                     color='blue', linewidth=1, label='Forecast Probability')
        axes[1].axhline(y=threshold_buy, color='green', linestyle='--', 
                        label=f'Buy Threshold ({threshold_buy})')
        axes[1].axhline(y=threshold_sell, color='red', linestyle='--',
                        label=f'Sell Threshold ({threshold_sell})')
        axes[1].fill_between(forecast.index, threshold_sell, threshold_buy,
                            color='gray', alpha=0.2, label='Neutral Zone')
        axes[1].set_ylabel('Probability')
        axes[1].set_title('Prediction Probability', fontsize=12)
        axes[1].legend(loc='upper left')
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)
        
        # 成交量
        volume_col = 'volume' if 'volume' in df.columns else 'Volume'
        if volume_col in df.columns:
            axes[2].bar(df.index, df[volume_col], color='steelblue', alpha=0.7)
            axes[2].set_ylabel('Volume')
            axes[2].set_title('Trading Volume', fontsize=12)
            axes[2].grid(True, alpha=0.3)
        
        axes[2].set_xlabel('Date')
        
        # 格式化x轴日期
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=self.config.get('dpi', 100), bbox_inches='tight')
            logger.info(f"价格信号图已保存到 {filepath}")
        
        return fig
    
    def generate_report(self, model_results: Dict,
                        feature_importance: pd.DataFrame,
                        backtest_stats: Dict,
                        equity_curve: pd.Series = None,
                        trades: pd.DataFrame = None,
                        y_true: np.ndarray = None,
                        y_pred: np.ndarray = None,
                        y_proba: np.ndarray = None):
        """
        生成完整报告
        
        Args:
            model_results: 模型评估结果
            feature_importance: 特征重要性
            backtest_stats: 回测统计
            equity_curve: 权益曲线
            trades: 交易记录
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 预测概率
        """
        logger.info("开始生成报告...")
        
        # 模型对比图
        if model_results:
            self.plot_model_comparison(model_results)
        
        # 特征重要性图
        if feature_importance is not None and not feature_importance.empty:
            self.plot_feature_importance(feature_importance)
        
        # 回测摘要图
        if backtest_stats:
            self.plot_backtest_summary(backtest_stats)
        
        # 权益曲线
        if equity_curve is not None and len(equity_curve) > 0:
            self.plot_equity_curve(equity_curve)
        
        # 收益分布
        if trades is not None and len(trades) > 0:
            self.plot_returns_distribution(trades)
        
        # 预测分析
        if y_true is not None and y_pred is not None and y_proba is not None:
            self.plot_prediction_analysis(y_true, y_pred, y_proba)
        
        logger.info(f"报告生成完成，所有图表已保存到 {self.output_dir} 目录")
        
        plt.show()


def main():
    """测试可视化功能"""
    import numpy as np
    
    # 创建示例数据
    np.random.seed(42)
    
    # 模型结果示例
    model_results = {
        'rf': {'accuracy': 0.65, 'precision': 0.63, 'recall': 0.68, 'f1': 0.65, 'auc_roc': 0.70},
        'xgb': {'accuracy': 0.64, 'precision': 0.62, 'recall': 0.67, 'f1': 0.64, 'auc_roc': 0.69},
        'bagging': {'accuracy': 0.62, 'precision': 0.60, 'recall': 0.65, 'f1': 0.62, 'auc_roc': 0.67},
        'ensemble': {'accuracy': 0.66, 'precision': 0.64, 'recall': 0.69, 'f1': 0.66, 'auc_roc': 0.72},
    }
    
    # 特征重要性示例
    feature_importance = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(20)],
        'importance': np.random.uniform(0.01, 0.1, 20)
    }).sort_values('importance', ascending=False)
    
    # 回测统计示例
    backtest_stats = {
        'total_return': 25.5,
        'annual_return': 12.3,
        'buy_hold_return': 15.2,
        'sharpe_ratio': 1.2,
        'max_drawdown': -15.6,
        'total_trades': 45,
        'win_rate': 55.0,
        'best_trade': 8.5,
        'worst_trade': -5.2,
        'avg_trade': 0.56,
        'initial_capital': 100000,
        'final_equity': 125500,
        'equity_peak': 130000,
    }
    
    # 可视化
    viz = Visualizer()
    viz.plot_model_comparison(model_results)
    viz.plot_feature_importance(feature_importance)
    viz.plot_backtest_summary(backtest_stats)
    
    plt.show()


if __name__ == '__main__':
    main()
