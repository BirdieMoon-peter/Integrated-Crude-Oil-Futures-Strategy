"""
策略回测模块 - 基于backtesting.py框架进行策略验证
原油期货多模型集成投资策略
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import logging
import warnings

from config import STRATEGY_CONFIG

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class MLStrategy(Strategy):
    """
    基于机器学习预测的交易策略
    
    使用预计算的预测概率进行交易决策
    """
    
    # 策略参数（可通过optimize调整）
    threshold_buy = 0.55    # 买入阈值
    threshold_sell = 0.45   # 卖出阈值
    position_size = 0.3     # 仓位比例
    stop_loss = 0.05        # 止损比例
    take_profit = 0.10      # 止盈比例
    
    def init(self):
        """
        策略初始化
        预计算的Forecast列已经在数据中
        """
        # 获取预测概率
        self.forecast = self.I(lambda x: x, self.data.Forecast, name='Forecast')
        
        # 记录入场价格
        self.entry_price = None
    
    def next(self):
        """
        每个时间步的交易逻辑
        """
        current_forecast = self.forecast[-1]
        current_price = self.data.Close[-1]
        
        # 如果有持仓，检查止盈止损
        if self.position:
            if self.entry_price is not None:
                # 计算当前收益率
                pnl_pct = (current_price - self.entry_price) / self.entry_price
                
                if self.position.is_long:
                    # 多头止损止盈
                    if pnl_pct <= -self.stop_loss:
                        self.position.close()
                        self.entry_price = None
                        return
                    elif pnl_pct >= self.take_profit:
                        self.position.close()
                        self.entry_price = None
                        return
                    # 反向信号平仓
                    elif current_forecast < self.threshold_sell:
                        self.position.close()
                        self.entry_price = None
                        return
                        
                elif self.position.is_short:
                    # 空头止损止盈
                    if pnl_pct >= self.stop_loss:
                        self.position.close()
                        self.entry_price = None
                        return
                    elif pnl_pct <= -self.take_profit:
                        self.position.close()
                        self.entry_price = None
                        return
                    # 反向信号平仓
                    elif current_forecast > self.threshold_buy:
                        self.position.close()
                        self.entry_price = None
                        return
        
        # 开仓逻辑
        if not self.position:
            # 计算可用仓位
            size = self.position_size
            
            if current_forecast > self.threshold_buy:
                # 买入信号
                self.buy(size=size)
                self.entry_price = current_price
                
            elif current_forecast < self.threshold_sell:
                # 卖出信号（做空）
                self.sell(size=size)
                self.entry_price = current_price


class LongOnlyMLStrategy(Strategy):
    """
    仅做多的机器学习策略
    适用于不支持做空的市场
    """
    
    threshold_buy = 0.55
    threshold_sell = 0.45
    position_size = 0.3
    stop_loss = 0.05
    take_profit = 0.10
    
    def init(self):
        self.forecast = self.I(lambda x: x, self.data.Forecast, name='Forecast')
        self.entry_price = None
    
    def next(self):
        current_forecast = self.forecast[-1]
        current_price = self.data.Close[-1]
        
        if self.position:
            if self.entry_price is not None:
                pnl_pct = (current_price - self.entry_price) / self.entry_price
                
                # 止损
                if pnl_pct <= -self.stop_loss:
                    self.position.close()
                    self.entry_price = None
                    return
                
                # 止盈
                if pnl_pct >= self.take_profit:
                    self.position.close()
                    self.entry_price = None
                    return
                
                # 卖出信号
                if current_forecast < self.threshold_sell:
                    self.position.close()
                    self.entry_price = None
                    return
        
        else:
            # 买入信号
            if current_forecast > self.threshold_buy:
                self.buy(size=self.position_size)
                self.entry_price = current_price


class BacktestEngine:
    """回测引擎类"""
    
    def __init__(self, config: dict = None):
        """
        初始化回测引擎
        
        Args:
            config: 策略配置
        """
        self.config = config or STRATEGY_CONFIG
        self.backtest = None
        self.results = None
        self.stats = None
    
    def prepare_backtest_data(self, df: pd.DataFrame, 
                               forecast: pd.Series) -> pd.DataFrame:
        """
        准备回测数据
        
        Args:
            df: 原始OHLCV数据
            forecast: 预测概率Series
            
        Returns:
            回测用DataFrame
        """
        # 确保列名符合backtesting.py的要求
        bt_data = df[['open', 'high', 'low', 'close', 'volume']].copy()
        bt_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # 添加预测概率
        bt_data['Forecast'] = forecast
        
        # 删除NaN
        bt_data = bt_data.dropna()
        
        # 确保索引是DatetimeIndex
        bt_data.index = pd.to_datetime(bt_data.index)
        
        return bt_data
    
    def run_backtest(self, df: pd.DataFrame, 
                     forecast: pd.Series,
                     strategy_class: Strategy = None,
                     long_only: bool = False) -> Dict:
        """
        运行回测
        
        Args:
            df: OHLCV数据
            forecast: 预测概率Series
            strategy_class: 策略类（可选）
            long_only: 是否仅做多
            
        Returns:
            回测统计结果
        """
        # 准备数据
        bt_data = self.prepare_backtest_data(df, forecast)
        
        logger.info(f"回测数据准备完成，共 {len(bt_data)} 条记录")
        logger.info(f"回测时间范围: {bt_data.index[0]} 至 {bt_data.index[-1]}")
        
        # 选择策略
        if strategy_class is None:
            strategy_class = LongOnlyMLStrategy if long_only else MLStrategy
        
        # 创建回测实例
        self.backtest = Backtest(
            bt_data,
            strategy_class,
            cash=self.config.get('initial_capital', 100000),
            commission=self.config.get('commission', 0.001),
            exclusive_orders=True
        )
        
        # 设置策略参数
        self.results = self.backtest.run(
            threshold_buy=self.config.get('threshold_buy', 0.55),
            threshold_sell=self.config.get('threshold_sell', 0.45),
            position_size=self.config.get('position_size', 0.3),
            stop_loss=self.config.get('stop_loss', 0.05),
            take_profit=self.config.get('take_profit', 0.10)
        )
        
        # 提取统计信息
        self.stats = self._extract_stats()
        
        return self.stats
    
    def _extract_stats(self) -> Dict:
        """
        提取回测统计信息
        
        Returns:
            统计信息字典
        """
        if self.results is None:
            return {}
        
        stats = {
            # 收益指标
            'total_return': self.results['Return [%]'],
            'annual_return': self.results.get('Return (Ann.) [%]', None),
            'buy_hold_return': self.results['Buy & Hold Return [%]'],
            
            # 风险指标
            'sharpe_ratio': self.results.get('Sharpe Ratio', None),
            'sortino_ratio': self.results.get('Sortino Ratio', None),
            'calmar_ratio': self.results.get('Calmar Ratio', None),
            'max_drawdown': self.results['Max. Drawdown [%]'],
            'avg_drawdown': self.results.get('Avg. Drawdown [%]', None),
            
            # 交易统计
            'total_trades': self.results['# Trades'],
            'win_rate': self.results['Win Rate [%]'],
            'best_trade': self.results['Best Trade [%]'],
            'worst_trade': self.results['Worst Trade [%]'],
            'avg_trade': self.results.get('Avg. Trade [%]', None),
            'profit_factor': self.results.get('Profit Factor', None),
            
            # 持仓统计
            'avg_trade_duration': str(self.results.get('Avg. Trade Duration', '')),
            'exposure_time': self.results['Exposure Time [%]'],
            
            # 资金
            'initial_capital': self.config.get('initial_capital', 100000),
            'final_equity': self.results['Equity Final [$]'],
            'equity_peak': self.results['Equity Peak [$]'],
        }
        
        return stats
    
    def optimize_strategy(self, df: pd.DataFrame, 
                          forecast: pd.Series,
                          maximize: str = 'Sharpe Ratio',
                          constraint = None) -> Tuple[Dict, Dict]:
        """
        优化策略参数
        
        Args:
            df: OHLCV数据
            forecast: 预测概率
            maximize: 优化目标
            constraint: 约束条件
            
        Returns:
            (最优参数, 最优结果)
        """
        bt_data = self.prepare_backtest_data(df, forecast)
        
        self.backtest = Backtest(
            bt_data,
            MLStrategy,
            cash=self.config.get('initial_capital', 100000),
            commission=self.config.get('commission', 0.001),
            exclusive_orders=True
        )
        
        # 参数搜索范围
        opt_results = self.backtest.optimize(
            threshold_buy=[0.50, 0.55, 0.60, 0.65],
            threshold_sell=[0.35, 0.40, 0.45, 0.50],
            position_size=[0.2, 0.3, 0.4, 0.5],
            stop_loss=[0.03, 0.05, 0.07, 0.10],
            take_profit=[0.05, 0.10, 0.15, 0.20],
            maximize=maximize,
            constraint=constraint,
            max_tries=100
        )
        
        best_params = {
            'threshold_buy': opt_results._strategy.threshold_buy,
            'threshold_sell': opt_results._strategy.threshold_sell,
            'position_size': opt_results._strategy.position_size,
            'stop_loss': opt_results._strategy.stop_loss,
            'take_profit': opt_results._strategy.take_profit,
        }
        
        self.results = opt_results
        self.stats = self._extract_stats()
        
        return best_params, self.stats
    
    def plot_results(self, filename: str = None, open_browser: bool = False):
        """
        绘制回测结果图表
        
        Args:
            filename: 保存文件名
            open_browser: 是否在浏览器中打开
        """
        if self.backtest is None:
            raise ValueError("请先运行回测")
        
        self.backtest.plot(
            filename=filename,
            open_browser=open_browser,
            resample=False
        )
    
    def get_trades(self) -> pd.DataFrame:
        """
        获取交易记录
        
        Returns:
            交易记录DataFrame
        """
        if self.results is None:
            return pd.DataFrame()
        
        return self.results._trades
    
    def get_equity_curve(self) -> pd.Series:
        """
        获取权益曲线
        
        Returns:
            权益曲线Series
        """
        if self.results is None:
            return pd.Series()
        
        return self.results._equity_curve['Equity']
    
    def print_summary(self):
        """打印回测摘要"""
        if self.stats is None:
            print("尚未运行回测")
            return
        
        print("\n" + "="*60)
        print("回测结果摘要")
        print("="*60)
        
        print(f"\n【收益指标】")
        print(f"  总收益率:          {self.stats['total_return']:.2f}%")
        if self.stats['annual_return']:
            print(f"  年化收益率:        {self.stats['annual_return']:.2f}%")
        print(f"  买入持有收益:      {self.stats['buy_hold_return']:.2f}%")
        
        print(f"\n【风险指标】")
        if self.stats['sharpe_ratio']:
            print(f"  夏普比率:          {self.stats['sharpe_ratio']:.4f}")
        if self.stats['sortino_ratio']:
            print(f"  索提诺比率:        {self.stats['sortino_ratio']:.4f}")
        print(f"  最大回撤:          {self.stats['max_drawdown']:.2f}%")
        
        print(f"\n【交易统计】")
        print(f"  总交易次数:        {self.stats['total_trades']}")
        print(f"  胜率:              {self.stats['win_rate']:.2f}%")
        print(f"  最佳交易:          {self.stats['best_trade']:.2f}%")
        print(f"  最差交易:          {self.stats['worst_trade']:.2f}%")
        if self.stats['profit_factor']:
            print(f"  盈亏比:            {self.stats['profit_factor']:.4f}")
        
        print(f"\n【资金状况】")
        print(f"  初始资金:          ${self.stats['initial_capital']:,.2f}")
        print(f"  最终权益:          ${self.stats['final_equity']:,.2f}")
        print(f"  权益峰值:          ${self.stats['equity_peak']:,.2f}")
        
        print("="*60)


def compute_forecasts(df: pd.DataFrame, 
                      trainer, 
                      feature_engineer) -> pd.Series:
    """
    预计算整个时间序列的预测概率
    
    Args:
        df: 特征数据DataFrame
        trainer: 训练好的ModelTrainer
        feature_engineer: 训练好的FeatureEngineer
        
    Returns:
        预测概率Series
    """
    from feature_engineering import FeatureEngineer
    
    # 获取特征列
    feature_cols = feature_engineer.get_feature_columns(df)
    X = df[feature_cols].copy()
    
    # 填充NaN
    X = X.fillna(0)
    
    # 对齐特征
    X_aligned = feature_engineer._align_features(X)
    
    # 标准化
    X_scaled = feature_engineer.scaler.transform(X_aligned)
    
    # 特征选择
    X_selected = feature_engineer.selector.transform(X_scaled)
    
    # 预测概率
    proba = trainer.voting_predict_proba(X_selected)
    
    # 返回上涨概率
    return pd.Series(proba[:, 1], index=df.index, name='Forecast')


def main():
    """测试回测功能"""
    from data_collector import DataCollector
    from feature_engineering import FeatureMatrix
    from model_trainer import ModelTrainer
    
    print("正在加载数据...")
    collector = DataCollector()
    data = collector.get_data()
    
    print("正在进行特征工程...")
    feature_matrix = FeatureMatrix()
    X_train, X_test, y_train, y_test = feature_matrix.fit_transform_pipeline(data)
    
    print("正在训练模型...")
    trainer = ModelTrainer()
    trainer.train(X_train, y_train)
    trainer.evaluate(X_test, y_test)
    
    print("正在计算预测...")
    # 获取测试集时间范围
    test_start_idx = feature_matrix.test_index[0]
    test_df = feature_matrix.featured_data.loc[test_start_idx:]
    
    # 预计算预测概率
    forecast = compute_forecasts(test_df, trainer, feature_matrix.engineer)
    
    print("正在运行回测...")
    engine = BacktestEngine()
    stats = engine.run_backtest(test_df, forecast, long_only=True)
    
    engine.print_summary()


if __name__ == '__main__':
    main()
