# 原油期货多模型集成投资策略

基于机器学习的WTI原油期货（CL=F）量化交易系统，覆盖数据获取、特征工程、模型训练、回测与可视化全流程。

## 运行环境
- Python 3.9+，推荐创建虚拟环境
- 安装依赖：`pip install -r requirements.txt`
- 依赖亮点：scikit-learn、xgboost、backtesting、optuna、yfinance、matplotlib

## 快速上手
- 运行完整管道（含回测与图表）：`python main.py`
- 无图形环境可设 `show_plots=False`：在 [main.py](main.py#L166-L239) 的 `run_full_pipeline` 调用中传参
- 快速小样本测试：在 [main.py](main.py#L241-L253) 调用 `pipeline.quick_test()`（时间范围缩短）
- 分模块调试：
	- 数据收集 [data_collector.py](data_collector.py)
	- 特征工程 [feature_engineering.py](feature_engineering.py)
	- 模型训练 [model_trainer.py](model_trainer.py)
	- 回测 [strategy_backtest.py](strategy_backtest.py)
	- 可视化 [visualization.py](visualization.py)

## 工作流概览
1) 数据：从 Yahoo Finance 获取 CL=F 及宏观因子（美债收益率、美元指数、标普500、VIX、USO、黄金），频率日线 [data_collector.py](data_collector.py#L23-L193)
2) 特征：技术指标（SMA/EMA/RSI/MACD/布林/ATR）、滞后、动量、交互特征，并生成未来5日上涨标签 [feature_engineering.py](feature_engineering.py#L18-L199)
3) 训练：时间序列切分（默认训练集70%），RandomForest + XGBoost + Bagging(LogReg) 软投票集成，交叉验证与测试评估 [model_trainer.py](model_trainer.py#L21-L214)
4) 回测：使用 backtesting.py，支持多空/仅多头，含滑点、佣金、止盈止损 [strategy_backtest.py](strategy_backtest.py#L15-L210)
5) 报告：输出模型对比、特征重要性、回测摘要、权益曲线、收益分布、预测分析、价格信号图 [visualization.py](visualization.py#L17-L326)

## 配置要点（默认值见 [config.py](config.py)）
- 数据：`symbol=CL=F`，时间范围 2016-01-01 至 2025-12-01，宏观因子 6 条，日频
- 特征：滞后期 `lag_periods=60`，动量/波动窗口多档，特征选择 `n_features=70`，预测窗口 `prediction_horizon=5`
- 模型：
	- RF `n_estimators=200`，`max_depth=10`
	- XGB `n_estimators=200`，`max_depth=8`，`learning_rate=0.05`
	- Bagging(LogReg) `n_estimators=60`
	- 集成权重：rf 0.30 / xgb 0.43 / bagging 0.27；时间序列 CV 5 折；训练集比例 0.7
- 策略：买入阈值 0.58，卖出阈值 0.38，单笔仓位 45%，佣金/滑点各 0.1%，止损 4%，止盈 14%
- 可视化：输出目录 `output/`，默认风格 seaborn whitegrid

## 输出物
- 数据：data/raw_data.csv
- 模型：models/rf_model.joblib、models/xgb_model.joblib、models/bagging_model.joblib、models/model_config.joblib
- 图表（输出目录）：model_comparison.png、feature_importance.png、backtest_summary.png、equity_curve.png、returns_distribution.png、prediction_analysis.png、price_signals.png
- 调参结果：output/best_params.json（Optuna 运行后生成）

## 调参指南
- 运行：`python tune.py --trials 50 --study-name cl_ml_tune`
- 搜索空间涵盖：RF/XGB/Bagging 参数、集成权重、特征选择数、交易阈值/仓位/止盈止损
- 评分：以 Sharpe 为主，兼顾最大回撤和年化收益，最优结果自动写入 output/best_params.json

## 注意事项
- yfinance 在部分地区需代理；如下载为空会自动重试 download 接口
- 特征选择与标准化仅在训练集拟合，测试/回测阶段保持特征对齐并填充缺失
- 目标使用 `close.shift(-horizon)` 生成，请避免前瞻偏差；尾部 NaN 行已丢弃
- 回测默认仅做多版本可切换 `long_only=True`，如需多空可使用 `MLStrategy`

本项目仅供学习研究使用。
