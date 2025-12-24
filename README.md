# 原油期货多模型集成投资策略

基于机器学习的WTI原油期货（CL=F）量化交易系统，涵盖数据获取、特征工程、模型训练、回测与可视化全流程。

## 项目结构

```
.
├── config.py                      # 配置文件（参数统一管理）
├── data_collector.py              # 数据收集模块
├── feature_engineering.py         # 特征工程模块
├── main.py                        # 主程序入口
├── model_trainer.py               # 模型训练模块
├── strategy_backtest.py           # 回测策略模块
├── tune.py                        # 参数调优模块（Optuna）
├── visualization.py               # 可视化与报告生成
├── predictor.py                   # 推理接口（模拟盘用）
├── paper_account.py               # 模拟账户模块
├── app.py                         # Streamlit交互式演示程序
├── requirements.txt               # 依赖包列表
├── README.md                      # 本文件
│
├── data/                          # 数据目录
│   └── raw_data.csv              # 原始数据（自动下载）
│
├── models/                        # 训练模型存储
│   ├── rf_model.joblib           # 随机森林模型
│   ├── xgb_model.joblib          # XGBoost模型
│   ├── bagging_model.joblib      # Bagging模型
│   ├── model_config.joblib       # 模型配置
│   ├── scaler.joblib             # 特征标准化器
│   ├── selector.joblib           # 特征选择器
│   ├── feature_names.joblib      # 特征名列表
│   └── selected_features.joblib  # 选中特征列表
│
└── output/                        # 输出结果目录
    ├── best_params.json          # 调参最优结果
    ├── model_comparison.png       # 模型对比
    ├── feature_importance.png     # 特征重要性
    ├── backtest_summary.png       # 回测摘要
    ├── equity_curve.png           # 权益曲线
    ├── returns_distribution.png   # 收益分布
    ├── prediction_analysis.png    # 预测分析
    ├── price_signals.png          # 价格信号图
    ├── trade_timeline.png         # 交易时间线与累计盈亏
    └── trades.json                # 交易记录明细
```

## 运行环境

- Python 3.9 或更高版本（推荐创建虚拟环境）
- 安装依赖：`pip install -r requirements.txt`
- 核心库：scikit-learn、xgboost、backtesting、optuna、yfinance、matplotlib、pandas、streamlit、plotly

## 快速开始

### 完整执行（回测模式）
```bash
python main.py
```
运行完整管道（数据收集 → 特征工程 → 模型训练 → 回测 → 可视化）。

### 🎮 交互式模拟盘演示
```bash
streamlit run app.py
```
启动交互式Web界面，进行步进式模拟交易演示。

**模拟盘功能：**
- 📊 逐日推进模拟交易，观察每日交易决策
- 💼 账户状态实时更新（现金、持仓、盈亏）
- 📈 交互式K线图、权益曲线、信号标记
- 📝 完整交易记录导出

**首次使用：**
1. 如果没有训练过模型，程序会提示训练
2. 点击"初始化/重置模拟"开始演示
3. 使用"下一天"/"前进5天"等按钮逐步推进模拟

### 无图形环境
如在服务器环境，设置 `show_plots=False`：
```python
pipeline.run_full_pipeline(show_plots=False)
```

### 快速测试
快速验证流程（数据范围缩短）：
```python
pipeline.quick_test()
```

### 分模块调试
| 模块 | 文件 | 功能 |
|------|------|------|
| 数据收集 | [data_collector.py](data_collector.py) | Yahoo Finance 数据下载 |
| 特征工程 | [feature_engineering.py](feature_engineering.py) | 技术指标与衍生特征 |
| 模型训练 | [model_trainer.py](model_trainer.py) | 集成模型与评估 |
| 策略回测 | [strategy_backtest.py](strategy_backtest.py) | 回测与性能统计 |
| 可视化报告 | [visualization.py](visualization.py) | 结果展示与分析 |
| 推理接口 | [predictor.py](predictor.py) | 单日/批量预测 |
| 模拟账户 | [paper_account.py](paper_account.py) | 模拟交易执行 |
| 交互演示 | [app.py](app.py) | Streamlit Web界面 |

## 技术方案

### 数据层
- **标的**：WTI原油期货（CL=F）
- **时间范围**：2016-01-01 至 2025-12-01
- **频率**：日线数据
- **宏观因子**：美债收益率、美元指数、标普500、VIX、USO、黄金（6类）

详见 [data_collector.py](data_collector.py#L23-L193)

### 特征层
- 技术指标：SMA、EMA、RSI、MACD、布林带、ATR
- 滞后特征：60期历史价格
- 动量特征：多窗口 (5, 10, 20, 60 日)
- 交互特征：衍生变量
- **目标**：预测未来5日上涨概率（二分类）
 - **特征选择**：默认筛选 70 个最重要特征（如加载 `output/best_params.json` 会用最优值 30）

详见 [feature_engineering.py](feature_engineering.py#L18-L199)

### 模型层（三模型集成）
| 模型 | 类型 | 配置 | 权重 |
|------|------|------|------|
| Random Forest | 并行集成 | n_estimators=200, max_depth=10 | 30% |
| XGBoost | 梯度提升 | n_estimators=200, max_depth=8, lr=0.05 | 43% |
| Bagging LogReg | 装袋集成 | n_estimators=60 | 27% |

- 时间序列交叉验证：5折
- 训练/测试分割：7/3
- 集成方式：软投票

详见 [model_trainer.py](model_trainer.py#L21-L214)

### 策略层
- **默认基线**（`config.py`）：买入 ≥ 0.58，卖出 ≤ 0.38，单笔仓位 45%，止损 4%，止盈 14%，佣金/滑点 0.1%，`long_only=False` 支持多空
- **最佳参数覆盖**（自动加载 `output/best_params.json`）：买入 0.63，卖出 0.37，单笔仓位 30%，止损 6%，止盈 10%，特征数 30，集成权重/模型超参同步更新
- **交易记录**：回测结束自动写入 `output/trades.json`

详见 [strategy_backtest.py](strategy_backtest.py#L15-L210)

### 输出层
生成 8 类分析图表：
- 模型性能对比
- 特征重要性排序
- 回测性能摘要
- 权益曲线走势
- 收益分布统计
- 预测准确性分析
- 价格与交易信号
- 交易时间线与累计盈亏

详见 [visualization.py](visualization.py#L17-L326)

## 配置说明

详见 [config.py](config.py)，主要参数如下：

## 输出物

| 文件 | 位置 | 说明 |
|------|------|------|
| 原始数据 | `data/raw_data.csv` | 下载的原油期货与宏观因子数据 |
| 随机森林模型 | `models/rf_model.joblib` | Random Forest 训练模型 |
| XGBoost 模型 | `models/xgb_model.joblib` | XGBoost 训练模型 |
| Bagging 模型 | `models/bagging_model.joblib` | Bagging(LogReg) 训练模型 |
| 模型配置 | `models/model_config.joblib` | 特征处理器与配置 |
| 模型对比 | `output/model_comparison.png` | 三模型性能对比 |
| 特征重要性 | `output/feature_importance.png` | 特征贡献度分析 |
| 回测摘要 | `output/backtest_summary.png` | 关键回测指标 |
| 权益曲线 | `output/equity_curve.png` | 账户资产变化曲线 |
| 收益分布 | `output/returns_distribution.png` | 收益率直方图 |
| 预测分析 | `output/prediction_analysis.png` | 预测准确率分析 |
| 交易信号 | `output/price_signals.png` | 价格与买卖信号图 |
| 交易时间线 | `output/trade_timeline.png` | 交易持仓区间与累计盈亏轨迹 |
| 交易记录 | `output/trades.json` | 回测期间的逐笔持仓记录 |
| 调参结果 | `output/best_params.json` | Optuna 最优参数 |

## 调参优化

参数优化采用 Optuna 框架，支持全局搜索和贝叶斯优化。

### 运行调参
```bash
python tune.py --trials 50 --study-name cl_ml_tune
```

### 搜索空间

优化的关键参数：
- **模型参数**：Random Forest、XGBoost、Bagging 的树深度、学习率等
- **集成权重**：三个模型的投票权重
- **特征数量**：选择前 K 个重要特征
- **策略参数**：买入/卖出阈值、持仓比例、止盈止损水平

### 评估指标
- 主指标：Sharpe 比率
- 辅助指标：年化收益率、最大回撤

最优参数自动保存至 `output/best_params.json`。

## 注意事项

1. **网络连接**：部分地区访问 Yahoo Finance 需要代理。如数据下载为空，系统会自动重试。
   
2. **数据处理**：
   - 特征选择与标准化仅在训练集进行
   - 测试集和回测阶段保持特征对齐，缺失值自动填充
   - 目标标签由 `close.shift(-horizon)` 生成，已规避前瞻偏差
   - 末尾 NaN 行已自动丢弃

3. **模型使用**：
   - 支持多空双向交易，`STRATEGY_CONFIG['long_only']` 控制是否禁用卖空（默认允许卖空）
   - 若需仅做多，将配置设为 `True`
   - 默认自动读取 `output/best_params.json` 覆盖模型与策略配置（可在 `config.py` 中将 `USE_BEST_PARAMS=False` 或通过 `BEST_PARAMS_PATH` 环境变量自定义路径）
   - 回测交易记录会保存到 `STRATEGY_CONFIG['trades_log_path']`（默认 `output/trades.json`），便于复盘与审计

4. **依赖问题**：
   - Windows 用户如遇 joblib 保存问题，可改用 pickle
   - macOS M1/M2 用户运行 XGBoost 可能需要指定平台

## 项目主要算法亮点

✓ **时间序列交叉验证** - 避免数据泄露，真实评估模型性能  
✓ **三模型软投票集成** - 结合 RF 的稳定性、XGB 的拟合力、Bagging 的鲁棒性  
✓ **特征交互与衍生** - 融合宏观因子与技术指标  
✓ **动态阈值策略** - 不同市场环境自适应交易信号  
✓ **完整回测框架** - 模拟真实交易成本（佣金、滑点、止盈止损）

## 免责声明

本项目仅供学习研究使用，**不构成任何投资建议**。使用者需自行承担由此产生的一切后果。量化交易存在风险，请谨慎参与。
