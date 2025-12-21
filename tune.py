"""
Optuna 调参脚本
- 调 RandomForest / XGBoost / Bagging(LogReg) 参数
- 调 集成权重 与 交易阈值 / 仓位 / 止损止盈
- 目标: 验证集综合评分 (Sharpe 优先，兼顾最大回撤)

使用方式:
    pip install -r requirements.txt
    python tune.py --trials 50 --study-name cl_ml_tune

输出:
- 最优参数打印到控制台
- 将 best_params.json 写入 output/best_params.json

说明:
- 采用时间序列切分: 前 70% 作为训练, 后 30% 作为验证/回测段
- 在验证段上跑一次 backtesting 回测，返回评分
- 评分函数: score = sharpe - 0.1 * max_drawdown/100 + 0.001 * annual_return/100
  若 Sharpe 缺失则用总收益率/100 代替

注意:
- 回测是否允许做空由 STRATEGY_CONFIG 中的 long_only 控制
- 为加快调参，采用较小搜索空间，可在下方自行扩展
- 若需要更快，可减少 trials 或收窄参数范围
"""

import argparse
import json
import os
import warnings
from typing import Dict, Any

import numpy as np
import optuna

from config import DATA_CONFIG, FEATURE_CONFIG, MODEL_CONFIG, STRATEGY_CONFIG
from data_collector import DataCollector
from feature_engineering import FeatureMatrix
from model_trainer import ModelTrainer
from strategy_backtest import BacktestEngine, compute_forecasts

warnings.filterwarnings("ignore")


def scoring(stats: Dict[str, Any]) -> float:
    """综合评分函数"""
    sharpe = stats.get("sharpe_ratio")
    max_dd = stats.get("max_drawdown", 0)  # %
    annual = stats.get("annual_return", 0)

    if sharpe is None:
        sharpe = stats.get("total_return", 0) / 100.0

    score = (sharpe or 0) - 0.1 * (abs(max_dd) / 100.0) + 0.001 * (annual or 0) / 100.0
    return score


def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    """定义搜索空间"""
    params = {}

    # RandomForest
    params["rf_n_estimators"] = trial.suggest_int("rf_n_estimators", 150, 400, step=50)
    params["rf_max_depth"] = trial.suggest_int("rf_max_depth", 6, 18, step=2)
    params["rf_min_samples_leaf"] = trial.suggest_int("rf_min_samples_leaf", 1, 5)

    # XGBoost
    params["xgb_n_estimators"] = trial.suggest_int("xgb_n_estimators", 200, 600, step=50)
    params["xgb_max_depth"] = trial.suggest_int("xgb_max_depth", 3, 8)
    params["xgb_learning_rate"] = trial.suggest_float("xgb_learning_rate", 0.03, 0.15, step=0.01)
    params["xgb_subsample"] = trial.suggest_float("xgb_subsample", 0.6, 1.0, step=0.1)
    params["xgb_colsample"] = trial.suggest_float("xgb_colsample", 0.6, 1.0, step=0.1)
    params["xgb_reg_lambda"] = trial.suggest_float("xgb_reg_lambda", 0.5, 5.0)
    params["xgb_reg_alpha"] = trial.suggest_float("xgb_reg_alpha", 0.0, 1.0)

    # Bagging(LogReg)
    params["bag_n_estimators"] = trial.suggest_int("bag_n_estimators", 30, 80, step=10)

    # Ensemble Weights (softmax归一)
    w_rf = trial.suggest_float("w_rf", 0.2, 0.6)
    w_xgb = trial.suggest_float("w_xgb", 0.2, 0.6)
    w_bag = trial.suggest_float("w_bag", 0.1, 0.4)
    total = w_rf + w_xgb + w_bag
    params["weights"] = {
        "rf": w_rf / total,
        "xgb": w_xgb / total,
        "bagging": w_bag / total,
    }

    # Strategy thresholds & risk
    params["threshold_buy"] = trial.suggest_float("threshold_buy", 0.52, 0.65, step=0.01)
    params["threshold_sell"] = trial.suggest_float("threshold_sell", 0.35, 0.50, step=0.01)
    params["position_size"] = trial.suggest_float("position_size", 0.15, 0.45, step=0.05)
    params["stop_loss"] = trial.suggest_float("stop_loss", 0.02, 0.08, step=0.01)
    params["take_profit"] = trial.suggest_float("take_profit", 0.06, 0.20, step=0.02)

    # 特征选择数量
    params["n_features"] = trial.suggest_int("n_features", 30, 80, step=5)

    return params


def apply_params(params: Dict[str, Any]):
    """将建议参数应用到全局配置（浅拷贝，不改原config.py）"""
    data_cfg = DATA_CONFIG.copy()
    feat_cfg = FEATURE_CONFIG.copy()
    model_cfg = MODEL_CONFIG.copy()
    strat_cfg = STRATEGY_CONFIG.copy()

    # 更新特征选择数量
    feat_cfg["n_features"] = params["n_features"]

    # RF
    model_cfg["rf"] = model_cfg["rf"].copy()
    model_cfg["rf"].update({
        "n_estimators": params["rf_n_estimators"],
        "max_depth": params["rf_max_depth"],
        "min_samples_leaf": params["rf_min_samples_leaf"],
    })

    # XGB
    model_cfg["xgb"] = model_cfg["xgb"].copy()
    model_cfg["xgb"].update({
        "n_estimators": params["xgb_n_estimators"],
        "max_depth": params["xgb_max_depth"],
        "learning_rate": params["xgb_learning_rate"],
        "subsample": params["xgb_subsample"],
        "colsample_bytree": params["xgb_colsample"],
        "reg_lambda": params["xgb_reg_lambda"],
        "reg_alpha": params["xgb_reg_alpha"],
    })

    # Bagging
    model_cfg["bagging"] = model_cfg["bagging"].copy()
    model_cfg["bagging"].update({
        "n_estimators": params["bag_n_estimators"],
    })

    # Ensemble weights
    model_cfg["ensemble_weights"] = params["weights"]

    # Strategy
    strat_cfg.update({
        "threshold_buy": params["threshold_buy"],
        "threshold_sell": params["threshold_sell"],
        "position_size": params["position_size"],
        "stop_loss": params["stop_loss"],
        "take_profit": params["take_profit"],
    })

    # 训练/验证划分比例保持 0.7，便于对齐全局使用
    model_cfg["train_size"] = 0.7

    return data_cfg, feat_cfg, model_cfg, strat_cfg


def objective(trial: optuna.Trial) -> float:
    params = suggest_params(trial)

    data_cfg, feat_cfg, model_cfg, strat_cfg = apply_params(params)

    # 数据收集
    collector = DataCollector(data_cfg)
    data = collector.get_data()

    # 特征工程
    fm = FeatureMatrix(feat_cfg)
    X_train, X_test, y_train, y_test = fm.fit_transform_pipeline(
        data, train_size=model_cfg.get("train_size", 0.7)
    )

    # 模型训练
    trainer = ModelTrainer(model_cfg)
    trainer.train(X_train, y_train)

    # 验证段数据
    test_df = fm.featured_data.loc[fm.test_index]

    # 预计算预测概率
    forecast = compute_forecasts(test_df, trainer, fm.engineer)

    # 回测
    engine = BacktestEngine(strat_cfg)
    stats = engine.run_backtest(test_df, forecast, long_only=strat_cfg.get("long_only", True))

    score = scoring(stats)

    # 为了分析，记录指标
    trial.set_user_attr("stats", stats)
    trial.set_user_attr("params_full", params)
    trial.report(score, step=0)

    return score


def main():
    parser = argparse.ArgumentParser(description="Optuna 调参")
    parser.add_argument("--trials", type=int, default=30, help="试验次数")
    parser.add_argument("--study-name", type=str, default="cl_ml_tune", help="Study 名称")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage，例如 sqlite:///optuna.db")
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
    )

    study.optimize(objective, n_trials=args.trials, n_jobs=1, gc_after_trial=True)

    best = study.best_trial
    best_stats = best.user_attrs.get("stats", {})
    best_params = best.user_attrs.get("params_full", {})

    print("\n===== 最优结果 =====")
    print(f"Trial #{best.number}")
    print(f"Score: {best.value:.4f}")
    print("Stats:", json.dumps(best_stats, ensure_ascii=False, indent=2))
    print("Params:", json.dumps(best_params, ensure_ascii=False, indent=2))

    with open("output/best_params.json", "w", encoding="utf-8") as f:
        json.dump({
            "score": best.value,
            "stats": best_stats,
            "params": best_params,
        }, f, ensure_ascii=False, indent=2)

    print("\n最佳参数已写入 output/best_params.json")


if __name__ == "__main__":
    main()
