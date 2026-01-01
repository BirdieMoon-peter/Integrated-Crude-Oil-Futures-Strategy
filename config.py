"""
配置模块 - 集中管理所有参数，并支持加载最优参数覆盖
原油期货多模型集成投资策略
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

# 数据配置
DATA_CONFIG = {
    # 核心标的
    'symbol': 'CL=F',  # WTI原油期货
    
    # 时间跨度
    'start_date': '2016-01-01',
    'end_date': '2025-12-20',
    
    # 宏观经济数据标的
    'macro_symbols': [
        '^TNX',      # 10年期美国国债收益率
        'DX-Y.NYB',  # 美元指数期货
        '^GSPC',     # 标普500指数
        '^VIX',      # 恐慌指数
        'USO',       # 美国原油基金ETF
        'GC=F',      # 黄金期货
        'HO=F',      # 取暖油期货（用于裂解价差）
        'RB=F',      # RBOB汽油期货（用于裂解价差）
    ],

    # 期限结构（次月）下载配置：
    # - Yahoo 上并不保证存在类似 CL2=F 的连续次月代码，所以默认关闭
    # - 若你找到了可用的次月代码（例如某些市场/后缀格式），把 enable 置 True 并填入 candidates
    'term_structure': {
        'enable': False,
        'next_month_candidates': ['CL2=F', 'CL1=F'],
        'next_month_col': 'cl_next_close',
    },
    
    # 数据频率
    'interval': '1d',  # 日频
}

# 特征工程配置
FEATURE_CONFIG = {
    # 技术指标参数
    'sma_periods': [5, 10, 20, 50],
    'ema_periods': [12, 26],
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bb_period': 20,
    'bb_std': 2,
    'atr_period': 14,
    'volume_sma_period': 20,
    
    # 滞后特征配置
    'lag_periods': 60,  # 回溯期
    
    # 动量特征窗口
    'momentum_windows': [1, 3, 5, 10, 20],
    'volatility_windows': [5, 10, 20],
    
    # 特征选择
    'n_features': 70,  # SelectKBest选择的特征数量
    
    # 目标变量预测窗口
    'prediction_horizon': 5,  # 预测未来N天

    # 期限结构/裂解价差特征
    'crude_price_col': 'close',  # 原油价格列名（前月合约）
    'time_spread_next_cols': ['cl_next_close', 'cl2f_close', 'cl2_close'],  # 次月价格列候选（Front - Next）
}

# 模型配置
MODEL_CONFIG = {
    # 随机森林配置
    'rf': {
        'n_estimators': 200,
        'max_depth': 10,
        'random_state': 42,
        'min_samples_leaf': 3,
        'n_jobs': -1,
    },
    
    # XGBoost配置
    'xgb': {
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.05,
        'eval_metric': 'logloss',
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'reg_lambda': 4.316815537303276,
        'reg_alpha': 0.591105432886858,
        'random_state': 42,
        'use_label_encoder': False,
    },
    
    # Bagging配置
    'bagging': {
        'n_estimators': 60,
        'random_state': 42,
        'n_jobs': -1,
    },
    
    # 集成权重
    'ensemble_weights': {
        'rf': 0.3,
        'xgb': 0.43,
        'bagging': 0.27,
    },
    
    # 训练配置
    'train_size': 0.7,  # 训练集比例
    'cv_splits': 5,  # 时间序列交叉验证折数
}

# 交易策略配置
STRATEGY_CONFIG = {
    # 是否仅做多（False 时启用卖空）
    'long_only': False,
    
    # 交易记录输出
    'trades_log_path': 'output/trades.json',
    
    # 信号阈值
    'threshold_buy': 0.58,   # 买入阈值
    'threshold_sell': 0.38,  # 卖出阈值
    
    # 仓位管理
    'initial_capital': 100000,  # 初始资金
    'position_size': 0.45,       # 单笔仓位比例 (45%)
    
    # 交易成本
    'commission': 0.001,  # 佣金 0.1%
    'slippage': 0.001,    # 滑点 0.1%
    
    # 风险控制
    'stop_loss': 0.04,    # 止损 4%
    'take_profit': 0.14,  # 止盈 14%
}

# 可视化配置
VIS_CONFIG = {
    'figure_size': (14, 8),
    'dpi': 100,
    'style': 'seaborn-v0_8-whitegrid',
    'top_features': 15,  # 显示前N个重要特征
    'output_dir': 'output',
}

# 日志配置
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
}

# 是否优先加载 output/best_params.json 覆盖默认配置
USE_BEST_PARAMS = True
BEST_PARAMS_PATH = os.environ.get("BEST_PARAMS_PATH", "output/best_params.json")


def _load_best_params(path: str = BEST_PARAMS_PATH):
    """从文件读取最优参数"""
    if not USE_BEST_PARAMS:
        return None
    
    if not os.path.exists(path):
        logger.info(f"未找到最佳参数文件: {path}，将使用默认配置")
        return None
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        return payload.get('params')
    except Exception as e:
        logger.warning(f"加载最佳参数失败，将使用默认配置。错误: {e}")
        return None


def apply_best_params():
    """
    将最佳参数覆盖到当前配置（原地修改全局 CONFIG）
    """
    params = _load_best_params()
    if not params:
        return
    
    # 特征选择
    FEATURE_CONFIG['n_features'] = params.get('n_features', FEATURE_CONFIG['n_features'])
    FEATURE_CONFIG['prediction_horizon'] = params.get('prediction_horizon', FEATURE_CONFIG['prediction_horizon'])
    FEATURE_CONFIG['lag_periods'] = params.get('lag_periods', FEATURE_CONFIG['lag_periods'])
    FEATURE_CONFIG['momentum_windows'] = params.get('momentum_windows', FEATURE_CONFIG['momentum_windows'])
    FEATURE_CONFIG['volatility_windows'] = params.get('volatility_windows', FEATURE_CONFIG['volatility_windows'])
    
    # 模型参数
    MODEL_CONFIG['rf'] = MODEL_CONFIG['rf'].copy()
    MODEL_CONFIG['rf'].update({
        'n_estimators': params.get('rf_n_estimators', MODEL_CONFIG['rf']['n_estimators']),
        'max_depth': params.get('rf_max_depth', MODEL_CONFIG['rf']['max_depth']),
        'min_samples_leaf': params.get('rf_min_samples_leaf', MODEL_CONFIG['rf']['min_samples_leaf']),
        'min_samples_split': params.get('rf_min_samples_split', MODEL_CONFIG['rf'].get('min_samples_split', 2)),
        'max_features': params.get('rf_max_features', MODEL_CONFIG['rf'].get('max_features', 'sqrt')),
        'bootstrap': params.get('rf_bootstrap', MODEL_CONFIG['rf'].get('bootstrap', True)),
    })
    
    MODEL_CONFIG['xgb'] = MODEL_CONFIG['xgb'].copy()
    MODEL_CONFIG['xgb'].update({
        'n_estimators': params.get('xgb_n_estimators', MODEL_CONFIG['xgb']['n_estimators']),
        'max_depth': params.get('xgb_max_depth', MODEL_CONFIG['xgb']['max_depth']),
        'learning_rate': params.get('xgb_learning_rate', MODEL_CONFIG['xgb']['learning_rate']),
        'subsample': params.get('xgb_subsample', MODEL_CONFIG['xgb']['subsample']),
        'colsample_bytree': params.get('xgb_colsample', MODEL_CONFIG['xgb']['colsample_bytree']),
        'reg_lambda': params.get('xgb_reg_lambda', MODEL_CONFIG['xgb']['reg_lambda']),
        'reg_alpha': params.get('xgb_reg_alpha', MODEL_CONFIG['xgb']['reg_alpha']),
        'min_child_weight': params.get('xgb_min_child_weight', MODEL_CONFIG['xgb'].get('min_child_weight', 1.0)),
        'gamma': params.get('xgb_gamma', MODEL_CONFIG['xgb'].get('gamma', 0.0)),
    })
    
    MODEL_CONFIG['bagging'] = MODEL_CONFIG['bagging'].copy()
    MODEL_CONFIG['bagging'].update({
        'n_estimators': params.get('bag_n_estimators', MODEL_CONFIG['bagging']['n_estimators']),
        'max_samples': params.get('bag_max_samples', MODEL_CONFIG['bagging'].get('max_samples', 1.0)),
        'max_features': params.get('bag_max_features', MODEL_CONFIG['bagging'].get('max_features', 1.0)),
        'bootstrap': params.get('bag_bootstrap', MODEL_CONFIG['bagging'].get('bootstrap', True)),
    })
    
    if 'weights' in params:
        MODEL_CONFIG['ensemble_weights'] = params['weights']
    
    # 策略参数
    STRATEGY_CONFIG.update({
        'threshold_buy': params.get('threshold_buy', STRATEGY_CONFIG['threshold_buy']),
        'threshold_sell': params.get('threshold_sell', STRATEGY_CONFIG['threshold_sell']),
        'position_size': params.get('position_size', STRATEGY_CONFIG['position_size']),
        'stop_loss': params.get('stop_loss', STRATEGY_CONFIG['stop_loss']),
        'take_profit': params.get('take_profit', STRATEGY_CONFIG['take_profit']),
    })
    
    logger.info(f"已加载最佳参数并覆盖默认配置，来源: {BEST_PARAMS_PATH}")


def export_current_config(output_path: str = 'output/current_config.json'):
    """
    导出当前配置到 JSON 文件，供 Agent 优化器读取
    
    Args:
        output_path: 输出文件路径
    """
    current_config = {
        'feature_config': {
            'n_features': FEATURE_CONFIG['n_features'],
            'prediction_horizon': FEATURE_CONFIG['prediction_horizon'],
            'lag_periods': FEATURE_CONFIG['lag_periods'],
            'momentum_windows': FEATURE_CONFIG['momentum_windows'],
            'volatility_windows': FEATURE_CONFIG['volatility_windows'],
            'sma_periods': FEATURE_CONFIG['sma_periods'],
            'ema_periods': FEATURE_CONFIG['ema_periods'],
            'rsi_period': FEATURE_CONFIG['rsi_period'],
        },
        'model_config': {
            'rf': MODEL_CONFIG['rf'].copy(),
            'xgb': MODEL_CONFIG['xgb'].copy(),
            'bagging': MODEL_CONFIG['bagging'].copy(),
            'ensemble_weights': MODEL_CONFIG['ensemble_weights'].copy(),
            'train_size': MODEL_CONFIG['train_size'],
        },
        'strategy_config': {
            'threshold_buy': STRATEGY_CONFIG['threshold_buy'],
            'threshold_sell': STRATEGY_CONFIG['threshold_sell'],
            'position_size': STRATEGY_CONFIG['position_size'],
            'stop_loss': STRATEGY_CONFIG['stop_loss'],
            'take_profit': STRATEGY_CONFIG['take_profit'],
            'long_only': STRATEGY_CONFIG['long_only'],
            'initial_capital': STRATEGY_CONFIG['initial_capital'],
            'commission': STRATEGY_CONFIG['commission'],
        }
    }
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(current_config, f, indent=2, ensure_ascii=False)
    logger.info(f"当前配置已导出到 {output_path}")
    
    return current_config


# 在模块导入时应用（若文件存在）
apply_best_params()
