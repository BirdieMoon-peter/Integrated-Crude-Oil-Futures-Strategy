"""
Agent 优化器模块 - LLM 作为策略动态优化器
原油期货多模型集成投资策略

该模块实现了一个基于大语言模型的自动策略优化器，通过分析回测结果
自动建议参数调整以提高策略性能。
"""

import os
import json
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# ============================================================================
# 配置常量
# ============================================================================

# LLM API 配置（支持 OpenAI、Anthropic 和 Qwen）
LLM_CONFIG = {
    'provider': os.environ.get('LLM_PROVIDER', 'qwen'),  # 'openai', 'anthropic' 或 'qwen'
    'openai': {
        'api_key': os.environ.get('OPENAI_API_KEY', ''),
        'model': os.environ.get('OPENAI_MODEL', 'gpt-4o'),
        'base_url': os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
        'max_tokens': 2000,
        'temperature': 0.3,  # 较低的温度保证输出稳定性
    },
    'anthropic': {
        'api_key': os.environ.get('ANTHROPIC_API_KEY', ''),
        'model': os.environ.get('ANTHROPIC_MODEL', 'claude-sonnet-4-20250514'),
        'max_tokens': 2000,
        'temperature': 0.3,
    },
    'qwen': {
        'api_key': os.environ.get('DASHSCOPE_API_KEY', ''),
        'model': os.environ.get('QWEN_MODEL', 'qwen-plus'),  # qwen-turbo, qwen-plus, qwen-max
        'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'max_tokens': 2000,
        'temperature': 0.3,
    }
}

# 参数边界配置 - 防止 LLM 产生幻觉给出离谱参数
PARAM_BOUNDS = {
    # 特征配置
    'n_features': {'min': 20, 'max': 150, 'type': int},
    'prediction_horizon': {'min': 1, 'max': 20, 'type': int},
    'lag_periods': {'min': 10, 'max': 120, 'type': int},
    
    # Random Forest
    'rf_n_estimators': {'min': 50, 'max': 500, 'type': int},
    'rf_max_depth': {'min': 3, 'max': 20, 'type': int},
    'rf_min_samples_leaf': {'min': 1, 'max': 20, 'type': int},
    'rf_min_samples_split': {'min': 2, 'max': 20, 'type': int},
    
    # XGBoost
    'xgb_n_estimators': {'min': 50, 'max': 800, 'type': int},
    'xgb_max_depth': {'min': 3, 'max': 15, 'type': int},
    'xgb_learning_rate': {'min': 0.01, 'max': 0.3, 'type': float},
    'xgb_subsample': {'min': 0.5, 'max': 1.0, 'type': float},
    'xgb_colsample': {'min': 0.5, 'max': 1.0, 'type': float},
    'xgb_reg_lambda': {'min': 0.0, 'max': 10.0, 'type': float},
    'xgb_reg_alpha': {'min': 0.0, 'max': 5.0, 'type': float},
    'xgb_min_child_weight': {'min': 1.0, 'max': 10.0, 'type': float},
    'xgb_gamma': {'min': 0.0, 'max': 1.0, 'type': float},
    
    # Bagging
    'bag_n_estimators': {'min': 10, 'max': 100, 'type': int},
    'bag_max_samples': {'min': 0.3, 'max': 1.0, 'type': float},
    'bag_max_features': {'min': 0.3, 'max': 1.0, 'type': float},
    
    # 策略参数
    'threshold_buy': {'min': 0.50, 'max': 0.75, 'type': float},
    'threshold_sell': {'min': 0.25, 'max': 0.50, 'type': float},
    'position_size': {'min': 0.1, 'max': 0.6, 'type': float},
    'stop_loss': {'min': 0.02, 'max': 0.15, 'type': float},
    'take_profit': {'min': 0.05, 'max': 0.30, 'type': float},
}

# Prompt 模板
SYSTEM_PROMPT = """你是一个专业的算法交易策略优化专家，专注于期货市场量化交易。
你的任务是分析策略的回测结果和交易记录，找出问题并提出参数优化建议。

你需要：
1. 分析当前策略的表现指标（夏普比率、最大回撤、胜率等）
2. 研究亏损交易的模式和原因
3. 提出具体的参数调整建议
4. 确保建议的参数在合理范围内

重要提示：
- 输出必须是纯 JSON 格式，不包含任何 Markdown 格式或代码块标记
- 参数调整应该是渐进的，每次调整幅度不宜过大
- 需要平衡收益和风险，不能只追求高收益而忽略风险控制
"""

# 高级 Prompt 模板（包含代码级建议）
ADVANCED_SYSTEM_PROMPT = """你是一个专业的算法交易策略优化专家，专注于期货市场量化交易。
你不仅可以调整参数，还可以对特征工程提出建议。

你需要：
1. 分析当前策略的表现指标
2. 研究亏损交易的模式和原因
3. 提出参数调整建议
4. 分析特征工程配置，判断是否存在过拟合或欠拟合
5. 提出特征选择建议（如调整滞后期数、动量窗口等）

重要提示：
- 输出必须是纯 JSON 格式
- 特征建议应该基于对交易模式的分析
- 如果发现过拟合迹象（如训练表现好但回测差），建议减少特征复杂度
"""

ANALYSIS_PROMPT_TEMPLATE = """
【策略背景】
目前策略运行在 WTI 原油期货 (CL=F) 上，使用的是集成机器学习模型 (Random Forest + XGBoost + Bagging)。

【当前表现】
- 总收益率: {total_return:.2f}%
- 年化收益率: {annual_return:.2f}%
- 夏普比率: {sharpe_ratio:.4f}
- 索提诺比率: {sortino_ratio:.4f}
- 最大回撤: {max_drawdown:.2f}%
- 胜率: {win_rate:.2f}%
- 交易次数: {trades_count}
- 盈亏比: {profit_factor:.2f}
- 最佳单笔交易: {best_trade:.2f}%
- 最差单笔交易: {worst_trade:.2f}%
- 买入持有收益: {buy_hold_return:.2f}%

【当前参数配置】
{current_config}

【亏损交易分析（前{n_loss}大亏损）】
{loss_trades}

【盈利交易分析（前{n_profit}大盈利）】
{profit_trades}

【优化目标】
{optimization_goal}

【任务】
请分析上述数据，找出策略的问题所在，并给出参数优化建议。

你的回复必须是纯 JSON 格式（不要用 Markdown 代码块包裹），结构如下：
{{
    "analysis": {{
        "current_assessment": "对当前策略表现的评估",
        "loss_pattern": "亏损交易的主要模式和原因",
        "improvement_areas": ["改进点1", "改进点2", ...]
    }},
    "reasoning": "参数调整的详细理由说明",
    "params": {{
        "threshold_buy": <新值>,
        "threshold_sell": <新值>,
        "stop_loss": <新值>,
        "take_profit": <新值>,
        "position_size": <新值>,
        "n_features": <新值>,
        ...其他需要调整的参数
    }},
    "expected_improvement": "预期的改进效果说明",
    "risk_warning": "调整后需要注意的风险点"
}}
"""

# 高级分析 Prompt（包含特征工程建议）
ADVANCED_ANALYSIS_PROMPT_TEMPLATE = """
【策略背景】
目前策略运行在 WTI 原油期货 (CL=F) 上，使用的是集成机器学习模型 (Random Forest + XGBoost + Bagging)。

【当前表现】
- 总收益率: {total_return:.2f}%
- 年化收益率: {annual_return:.2f}%
- 夏普比率: {sharpe_ratio:.4f}
- 索提诺比率: {sortino_ratio:.4f}
- 最大回撤: {max_drawdown:.2f}%
- 胜率: {win_rate:.2f}%
- 交易次数: {trades_count}
- 盈亏比: {profit_factor:.2f}
- 买入持有收益: {buy_hold_return:.2f}%

【当前参数配置】
{current_config}

【特征工程配置详情】
- 滞后期数 (lag_periods): {lag_periods}
- 动量窗口 (momentum_windows): {momentum_windows}
- 波动率窗口 (volatility_windows): {volatility_windows}
- 选择特征数 (n_features): {n_features}
- 预测窗口 (prediction_horizon): {prediction_horizon}

【亏损交易分析（前{n_loss}大亏损）】
{loss_trades}

【盈利交易分析（前{n_profit}大盈利）】
{profit_trades}

【优化目标】
{optimization_goal}

【任务】
请深入分析上述数据，不仅给出参数建议，还要分析特征工程是否存在问题。

你的回复必须是纯 JSON 格式：
{{
    "analysis": {{
        "current_assessment": "对当前策略表现的评估",
        "loss_pattern": "亏损交易的主要模式和原因",
        "overfitting_risk": "过拟合风险评估（高/中/低）",
        "feature_assessment": "对当前特征工程的评估",
        "improvement_areas": ["改进点1", "改进点2", ...]
    }},
    "reasoning": "参数和特征调整的详细理由说明",
    "params": {{
        "threshold_buy": <新值>,
        "threshold_sell": <新值>,
        "stop_loss": <新值>,
        "take_profit": <新值>,
        "position_size": <新值>,
        "n_features": <新值>,
        "lag_periods": <新值>,
        "momentum_windows": [<窗口列表>],
        "volatility_windows": [<窗口列表>],
        "prediction_horizon": <新值>,
        ...其他参数
    }},
    "feature_recommendations": {{
        "reduce_complexity": <true/false>,
        "suggested_lag_periods": <建议的滞后期数>,
        "rationale": "特征调整的理由"
    }},
    "expected_improvement": "预期的改进效果说明",
    "risk_warning": "调整后需要注意的风险点"
}}
"""


class ContextReader:
    """上下文读取器 - 读取回测结果和配置文件"""
    
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
    
    def load_json(self, filename: str) -> Optional[Dict]:
        """加载 JSON 文件"""
        filepath = os.path.join(self.output_dir, filename)
        if not os.path.exists(filepath):
            logger.warning(f"文件不存在: {filepath}")
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载文件失败 {filepath}: {e}")
            return None
    
    def load_metrics(self) -> Optional[Dict]:
        """加载性能指标"""
        return self.load_json('metrics.json')
    
    def load_trades(self) -> Optional[List[Dict]]:
        """加载交易记录"""
        return self.load_json('trades.json')
    
    def load_config(self) -> Optional[Dict]:
        """加载当前配置"""
        return self.load_json('current_config.json')
    
    def get_top_loss_trades(self, trades: List[Dict], n: int = 10) -> List[Dict]:
        """获取前 N 大亏损交易"""
        if not trades:
            return []
        loss_trades = [t for t in trades if t.get('PnL', 0) < 0]
        loss_trades.sort(key=lambda x: x.get('PnL', 0))
        return loss_trades[:n]
    
    def get_top_profit_trades(self, trades: List[Dict], n: int = 10) -> List[Dict]:
        """获取前 N 大盈利交易"""
        if not trades:
            return []
        profit_trades = [t for t in trades if t.get('PnL', 0) > 0]
        profit_trades.sort(key=lambda x: x.get('PnL', 0), reverse=True)
        return profit_trades[:n]
    
    def format_trades_for_prompt(self, trades: List[Dict]) -> str:
        """格式化交易记录用于 Prompt"""
        if not trades:
            return "无数据"
        
        formatted = []
        for i, t in enumerate(trades, 1):
            trade_type = t.get('TradeType', 'LONG' if t.get('Size', 0) > 0 else 'SHORT')
            formatted.append(
                f"  {i}. [{trade_type}] 入场: {t.get('EntryTime', 'N/A')} @ ${t.get('EntryPrice', 0):.2f}, "
                f"出场: {t.get('ExitTime', 'N/A')} @ ${t.get('ExitPrice', 0):.2f}, "
                f"盈亏: ${t.get('PnL', 0):.2f} ({t.get('ReturnPct', 0)*100:.2f}%), "
                f"持仓时长: {t.get('Duration', 0)//86400000}天, "
                f"入场信号: {t.get('Entry_Forecast', 0):.4f}"
            )
        return '\n'.join(formatted)
    
    def format_config_for_prompt(self, config: Dict) -> str:
        """格式化配置用于 Prompt"""
        if not config:
            return "无配置数据"
        
        lines = []
        
        # 策略参数
        strategy = config.get('strategy_config', {})
        lines.append("策略参数:")
        lines.append(f"  - 买入阈值: {strategy.get('threshold_buy', 'N/A')}")
        lines.append(f"  - 卖出阈值: {strategy.get('threshold_sell', 'N/A')}")
        lines.append(f"  - 止损: {strategy.get('stop_loss', 'N/A')}")
        lines.append(f"  - 止盈: {strategy.get('take_profit', 'N/A')}")
        lines.append(f"  - 仓位比例: {strategy.get('position_size', 'N/A')}")
        
        # 特征配置
        feature = config.get('feature_config', {})
        lines.append("\n特征配置:")
        lines.append(f"  - 特征数量: {feature.get('n_features', 'N/A')}")
        lines.append(f"  - 预测窗口: {feature.get('prediction_horizon', 'N/A')}")
        lines.append(f"  - 滞后期数: {feature.get('lag_periods', 'N/A')}")
        
        # 模型配置
        model = config.get('model_config', {})
        rf = model.get('rf', {})
        xgb = model.get('xgb', {})
        weights = model.get('ensemble_weights', {})
        
        lines.append("\n模型配置:")
        lines.append(f"  RF: n_estimators={rf.get('n_estimators')}, max_depth={rf.get('max_depth')}")
        lines.append(f"  XGB: n_estimators={xgb.get('n_estimators')}, max_depth={xgb.get('max_depth')}, lr={xgb.get('learning_rate')}")
        lines.append(f"  集成权重: RF={weights.get('rf', 0):.2f}, XGB={weights.get('xgb', 0):.2f}, Bagging={weights.get('bagging', 0):.2f}")
        
        return '\n'.join(lines)


class LLMClient:
    """LLM 客户端 - 支持 OpenAI 和 Anthropic"""
    
    def __init__(self, config: Dict = None):
        self.config = config or LLM_CONFIG
        self.provider = self.config.get('provider', 'openai')
        self._client = None
    
    def _get_openai_client(self):
        """获取 OpenAI 客户端"""
        try:
            from openai import OpenAI
            cfg = self.config['openai']
            return OpenAI(
                api_key=cfg['api_key'],
                base_url=cfg.get('base_url')
            )
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")
    
    def _get_anthropic_client(self):
        """获取 Anthropic 客户端"""
        try:
            import anthropic
            cfg = self.config['anthropic']
            return anthropic.Anthropic(api_key=cfg['api_key'])
        except ImportError:
            raise ImportError("请安装 anthropic: pip install anthropic")
    
    def chat_completion(self, system_prompt: str, user_prompt: str) -> str:
        """调用 LLM 生成回复"""
        if self.provider == 'openai':
            return self._openai_completion(system_prompt, user_prompt)
        elif self.provider == 'anthropic':
            return self._anthropic_completion(system_prompt, user_prompt)
        elif self.provider == 'qwen':
            return self._qwen_completion(system_prompt, user_prompt)
        else:
            raise ValueError(f"不支持的 LLM 提供商: {self.provider}")
    
    def _openai_completion(self, system_prompt: str, user_prompt: str) -> str:
        """OpenAI API 调用"""
        client = self._get_openai_client()
        cfg = self.config['openai']
        
        response = client.chat.completions.create(
            model=cfg['model'],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=cfg['max_tokens'],
            temperature=cfg['temperature']
        )
        
        return response.choices[0].message.content
    
    def _anthropic_completion(self, system_prompt: str, user_prompt: str) -> str:
        """Anthropic API 调用"""
        client = self._get_anthropic_client()
        cfg = self.config['anthropic']
        
        response = client.messages.create(
            model=cfg['model'],
            max_tokens=cfg['max_tokens'],
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        return response.content[0].text
    
    def _qwen_completion(self, system_prompt: str, user_prompt: str) -> str:
        """Qwen (通义千问) API 调用 - 使用 OpenAI 兼容模式"""
        try:
            from openai import OpenAI
            cfg = self.config['qwen']
            client = OpenAI(
                api_key=cfg['api_key'],
                base_url=cfg['base_url']
            )
            
            response = client.chat.completions.create(
                model=cfg['model'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=cfg['max_tokens'],
                temperature=cfg['temperature']
            )
            
            return response.choices[0].message.content
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")


class ParamValidator:
    """参数验证器 - 防止 LLM 幻觉"""
    
    def __init__(self, bounds: Dict = None):
        self.bounds = bounds or PARAM_BOUNDS
    
    def validate_and_clip(self, params: Dict) -> Tuple[Dict, List[str]]:
        """
        验证并修正参数
        
        Args:
            params: LLM 建议的参数字典
            
        Returns:
            (修正后的参数, 警告列表)
        """
        validated = {}
        warnings = []
        
        for key, value in params.items():
            if key not in self.bounds:
                # 未知参数，保留原值但记录警告
                validated[key] = value
                warnings.append(f"未知参数 '{key}'，保留原值")
                continue
            
            bound = self.bounds[key]
            param_type = bound['type']
            min_val = bound['min']
            max_val = bound['max']
            
            # 类型转换
            try:
                if param_type == int:
                    value = int(round(value))
                elif param_type == float:
                    value = float(value)
            except (ValueError, TypeError):
                warnings.append(f"参数 '{key}' 类型转换失败，使用边界中值")
                value = (min_val + max_val) / 2
                if param_type == int:
                    value = int(value)
            
            # 边界检查
            original = value
            if value < min_val:
                value = min_val
                warnings.append(f"参数 '{key}' 值 {original} 低于最小值，修正为 {min_val}")
            elif value > max_val:
                value = max_val
                warnings.append(f"参数 '{key}' 值 {original} 高于最大值，修正为 {max_val}")
            
            validated[key] = value
        
        return validated, warnings
    
    def check_logical_constraints(self, params: Dict) -> List[str]:
        """检查参数的逻辑约束"""
        warnings = []
        
        # 买入阈值应大于卖出阈值
        buy = params.get('threshold_buy')
        sell = params.get('threshold_sell')
        if buy is not None and sell is not None and buy <= sell:
            warnings.append(f"买入阈值 ({buy}) 应大于卖出阈值 ({sell})")
        
        # 止盈应大于止损
        sl = params.get('stop_loss')
        tp = params.get('take_profit')
        if sl is not None and tp is not None and tp <= sl:
            warnings.append(f"止盈 ({tp}) 应大于止损 ({sl})")
        
        return warnings


class StrategyOptimizer:
    """策略优化器 - 核心类"""
    
    def __init__(self, 
                 output_dir: str = 'output',
                 llm_config: Dict = None):
        """
        初始化策略优化器
        
        Args:
            output_dir: 输出目录
            llm_config: LLM 配置
        """
        self.output_dir = output_dir
        self.context_reader = ContextReader(output_dir)
        self.llm_client = LLMClient(llm_config)
        self.validator = ParamValidator()
        
        # 优化历史
        self.optimization_history = []
    
    def analyze_and_suggest(self,
                            optimization_goal: str = "提高夏普比率，同时控制最大回撤在 10% 以内",
                            n_loss_trades: int = 10,
                            n_profit_trades: int = 5) -> Optional[Dict]:
        """
        分析当前策略并生成优化建议
        
        Args:
            optimization_goal: 优化目标描述
            n_loss_trades: 分析的亏损交易数量
            n_profit_trades: 分析的盈利交易数量
            
        Returns:
            包含分析结果和建议参数的字典
        """
        logger.info("开始策略分析...")
        
        # 1. 读取当前状态
        metrics = self.context_reader.load_metrics()
        trades = self.context_reader.load_trades()
        config = self.context_reader.load_config()
        
        if metrics is None:
            logger.error("无法读取性能指标文件，请先运行回测")
            return None
        
        # 2. 准备交易分析数据
        loss_trades = self.context_reader.get_top_loss_trades(trades or [], n_loss_trades)
        profit_trades = self.context_reader.get_top_profit_trades(trades or [], n_profit_trades)
        
        # 3. 构建 Prompt
        user_prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            total_return=metrics.get('total_return', 0),
            annual_return=metrics.get('annual_return', 0) or 0,
            sharpe_ratio=metrics.get('sharpe_ratio', 0) or 0,
            sortino_ratio=metrics.get('sortino_ratio', 0) or 0,
            max_drawdown=metrics.get('max_drawdown', 0),
            win_rate=metrics.get('win_rate', 0),
            trades_count=metrics.get('trades_count', 0),
            profit_factor=metrics.get('profit_factor', 0) or 0,
            best_trade=metrics.get('best_trade', 0),
            worst_trade=metrics.get('worst_trade', 0),
            buy_hold_return=metrics.get('buy_hold_return', 0),
            current_config=self.context_reader.format_config_for_prompt(config),
            n_loss=len(loss_trades),
            loss_trades=self.context_reader.format_trades_for_prompt(loss_trades),
            n_profit=len(profit_trades),
            profit_trades=self.context_reader.format_trades_for_prompt(profit_trades),
            optimization_goal=optimization_goal
        )
        
        # 4. 调用 LLM
        logger.info("调用 LLM 进行分析...")
        try:
            response = self.llm_client.chat_completion(SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return None
        
        # 5. 解析 JSON 响应
        result = self._parse_llm_response(response)
        if result is None:
            return None
        
        # 6. 验证和修正参数
        if 'params' in result:
            validated_params, warnings = self.validator.validate_and_clip(result['params'])
            logical_warnings = self.validator.check_logical_constraints(validated_params)
            
            result['params'] = validated_params
            result['validation_warnings'] = warnings + logical_warnings
            
            if warnings or logical_warnings:
                logger.warning(f"参数验证警告: {warnings + logical_warnings}")
        
        # 7. 记录优化历史
        self._record_optimization(metrics, result)
        
        return result
    
    def _parse_llm_response(self, response: str) -> Optional[Dict]:
        """解析 LLM 的 JSON 响应"""
        try:
            # 尝试直接解析
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # 尝试从 Markdown 代码块中提取
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 尝试找到 JSON 对象的边界
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(response[start:end+1])
            except json.JSONDecodeError:
                pass
        
        logger.error(f"无法解析 LLM 响应为 JSON:\n{response[:500]}...")
        return None
    
    def _record_optimization(self, metrics: Dict, result: Dict):
        """记录优化历史"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'metrics_before': metrics,
            'suggestion': result,
        }
        self.optimization_history.append(record)
    
    def save_suggested_params(self, result: Dict, 
                               output_path: str = 'output/best_params.json') -> bool:
        """
        将 LLM 建议的参数保存到文件
        
        Args:
            result: analyze_and_suggest 的返回结果
            output_path: 输出路径
            
        Returns:
            是否保存成功
        """
        if result is None or 'params' not in result:
            logger.error("无有效参数可保存")
            return False
        
        # 构建与 config.py 兼容的参数格式
        params_to_save = {
            'timestamp': datetime.now().isoformat(),
            'source': 'agent_optimizer',
            'reasoning': result.get('reasoning', ''),
            'analysis': result.get('analysis', {}),
            'expected_improvement': result.get('expected_improvement', ''),
            'risk_warning': result.get('risk_warning', ''),
            'validation_warnings': result.get('validation_warnings', []),
            'params': result['params'],
        }
        
        # 备份现有文件
        if os.path.exists(output_path):
            backup_path = output_path.replace('.json', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            try:
                import shutil
                shutil.copy(output_path, backup_path)
                logger.info(f"已备份原参数文件到 {backup_path}")
            except Exception as e:
                logger.warning(f"备份失败: {e}")
        
        # 保存新参数
        try:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(params_to_save, f, indent=2, ensure_ascii=False)
            logger.info(f"新参数已保存到 {output_path}")
            return True
        except Exception as e:
            logger.error(f"保存参数失败: {e}")
            return False
    
    def get_optimization_history(self) -> List[Dict]:
        """获取优化历史"""
        return self.optimization_history
    
    def analyze_with_feature_suggestions(self,
                                          optimization_goal: str = "提高夏普比率，同时控制最大回撤",
                                          n_loss_trades: int = 10,
                                          n_profit_trades: int = 5) -> Optional[Dict]:
        """
        高级分析 - 包含特征工程建议
        
        Args:
            optimization_goal: 优化目标描述
            n_loss_trades: 分析的亏损交易数量
            n_profit_trades: 分析的盈利交易数量
            
        Returns:
            包含分析结果、参数建议和特征工程建议的字典
        """
        logger.info("开始高级策略分析（含特征工程建议）...")
        
        # 1. 读取当前状态
        metrics = self.context_reader.load_metrics()
        trades = self.context_reader.load_trades()
        config = self.context_reader.load_config()
        
        if metrics is None:
            logger.error("无法读取性能指标文件，请先运行回测")
            return None
        
        # 2. 准备交易分析数据
        loss_trades = self.context_reader.get_top_loss_trades(trades or [], n_loss_trades)
        profit_trades = self.context_reader.get_top_profit_trades(trades or [], n_profit_trades)
        
        # 3. 提取特征配置
        feature_config = config.get('feature_config', {}) if config else {}
        
        # 4. 构建高级 Prompt
        user_prompt = ADVANCED_ANALYSIS_PROMPT_TEMPLATE.format(
            total_return=metrics.get('total_return', 0),
            annual_return=metrics.get('annual_return', 0) or 0,
            sharpe_ratio=metrics.get('sharpe_ratio', 0) or 0,
            sortino_ratio=metrics.get('sortino_ratio', 0) or 0,
            max_drawdown=metrics.get('max_drawdown', 0),
            win_rate=metrics.get('win_rate', 0),
            trades_count=metrics.get('trades_count', 0),
            profit_factor=metrics.get('profit_factor', 0) or 0,
            buy_hold_return=metrics.get('buy_hold_return', 0),
            current_config=self.context_reader.format_config_for_prompt(config),
            lag_periods=feature_config.get('lag_periods', 60),
            momentum_windows=feature_config.get('momentum_windows', [1, 3, 5, 10, 20]),
            volatility_windows=feature_config.get('volatility_windows', [5, 10, 20]),
            n_features=feature_config.get('n_features', 70),
            prediction_horizon=feature_config.get('prediction_horizon', 5),
            n_loss=len(loss_trades),
            loss_trades=self.context_reader.format_trades_for_prompt(loss_trades),
            n_profit=len(profit_trades),
            profit_trades=self.context_reader.format_trades_for_prompt(profit_trades),
            optimization_goal=optimization_goal
        )
        
        # 5. 调用 LLM
        logger.info("调用 LLM 进行高级分析...")
        try:
            response = self.llm_client.chat_completion(ADVANCED_SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return None
        
        # 6. 解析 JSON 响应
        result = self._parse_llm_response(response)
        if result is None:
            return None
        
        # 7. 验证和修正参数
        if 'params' in result:
            validated_params, warnings = self.validator.validate_and_clip(result['params'])
            logical_warnings = self.validator.check_logical_constraints(validated_params)
            
            result['params'] = validated_params
            result['validation_warnings'] = warnings + logical_warnings
        
        # 8. 记录优化历史
        self._record_optimization(metrics, result)
        
        return result


class ErrorRecovery:
    """错误恢复机制"""
    
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        self.backup_dir = os.path.join(output_dir, 'backups')
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def backup_current_params(self) -> Optional[str]:
        """
        备份当前参数文件
        
        Returns:
            备份文件路径
        """
        params_path = os.path.join(self.output_dir, 'best_params.json')
        if not os.path.exists(params_path):
            logger.warning("无参数文件可备份")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.backup_dir, f'best_params_{timestamp}.json')
        
        try:
            shutil.copy(params_path, backup_path)
            logger.info(f"参数已备份到 {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"备份失败: {e}")
            return None
    
    def list_backups(self) -> List[Dict]:
        """列出所有备份"""
        backups = []
        for f in os.listdir(self.backup_dir):
            if f.startswith('best_params_') and f.endswith('.json'):
                path = os.path.join(self.backup_dir, f)
                stat = os.stat(path)
                backups.append({
                    'filename': f,
                    'path': path,
                    'timestamp': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'size': stat.st_size
                })
        
        # 按时间排序
        backups.sort(key=lambda x: x['timestamp'], reverse=True)
        return backups
    
    def restore_backup(self, backup_path: str = None) -> bool:
        """
        恢复备份
        
        Args:
            backup_path: 备份文件路径，如果为 None 则恢复最近的备份
            
        Returns:
            是否成功
        """
        if backup_path is None:
            backups = self.list_backups()
            if not backups:
                logger.error("无备份可恢复")
                return False
            backup_path = backups[0]['path']
        
        if not os.path.exists(backup_path):
            logger.error(f"备份文件不存在: {backup_path}")
            return False
        
        params_path = os.path.join(self.output_dir, 'best_params.json')
        
        try:
            shutil.copy(backup_path, params_path)
            logger.info(f"已从 {backup_path} 恢复参数")
            return True
        except Exception as e:
            logger.error(f"恢复失败: {e}")
            return False
    
    def validate_params_file(self) -> Tuple[bool, str]:
        """
        验证参数文件是否有效
        
        Returns:
            (是否有效, 错误信息)
        """
        params_path = os.path.join(self.output_dir, 'best_params.json')
        
        if not os.path.exists(params_path):
            return True, "参数文件不存在，将使用默认配置"
        
        try:
            with open(params_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'params' not in data:
                return False, "参数文件缺少 'params' 字段"
            
            # 基本参数检查
            params = data['params']
            required_keys = ['threshold_buy', 'stop_loss']
            for key in required_keys:
                if key not in params:
                    return False, f"参数文件缺少必需字段: {key}"
            
            return True, "参数文件有效"
            
        except json.JSONDecodeError as e:
            return False, f"JSON 解析错误: {e}"
        except Exception as e:
            return False, f"验证错误: {e}"
    
    def safe_run_with_recovery(self, func, *args, **kwargs):
        """
        安全运行函数，出错时自动恢复
        
        Args:
            func: 要执行的函数
            *args, **kwargs: 函数参数
            
        Returns:
            函数返回值或 None
        """
        # 先备份
        backup_path = self.backup_current_params()
        
        try:
            result = func(*args, **kwargs)
            
            # 验证结果
            is_valid, msg = self.validate_params_file()
            if not is_valid:
                logger.error(f"参数文件验证失败: {msg}")
                if backup_path:
                    self.restore_backup(backup_path)
                return None
            
            return result
            
        except Exception as e:
            logger.error(f"执行出错: {e}")
            if backup_path:
                logger.info("正在恢复到备份状态...")
                self.restore_backup(backup_path)
            return None


# 导入 shutil 用于文件操作
import shutil


def main():
    """命令行入口"""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='LLM 策略优化器')
    parser.add_argument('--output-dir', default='output', help='输出目录')
    parser.add_argument('--goal', default='提高夏普比率，同时控制最大回撤在 10% 以内', 
                        help='优化目标')
    parser.add_argument('--save', action='store_true', help='是否保存建议参数')
    parser.add_argument('--provider', choices=['openai', 'anthropic', 'qwen'], 
                        default='qwen', help='LLM 提供商（默认: qwen）')
    
    args = parser.parse_args()
    
    # 检查 API Key
    if args.provider == 'openai' and not os.environ.get('OPENAI_API_KEY'):
        print("请设置环境变量 OPENAI_API_KEY")
        return
    elif args.provider == 'anthropic' and not os.environ.get('ANTHROPIC_API_KEY'):
        print("请设置环境变量 ANTHROPIC_API_KEY")
        return
    elif args.provider == 'qwen' and not os.environ.get('DASHSCOPE_API_KEY'):
        print("请设置环境变量 DASHSCOPE_API_KEY")
        print("获取方式: https://dashscope.console.aliyun.com/apiKey")
        return
    
    # 更新配置
    LLM_CONFIG['provider'] = args.provider
    
    # 创建优化器
    optimizer = StrategyOptimizer(output_dir=args.output_dir)
    
    print("\n" + "="*60)
    print("LLM 策略优化器")
    print("="*60)
    
    # 运行分析
    result = optimizer.analyze_and_suggest(optimization_goal=args.goal)
    
    if result:
        print("\n【分析结果】")
        if 'analysis' in result:
            analysis = result['analysis']
            print(f"\n当前评估: {analysis.get('current_assessment', 'N/A')}")
            print(f"\n亏损模式: {analysis.get('loss_pattern', 'N/A')}")
            print(f"\n改进建议: {analysis.get('improvement_areas', [])}")
        
        print(f"\n【优化理由】\n{result.get('reasoning', 'N/A')}")
        
        print("\n【建议参数】")
        for key, value in result.get('params', {}).items():
            print(f"  {key}: {value}")
        
        if result.get('validation_warnings'):
            print("\n【参数验证警告】")
            for w in result['validation_warnings']:
                print(f"  ⚠️ {w}")
        
        print(f"\n【预期改进】\n{result.get('expected_improvement', 'N/A')}")
        print(f"\n【风险提示】\n{result.get('risk_warning', 'N/A')}")
        
        # 保存参数
        if args.save:
            if optimizer.save_suggested_params(result):
                print("\n✅ 参数已保存，下次运行 main.py 将自动使用新参数")
            else:
                print("\n❌ 参数保存失败")
    else:
        print("\n❌ 分析失败，请检查日志")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
