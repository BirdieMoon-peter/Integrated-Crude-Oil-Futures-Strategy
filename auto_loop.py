"""
自动优化循环模块 - 串联整个 Agentic Workflow
原油期货多模型集成投资策略

该模块实现了完整的自动化优化循环：
1. 运行 main.py 进行训练和回测
2. 运行 agent_optimizer.py 分析结果并生成新参数
3. 检查停止条件
4. 循环迭代直到达到目标或停止条件
"""

import os
import sys
import json
import csv
import logging
import subprocess
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import time

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_optimizer import StrategyOptimizer, LLM_CONFIG

logger = logging.getLogger(__name__)

# ============================================================================
# 配置常量
# ============================================================================

LOOP_CONFIG = {
    # 停止条件
    'max_rounds': 10,                    # 最大迭代轮数
    'no_improvement_rounds': 3,           # 连续多少轮无提升则停止
    'target_sharpe': 2.5,                 # 目标夏普比率
    'max_drawdown_limit': -15.0,          # 最大回撤限制（超过则停止）
    
    # 改进阈值
    'min_sharpe_improvement': 0.05,       # 最小夏普比率改进幅度
    
    # 输出配置
    'output_dir': 'output',
    'log_file': 'output/optimization_log.csv',
    'history_file': 'output/optimization_history.json',
    
    # 执行配置
    'python_executable': sys.executable,  # Python 解释器路径
    'main_script': 'main.py',            # 主脚本
    'show_plots': False,                  # 是否显示图表
    
    # 优化目标
    'optimization_goal': '提高夏普比率至 2.0 以上，同时将最大回撤控制在 10% 以内',
}


class OptimizationLogger:
    """优化日志记录器"""
    
    def __init__(self, log_file: str, history_file: str):
        self.log_file = log_file
        self.history_file = history_file
        self.history = []
        
        # 加载现有历史
        self._load_history()
    
    def _load_history(self):
        """加载现有历史记录"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
            except Exception as e:
                logger.warning(f"加载历史记录失败: {e}")
                self.history = []
    
    def _ensure_csv_header(self):
        """确保 CSV 文件有表头"""
        if not os.path.exists(self.log_file):
            os.makedirs(os.path.dirname(self.log_file) or '.', exist_ok=True)
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Round', 'Timestamp', 'Sharpe_Ratio', 'Total_Return',
                    'Max_Drawdown', 'Win_Rate', 'Trades_Count',
                    'Improvement', 'Key_Params', 'Reasoning'
                ])
    
    def log_round(self, round_id: int, metrics: Dict, params: Dict,
                  improvement: float, reasoning: str):
        """记录一轮优化结果"""
        self._ensure_csv_header()
        
        timestamp = datetime.now().isoformat()
        
        # 提取关键参数
        key_params = {
            'threshold_buy': params.get('threshold_buy'),
            'stop_loss': params.get('stop_loss'),
            'take_profit': params.get('take_profit'),
        }
        
        # 写入 CSV
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                round_id,
                timestamp,
                f"{metrics.get('sharpe_ratio', 0):.4f}",
                f"{metrics.get('total_return', 0):.2f}",
                f"{metrics.get('max_drawdown', 0):.2f}",
                f"{metrics.get('win_rate', 0):.2f}",
                metrics.get('trades_count', 0),
                f"{improvement:.4f}",
                json.dumps(key_params),
                reasoning[:200] if reasoning else ''
            ])
        
        # 添加到历史
        record = {
            'round_id': round_id,
            'timestamp': timestamp,
            'metrics': metrics,
            'params': params,
            'improvement': improvement,
            'reasoning': reasoning,
        }
        self.history.append(record)
        
        # 保存历史 JSON
        self._save_history()
    
    def _save_history(self):
        """保存历史记录到 JSON"""
        try:
            os.makedirs(os.path.dirname(self.history_file) or '.', exist_ok=True)
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存历史记录失败: {e}")
    
    def get_best_round(self) -> Optional[Dict]:
        """获取最佳轮次"""
        if not self.history:
            return None
        
        best = max(self.history, 
                   key=lambda x: x['metrics'].get('sharpe_ratio', 0) or 0)
        return best
    
    def get_recent_improvements(self, n: int = 3) -> List[float]:
        """获取最近 N 轮的改进幅度"""
        if len(self.history) < n:
            return [h.get('improvement', 0) for h in self.history]
        return [h.get('improvement', 0) for h in self.history[-n:]]


class AutoOptimizationLoop:
    """自动优化循环控制器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化自动优化循环
        
        Args:
            config: 循环配置
        """
        self.config = config or LOOP_CONFIG
        self.output_dir = self.config['output_dir']
        
        # 初始化日志记录器
        self.logger = OptimizationLogger(
            self.config['log_file'],
            self.config['history_file']
        )
        
        # 初始化优化器
        self.optimizer = StrategyOptimizer(output_dir=self.output_dir)
        
        # 状态
        self.current_round = len(self.logger.history)
        self.best_sharpe = self._get_current_best_sharpe()
        self.consecutive_no_improvement = 0
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _get_current_best_sharpe(self) -> float:
        """获取当前最佳夏普比率"""
        best = self.logger.get_best_round()
        if best:
            return best['metrics'].get('sharpe_ratio', 0) or 0
        return 0
    
    def run_main_pipeline(self) -> bool:
        """
        运行主程序管道
        
        Returns:
            是否成功
        """
        logger.info("运行主程序管道...")
        
        try:
            # 导出当前配置
            from config import export_current_config
            export_current_config(os.path.join(self.output_dir, 'current_config.json'))
        except Exception as e:
            logger.warning(f"导出配置失败: {e}")
        
        # 构建命令
        cmd = [
            self.config['python_executable'],
            self.config['main_script']
        ]
        
        # 设置环境变量以禁用图表显示
        env = os.environ.copy()
        if not self.config.get('show_plots', False):
            env['MPLBACKEND'] = 'Agg'  # 使用非交互式后端
        
        try:
            # 运行主程序
            result = subprocess.run(
                cmd,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                env=env,
                capture_output=True,
                text=True,
                timeout=600  # 10 分钟超时
            )
            
            if result.returncode != 0:
                logger.error(f"主程序运行失败:\n{result.stderr}")
                return False
            
            logger.info("主程序运行成功")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("主程序运行超时")
            return False
        except Exception as e:
            logger.error(f"运行主程序出错: {e}")
            return False
    
    def load_current_metrics(self) -> Optional[Dict]:
        """加载当前回测指标"""
        metrics_path = os.path.join(self.output_dir, 'metrics.json')
        if not os.path.exists(metrics_path):
            logger.error(f"指标文件不存在: {metrics_path}")
            return None
        
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载指标文件失败: {e}")
            return None
    
    def load_current_params(self) -> Optional[Dict]:
        """加载当前参数"""
        params_path = os.path.join(self.output_dir, 'best_params.json')
        if not os.path.exists(params_path):
            return {}
        
        try:
            with open(params_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('params', {})
        except Exception as e:
            logger.warning(f"加载参数文件失败: {e}")
            return {}
    
    def check_stop_conditions(self, metrics: Dict) -> Tuple[bool, str]:
        """
        检查停止条件
        
        Args:
            metrics: 当前指标
            
        Returns:
            (是否停止, 原因)
        """
        sharpe = metrics.get('sharpe_ratio', 0) or 0
        max_dd = metrics.get('max_drawdown', 0)
        
        # 达到目标夏普比率
        if sharpe >= self.config['target_sharpe']:
            return True, f"已达到目标夏普比率 {self.config['target_sharpe']}"
        
        # 最大回撤超限
        if max_dd < self.config['max_drawdown_limit']:
            return True, f"最大回撤 {max_dd:.2f}% 超过限制 {self.config['max_drawdown_limit']}%"
        
        # 达到最大轮数
        if self.current_round >= self.config['max_rounds']:
            return True, f"已达到最大迭代轮数 {self.config['max_rounds']}"
        
        # 连续无改进
        if self.consecutive_no_improvement >= self.config['no_improvement_rounds']:
            return True, f"连续 {self.consecutive_no_improvement} 轮无显著改进"
        
        return False, ""
    
    def run_optimization_round(self) -> Tuple[bool, str]:
        """
        运行一轮优化
        
        Returns:
            (是否继续, 状态消息)
        """
        self.current_round += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"开始第 {self.current_round} 轮优化")
        logger.info(f"{'='*60}")
        
        # 1. 运行主程序
        if not self.run_main_pipeline():
            return False, "主程序运行失败"
        
        # 2. 加载当前指标
        metrics = self.load_current_metrics()
        if metrics is None:
            return False, "无法加载回测指标"
        
        current_sharpe = metrics.get('sharpe_ratio', 0) or 0
        logger.info(f"当前夏普比率: {current_sharpe:.4f}")
        logger.info(f"历史最佳夏普比率: {self.best_sharpe:.4f}")
        
        # 3. 检查停止条件
        should_stop, reason = self.check_stop_conditions(metrics)
        if should_stop:
            # 记录最后一轮
            params = self.load_current_params()
            self.logger.log_round(
                self.current_round, metrics, params,
                current_sharpe - self.best_sharpe, f"停止: {reason}"
            )
            return False, reason
        
        # 4. 计算改进幅度
        improvement = current_sharpe - self.best_sharpe
        params = self.load_current_params()
        
        if improvement > self.config['min_sharpe_improvement']:
            self.best_sharpe = current_sharpe
            self.consecutive_no_improvement = 0
            logger.info(f"✅ 夏普比率改进 {improvement:.4f}")
            
            # 立即将当前最佳参数保存到 best_params.json
            self._save_best_params(params, metrics, f"第 {self.current_round} 轮优化改进")
        else:
            self.consecutive_no_improvement += 1
            logger.info(f"⚠️ 无显著改进 (连续 {self.consecutive_no_improvement} 轮)")
        
        # 5. 调用 LLM 优化器
        logger.info("调用 LLM 分析并生成优化建议...")
        result = self.optimizer.analyze_and_suggest(
            optimization_goal=self.config['optimization_goal']
        )
        
        if result is None:
            # 记录失败轮次
            params = self.load_current_params()
            self.logger.log_round(
                self.current_round, metrics, params,
                improvement, "LLM 分析失败"
            )
            return False, "LLM 分析失败"
        
        # 6. 保存 LLM 建议的新参数（用于下一轮）
        if not self.optimizer.save_suggested_params(result):
            self.logger.log_round(
                self.current_round, metrics, params,
                improvement, "参数保存失败"
            )
            return False, "参数保存失败"
        
        # 7. 记录本轮结果
        reasoning = result.get('reasoning', '')
        self.logger.log_round(
            self.current_round, metrics, result.get('params', {}),
            improvement, reasoning
        )
        
        logger.info(f"第 {self.current_round} 轮完成，新参数已保存")
        return True, f"夏普比率: {current_sharpe:.4f}, 改进: {improvement:.4f}"
    
    def run(self) -> Dict:
        """
        运行完整的自动优化循环
        
        Returns:
            优化结果摘要
        """
        start_time = datetime.now()
        
        print("\n" + "="*60)
        print("自动策略优化循环")
        print("="*60)
        print(f"目标夏普比率: {self.config['target_sharpe']}")
        print(f"最大迭代轮数: {self.config['max_rounds']}")
        print(f"优化目标: {self.config['optimization_goal']}")
        print("="*60 + "\n")
        
        # 主循环
        while True:
            should_continue, status = self.run_optimization_round()
            
            print(f"\n轮次 {self.current_round} 状态: {status}")
            
            if not should_continue:
                break
            
            # 短暂延迟，避免 API 限流
            time.sleep(2)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # 生成结果摘要
        best_round = self.logger.get_best_round()
        
        result = {
            'total_rounds': self.current_round,
            'duration': str(duration),
            'best_round': best_round,
            'final_sharpe': self.best_sharpe,
            'stop_reason': status,
        }
        
        # 打印结果
        print("\n" + "="*60)
        print("优化完成")
        print("="*60)
        print(f"总轮数: {self.current_round}")
        print(f"耗时: {duration}")
        print(f"最终最佳夏普比率: {self.best_sharpe:.4f}")
        if best_round:
            print(f"最佳轮次: {best_round['round_id']}")
            print(f"最佳回测指标:")
            print(f"  - 总收益率: {best_round['metrics'].get('total_return', 0):.2f}%")
            print(f"  - 最大回撤: {best_round['metrics'].get('max_drawdown', 0):.2f}%")
            print(f"  - 胜率: {best_round['metrics'].get('win_rate', 0):.2f}%")
        print(f"停止原因: {status}")
        print("="*60)
        
        # 确保 best_params.json 保存的是历史最佳参数
        best_round = self.logger.get_best_round()
        if best_round:
            logger.info(f"\n保存历史最佳参数（来自第 {best_round['round_id']} 轮）到 best_params.json...")
            self._save_best_params(
                best_round['params'],
                best_round['metrics'],
                f"优化完成，历史最佳（第 {best_round['round_id']} 轮）"
            )
            print(f"\n✅ 历史最佳参数已保存到 {self.output_dir}/best_params.json")
        
        return result
    
    def _save_best_params(self, params: Dict, metrics: Dict, source: str) -> bool:
        """
        保存最佳参数到 best_params.json
        
        Args:
            params: 参数字典
            metrics: 性能指标
            source: 来源说明
            
        Returns:
            是否成功
        """
        params_to_save = {
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'score': metrics.get('sharpe_ratio', 0),
            'stats': metrics,
            'params': params,
        }
        
        params_path = os.path.join(self.output_dir, 'best_params.json')
        try:
            with open(params_path, 'w', encoding='utf-8') as f:
                json.dump(params_to_save, f, indent=2, ensure_ascii=False)
            logger.info(f"最佳参数已保存: {source}")
            return True
        except Exception as e:
            logger.error(f"保存最佳参数失败: {e}")
            return False
    
    def rollback_to_best(self) -> bool:
        """
        回滚到历史最佳参数
        
        Returns:
            是否成功
        """
        best_round = self.logger.get_best_round()
        if not best_round:
            logger.error("无历史最佳记录")
            return False
        
        return self._save_best_params(
            best_round['params'],
            best_round['metrics'],
            f"回滚到第 {best_round['round_id']} 轮"
        )


def main():
    """命令行入口"""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='自动策略优化循环')
    parser.add_argument('--max-rounds', type=int, default=10, 
                        help='最大迭代轮数')
    parser.add_argument('--target-sharpe', type=float, default=2.5,
                        help='目标夏普比率')
    parser.add_argument('--no-improvement-limit', type=int, default=3,
                        help='连续无改进停止轮数')
    parser.add_argument('--goal', type=str,
                        default='提高夏普比率至 2.0 以上，同时将最大回撤控制在 10% 以内',
                        help='优化目标描述')
    parser.add_argument('--show-plots', action='store_true',
                        help='显示图表')
    parser.add_argument('--rollback', action='store_true',
                        help='回滚到历史最佳参数')
    parser.add_argument('--provider', choices=['openai', 'anthropic', 'qwen'],
                        default='qwen', help='LLM 提供商（默认: qwen）')
    
    args = parser.parse_args()
    
    # 检查 API Key
    if args.provider == 'openai' and not os.environ.get('OPENAI_API_KEY'):
        print("请设置环境变量 OPENAI_API_KEY")
        print("例如: export OPENAI_API_KEY='your-api-key'")
        return
    elif args.provider == 'anthropic' and not os.environ.get('ANTHROPIC_API_KEY'):
        print("请设置环境变量 ANTHROPIC_API_KEY")
        print("例如: export ANTHROPIC_API_KEY='your-api-key'")
        return
    elif args.provider == 'qwen' and not os.environ.get('DASHSCOPE_API_KEY'):
        print("请设置环境变量 DASHSCOPE_API_KEY")
        print("获取方式: https://dashscope.console.aliyun.com/apiKey")
        print("例如: export DASHSCOPE_API_KEY='your-api-key'")
        return
    
    # 更新 LLM 配置
    LLM_CONFIG['provider'] = args.provider
    
    # 更新循环配置
    config = LOOP_CONFIG.copy()
    config['max_rounds'] = args.max_rounds
    config['target_sharpe'] = args.target_sharpe
    config['no_improvement_rounds'] = args.no_improvement_limit
    config['optimization_goal'] = args.goal
    config['show_plots'] = args.show_plots
    
    # 创建优化循环
    loop = AutoOptimizationLoop(config)
    
    if args.rollback:
        # 回滚模式
        if loop.rollback_to_best():
            print("已回滚到历史最佳参数")
        else:
            print("回滚失败")
    else:
        # 正常运行模式
        loop.run()


if __name__ == '__main__':
    main()
