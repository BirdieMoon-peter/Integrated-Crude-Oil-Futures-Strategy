"""
模拟账户模块 - 用于模拟盘交易
原油期货多模型集成投资策略

提供 PaperAccount 类，模拟交易所账户功能：
1. 维护账户状态：现金、持仓、均价、已实现/未实现盈亏
2. 订单执行：买入、卖出、平仓
3. 交易记录和绩效统计
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime
from dataclasses import dataclass, field
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """交易记录"""
    date: str
    action: str  # 'BUY', 'SELL', 'CLOSE_LONG', 'CLOSE_SHORT'
    price: float
    quantity: float
    value: float
    commission: float = 0.0
    pnl: float = 0.0  # 平仓时的盈亏
    note: str = ""
    
    def to_dict(self) -> dict:
        return {
            'date': self.date,
            'action': self.action,
            'price': self.price,
            'quantity': self.quantity,
            'value': self.value,
            'commission': self.commission,
            'pnl': self.pnl,
            'note': self.note
        }


@dataclass
class Position:
    """持仓信息"""
    quantity: float = 0.0  # 正数为多头，负数为空头
    avg_price: float = 0.0  # 持仓均价
    entry_date: str = ""  # 入场日期
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0
    
    @property
    def is_empty(self) -> bool:
        return self.quantity == 0
    
    def market_value(self, current_price: float) -> float:
        """计算持仓市值"""
        return abs(self.quantity) * current_price
    
    def unrealized_pnl(self, current_price: float) -> float:
        """计算未实现盈亏"""
        if self.is_empty:
            return 0.0
        if self.is_long:
            return self.quantity * (current_price - self.avg_price)
        else:  # short
            return abs(self.quantity) * (self.avg_price - current_price)


class PaperAccount:
    """
    模拟账户类
    
    模拟交易所账户，支持做多做空，记录所有交易。
    """
    
    def __init__(self, 
                 initial_cash: float = 1_000_000.0,
                 commission_rate: float = 0.0001,  # 万分之一手续费
                 contract_multiplier: float = 1.0,  # 合约乘数
                 allow_short: bool = False):  # 是否允许做空
        """
        初始化模拟账户
        
        Args:
            initial_cash: 初始资金
            commission_rate: 手续费率
            contract_multiplier: 合约乘数（期货用）
            allow_short: 是否允许做空
        """
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.contract_multiplier = contract_multiplier
        self.allow_short = allow_short
        
        # 账户状态
        self.cash = initial_cash
        self.position = Position()
        self.realized_pnl = 0.0
        
        # 交易历史
        self.trades: List[Trade] = []
        
        # 每日账户快照（用于绘图）
        self.daily_snapshots: List[Dict] = []
        
        # 当前日期
        self.current_date = ""
    
    def reset(self):
        """重置账户"""
        self.cash = self.initial_cash
        self.position = Position()
        self.realized_pnl = 0.0
        self.trades = []
        self.daily_snapshots = []
        self.current_date = ""
    
    def _calculate_commission(self, value: float) -> float:
        """计算手续费"""
        return value * self.commission_rate
    
    def buy(self, price: float, quantity: float = None, 
            value: float = None, date: str = "") -> Optional[Trade]:
        """
        买入
        
        Args:
            price: 买入价格
            quantity: 买入数量（与value二选一）
            value: 买入金额（与quantity二选一）
            date: 交易日期
            
        Returns:
            交易记录，如果失败返回None
        """
        if quantity is None and value is None:
            logger.error("买入需要指定数量或金额")
            return None
        
        if value is not None:
            quantity = value / (price * self.contract_multiplier)
        
        trade_value = quantity * price * self.contract_multiplier
        commission = self._calculate_commission(trade_value)
        total_cost = trade_value + commission
        
        # 检查资金是否充足
        if total_cost > self.cash:
            logger.warning(f"资金不足: 需要 {total_cost:.2f}, 可用 {self.cash:.2f}")
            # 调整数量
            available = self.cash - commission
            quantity = available / (price * self.contract_multiplier)
            if quantity <= 0:
                return None
            trade_value = quantity * price * self.contract_multiplier
            commission = self._calculate_commission(trade_value)
            total_cost = trade_value + commission
        
        # 如果有空头持仓，先平仓
        if self.position.is_short:
            # 平空仓
            close_pnl = self.position.unrealized_pnl(price)
            self.realized_pnl += close_pnl
            self.cash += abs(self.position.quantity) * self.position.avg_price * self.contract_multiplier + close_pnl
            
            close_trade = Trade(
                date=date,
                action='CLOSE_SHORT',
                price=price,
                quantity=abs(self.position.quantity),
                value=abs(self.position.quantity) * price * self.contract_multiplier,
                commission=0,
                pnl=close_pnl,
                note="平空仓后买入"
            )
            self.trades.append(close_trade)
            
            self.position = Position()
        
        # 执行买入
        if self.position.is_empty:
            self.position.avg_price = price
            self.position.quantity = quantity
            self.position.entry_date = date
        else:
            # 加仓：计算新均价
            total_quantity = self.position.quantity + quantity
            total_cost_basis = self.position.quantity * self.position.avg_price + quantity * price
            self.position.avg_price = total_cost_basis / total_quantity
            self.position.quantity = total_quantity
        
        self.cash -= total_cost
        
        trade = Trade(
            date=date,
            action='BUY',
            price=price,
            quantity=quantity,
            value=trade_value,
            commission=commission,
            note=f"买入 {quantity:.4f} @ {price:.2f}"
        )
        self.trades.append(trade)
        
        logger.info(f"[{date}] 买入: {quantity:.4f} @ {price:.2f}, 花费: {total_cost:.2f}")
        
        return trade
    
    def sell(self, price: float, quantity: float = None,
             value: float = None, date: str = "") -> Optional[Trade]:
        """
        卖出/做空
        
        Args:
            price: 卖出价格
            quantity: 卖出数量
            value: 卖出金额
            date: 交易日期
            
        Returns:
            交易记录
        """
        if quantity is None and value is None:
            logger.error("卖出需要指定数量或金额")
            return None
        
        if value is not None:
            quantity = value / (price * self.contract_multiplier)
        
        # 如果有多头持仓
        if self.position.is_long:
            close_quantity = min(quantity, self.position.quantity)
            
            # 计算平仓盈亏
            close_pnl = close_quantity * (price - self.position.avg_price) * self.contract_multiplier
            commission = self._calculate_commission(close_quantity * price * self.contract_multiplier)
            
            self.realized_pnl += close_pnl
            self.cash += close_quantity * price * self.contract_multiplier - commission
            
            trade = Trade(
                date=date,
                action='CLOSE_LONG' if close_quantity == self.position.quantity else 'SELL',
                price=price,
                quantity=close_quantity,
                value=close_quantity * price * self.contract_multiplier,
                commission=commission,
                pnl=close_pnl,
                note=f"卖出 {close_quantity:.4f} @ {price:.2f}, 盈亏: {close_pnl:.2f}"
            )
            self.trades.append(trade)
            
            # 更新持仓
            self.position.quantity -= close_quantity
            if self.position.quantity <= 0:
                self.position = Position()
            
            logger.info(f"[{date}] 平多仓: {close_quantity:.4f} @ {price:.2f}, 盈亏: {close_pnl:.2f}")
            
            # 如果还有剩余数量需要做空
            remaining = quantity - close_quantity
            if remaining > 0 and self.allow_short:
                return self._open_short(price, remaining, date)
            
            return trade
        
        # 没有多头持仓，执行做空
        if self.allow_short:
            return self._open_short(price, quantity, date)
        else:
            logger.warning("不允许做空，无持仓可平")
            return None
    
    def _open_short(self, price: float, quantity: float, date: str) -> Trade:
        """开空仓"""
        trade_value = quantity * price * self.contract_multiplier
        commission = self._calculate_commission(trade_value)
        
        # 做空：保证金模式简化处理，直接记录
        if self.position.is_empty:
            self.position.avg_price = price
            self.position.quantity = -quantity
            self.position.entry_date = date
        else:
            # 加空仓
            total_quantity = self.position.quantity - quantity
            total_cost_basis = abs(self.position.quantity) * self.position.avg_price + quantity * price
            self.position.avg_price = total_cost_basis / abs(total_quantity)
            self.position.quantity = total_quantity
        
        self.cash -= commission
        
        trade = Trade(
            date=date,
            action='SELL',
            price=price,
            quantity=quantity,
            value=trade_value,
            commission=commission,
            note=f"做空 {quantity:.4f} @ {price:.2f}"
        )
        self.trades.append(trade)
        
        logger.info(f"[{date}] 做空: {quantity:.4f} @ {price:.2f}")
        
        return trade
    
    def close_position(self, price: float, date: str = "") -> Optional[Trade]:
        """
        平掉所有仓位
        
        Args:
            price: 平仓价格
            date: 交易日期
            
        Returns:
            交易记录
        """
        if self.position.is_empty:
            logger.info("没有持仓可平")
            return None
        
        if self.position.is_long:
            return self.sell(price, quantity=self.position.quantity, date=date)
        else:
            return self.buy(price, quantity=abs(self.position.quantity), date=date)
    
    def take_snapshot(self, date: str, current_price: float):
        """
        记录每日账户快照
        
        Args:
            date: 日期
            current_price: 当前价格
        """
        self.current_date = date
        
        unrealized_pnl = self.position.unrealized_pnl(current_price)
        position_value = self.position.market_value(current_price) if not self.position.is_empty else 0
        
        snapshot = {
            'date': date,
            'cash': self.cash,
            'position_quantity': self.position.quantity,
            'position_avg_price': self.position.avg_price,
            'position_value': position_value,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_equity': self.cash + position_value + unrealized_pnl,
            'current_price': current_price
        }
        
        self.daily_snapshots.append(snapshot)
    
    def get_equity(self, current_price: float) -> float:
        """
        获取当前总权益
        
        Args:
            current_price: 当前价格
            
        Returns:
            总权益
        """
        unrealized = self.position.unrealized_pnl(current_price)
        position_value = self.position.market_value(current_price) if not self.position.is_empty else 0
        return self.cash + position_value + unrealized
    
    def get_return(self, current_price: float) -> float:
        """
        获取收益率
        
        Args:
            current_price: 当前价格
            
        Returns:
            收益率
        """
        equity = self.get_equity(current_price)
        return (equity - self.initial_cash) / self.initial_cash
    
    def get_stats(self, current_price: float) -> Dict:
        """
        获取账户统计信息
        
        Args:
            current_price: 当前价格
            
        Returns:
            统计信息字典
        """
        equity = self.get_equity(current_price)
        unrealized = self.position.unrealized_pnl(current_price)
        
        # 计算交易统计
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]

        # 只在有非零盈亏的交易时计算胜率，避免除零
        non_zero_trades = [t for t in self.trades if t.pnl != 0]
        win_rate = len(winning_trades) / len(non_zero_trades) if len(non_zero_trades) > 0 else 0
        
        return {
            'initial_cash': self.initial_cash,
            'current_cash': self.cash,
            'position_quantity': self.position.quantity,
            'position_avg_price': self.position.avg_price,
            'position_side': '多头' if self.position.is_long else ('空头' if self.position.is_short else '空仓'),
            'unrealized_pnl': unrealized,
            'realized_pnl': self.realized_pnl,
            'total_equity': equity,
            'total_return': (equity - self.initial_cash) / self.initial_cash,
            'total_return_pct': f"{((equity - self.initial_cash) / self.initial_cash * 100):.2f}%",
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'current_price': current_price
        }
    
    def get_trades_df(self) -> pd.DataFrame:
        """获取交易记录DataFrame"""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trades])
    
    def get_snapshots_df(self) -> pd.DataFrame:
        """获取每日快照DataFrame"""
        if not self.daily_snapshots:
            return pd.DataFrame()
        return pd.DataFrame(self.daily_snapshots)
    
    def export_trades(self, filepath: str):
        """导出交易记录到JSON"""
        trades_data = [t.to_dict() for t in self.trades]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(trades_data, f, ensure_ascii=False, indent=2)
        logger.info(f"交易记录已导出到 {filepath}")


class SimulationEngine:
    """
    模拟交易引擎
    
    整合预测器和模拟账户，执行步进式模拟交易。
    """
    
    def __init__(self, 
                 predictor,
                 account: PaperAccount,
                 position_size: float = 0.3,  # 仓位比例
                 threshold_buy: float = 0.55,
                 threshold_sell: float = 0.45,
                 stop_loss: float = 0.05,
                 take_profit: float = 0.10):
        """
        初始化模拟引擎
        
        Args:
            predictor: Predictor实例
            account: PaperAccount实例
            position_size: 仓位比例
            threshold_buy: 买入阈值
            threshold_sell: 卖出阈值
            stop_loss: 止损比例
            take_profit: 止盈比例
        """
        self.predictor = predictor
        self.account = account
        self.position_size = position_size
        self.threshold_buy = threshold_buy
        self.threshold_sell = threshold_sell
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # 模拟状态
        self.current_step = 0
        self.simulation_data: Optional[pd.DataFrame] = None
        self.signals_history: List[Dict] = []
    
    def load_simulation_data(self, data: pd.DataFrame, start_idx: int = 0):
        """
        加载模拟数据
        
        Args:
            data: 完整数据
            start_idx: 开始的索引位置
        """
        self.simulation_data = data.copy()
        self.current_step = start_idx
        
        # 使用前start_idx的数据初始化预测器缓冲区
        if start_idx > 0:
            init_data = data.iloc[:start_idx]
            self.predictor.initialize_buffer(init_data)
    
    def step(self) -> Dict:
        """
        执行一步模拟
        
        Returns:
            当前步骤的信息
        """
        if self.simulation_data is None:
            raise ValueError("请先调用 load_simulation_data() 加载数据")
        
        if self.current_step >= len(self.simulation_data):
            return {'status': 'finished', 'message': '模拟已结束'}
        
        # 获取当日数据
        current_row = self.simulation_data.iloc[self.current_step:self.current_step+1]
        current_date = str(current_row.index[0].date()) if hasattr(current_row.index[0], 'date') else str(current_row.index[0])
        current_price = float(current_row['close'].iloc[0])
        
        # 追加数据到缓冲区
        self.predictor.append_data(current_row)
        
        # 获取预测信号
        prediction = self.predictor.predict(as_of_date=current_date)
        
        # 执行交易逻辑
        trade_action = None
        
        # 检查止盈止损
        if not self.account.position.is_empty:
            pnl_pct = self.account.position.unrealized_pnl(current_price) / (
                abs(self.account.position.quantity) * self.account.position.avg_price
            )
            
            if pnl_pct <= -self.stop_loss:
                # 止损
                trade_action = 'STOP_LOSS'
                self.account.close_position(current_price, date=current_date)
            elif pnl_pct >= self.take_profit:
                # 止盈
                trade_action = 'TAKE_PROFIT'
                self.account.close_position(current_price, date=current_date)
        
        # 根据信号交易
        if trade_action is None:
            if prediction['signal'] == 1:  # 买入信号
                if self.account.position.is_empty:
                    # 开多仓
                    trade_value = self.account.cash * self.position_size
                    self.account.buy(current_price, value=trade_value, date=current_date)
                    trade_action = 'OPEN_LONG'
                elif self.account.position.is_short:
                    # 平空开多
                    self.account.close_position(current_price, date=current_date)
                    trade_value = self.account.cash * self.position_size
                    self.account.buy(current_price, value=trade_value, date=current_date)
                    trade_action = 'REVERSE_TO_LONG'
                    
            elif prediction['signal'] == -1:  # 卖出信号
                if self.account.position.is_long:
                    # 平多仓
                    self.account.close_position(current_price, date=current_date)
                    trade_action = 'CLOSE_LONG'
                elif self.account.position.is_empty and self.account.allow_short:
                    # 开空仓
                    trade_value = self.account.cash * self.position_size
                    self.account.sell(current_price, value=trade_value, date=current_date)
                    trade_action = 'OPEN_SHORT'
        
        # 记录每日快照
        self.account.take_snapshot(current_date, current_price)
        
        # 记录信号历史
        signal_record = {
            **prediction,
            'trade_action': trade_action,
            'equity': self.account.get_equity(current_price),
            'position': self.account.position.quantity,
            'cash': self.account.cash
        }
        self.signals_history.append(signal_record)
        
        # 前进一步
        self.current_step += 1
        
        return {
            'status': 'running',
            'step': self.current_step,
            'total_steps': len(self.simulation_data),
            'date': current_date,
            'price': current_price,
            'prediction': prediction,
            'trade_action': trade_action,
            'account_stats': self.account.get_stats(current_price)
        }
    
    def run_to_end(self) -> List[Dict]:
        """运行到结束"""
        results = []
        while self.current_step < len(self.simulation_data):
            result = self.step()
            results.append(result)
        return results
    
    def get_signals_df(self) -> pd.DataFrame:
        """获取信号历史DataFrame"""
        if not self.signals_history:
            return pd.DataFrame()
        return pd.DataFrame(self.signals_history)


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建账户
    account = PaperAccount(initial_cash=100000, allow_short=False)
    
    # 模拟交易
    account.buy(price=70.0, value=30000, date='2024-01-01')
    account.take_snapshot('2024-01-01', 70.0)
    
    account.take_snapshot('2024-01-02', 72.0)
    
    account.sell(price=75.0, quantity=account.position.quantity, date='2024-01-03')
    account.take_snapshot('2024-01-03', 75.0)
    
    # 输出统计
    stats = account.get_stats(75.0)
    print("\n账户统计:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("\n交易记录:")
    print(account.get_trades_df())
    
    print("\n每日快照:")
    print(account.get_snapshots_df())
