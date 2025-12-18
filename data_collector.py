"""
数据收集模块 - 负责从Yahoo Finance获取期货及宏观数据
原油期货多模型集成投资策略
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Tuple
import logging
import warnings

from config import DATA_CONFIG, FEATURE_CONFIG

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCollector:
    """数据收集器类"""
    
    def __init__(self, config: Dict = None):
        """
        初始化数据收集器
        
        Args:
            config: 数据配置字典
        """
        self.config = config or DATA_CONFIG
        self.symbol = self.config['symbol']
        self.macro_symbols = self.config['macro_symbols']
        self.start_date = self.config['start_date']
        self.end_date = self.config['end_date']
        self.interval = self.config['interval']
        
        self.futures_data = None
        self.macro_data = {}
        self.merged_data = None
    
    def _flatten_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        展平多层列名并标准化
        
        Args:
            df: 原始DataFrame
            
        Returns:
            展平后的DataFrame
        """
        if isinstance(df.columns, pd.MultiIndex):
            # 展平多层列名，取第一层
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        
        # 标准化列名：转小写、去特殊字符
        df.columns = [str(col).lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        
        return df
    
    def _download_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        从Yahoo Finance下载数据
        
        Args:
            symbol: 股票/期货代码
            
        Returns:
            下载的DataFrame或None
        """
        try:
            logger.info(f"正在下载 {symbol} 数据...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=self.start_date,
                end=self.end_date,
                interval=self.interval
            )
            
            if df.empty:
                logger.warning(f"{symbol} 数据为空，尝试使用download方法...")
                df = yf.download(
                    symbol,
                    start=self.start_date,
                    end=self.end_date,
                    interval=self.interval,
                    progress=False
                )
            
            if not df.empty:
                df = self._flatten_columns(df)
                # 确保索引是DatetimeIndex且没有时区信息
                df.index = pd.to_datetime(df.index).tz_localize(None)
                logger.info(f"{symbol} 下载成功，共 {len(df)} 条记录")
                return df
            else:
                logger.error(f"{symbol} 数据下载失败")
                return None
                
        except Exception as e:
            logger.error(f"下载 {symbol} 数据时出错: {e}")
            return None
    
    def collect_futures_data(self) -> pd.DataFrame:
        """
        收集期货主数据
        
        Returns:
            期货数据DataFrame
        """
        self.futures_data = self._download_data(self.symbol)
        
        if self.futures_data is not None:
            # 计算基础技术指标
            self.futures_data = self._calculate_technical_indicators(self.futures_data)
            
        return self.futures_data
    
    def collect_macro_data(self) -> Dict[str, pd.DataFrame]:
        """
        收集宏观经济数据
        
        Returns:
            宏观数据字典
        """
        for symbol in self.macro_symbols:
            df = self._download_data(symbol)
            if df is not None:
                # 只保留收盘价作为宏观特征
                clean_symbol = symbol.replace('^', '').replace('=', '').replace('.', '').replace('-', '_').lower()
                self.macro_data[clean_symbol] = df[['close']].rename(columns={'close': f'{clean_symbol}_close'})
        
        return self.macro_data
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算基础技术指标
        
        Args:
            df: 原始OHLCV数据
            
        Returns:
            添加技术指标后的DataFrame
        """
        df = df.copy()
        
        # 确保必要的列存在
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"缺少列: {col}")
                return df
        
        # ====== 均线指标 ======
        for period in FEATURE_CONFIG['sma_periods']:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        for period in FEATURE_CONFIG['ema_periods']:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # ====== RSI ======
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=FEATURE_CONFIG['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=FEATURE_CONFIG['rsi_period']).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ====== MACD ======
        ema_fast = df['close'].ewm(span=FEATURE_CONFIG['macd_fast'], adjust=False).mean()
        ema_slow = df['close'].ewm(span=FEATURE_CONFIG['macd_slow'], adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=FEATURE_CONFIG['macd_signal'], adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ====== Stochastic ======
        low_min = df['low'].rolling(window=14).min()
        high_max = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # ====== 布林带 ======
        bb_period = FEATURE_CONFIG['bb_period']
        bb_std = FEATURE_CONFIG['bb_std']
        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        bb_std_val = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_val * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_val * bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ====== ATR ======
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=FEATURE_CONFIG['atr_period']).mean()
        
        # ====== 成交量指标 ======
        df['volume_sma'] = df['volume'].rolling(window=FEATURE_CONFIG['volume_sma_period']).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # ====== 对数收益率 ======
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # ====== 价格变动 ======
        df['price_change'] = df['close'].pct_change()
        df['high_low_pct'] = (df['high'] - df['low']) / df['close']
        df['close_open_pct'] = (df['close'] - df['open']) / df['open']
        
        return df
    
    def merge_data(self) -> pd.DataFrame:
        """
        合并期货数据和宏观数据
        
        Returns:
            合并后的DataFrame
        """
        if self.futures_data is None:
            self.collect_futures_data()
        
        if not self.macro_data:
            self.collect_macro_data()
        
        # 以期货数据为基础
        self.merged_data = self.futures_data.copy()
        
        # 合并宏观数据
        for symbol, macro_df in self.macro_data.items():
            self.merged_data = self.merged_data.join(macro_df, how='left')
        
        # 缺失值处理：前向填充 + 后向填充
        self.merged_data = self.merged_data.ffill().bfill()
        
        # 删除仍然存在NaN的行
        initial_len = len(self.merged_data)
        self.merged_data = self.merged_data.dropna()
        final_len = len(self.merged_data)
        
        if initial_len > final_len:
            logger.info(f"删除了 {initial_len - final_len} 行含NaN的数据")
        
        logger.info(f"数据合并完成，最终数据量: {len(self.merged_data)} 行")
        
        return self.merged_data
    
    def get_data(self) -> pd.DataFrame:
        """
        获取处理后的完整数据
        
        Returns:
            完整的DataFrame
        """
        if self.merged_data is None:
            self.merge_data()
        
        return self.merged_data
    
    def save_data(self, filepath: str = 'data/raw_data.csv'):
        """
        保存数据到CSV文件
        
        Args:
            filepath: 保存路径
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if self.merged_data is not None:
            self.merged_data.to_csv(filepath)
            logger.info(f"数据已保存到 {filepath}")
    
    def load_data(self, filepath: str = 'data/raw_data.csv') -> pd.DataFrame:
        """
        从CSV文件加载数据
        
        Args:
            filepath: 文件路径
            
        Returns:
            加载的DataFrame
        """
        self.merged_data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logger.info(f"从 {filepath} 加载了 {len(self.merged_data)} 条数据")
        return self.merged_data


def main():
    """测试数据收集功能"""
    collector = DataCollector()
    data = collector.get_data()
    print(f"\n数据形状: {data.shape}")
    print(f"\n数据列: {data.columns.tolist()}")
    print(f"\n数据前5行:\n{data.head()}")
    print(f"\n数据统计:\n{data.describe()}")


if __name__ == '__main__':
    main()
