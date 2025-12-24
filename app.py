"""
Streamlit 交互式演示程序
原油期货多模型集成投资策略 - 模拟盘演示

功能：
1. 步进式模拟交易：逐日推进，观察每日交易决策
2. 账户状态展示：现金、持仓、盈亏实时更新
3. 交互式图表：K线图、权益曲线、信号标记
4. 交易记录查看
"""

import os
import sys
import logging
import warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# 设置日志
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings('ignore')

# 导入项目模块
from config import DATA_CONFIG, FEATURE_CONFIG, MODEL_CONFIG, STRATEGY_CONFIG
from data_collector import DataCollector
from predictor import Predictor, save_feature_params
from paper_account import PaperAccount, SimulationEngine
from feature_engineering import FeatureMatrix

# 页面配置
st.set_page_config(
    page_title="原油期货策略模拟盘",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
    }
    .positive { color: #00c853; }
    .negative { color: #ff1744; }
    .signal-buy { background-color: #e8f5e9; border-left: 4px solid #4caf50; }
    .signal-sell { background-color: #ffebee; border-left: 4px solid #f44336; }
    .signal-hold { background-color: #fff3e0; border-left: 4px solid #ff9800; }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """初始化 Session State"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.data = None
        st.session_state.predictor = None
        st.session_state.account = None
        st.session_state.engine = None
        st.session_state.simulation_started = False
        st.session_state.current_step = 0
        st.session_state.train_end_idx = 0
        st.session_state.step_results = []
        st.session_state.auto_running = False


def load_data():
    """加载数据"""
    with st.spinner("正在加载数据..."):
        collector = DataCollector(DATA_CONFIG)
        data = collector.get_data()
        return data


def check_and_train_models():
    """检查模型是否存在，如果不存在则训练"""
    models_dir = 'models'
    required_files = [
        'rf_model.joblib', 'xgb_model.joblib', 'bagging_model.joblib',
        'scaler.joblib', 'selector.joblib', 'feature_names.joblib', 'selected_features.joblib'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(models_dir, f))]
    
    if missing_files:
        st.warning(f"检测到模型文件缺失: {missing_files}")
        st.info("需要先训练模型才能进行模拟交易...")
        
        if st.button("🚀 开始训练模型", type="primary"):
            with st.spinner("正在训练模型，请稍候（这可能需要几分钟）..."):
                train_models()
            st.success("模型训练完成！")
            st.rerun()
        return False
    return True


def train_models():
    """训练模型并保存特征工程参数"""
    from model_trainer import ModelTrainer
    
    # 加载数据
    collector = DataCollector(DATA_CONFIG)
    data = collector.get_data()
    
    # 特征工程
    feature_matrix = FeatureMatrix()
    X_train, X_test, y_train, y_test = feature_matrix.fit_transform_pipeline(data, train_size=0.8)
    
    # 保存特征工程参数
    save_feature_params(feature_matrix.engineer, 'models')
    
    # 训练模型
    trainer = ModelTrainer()
    trainer.train(X_train, y_train)
    trainer.save_models('models')


def initialize_simulation(data: pd.DataFrame, train_ratio: float, initial_cash: float, 
                          position_size: float, allow_short: bool):
    """初始化模拟环境"""
    # 计算训练集结束位置
    train_end_idx = int(len(data) * train_ratio)
    
    # 创建预测器
    predictor = Predictor(models_dir='models', buffer_size=100)
    
    # 使用训练数据初始化缓冲区
    predictor.initialize_buffer(data.iloc[:train_end_idx])
    
    # 创建模拟账户
    account = PaperAccount(
        initial_cash=initial_cash,
        commission_rate=0.0001,
        allow_short=allow_short
    )
    
    # 创建模拟引擎
    engine = SimulationEngine(
        predictor=predictor,
        account=account,
        position_size=position_size,
        threshold_buy=STRATEGY_CONFIG.get('threshold_buy', 0.55),
        threshold_sell=STRATEGY_CONFIG.get('threshold_sell', 0.45),
        stop_loss=STRATEGY_CONFIG.get('stop_loss', 0.05),
        take_profit=STRATEGY_CONFIG.get('take_profit', 0.10)
    )
    
    # 加载模拟数据（测试集部分）
    engine.load_simulation_data(data, start_idx=train_end_idx)
    
    return predictor, account, engine, train_end_idx


def create_price_chart(data: pd.DataFrame, signals_df: pd.DataFrame = None, 
                       current_idx: int = None):
    """创建价格K线图"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('价格走势', '预测概率', '成交量')
    )
    
    # K线图
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='K线',
            increasing_line_color='#00c853',
            decreasing_line_color='#ff1744'
        ),
        row=1, col=1
    )
    
    # 添加均线
    if 'sma_20' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['sma_20'], name='MA20', 
                      line=dict(color='orange', width=1)),
            row=1, col=1
        )
    if 'sma_50' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['sma_50'], name='MA50',
                      line=dict(color='purple', width=1)),
            row=1, col=1
        )
    
    # 添加信号标记
    if signals_df is not None and len(signals_df) > 0:
        # 买入信号
        buy_signals = signals_df[signals_df['signal'] == 1]
        if len(buy_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(buy_signals['date']),
                    y=buy_signals['close_price'],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=12, color='green'),
                    name='买入信号'
                ),
                row=1, col=1
            )
        
        # 卖出信号
        sell_signals = signals_df[signals_df['signal'] == -1]
        if len(sell_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(sell_signals['date']),
                    y=sell_signals['close_price'],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=12, color='red'),
                    name='卖出信号'
                ),
                row=1, col=1
            )
        
        # 预测概率
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(signals_df['date']),
                y=signals_df['probability'],
                name='上涨概率',
                line=dict(color='blue', width=2),
                fill='tozeroy',
                fillcolor='rgba(0,100,255,0.1)'
            ),
            row=2, col=1
        )
        
        # 阈值线
        fig.add_hline(y=0.55, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=0.45, line_dash="dash", line_color="red", row=2, col=1)
    
    # 成交量
    if 'volume' in data.columns:
        colors = ['green' if data['close'].iloc[i] >= data['open'].iloc[i] else 'red' 
                  for i in range(len(data))]
        fig.add_trace(
            go.Bar(x=data.index, y=data['volume'], name='成交量', 
                   marker_color=colors, opacity=0.5),
            row=3, col=1
        )
    
    # 添加当前位置标记
    if current_idx is not None and current_idx < len(data):
        # Plotly 5.16+ 与 pandas 2.2 组合下，给 datetime 轴添加注释线时会在内部对
        # Timestamp 做加法导致 TypeError，这里去掉 annotation，保留 vline 标记即可。
        current_date = data.index[current_idx]
        fig.add_vline(x=current_date, line_dash="dash", line_color="blue", row=1, col=1)
    
    fig.update_layout(
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_xaxes(type='date')
    
    return fig


def create_equity_chart(snapshots_df: pd.DataFrame):
    """创建权益曲线图"""
    if snapshots_df is None or len(snapshots_df) == 0:
        return None
    
    fig = go.Figure()
    
    # 权益曲线
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(snapshots_df['date']),
            y=snapshots_df['total_equity'],
            name='总权益',
            line=dict(color='blue', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,100,255,0.1)'
        )
    )
    
    # 初始资金线
    initial_cash = snapshots_df['total_equity'].iloc[0] if len(snapshots_df) > 0 else 0
    fig.add_hline(y=initial_cash, line_dash="dash", line_color="gray", 
                  annotation_text="初始资金")
    
    fig.update_layout(
        title='账户权益曲线',
        xaxis_title='日期',
        yaxis_title='权益',
        height=300,
        showlegend=True
    )
    
    return fig


def create_pnl_chart(snapshots_df: pd.DataFrame):
    """创建盈亏图"""
    if snapshots_df is None or len(snapshots_df) == 0:
        return None
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 已实现盈亏
    fig.add_trace(
        go.Bar(
            x=pd.to_datetime(snapshots_df['date']),
            y=snapshots_df['realized_pnl'],
            name='已实现盈亏',
            marker_color='green',
            opacity=0.6
        ),
        secondary_y=False
    )
    
    # 未实现盈亏
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(snapshots_df['date']),
            y=snapshots_df['unrealized_pnl'],
            name='未实现盈亏',
            line=dict(color='orange', width=2)
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title='盈亏情况',
        height=300,
        showlegend=True
    )
    
    fig.update_yaxes(title_text="已实现盈亏", secondary_y=False)
    fig.update_yaxes(title_text="未实现盈亏", secondary_y=True)
    
    return fig


def display_account_metrics(stats: dict):
    """显示账户指标"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 总权益",
            value=f"¥{stats['total_equity']:,.2f}",
            delta=f"{stats['total_return_pct']}"
        )
    
    with col2:
        pnl_color = "green" if stats['realized_pnl'] >= 0 else "red"
        st.metric(
            label="📈 已实现盈亏",
            value=f"¥{stats['realized_pnl']:,.2f}"
        )
    
    with col3:
        st.metric(
            label="📊 当前持仓",
            value=f"{stats['position_quantity']:.4f}",
            delta=stats['position_side']
        )
    
    with col4:
        st.metric(
            label="💵 可用现金",
            value=f"¥{stats['current_cash']:,.2f}"
        )


def display_signal_card(prediction: dict, trade_action: str):
    """显示当前信号卡片"""
    signal = prediction['signal']
    signal_text = prediction['signal_text']
    probability = prediction['probability']
    
    if signal == 1:
        signal_class = "signal-buy"
        emoji = "🟢"
    elif signal == -1:
        signal_class = "signal-sell"
        emoji = "🔴"
    else:
        signal_class = "signal-hold"
        emoji = "🟡"
    
    st.markdown(f"""
    <div class="{signal_class}" style="padding: 15px; border-radius: 5px; margin: 10px 0;">
        <h3>{emoji} 当前信号: {signal_text}</h3>
        <p><strong>上涨概率:</strong> {probability:.2%}</p>
        <p><strong>执行动作:</strong> {trade_action or '无操作'}</p>
        <p><strong>各模型预测:</strong></p>
        <ul>
            {''.join([f"<li>{k}: {v:.2%}</li>" for k, v in prediction['individual_proba'].items()])}
        </ul>
    </div>
    """, unsafe_allow_html=True)


def main():
    """主函数"""
    st.title("📈 原油期货策略模拟盘")
    st.markdown("---")
    
    # 初始化 Session State
    init_session_state()
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 模拟配置")
        
        # 检查模型
        models_ready = check_and_train_models()
        if not models_ready:
            st.stop()
        
        st.success("✅ 模型已就绪")
        
        st.subheader("账户设置")
        initial_cash = st.number_input("初始资金 (¥)", 
                                       min_value=10000, 
                                       max_value=10000000, 
                                       value=1000000, 
                                       step=100000)
        
        position_size = st.slider("仓位比例", 
                                  min_value=0.1, 
                                  max_value=1.0, 
                                  value=0.3, 
                                  step=0.1)
        
        allow_short = st.checkbox("允许做空", value=False)
        
        st.subheader("数据设置")
        train_ratio = st.slider("训练集比例", 
                                min_value=0.5, 
                                max_value=0.9, 
                                value=0.8, 
                                step=0.05)
        
        st.markdown("---")
        
        # 初始化/重置按钮
        if st.button("🔄 初始化/重置模拟", type="primary", use_container_width=True):
            with st.spinner("正在初始化..."):
                # 加载数据
                data = load_data()
                
                # 初始化模拟
                predictor, account, engine, train_end_idx = initialize_simulation(
                    data, train_ratio, initial_cash, position_size, allow_short
                )
                
                # 保存到 Session State
                st.session_state.data = data
                st.session_state.predictor = predictor
                st.session_state.account = account
                st.session_state.engine = engine
                st.session_state.train_end_idx = train_end_idx
                st.session_state.simulation_started = True
                st.session_state.current_step = train_end_idx
                st.session_state.step_results = []
                st.session_state.initialized = True
            
            st.success(f"模拟已初始化! 训练集: {train_end_idx}天, 测试集: {len(data)-train_end_idx}天")
            st.rerun()
    
    # 主界面
    if not st.session_state.initialized:
        st.info("👈 请在左侧配置参数并点击\"初始化/重置模拟\"开始")
        
        # 显示数据预览
        st.subheader("📊 数据预览")
        with st.spinner("加载数据预览..."):
            preview_data = load_data()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("数据起始", str(preview_data.index[0].date()))
            with col2:
                st.metric("数据结束", str(preview_data.index[-1].date()))
            with col3:
                st.metric("总交易日", len(preview_data))
            
            # 显示简单图表
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=preview_data.index, y=preview_data['close'], 
                                     name='收盘价', line=dict(color='blue')))
            fig.update_layout(title='WTI原油期货价格走势', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.stop()
    
    # 模拟控制面板
    st.subheader("🎮 模拟控制")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        step_button = st.button("▶️ 下一天", use_container_width=True)
    
    with col2:
        step_5_button = st.button("⏩ 前进5天", use_container_width=True)
    
    with col3:
        step_20_button = st.button("⏭️ 前进20天", use_container_width=True)
    
    with col4:
        run_all_button = st.button("🏁 运行到结束", use_container_width=True)
    
    with col5:
        export_button = st.button("📥 导出交易记录", use_container_width=True)
    
    # 执行步进
    engine = st.session_state.engine
    data = st.session_state.data
    
    steps_to_run = 0
    if step_button:
        steps_to_run = 1
    elif step_5_button:
        steps_to_run = 5
    elif step_20_button:
        steps_to_run = 20
    elif run_all_button:
        steps_to_run = len(data) - engine.current_step
    
    if steps_to_run > 0:
        progress_bar = st.progress(0)
        for i in range(steps_to_run):
            if engine.current_step >= len(data):
                break
            result = engine.step()
            st.session_state.step_results.append(result)
            st.session_state.current_step = engine.current_step
            progress_bar.progress((i + 1) / steps_to_run)
        progress_bar.empty()
        st.rerun()
    
    if export_button:
        trades_df = engine.account.get_trades_df()
        if len(trades_df) > 0:
            csv = trades_df.to_csv(index=False)
            st.download_button(
                label="下载CSV",
                data=csv,
                file_name=f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("暂无交易记录")
    
    # 显示进度
    total_steps = len(data) - st.session_state.train_end_idx
    current_progress = engine.current_step - st.session_state.train_end_idx
    st.progress(current_progress / total_steps if total_steps > 0 else 0)
    st.caption(f"模拟进度: {current_progress}/{total_steps} 天")
    
    st.markdown("---")
    
    # 账户状态
    if len(st.session_state.step_results) > 0:
        last_result = st.session_state.step_results[-1]
        
        st.subheader("💼 账户状态")
        display_account_metrics(last_result['account_stats'])
        
        # 当前信号
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🎯 当前交易信号")
            display_signal_card(last_result['prediction'], last_result['trade_action'])
            
            st.markdown(f"""
            **当前日期:** {last_result['date']}  
            **当前价格:** ¥{last_result['price']:.2f}
            """)
        
        with col2:
            # 权益曲线
            snapshots_df = engine.account.get_snapshots_df()
            equity_fig = create_equity_chart(snapshots_df)
            if equity_fig:
                st.plotly_chart(equity_fig, use_container_width=True)
    
    st.markdown("---")
    
    # 图表区域
    st.subheader("📊 行情图表")
    
    # 获取信号数据
    signals_df = engine.get_signals_df() if len(engine.signals_history) > 0 else None
    
    # 确定显示范围
    display_start = max(0, st.session_state.train_end_idx - 50)
    display_end = min(len(data), engine.current_step + 10)
    display_data = data.iloc[display_start:display_end]
    
    # 创建价格图表
    price_fig = create_price_chart(
        display_data, 
        signals_df,
        current_idx=engine.current_step - display_start if engine.current_step >= display_start else None
    )
    st.plotly_chart(price_fig, use_container_width=True)
    
    # 盈亏图表
    if len(st.session_state.step_results) > 0:
        snapshots_df = engine.account.get_snapshots_df()
        pnl_fig = create_pnl_chart(snapshots_df)
        if pnl_fig:
            st.plotly_chart(pnl_fig, use_container_width=True)
    
    st.markdown("---")
    
    # 交易记录
    st.subheader("📝 交易记录")
    
    trades_df = engine.account.get_trades_df()
    if len(trades_df) > 0:
        # 格式化显示
        display_trades = trades_df.copy()
        display_trades['price'] = display_trades['price'].apply(lambda x: f"¥{x:.2f}")
        display_trades['value'] = display_trades['value'].apply(lambda x: f"¥{x:,.2f}")
        display_trades['pnl'] = display_trades['pnl'].apply(lambda x: f"¥{x:,.2f}")
        display_trades['commission'] = display_trades['commission'].apply(lambda x: f"¥{x:.2f}")
        
        st.dataframe(
            display_trades[['date', 'action', 'price', 'quantity', 'value', 'pnl', 'note']],
            use_container_width=True,
            hide_index=True
        )
        
        # 交易统计
        st.subheader("📈 交易统计")
        
        stats = engine.account.get_stats(data.iloc[engine.current_step - 1]['close'] if engine.current_step > 0 else data.iloc[-1]['close'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("总交易次数", stats['total_trades'])
        with col2:
            st.metric("盈利次数", stats['winning_trades'])
        with col3:
            st.metric("亏损次数", stats['losing_trades'])
        with col4:
            st.metric("胜率", f"{stats['win_rate']:.1%}")
    else:
        st.info("暂无交易记录")
    
    # 页脚
    st.markdown("---")
    st.caption("原油期货多模型集成投资策略 - 模拟盘演示系统 | 仅供学习研究使用")


if __name__ == "__main__":
    main()
