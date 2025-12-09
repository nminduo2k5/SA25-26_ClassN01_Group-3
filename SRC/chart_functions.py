import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

def create_stock_prediction_chart(pred, symbol, current_price, predictions):
    """Tạo biểu đồ dự đoán giá chứng khoán chuyên nghiệp"""
    
    # Tạo dữ liệu lịch sử (30 ngày gần đây)
    historical_dates = []
    historical_prices = []
    base_date = datetime.now() - timedelta(days=30)
    
    # Tạo dữ liệu lịch sử giả lập với xu hướng thực tế
    for i in range(30):
        date = base_date + timedelta(days=i)
        if date.weekday() < 5:  # Chỉ ngày làm việc
            # Tạo giá với biến động thực tế
            price_variation = current_price * (0.95 + (i/30) * 0.1)  # Xu hướng tăng nhẹ
            noise = current_price * 0.02 * (0.5 - hash(str(date)) % 100 / 100)  # Nhiễu ngẫu nhiên
            price = price_variation + noise
            historical_dates.append(date)
            historical_prices.append(price)
    
    # Tạo dữ liệu dự đoán tương lai
    future_dates = []
    future_prices = []
    
    # Lấy dự đoán từ predictions
    prediction_points = []
    
    # Short term predictions
    short_term = predictions.get('short_term', {})
    for period, data in short_term.items():
        if 'days' in period:
            days = int(period.split('_')[0])
            price = data.get('price', current_price)
            date = datetime.now() + timedelta(days=days)
            prediction_points.append((date, price, 'short'))
    
    # Medium term predictions
    medium_term = predictions.get('medium_term', {})
    for period, data in medium_term.items():
        if 'days' in period:
            days = int(period.split('_')[0])
            price = data.get('price', current_price)
            date = datetime.now() + timedelta(days=days)
            prediction_points.append((date, price, 'medium'))
    
    # Long term predictions
    long_term = predictions.get('long_term', {})
    for period, data in long_term.items():
        if 'days' in period:
            days = int(period.split('_')[0])
            price = data.get('price', current_price)
            date = datetime.now() + timedelta(days=days)
            prediction_points.append((date, price, 'long'))
    
    # Sắp xếp theo ngày
    prediction_points.sort(key=lambda x: x[0])
    
    # Tạo đường dự đoán liên tục
    if prediction_points:
        # Thêm điểm hiện tại
        future_dates = [datetime.now()] + [p[0] for p in prediction_points]
        future_prices = [current_price] + [p[1] for p in prediction_points]
    
    # Tạo biểu đồ
    fig = go.Figure()
    
    # Đường giá lịch sử
    fig.add_trace(go.Scatter(
        x=historical_dates,
        y=historical_prices,
        mode='lines',
        name='Giá lịch sử',
        line=dict(color='#2E86AB', width=2),
        hovertemplate='<b>Lịch sử</b><br>Ngày: %{x}<br>Giá: %{y:,.0f} VND<extra></extra>'
    ))
    
    # Điểm giá hiện tại
    fig.add_trace(go.Scatter(
        x=[datetime.now()],
        y=[current_price],
        mode='markers',
        name='Giá hiện tại',
        marker=dict(color='#F18F01', size=12, symbol='circle'),
        hovertemplate='<b>Hiện tại</b><br>Ngày: %{x}<br>Giá: %{y:,.0f} VND<extra></extra>'
    ))
    
    # Đường dự đoán
    if future_dates and future_prices:
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_prices,
            mode='lines+markers',
            name='Dự đoán AI',
            line=dict(color='#C73E1D', width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond'),
            hovertemplate='<b>Dự đoán</b><br>Ngày: %{x}<br>Giá: %{y:,.0f} VND<extra></extra>'
        ))
        
        # Vùng tin cậy (confidence interval)
        confidence = pred.get('confidence', 50)
        confidence_factor = (100 - confidence) / 100 * 0.1  # 10% max uncertainty
        
        upper_bound = [p * (1 + confidence_factor) for p in future_prices]
        lower_bound = [p * (1 - confidence_factor) for p in future_prices]
        
        fig.add_trace(go.Scatter(
            x=future_dates + future_dates[::-1],
            y=upper_bound + lower_bound[::-1],
            fill='toself',
            fillcolor='rgba(199, 62, 29, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'Vùng tin cậy ({confidence:.0f}%)',
            hoverinfo='skip'
        ))
    
    # Thêm các điểm dự đoán quan trọng
    important_predictions = []
    colors = {'short': '#28a745', 'medium': '#ffc107', 'long': '#dc3545'}
    
    for date, price, term in prediction_points:
        if term in ['short', 'medium', 'long']:
            fig.add_trace(go.Scatter(
                x=[date],
                y=[price],
                mode='markers',
                name=f'Mục tiêu {term}',
                marker=dict(
                    color=colors[term],
                    size=10,
                    symbol='star',
                    line=dict(color='white', width=2)
                ),
                hovertemplate=f'<b>Mục tiêu {term}</b><br>Ngày: %{{x}}<br>Giá: %{{y:,.0f}} VND<extra></extra>'
            ))
    
    # Support và Resistance levels
    trend_analysis = pred.get('trend_analysis', {})
    support = trend_analysis.get('support_level', current_price * 0.95)
    resistance = trend_analysis.get('resistance_level', current_price * 1.05)
    
    # Đường support
    fig.add_hline(
        y=support,
        line_dash="dot",
        line_color="green",
        annotation_text=f"Support: {support:,.0f}",
        annotation_position="bottom right"
    )
    
    # Đường resistance
    fig.add_hline(
        y=resistance,
        line_dash="dot", 
        line_color="red",
        annotation_text=f"Resistance: {resistance:,.0f}",
        annotation_position="top right"
    )
    
    # Cập nhật layout
    fig.update_layout(
        title=dict(
            text=f'📈 Biểu đồ Dự đoán Giá {symbol}',
            x=0.5,
            font=dict(size=20, color='#2E86AB')
        ),
        xaxis=dict(
            title='Thời gian',
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            type='date'
        ),
        yaxis=dict(
            title='Giá (VND)',
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            tickformat=',.0f'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_technical_indicators_chart(pred, symbol):
    """Tạo biểu đồ chỉ báo kỹ thuật"""
    
    # Lấy dữ liệu technical indicators
    tech_indicators = pred.get('technical_indicators', {})
    trend_analysis = pred.get('trend_analysis', {})
    
    rsi = trend_analysis.get('rsi', tech_indicators.get('rsi', 50))
    macd = trend_analysis.get('macd', tech_indicators.get('macd', 0))
    momentum_5d = trend_analysis.get('momentum_5d', 0)
    momentum_20d = trend_analysis.get('momentum_20d', 0)
    
    # Tạo subplot với 2 hàng - chỉ dùng xy plots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('RSI (Relative Strength Index)', 'MACD', 'Momentum 5D', 'Momentum 20D')
    )
    
    # RSI Bar Chart thay vì Gauge
    rsi_color = '#dc3545' if rsi > 70 else '#28a745' if rsi < 30 else '#007bff'
    fig.add_trace(
        go.Bar(
            x=['RSI'],
            y=[rsi],
            marker_color=rsi_color,
            name='RSI',
            text=[f'{rsi:.1f}'],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Thêm đường reference cho RSI
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
    
    # MACD Bar
    macd_color = '#28a745' if macd > 0 else '#dc3545'
    fig.add_trace(
        go.Bar(
            x=['MACD'],
            y=[macd],
            marker_color=macd_color,
            name='MACD',
            text=[f'{macd:.4f}'],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    # Momentum 5D
    momentum_5d_color = '#28a745' if momentum_5d > 0 else '#dc3545'
    fig.add_trace(
        go.Bar(
            x=['5D'],
            y=[momentum_5d],
            marker_color=momentum_5d_color,
            name='Momentum 5D',
            text=[f'{momentum_5d:.2f}%'],
            textposition='auto'
        ),
        row=2, col=1
    )
    
    # Momentum 20D
    momentum_20d_color = '#28a745' if momentum_20d > 0 else '#dc3545'
    fig.add_trace(
        go.Bar(
            x=['20D'],
            y=[momentum_20d],
            marker_color=momentum_20d_color,
            name='Momentum 20D',
            text=[f'{momentum_20d:.2f}%'],
            textposition='auto'
        ),
        row=2, col=2
    )
    
    # Cập nhật layout
    fig.update_layout(
        title=f'📊 Chỉ báo Kỹ thuật - {symbol}',
        height=500,
        showlegend=False
    )
    
    # Cập nhật trục y cho RSI
    fig.update_yaxes(range=[0, 100], title_text="RSI", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=1, col=2)
    fig.update_yaxes(title_text="Momentum %", row=2, col=1)
    fig.update_yaxes(title_text="Momentum %", row=2, col=2)
    
    return fig

def create_volume_analysis_chart(symbol, current_price):
    """Tạo biểu đồ phân tích khối lượng"""
    
    # Tạo dữ liệu volume giả lập cho 30 ngày
    dates = []
    volumes = []
    prices = []
    
    base_date = datetime.now() - timedelta(days=30)
    base_volume = 1000000  # 1M shares
    
    for i in range(30):
        date = base_date + timedelta(days=i)
        if date.weekday() < 5:  # Chỉ ngày làm việc
            # Volume với xu hướng và nhiễu
            volume = base_volume * (0.8 + 0.4 * (hash(str(date)) % 100 / 100))
            price = current_price * (0.95 + (i/30) * 0.1)
            
            dates.append(date)
            volumes.append(volume)
            prices.append(price)
    
    # Tạo subplot
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Giá cổ phiếu', 'Khối lượng giao dịch'),
        row_heights=[0.7, 0.3]
    )
    
    # Biểu đồ giá
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=prices,
            mode='lines',
            name='Giá',
            line=dict(color='#2E86AB', width=2)
        ),
        row=1, col=1
    )
    
    # Biểu đồ volume
    colors = ['green' if i % 2 == 0 else 'red' for i in range(len(volumes))]
    fig.add_trace(
        go.Bar(
            x=dates,
            y=volumes,
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'📊 Phân tích Giá & Khối lượng - {symbol}',
        height=600,
        xaxis2_title='Thời gian',
        yaxis_title='Giá (VND)',
        yaxis2_title='Khối lượng'
    )
    
    return fig