import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def plot_top_performers(top_10_green, top_10_loss):
    """
    Create a bar chart showing top 10 gainers and losers
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Top 10 Gaining Stocks', 'Top 10 Losing Stocks')
    )
    
    # Top gainers
    fig.add_trace(
        go.Bar(
            x=top_10_green['Ticker'],
            y=top_10_green['yearly_return'],
            name='Gainers',
            marker_color='green',
            text=top_10_green['yearly_return'].round(2),
            textposition='auto',
        ),
        row=1, col=1
    )
    
    # Top losers
    fig.add_trace(
        go.Bar(
            x=top_10_loss['Ticker'],
            y=top_10_loss['yearly_return'],
            name='Losers',
            marker_color='red',
            text=top_10_loss['yearly_return'].round(2),
            textposition='auto',
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="Top Performing and Worst Performing Stocks",
        showlegend=False,
        height=500,
        template='plotly_white'
    )
    
    fig.update_xaxes(title_text="Stock Ticker", row=1, col=1)
    fig.update_xaxes(title_text="Stock Ticker", row=1, col=2)
    fig.update_yaxes(title_text="Yearly Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Yearly Return (%)", row=1, col=2)
    
    return fig

def plot_volatility_analysis(volatility_data, top_n=10):
    """
    Create a bar chart showing the most volatile stocks
    """
    top_volatile = volatility_data.head(top_n)
    
    fig = go.Figure(data=[
        go.Bar(
            x=top_volatile['Ticker'],
            y=top_volatile['Volatility'],
            marker_color=top_volatile['Volatility'],
            marker_colorscale='Reds',
            text=top_volatile['Volatility'].round(2),
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f'Top {top_n} Most Volatile Stocks',
        xaxis_title='Stock Ticker',
        yaxis_title='Volatility (Standard Deviation %)',
        template='plotly_white',
        showlegend=False,
        height=500
    )
    
    return fig

def plot_cumulative_returns(cumulative_returns_df, top_n=5):
    """
    Create a line chart showing cumulative returns over time for top performers
    """
    # Get the final cumulative return for each stock
    final_returns = cumulative_returns_df.groupby('Ticker')['cumulative_return'].last().sort_values(ascending=False)
    top_tickers = final_returns.head(top_n).index
    
    # Filter data for top performers
    top_performers_data = cumulative_returns_df[cumulative_returns_df['Ticker'].isin(top_tickers)]
    
    fig = go.Figure()
    
    for ticker in top_tickers:
        ticker_data = top_performers_data[top_performers_data['Ticker'] == ticker]
        fig.add_trace(go.Scatter(
            x=ticker_data['date'],
            y=ticker_data['cumulative_return'],
            mode='lines',
            name=ticker,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title=f'Cumulative Returns Over Time - Top {top_n} Performers',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        template='plotly_white',
        hovermode='x unified',
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def plot_sector_performance(df, sector_mapping=None):
    """
    Create a bar chart showing average performance by sector
    """
    if sector_mapping is None:
        # Create a sample sector mapping (you should replace this with actual sector data)
        unique_tickers = df['Ticker'].unique()
        np.random.seed(42)  # For reproducibility
        sectors = np.random.choice(['Technology', 'Finance', 'Healthcare', 'Energy', 'Consumer', 'Industrial'], 
                                  size=len(unique_tickers))
        sector_mapping = dict(zip(unique_tickers, sectors))
    
    # Add sector column to dataframe
    df['Sector'] = df['Ticker'].map(sector_mapping)
    
    # Calculate average yearly return by sector
    sector_performance = df.groupby('Sector')['yearly_return'].mean().reset_index()
    sector_performance = sector_performance.sort_values('yearly_return', ascending=False)
    
    # Create color mapping based on positive/negative returns
    colors = ['green' if x > 0 else 'red' for x in sector_performance['yearly_return']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=sector_performance['Sector'],
            y=sector_performance['yearly_return'],
            marker_color=colors,
            text=sector_performance['yearly_return'].round(2),
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Average Stock Performance by Sector',
        xaxis_title='Sector',
        yaxis_title='Average Yearly Return (%)',
        template='plotly_white',
        showlegend=False,
        height=500
    )
    
    return fig, df

def plot_correlation_heatmap(correlation_matrix, top_n=20):
    """
    Create a heatmap showing correlation between stock prices
    """
    # Select top N stocks for better visualization
    if len(correlation_matrix) > top_n:
        correlation_matrix = correlation_matrix.iloc[:top_n, :top_n]
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title=f'Stock Price Correlation Matrix (Top {top_n} Stocks)',
        xaxis_title='Stock Ticker',
        yaxis_title='Stock Ticker',
        template='plotly_white',
        height=700,
        width=800
    )
    
    return fig

def plot_monthly_performance(monthly_performance, month_key):
    """
    Create bar charts for monthly top gainers and losers
    """
    month_data = monthly_performance[month_key]
    gainers = month_data['gainers']
    losers = month_data['losers']
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'Top 5 Gainers - {month_key}', f'Top 5 Losers - {month_key}')
    )
    
    # Gainers
    fig.add_trace(
        go.Bar(
            x=gainers['Ticker'],
            y=gainers['Monthly_Return'],
            marker_color='green',
            text=gainers['Monthly_Return'].round(2),
            textposition='auto',
            name='Gainers'
        ),
        row=1, col=1
    )
    
    # Losers
    fig.add_trace(
        go.Bar(
            x=losers['Ticker'],
            y=losers['Monthly_Return'],
            marker_color='red',
            text=losers['Monthly_Return'].round(2),
            textposition='auto',
            name='Losers'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        template='plotly_white'
    )
    
    fig.update_xaxes(title_text="Stock Ticker", row=1, col=1)
    fig.update_xaxes(title_text="Stock Ticker", row=1, col=2)
    fig.update_yaxes(title_text="Monthly Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Monthly Return (%)", row=1, col=2)
    
    return fig

def plot_market_summary_pie(market_summary):
    """
    Create a pie chart showing market composition (green vs red stocks)
    """
    labels = ['Green Stocks', 'Red Stocks']
    values = [market_summary['Green Stocks'], market_summary['Red Stocks']]
    colors = ['green', 'red']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        marker_colors=colors,
        textinfo='label+percent',
        textposition='auto'
    )])
    
    fig.update_layout(
        title='Market Composition: Green vs Red Stocks',
        template='plotly_white',
        height=400,
        annotations=[dict(text='Market<br>Overview', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    
    return fig

def plot_volume_distribution(df):
    """
    Create a histogram showing volume distribution across all stocks
    """
    fig = go.Figure(data=[go.Histogram(
        x=df['volume'] / 1000000,  # Convert to millions
        nbinsx=50,
        marker_color='blue',
        opacity=0.7
    )])
    
    fig.update_layout(
        title='Trading Volume Distribution',
        xaxis_title='Volume (Millions)',
        yaxis_title='Frequency',
        template='plotly_white',
        showlegend=False,
        height=400
    )
    
    return fig

def plot_price_trend(df, ticker):
    """
    Create a candlestick chart for a specific stock
    """
    stock_data = df[df['Ticker'] == ticker].sort_values('date')
    
    fig = go.Figure(data=[go.Candlestick(
        x=stock_data['date'],
        open=stock_data['open'],
        high=stock_data['high'],
        low=stock_data['low'],
        close=stock_data['close'],
        name=ticker
    )])
    
    fig.update_layout(
        title=f'{ticker} - Stock Price Movement',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        height=500
    )
    
    return fig

def create_summary_dashboard(df, top_10_green, top_10_loss, market_summary, volatility_data):
    """
    Create a comprehensive dashboard with multiple visualizations
    """
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Market Composition', 'Top Gainers', 
                       'Top Losers', 'Most Volatile Stocks',
                       'Volume Distribution', 'Market Statistics'),
        specs=[[{'type': 'pie'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'histogram'}, {'type': 'table'}]]
    )
    
    # 1. Market Composition Pie Chart
    fig.add_trace(
        go.Pie(
            labels=['Green Stocks', 'Red Stocks'],
            values=[market_summary['Green Stocks'], market_summary['Red Stocks']],
            marker_colors=['green', 'red'],
            hole=.3
        ),
        row=1, col=1
    )
    
    # 2. Top Gainers
    fig.add_trace(
        go.Bar(
            x=top_10_green['Ticker'].head(5),
            y=top_10_green['yearly_return'].head(5),
            marker_color='green',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. Top Losers
    fig.add_trace(
        go.Bar(
            x=top_10_loss['Ticker'].head(5),
            y=top_10_loss['yearly_return'].head(5),
            marker_color='red',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 4. Most Volatile
    fig.add_trace(
        go.Bar(
            x=volatility_data['Ticker'].head(5),
            y=volatility_data['Volatility'].head(5),
            marker_color='orange',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # 5. Volume Distribution
    fig.add_trace(
        go.Histogram(
            x=df['volume'] / 1000000,
            nbinsx=30,
            marker_color='blue',
            showlegend=False
        ),
        row=3, col=1
    )
    
    # 6. Market Statistics Table
    stats_data = [
        ['Total Stocks', str(market_summary['Total Stocks'])],
        ['Avg Price', f"${market_summary['Average Price']:.2f}"],
        ['Avg Volume', f"{market_summary['Average Volume']/1000000:.2f}M"],
        ['Market Sentiment', market_summary['Market Sentiment']]
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(values=['Metric', 'Value'],
                       fill_color='paleturquoise',
                       align='left'),
            cells=dict(values=list(zip(*stats_data)),
                      fill_color='lavender',
                      align='left')
        ),
        row=3, col=2
    )
    
    fig.update_layout(
        title_text="Stock Market Analysis Dashboard",
        showlegend=False,
        height=1000,
        template='plotly_white'
    )
    
    return fig