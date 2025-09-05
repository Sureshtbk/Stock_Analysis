import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from pathlib import Path

# Import our custom modules
from main import (
    calculate_key_metrics, 
    calculate_volatility,
    calculate_cumulative_returns,
    calculate_correlation_matrix,
    calculate_monthly_performance
)
from visualizations import (
    plot_top_performers,
    plot_volatility_analysis,
    plot_cumulative_returns,
    plot_sector_performance,
    plot_correlation_heatmap,
    plot_monthly_performance,
    plot_market_summary_pie,
    plot_volume_distribution,
    plot_price_trend,
    create_summary_dashboard
)

# Page configuration
st.set_page_config(
    page_title="Stock Market Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
        padding: 20px 0;
    }
    h2 {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    """
    Load the processed stock data
    """
    try:
        # Try to load the merged data file
        df = pd.read_csv("All_stock_data.csv")
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        st.error("❌ Data file not found! Please run main.py first to process the data.")
        return None

def main():
    # Title and description
    st.title("📈 Stock Market Analysis Dashboard")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["📊 Market Overview", 
         "💹 Performance Analysis",
         "⚡ Volatility Analysis",
         "📈 Cumulative Returns",
         "🏢 Sector Analysis",
         "🔗 Correlation Analysis",
         "📅 Monthly Performance",
         "🕯️ Individual Stock Analysis"]
    )
    
    # Load data
    df = load_data()
    
    if df is None:
        st.stop()
    
    # Calculate all metrics
    with st.spinner("Calculating metrics..."):
        top_10_green, top_10_loss, market_summary = calculate_key_metrics(df)
        volatility_data = calculate_volatility(df)
        cumulative_returns = calculate_cumulative_returns(df)
        correlation_matrix = calculate_correlation_matrix(df)
        monthly_performance = calculate_monthly_performance(df)
    
    # Page routing
    if page == "📊 Market Overview":
        show_market_overview(df, top_10_green, top_10_loss, market_summary, volatility_data)
    
    elif page == "💹 Performance Analysis":
        show_performance_analysis(top_10_green, top_10_loss)
    
    elif page == "⚡ Volatility Analysis":
        show_volatility_analysis(volatility_data)
    
    elif page == "📈 Cumulative Returns":
        show_cumulative_returns(cumulative_returns)
    
    elif page == "🏢 Sector Analysis":
        show_sector_analysis(df)
    
    elif page == "🔗 Correlation Analysis":
        show_correlation_analysis(correlation_matrix)
    
    elif page == "📅 Monthly Performance":
        show_monthly_performance(monthly_performance)
    
    elif page == "🕯️ Individual Stock Analysis":
        show_individual_stock_analysis(df)
    
    # Footer
    st.markdown("---")
    st.markdown("📊 **Stock Market Analysis Dashboard** | Created for AI/ML Course Mini Project")

def show_market_overview(df, top_10_green, top_10_loss, market_summary, volatility_data):
    """Display market overview page"""
    st.header("📊 Market Overview")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Stocks", market_summary['Total Stocks'])
    
    with col2:
        st.metric("Green Stocks 🟢", market_summary['Green Stocks'])
    
    with col3:
        st.metric("Red Stocks 🔴", market_summary['Red Stocks'])
    
    with col4:
        st.metric("Market Sentiment", market_summary['Market Sentiment'],
                 delta="Bullish" if market_summary['Market Sentiment'] == 'Bullish' else "Bearish")
    
    st.markdown("---")
    
    # Additional metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Average Stock Price", f"${market_summary['Average Price']:.2f}")
    
    with col2:
        st.metric("Average Trading Volume", f"{market_summary['Average Volume']/1000000:.2f}M")
    
    st.markdown("---")
    
    # Market composition pie chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Market Composition")
        fig = plot_market_summary_pie(market_summary)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Volume Distribution")
        fig = plot_volume_distribution(df)
        st.plotly_chart(fig, use_container_width=True)
    
    # Top performers tables
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🟢 Top 10 Gaining Stocks")
        st.dataframe(
            top_10_green.style.format({'yearly_return': '{:.2f}%', 'close': '${:.2f}'}),
            use_container_width=True
        )
    
    with col2:
        st.subheader("🔴 Top 10 Losing Stocks")
        st.dataframe(
            top_10_loss.style.format({'yearly_return': '{:.2f}%', 'close': '${:.2f}'}),
            use_container_width=True
        )

def show_performance_analysis(top_10_green, top_10_loss):
    """Display performance analysis page"""
    st.header("💹 Stock Performance Analysis")
    
    # Performance comparison chart
    fig = plot_top_performers(top_10_green, top_10_loss)
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance statistics
    st.markdown("---")
    st.subheader("Performance Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🟢 Gainers Statistics")
        st.write(f"**Best Performer:** {top_10_green.iloc[0]['Ticker']}")
        st.write(f"**Best Return:** {top_10_green.iloc[0]['yearly_return']:.2f}%")
        st.write(f"**Average Return:** {top_10_green['yearly_return'].mean():.2f}%")
        st.write(f"**Median Return:** {top_10_green['yearly_return'].median():.2f}%")
    
    with col2:
        st.markdown("### 🔴 Losers Statistics")
        st.write(f"**Worst Performer:** {top_10_loss.iloc[0]['Ticker']}")
        st.write(f"**Worst Return:** {top_10_loss.iloc[0]['yearly_return']:.2f}%")
        st.write(f"**Average Loss:** {top_10_loss['yearly_return'].mean():.2f}%")
        st.write(f"**Median Loss:** {top_10_loss['yearly_return'].median():.2f}%")

def show_volatility_analysis(volatility_data):
    """Display volatility analysis page"""
    st.header("⚡ Volatility Analysis")
    
    st.markdown("""
    **Volatility** measures the degree of variation in a stock's price over time. 
    Higher volatility indicates greater price fluctuations and potentially higher risk.
    """)
    
    # Volatility chart
    top_n = st.slider("Select number of stocks to display:", 5, 20, 10)
    fig = plot_volatility_analysis(volatility_data, top_n)
    st.plotly_chart(fig, use_container_width=True)
    
    # Volatility statistics
    st.markdown("---")
    st.subheader("Volatility Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Most Volatile Stock", 
                 volatility_data.iloc[0]['Ticker'],
                 f"{volatility_data.iloc[0]['Volatility']:.2f}%")
    
    with col2:
        st.metric("Average Volatility", 
                 f"{volatility_data['Volatility'].mean():.2f}%")
    
    with col3:
        st.metric("Median Volatility", 
                 f"{volatility_data['Volatility'].median():.2f}%")
    
    # Volatility table
    st.markdown("---")
    st.subheader("Volatility Rankings")
    st.dataframe(
        volatility_data.head(20).style.format({'Volatility': '{:.2f}%'}),
        use_container_width=True
    )

def show_cumulative_returns(cumulative_returns):
    """Display cumulative returns analysis page"""
    st.header("📈 Cumulative Returns Analysis")
    
    st.markdown("""
    **Cumulative returns** show the total return of an investment over time, 
    accounting for all gains and losses from the starting point.
    """)
    
    # Select number of top performers
    top_n = st.slider("Select number of top performers to display:", 3, 10, 5)
    
    # Cumulative returns chart
    fig = plot_cumulative_returns(cumulative_returns, top_n)
    st.plotly_chart(fig, use_container_width=True)
    
    # Final returns summary
    st.markdown("---")
    st.subheader("Final Cumulative Returns")
    
    final_returns = cumulative_returns.groupby('Ticker')['cumulative_return'].last().sort_values(ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Top 10 Performers")
        top_performers = final_returns.head(10).reset_index()
        st.dataframe(
            top_performers.style.format({'cumulative_return': '{:.2f}%'}),
            use_container_width=True
        )
    
    with col2:
        st.markdown("### Bottom 10 Performers")
        bottom_performers = final_returns.tail(10).reset_index()
        st.dataframe(
            bottom_performers.style.format({'cumulative_return': '{:.2f}%'}),
            use_container_width=True
        )

def show_sector_analysis(df):
    """Display sector analysis page"""
    st.header("🏢 Sector-wise Performance Analysis")

    # Load sector data (from a static file)
    try:
        sector_df = pd.read_csv('sector_data.csv')
        sector_mapping = dict(zip(sector_df['Ticker'], sector_df['Sector']))
    except FileNotFoundError:
        st.error("Sector data file 'sector_data.csv' not found. Please add the file to your project directory.")
        return

    fig, df_with_sectors = plot_sector_performance(df, sector_mapping)
    st.plotly_chart(fig, use_container_width=True)

    
    # Sector statistics
    st.markdown("---")
    st.subheader("Sector Statistics")
    
    sector_stats = df_with_sectors.groupby('Sector').agg({
        'yearly_return': ['mean', 'median', 'std', 'count']
    }).round(2)
    
    sector_stats.columns = ['Average Return (%)', 'Median Return (%)', 'Std Dev (%)', 'Stock Count']
    st.dataframe(sector_stats, use_container_width=True)

def show_correlation_analysis(correlation_matrix):
    """Display correlation analysis page"""
    st.header("🔗 Stock Price Correlation Analysis")
    
    st.markdown("""
    **Correlation analysis** helps identify stocks that move together. 
    - Values close to **1** indicate strong positive correlation (stocks move in the same direction)
    - Values close to **-1** indicate strong negative correlation (stocks move in opposite directions)
    - Values close to **0** indicate no correlation
    """)
    
    # Select number of stocks for correlation matrix
    top_n = st.slider("Select number of stocks for correlation matrix:", 10, 30, 20)
    
    # Correlation heatmap
    fig = plot_correlation_heatmap(correlation_matrix, top_n)
    st.plotly_chart(fig, use_container_width=True)
    
    # Most correlated pairs
    st.markdown("---")
    st.subheader("Highly Correlated Stock Pairs")
    
    # Find highly correlated pairs
    corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:  # Threshold for high correlation
                corr_pairs.append({
                    'Stock 1': correlation_matrix.columns[i],
                    'Stock 2': correlation_matrix.columns[j],
                    'Correlation': corr_value
                })
    
    if corr_pairs:
        corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False)
        st.dataframe(
            corr_df.style.format({'Correlation': '{:.3f}'}),
            use_container_width=True
        )
    else:
        st.info("No highly correlated pairs found (threshold: |correlation| > 0.7)")

def show_monthly_performance(monthly_performance):
    """Display monthly performance analysis page"""
    st.header("📅 Monthly Performance Analysis")
    
    # Month selector
    available_months = list(monthly_performance.keys())
    selected_month = st.selectbox("Select Month:", available_months)
    
    # Display monthly performance chart
    if selected_month:
        fig = plot_monthly_performance(monthly_performance, selected_month)
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly statistics
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🟢 Top 5 Gainers")
            gainers = monthly_performance[selected_month]['gainers']
            st.dataframe(
                gainers.style.format({'Monthly_Return': '{:.2f}%'}),
                use_container_width=True
            )
        
        with col2:
            st.subheader("🔴 Top 5 Losers")
            losers = monthly_performance[selected_month]['losers']
            st.dataframe(
                losers.style.format({'Monthly_Return': '{:.2f}%'}),
                use_container_width=True
            )
    
    # Monthly trends overview
    st.markdown("---")
    st.subheader("Monthly Trends Overview")
    
    # Create summary of all months
    monthly_summary = []
    for month, data in monthly_performance.items():
        gainers_avg = data['gainers']['Monthly_Return'].mean()
        losers_avg = data['losers']['Monthly_Return'].mean()
        monthly_summary.append({
            'Month': month,
            'Avg Gain (Top 5)': gainers_avg,
            'Avg Loss (Bottom 5)': losers_avg
        })
    
    summary_df = pd.DataFrame(monthly_summary)
    st.dataframe(
        summary_df.style.format({'Avg Gain (Top 5)': '{:.2f}%', 'Avg Loss (Bottom 5)': '{:.2f}%'}),
        use_container_width=True
    )

def show_individual_stock_analysis(df):
    """Display individual stock analysis page"""
    st.header("🕯️ Individual Stock Analysis")
    
    # Stock selector
    stocks = sorted(df['Ticker'].unique())
    selected_stock = st.selectbox("Select a Stock:", stocks)
    
    if selected_stock:
        stock_data = df[df['Ticker'] == selected_stock].sort_values('date')
        
        # Stock metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            latest_price = stock_data['close'].iloc[-1]
            st.metric("Latest Close Price", f"${latest_price:.2f}")
        
        with col2:
            yearly_return = stock_data['yearly_return'].iloc[-1]
            st.metric("Yearly Return", f"{yearly_return:.2f}%",
                     delta="Positive" if yearly_return > 0 else "Negative")
        
        with col3:
            avg_volume = stock_data['volume'].mean()
            st.metric("Avg Volume", f"{avg_volume/1000000:.2f}M")
        
        with col4:
            volatility = stock_data['daily_return'].std() * 100
            st.metric("Volatility", f"{volatility:.2f}%")
        
        # Price chart
        st.markdown("---")
        st.subheader(f"{selected_stock} Price Movement")
        
        # Chart type selector
        chart_type = st.radio("Select Chart Type:", ["Candlestick", "Line Chart"])
        
        if chart_type == "Candlestick":
            fig = plot_price_trend(df, selected_stock)
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=stock_data['date'],
                y=stock_data['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ))
            fig.update_layout(
                title=f'{selected_stock} - Closing Price Trend',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                template='plotly_white',
                height=500
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart
        st.markdown("---")
        st.subheader(f"{selected_stock} Trading Volume")
        
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(
            x=stock_data['date'],
            y=stock_data['volume']/1000000,
            name='Volume',
            marker_color='lightblue'
        ))
        fig_volume.update_layout(
            xaxis_title='Date',
            yaxis_title='Volume (Millions)',
            template='plotly_white',
            height=300
        )
        
        st.plotly_chart(fig_volume, use_container_width=True)
        
        # Stock statistics
        st.markdown("---")
        st.subheader("Stock Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**52-Week High:** ${stock_data['high'].max():.2f}")
            st.write(f"**52-Week Low:** ${stock_data['low'].min():.2f}")
            st.write(f"**Average Price:** ${stock_data['close'].mean():.2f}")
        
        with col2:
            st.write(f"**Price Range:** ${stock_data['high'].max() - stock_data['low'].min():.2f}")
            st.write(f"**Total Trading Days:** {len(stock_data)}")
            st.write(f"**First Trade Date:** {stock_data['date'].min().strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    main()