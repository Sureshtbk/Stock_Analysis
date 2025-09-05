import os
from pathlib import Path
import yaml
import pandas as pd
import mysql.connector as db
import numpy as np
from datetime import datetime

def convert_flat_yaml_by_ticker(infolder, outfolder):
    """
    Convert YAML files to CSV files organized by ticker symbol
    """
    print("🚀 Starting YAML to CSV by Ticker...")
    
    # Store data for each ticker
    ticker_data = {}
    
    os.makedirs(outfolder, exist_ok=True)
    
    for root, dirs, files in os.walk(infolder):
        for file in files:
            if file.endswith(".yaml") or file.endswith(".yml"):
                fpath = os.path.join(root, file)
                try:
                    with open(fpath, 'r') as f:
                        data = yaml.safe_load(f)
                    
                    if isinstance(data, list):
                        print(f"📄 {file} has {len(data)} records")
                        for row in data:
                            ticker = row.get("Ticker")
                            if ticker:
                                df = pd.DataFrame([row])
                                if ticker in ticker_data:
                                    ticker_data[ticker] = pd.concat([ticker_data[ticker], df], ignore_index=True)
                                else:
                                    ticker_data[ticker] = df
                    else:
                        print(f"⚠️ Unexpected structure in {file}: not a list")
                except Exception as e:
                    print(f"❌ Error reading {file}: {e}")
    
    # Desired column order
    col_name = ["date", "open", "high", "low", "close", "volume", "Ticker"]
    
    for ticker, df in ticker_data.items():
        # Ensure the column order
        for col in col_name:
            if col not in df.columns:
                df[col] = None  # Add missing column as empty
        
        df = df[col_name]  # Reorder columns
        outpath = os.path.join(outfolder, f"{ticker}.csv")
        df.to_csv(outpath, index=False)
        print(f"✅ Saved {ticker}.csv with {len(df)} rows")
    
    print("✅ All files processed.")

def merge_alldata_csv(output_folder):
    """
    Merge all individual stock CSV files into one combined DataFrame
    """
    print("📊 Merging all stock data...")
    combined_df = pd.DataFrame()
    
    for file in os.listdir(output_folder):
        if file.endswith(".csv"):
            symbol = file.replace(".csv", "")
            df = pd.read_csv(os.path.join(output_folder, file))
            df['Ticker'] = symbol
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Calculate daily returns
            df['daily_return'] = df['close'].pct_change()
            
            # Calculate yearly return (from first to last close price)
            if len(df) > 0:
                first_close = df['close'].iloc[0]
                last_close = df['close'].iloc[-1]
                df['yearly_return'] = ((last_close - first_close) / first_close) * 100
            else:
                df['yearly_return'] = 0
            
            combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    return combined_df

def database_connect(all_data_df):
    """
    Connect to MySQL and insert rows, converting all NaN/NaT/±inf -> NULL (None).
    Includes per-tuple final scrub and a null summary for debugging.
    """
    import math
    import numpy as np
    import pandas as pd
    import mysql.connector as db

    print("💾 Connecting to database...")

    try:
        conn = db.connect(
            host="localhost",
            user="root",
            password="test123#",
            database="stock_analysis",
            raise_on_warnings=True,
        )
        cur = conn.cursor()

        # Recreate table
        cur.execute("DROP TABLE IF EXISTS stock_data")
        cur.execute("""
            CREATE TABLE stock_data (
                stdata DATETIME,
                stopen DOUBLE,
                sthigh DOUBLE,
                stlow  DOUBLE,
                stclose DOUBLE,
                stvolume BIGINT,
                stticker VARCHAR(50),
                daily_return DOUBLE,
                yearly_return DOUBLE
            )
        """)

        # --- Build a clean records frame in the right order ---
        required = ['date','open','high','low','close','volume','Ticker','daily_return','yearly_return']
        miss = [c for c in required if c not in all_data_df.columns]
        if miss:
            raise ValueError(f"Missing columns for DB insert: {miss}")

        records = all_data_df[required].copy()

        # Normalize dtypes
        records['date'] = pd.to_datetime(records['date'], errors='coerce')

        # IMPORTANT: use object dtype so None survives
        dtype_map = {c: 'object' for c in required}
        records = records.astype(dtype_map)

        # Date: Timestamp/NaT -> python datetime/None
        records['date'] = records['date'].apply(lambda x: (x.to_pydatetime() if pd.notna(x) else None))

        # Numeric columns: coerce, inf -> NaN -> None
        num_cols = ['open','high','low','close','volume','daily_return','yearly_return']
        for c in num_cols:
            records[c] = pd.to_numeric(records[c], errors='coerce')
            # Replace infinities with NaN, then NaN -> None
            records[c] = records[c].replace([np.inf, -np.inf], np.nan)
            records[c] = records[c].where(records[c].notna(), None)

        # Ticker: keep as object; NaN -> None (do NOT cast to str)
        records['Ticker'] = records['Ticker'].where(records['Ticker'].notna(), None)

        # Final DataFrame-level sweep
        records = records.where(records.notna(), None)

        # --- Debug: show null counts per column (helps pinpoint sources) ---
        null_summary = {c: int(records[c].isna().sum()) for c in records.columns}
        print("🔎 NULL summary before insert:", null_summary)

        # Per-tuple final scrub: convert any lingering float NaN to None
        rows = []
        for tup in records.itertuples(index=False, name=None):
            cleaned = []
            for v in tup:
                if isinstance(v, float) and math.isnan(v):
                    cleaned.append(None)
                else:
                    cleaned.append(v)
            rows.append(tuple(cleaned))

        insert_sql = """
            INSERT INTO stock_data
            (stdata, stopen, sthigh, stlow, stclose, stvolume, stticker, daily_return, yearly_return)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """
        cur.executemany(insert_sql, rows)
        conn.commit()
        cur.close()
        conn.close()

        print(f"✅ Data inserted into stock_data successfully. Rows: {len(rows)}")

    except Exception as e:
        print(f"❌ Database error: {e}")



def calculate_key_metrics(df):
    """
    Calculate key metrics for the analysis
    """
    print("📈 Calculating key metrics...")
    
    # Get latest yearly return for each stock
    latest_data = df.groupby('Ticker').last().reset_index()
    
    # Top 10 Green Stocks (highest yearly returns)
    top_10_green = latest_data.nlargest(10, 'yearly_return')[['Ticker', 'yearly_return', 'close']]
    
    # Top 10 Loss Stocks (lowest yearly returns)
    top_10_loss = latest_data.nsmallest(10, 'yearly_return')[['Ticker', 'yearly_return', 'close']]
    
    # Market Summary
    green_stocks = len(latest_data[latest_data['yearly_return'] > 0])
    red_stocks = len(latest_data[latest_data['yearly_return'] <= 0])
    avg_price = df['close'].mean()
    avg_volume = df['volume'].mean()
    
    market_summary = {
        'Total Stocks': len(latest_data),
        'Green Stocks': green_stocks,
        'Red Stocks': red_stocks,
        'Average Price': avg_price,
        'Average Volume': avg_volume,
        'Market Sentiment': 'Bullish' if green_stocks > red_stocks else 'Bearish'
    }
    
    return top_10_green, top_10_loss, market_summary

def calculate_volatility(df):
    """
    Calculate volatility (standard deviation of daily returns) for each stock
    """
    print("📊 Calculating volatility...")
    
    volatility_data = df.groupby('Ticker')['daily_return'].std().reset_index()
    volatility_data.columns = ['Ticker', 'Volatility']
    volatility_data['Volatility'] = volatility_data['Volatility'] * 100  # Convert to percentage
    volatility_data = volatility_data.sort_values('Volatility', ascending=False)
    
    return volatility_data

def calculate_cumulative_returns(df):
    """
    Calculate cumulative returns over time for each stock
    """
    print("📈 Calculating cumulative returns...")
    
    cumulative_returns = pd.DataFrame()
    
    for ticker in df['Ticker'].unique():
        stock_data = df[df['Ticker'] == ticker].copy()
        stock_data = stock_data.sort_values('date')
        
        # Calculate cumulative return
        stock_data['cumulative_return'] = (1 + stock_data['daily_return'].fillna(0)).cumprod() - 1
        stock_data['cumulative_return'] = stock_data['cumulative_return'] * 100  # Convert to percentage
        
        cumulative_returns = pd.concat([cumulative_returns, stock_data])
    
    return cumulative_returns

def calculate_correlation_matrixBI(df):
    """
    Calculates the correlation matrix and saves it to a CSV file.
    The data is unpivoted to be in a format suitable for a Power BI matrix/heatmap.
    """
    print("📈 Calculating stock price correlation matrix...")
    
    # Pivot the data to have tickers as columns and dates as index
    df_pivot = df.pivot_table(index='date', columns='Ticker', values='close')
    
    # Calculate the correlation matrix
    correlation_matrix = df_pivot.corr()
    
    # Unpivot the correlation matrix to a long format
    correlation_matrix_unpivoted = correlation_matrix.reset_index().melt(
        id_vars='Ticker',
        var_name='Correlated Ticker',
        value_name='Correlation'
    )
    
    # Save the unpivoted correlation data to a CSV
    correlation_matrix_unpivoted.to_csv("correlation_matrix.csv", index=False)
    print("✅ Correlation matrix saved to correlation_matrix.csv")

def calculate_correlation_matrix(df):
    """
    Calculate correlation matrix between stock closing prices
    """
    print("🔗 Calculating stock price correlations...")
    
    # Pivot the data to have tickers as columns and dates as rows
    pivot_df = df.pivot_table(values='close', index='date', columns='Ticker')
    
    # Calculate correlation matrix
    correlation_matrix = pivot_df.corr()
    
    return correlation_matrix

def calculate_monthly_performance(df):
    """
    Calculate monthly top gainers and losers
    """
    print("📅 Calculating monthly performance...")
    
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
    monthly_performance = {}
    
    for month in df['month'].unique():
        month_data = df[df['month'] == month]
        
        # Calculate monthly returns for each stock
        monthly_returns = []
        for ticker in month_data['Ticker'].unique():
            stock_month = month_data[month_data['Ticker'] == ticker].sort_values('date')
            if len(stock_month) > 0:
                first_close = stock_month['close'].iloc[0]
                last_close = stock_month['close'].iloc[-1]
                monthly_return = ((last_close - first_close) / first_close) * 100
                monthly_returns.append({'Ticker': ticker, 'Monthly_Return': monthly_return})
        
        month_df = pd.DataFrame(monthly_returns)
        
        # Get top 5 gainers and losers
        top_5_gainers = month_df.nlargest(5, 'Monthly_Return')
        top_5_losers = month_df.nsmallest(5, 'Monthly_Return')
        
        monthly_performance[str(month)] = {
            'gainers': top_5_gainers,
            'losers': top_5_losers
        }
    
    return monthly_performance

def main():
    """
    Main execution function
    """
    print("🎯 Starting Stock Market Analysis Project")
    print("=" * 50)
    
    # Paths
    current_dir = Path(__file__).resolve().parent
    input_folder = current_dir.parent / "Requirement Docs" / "data"
    output_folder = current_dir.parent / "stock_data_csv"
    
    # Step 1: Convert YAML to CSV
    convert_flat_yaml_by_ticker(str(input_folder), str(output_folder))
    
    # Step 2: Merge all data
    df = merge_alldata_csv(output_folder)
    
     # headmap data for BI
    calculate_correlation_matrixBI(df)

     # NEW: Monthly Top Performers
    calculate_monthly_top_performers(df)

    # Step 3: Save merged data
    df.to_csv("All_stock_data.csv", index=False)
    print("✅ Merged & cleaned data saved as 'All_stock_data.csv'")
    
    # Step 4: Insert into database
    database_connect(df)
    
    # Step 5: Calculate all metrics
    print("\n" + "=" * 50)
    print("📊 ANALYSIS RESULTS")
    print("=" * 50)
    
    # Key Metrics
    top_10_green, top_10_loss, market_summary = calculate_key_metrics(df)
    
    print("\n🟢 Top 10 Green Stocks:")
    print(top_10_green.to_string(index=False))
    
    print("\n🔴 Top 10 Loss Stocks:")
    print(top_10_loss.to_string(index=False))
    
    print("\n📊 Market Summary:")
    for key, value in market_summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Volatility
    volatility = calculate_volatility(df)
    print("\n⚡ Top 10 Most Volatile Stocks:")
    print(volatility.head(10).to_string(index=False))
    
    # Save all analysis results
    print("\n💾 Saving analysis results...")
    top_10_green.to_csv("top_10_green_stocks.csv", index=False)
    top_10_loss.to_csv("top_10_loss_stocks.csv", index=False)
    volatility.to_csv("volatility_analysis.csv", index=False)
    
    print("\n✅ Analysis complete! Check the generated CSV files and run streamlit_app.py for visualizations.")

def calculate_monthly_top_performers(df):
    """
    Calculates top 5 gainers and losers for each month and saves the data to a CSV file.
    """
    print("🗓️ Calculating monthly top gainers and losers...")
    
    # Ensure 'date' is a datetime object and sort the data
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['Ticker', 'date'])
    
    # Get the first and last closing price for each month for each ticker
    monthly_summary = df.groupby([
        'Ticker', 
        df['date'].dt.year.rename('year'), 
        df['date'].dt.month.rename('month')
    ])['close'].agg(['first', 'last']).reset_index()
    
    # Calculate monthly return
    monthly_summary['monthly_return'] = ((monthly_summary['last'] - monthly_summary['first']) / monthly_summary['first']) * 100
    
    # Create an empty list to store the top and bottom performers
    monthly_results = []
    
    # Loop through each unique month
    for year, month in monthly_summary.groupby(['year', 'month']).groups.keys():
        monthly_data = monthly_summary[(monthly_summary['year'] == year) & (monthly_summary['month'] == month)]
        
        # Get top 5 gainers
        top_5_gainers = monthly_data.nlargest(5, 'monthly_return')
        top_5_gainers['type'] = 'Gainer'
        
        # Get top 5 losers
        top_5_losers = monthly_data.nsmallest(5, 'monthly_return')
        top_5_losers['type'] = 'Loser'
        
        # Combine and add to the results list
        monthly_results.append(pd.concat([top_5_gainers, top_5_losers]))
        
    # Concatenate all monthly results into a single DataFrame
    final_df = pd.concat(monthly_results, ignore_index=True)
    
    # Save the final DataFrame to CSV
    final_df.to_csv("monthly_top_performers.csv", index=False)
    print("✅ Monthly top gainers and losers data saved to monthly_top_performers.csv")
    

def calculate_sector_performance(df, sector_file_path):
    """
    Calculates the average yearly return for each sector.
    """
    try:
        sector_df = pd.read_csv(sector_file_path)
        sector_mapping = dict(zip(sector_df['Ticker'], sector_df['Sector']))
        df['Sector'] = df['Ticker'].map(sector_mapping)

        # Group by sector and calculate average yearly return
        sector_performance = df.groupby('Sector')['yearly_return'].mean().reset_index()
        return sector_performance
    except FileNotFoundError:
        print(f"❌ Sector file '{sector_file_path}' not found. Sector analysis will be skipped.")
        return None

if __name__ == "__main__":
    main()