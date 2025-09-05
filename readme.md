# 📈 Stock Market Analysis Dashboard

## 🎯 Project Overview
This is a comprehensive stock market analysis system built for an AI/ML course mini project. It analyzes stock performance data, calculates key metrics, and provides interactive visualizations through a Streamlit dashboard.

## 🚀 Features

### 1. **Data Processing**
- Convert YAML files to CSV format
- Merge and clean stock data
- Store data in MySQL database
- Calculate daily and yearly returns

### 2. **Analysis Capabilities**
- **Top Performers**: Identify top 10 gaining and losing stocks
- **Volatility Analysis**: Calculate and visualize stock volatility
- **Cumulative Returns**: Track performance over time
- **Sector Performance**: Analyze stocks by industry sector
- **Correlation Analysis**: Identify relationships between stocks
- **Monthly Performance**: Track top gainers/losers by month

### 3. **Visualizations**
- Interactive charts using Plotly
- Candlestick charts for price movements
- Heatmaps for correlation analysis
- Bar charts for performance comparison
- Line charts for trends over time

### 4. **Dashboard**
- User-friendly Streamlit interface
- Multiple analysis pages
- Real-time metric calculations
- Export capabilities for reports

## 📋 Prerequisites

- Python 3.8 or higher
- MySQL Server installed and running
- Git (for version control)

## 🔧 Installation

1. **Clone the repository** (or create project folder):
```bash
mkdir stock_analysis_project
cd stock_analysis_project
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

3. **Install required packages**:
```bash
pip install -r requirements.txt
```

4. **Set up MySQL Database**:
- Make sure MySQL is running
- Create a database named `stock_analysis`:
```sql
CREATE DATABASE stock_analysis;
```
- Update database credentials in `main.py` if needed

## 📁 Project Structure

```
stock_analysis_project/
│
├── main.py                 # Main data processing script
├── visualizations.py       # Visualization functions
├── streamlit_app.py       # Streamlit dashboard
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
│
├── Requirement Docs/     # Input data folder
│   └── data/            # YAML files go here
│
├── stock_data_csv/      # Generated CSV files (created automatically)
└── All_stock_data.csv   # Merged dataset (created automatically)
```

## 🎮 How to Run

### Step 1: Process Data
First, run the main script to process YAML files and prepare the data:

```bash
python main.py
```

This will:
- Convert YAML files to CSV
- Calculate all metrics
- Store data in MySQL database
- Generate analysis CSV files

### Step 2: Launch Dashboard
Start the Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

The dashboard will open in your browser (usually at `http://localhost:8501`)

## 📊 Using the Dashboard

### Navigation
Use the sidebar to navigate between different analysis pages:

1. **Market Overview**: View overall market statistics and top performers
2. **Performance Analysis**: Compare top gaining and losing stocks
3. **Volatility Analysis**: Identify high-risk stocks
4. **Cumulative Returns**: Track long-term performance
5. **Sector Analysis**: Compare industry sectors
6. **Correlation Analysis**: Find related stocks
7. **Monthly Performance**: View monthly trends
8. **Individual Stock Analysis**: Deep dive into specific stocks

### Interactive Features
- **Sliders**: Adjust the number of stocks displayed
- **Dropdowns**: Select specific stocks or time periods
- **File Upload**: Upload sector mapping CSV (optional)
- **Charts**: Hover for detailed information, zoom, and pan

## 💾 Database Schema

The MySQL database contains a single table `stock_data` with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| stdata | DATETIME | Trading date |
| stopen | FLOAT | Opening price |
| sthigh | FLOAT | Highest price |
| stlow | FLOAT | Lowest price |
| stclose | FLOAT | Closing price |
| stvolume | BIGINT | Trading volume |
| stticker | VARCHAR(50) | Stock symbol |
| daily_return | FLOAT | Daily percentage return |
| yearly_return | FLOAT | Yearly percentage return |

## 📈 Key Metrics Explained

### 1. **Yearly Return**
- Percentage change from first to last closing price of the year
- Formula: `((Last Close - First Close) / First Close) * 100`

### 2. **Daily Return**
- Percentage change between consecutive trading days
- Formula: `((Today's Close - Yesterday's Close) / Yesterday's Close) * 100`

### 3. **Volatility**
- Standard deviation of daily returns
- Higher volatility = Higher risk

### 4. **Cumulative Return**
- Running total of returns over time
- Shows overall growth trajectory

### 5. **Correlation**
- Relationship between stock price movements
- Range: -1 (perfect negative) to +1 (perfect positive)

## 🐛 Troubleshooting

### Common Issues and Solutions

1. **Database Connection Error**:
   - Check if MySQL is running
   - Verify username/password in `main.py`
   - Ensure `stock_analysis` database exists

2. **File Not Found Error**:
   - Run `main.py` before launching dashboard
   - Check if YAML files are in correct folder

3. **Import Error**:
   - Install all requirements: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

4. **Streamlit Not Opening**:
   - Try different port: `streamlit run streamlit_app.py --server.port 8502`
   - Check firewall settings

## 🎯 Learning Objectives

This project demonstrates:
- **Data Processing**: ETL pipeline from YAML to database
- **Data Analysis**: Statistical calculations and metrics
- **Visualization**: Creating meaningful charts and graphs
- **Database Management**: SQL operations and data storage
- **Web Development**: Building interactive dashboards
- **Python Programming**: Modular code organization

## 📝 Notes for Beginners

### Understanding the Code

1. **main.py**: 
   - Start here to understand data flow
   - Each function has clear documentation
   - Comments explain complex operations

2. **visualizations.py**:
   - Contains all plotting functions
   - Each function creates one type of chart
   - Uses Plotly for interactive visualizations

3. **streamlit_app.py**:
   - Main dashboard application
   - Organized into pages for different analyses
   - Uses caching for performance

### Best Practices Demonstrated
- **Modular Design**: Separate files for different functionalities
- **Error Handling**: Try-except blocks for robust code
- **Documentation**: Clear comments and docstrings
- **User Feedback**: Progress messages and error notifications





