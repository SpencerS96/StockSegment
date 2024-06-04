import yfinance as yf
import pandas as pd

# Constants
FILE_PATH = 'Data/constituents.csv'  # Path to the CSV file
START_DATE = '2023-01-01'  # Start date for data download
END_DATE = '2023-12-31'    # End date for data download
OUTPUT_FILE_PATH = 'Data/sp500_2023.csv'  # Path to save the downloaded data

def load_sp500_tickers(file_path):
    """
    Load the list of S&P 500 tickers from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file containing the tickers.
    
    Returns:
    list: List of ticker symbols.
    """
    try:
        sp500_data = pd.read_csv(file_path)
        return sp500_data['Symbol'].tolist()
    except Exception as e:
        print(f"error loading the list of tickers")
        exit(1)

def download_stock_data(ticker, start_date, end_date):
    """
    Download stock data for a given ticker and time period.
    
    Parameters:
    ticker (str): The stock ticker symbol.
    start_date (str): The start date for data download.
    end_date (str): The end date for data download.
    
    Returns:
    DataFrame: DataFrame containing the stock data for the given ticker.
    """
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if not stock_data.empty:
            stock_data['Ticker'] = ticker
            return stock_data
        else:
            print(f'No data for ticker: {ticker}')
            return pd.DataFrame()
    except Exception as e:
        print(f'Error for ticker {ticker}: {e}')
        return pd.DataFrame()

def main():
    """
    Main function to load tickers, download stock data, and save the data to a CSV file.
    """
    sp500_tickers = load_sp500_tickers(FILE_PATH)
    
    all_data = pd.DataFrame()
    for ticker in sp500_tickers:
        stock_data = download_stock_data(ticker, START_DATE, END_DATE)
        all_data = pd.concat([all_data, stock_data])
    
    print(all_data.head())
    all_data.to_csv(OUTPUT_FILE_PATH)
    print(f'Data saved to {OUTPUT_FILE_PATH}')

if __name__ == "__main__":
    main()
