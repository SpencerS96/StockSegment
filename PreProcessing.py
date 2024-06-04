import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Constants
FILE_PATH = 'Data/sp500_2023.csv'  # Path to the CSV file
FINAL_OUTPUT_FILE_PATH = 'Data/sp500_final_2023.csv'  # Path to save the final data
PCA_PLOT_PATH = 'Data/pca_plot.png'  # Path to save the PCA plot

def load_stock_data(file_path):
    """
    Load stock data from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file.
    
    Returns:
    DataFrame: Loaded stock data.
    """
    return pd.read_csv(file_path, parse_dates=['Date'])

def preprocess_data(stock_data):
    """
    Preprocess stock data: fill missing values, calculate returns, moving averages, and other indicators.
    
    Parameters:
    stock_data (DataFrame): Raw stock data.
    
    Returns:
    DataFrame: Preprocessed stock data with new features.
    """
    stock_data.ffill(inplace=True)
    stock_data['Daily Return'] = stock_data.groupby('Ticker')['Adj Close'].pct_change()
    stock_data['Cumulative Return'] = stock_data.groupby('Ticker')['Adj Close'].transform(lambda x: (x / x.iloc[0]) - 1)
    stock_data['30 Day MA'] = stock_data.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=30).mean())
    stock_data['30 Day STD'] = stock_data.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=30).std())
    stock_data['30 Day EMA'] = stock_data.groupby('Ticker')['Adj Close'].transform(lambda x: x.ewm(span=30, adjust=False).mean())
    stock_data['RSI'] = stock_data.groupby('Ticker')['Adj Close'].transform(lambda x: calculate_RSI(x))
    stock_data['Bollinger High'], stock_data['Bollinger Low'] = np.nan, np.nan

    for ticker in stock_data['Ticker'].unique():
        upper_band, lower_band = calculate_bollinger_bands(stock_data.loc[stock_data['Ticker'] == ticker, 'Adj Close'])
        stock_data.loc[stock_data['Ticker'] == ticker, 'Bollinger High'] = upper_band
        stock_data.loc[stock_data['Ticker'] == ticker, 'Bollinger Low'] = lower_band

    stock_data.dropna(inplace=True)
    return stock_data

def calculate_RSI(series, period=14):
    """
    Calculate the Relative Strength Index (RSI) for a given series.
    
    Parameters:
    series (Series): Time series data.
    period (int): The period over which to calculate the RSI.
    
    Returns:
    Series: RSI values.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    RS = gain / loss
    return 100 - (100 / (1 + RS))

def calculate_bollinger_bands(series, window=30, num_of_std=2):
    """
    Calculate Bollinger Bands for a given series.
    
    Parameters:
    series (Series): Time series data.
    window (int): The window size for the rolling mean and standard deviation.
    num_of_std (int): Number of standard deviations for the bands.
    
    Returns:
    tuple: Upper and lower Bollinger Bands.
    """
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return upper_band, lower_band

def aggregate_features(stock_data):
    """
    Aggregate features to summarize stock performance over the period.
    
    Parameters:
    stock_data (DataFrame): Preprocessed stock data.
    
    Returns:
    DataFrame: Aggregated feature data.
    """
    summary_data = stock_data.groupby('Ticker').agg({
        'Daily Return': ['mean', 'std'],
        'Cumulative Return': 'mean',
        '30 Day MA': 'mean',
        '30 Day STD': 'mean',
        '30 Day EMA': 'mean',
        'RSI': 'mean',
        'Bollinger High': 'mean',
        'Bollinger Low': 'mean',
        'Volume': 'mean'
    })
    summary_data.columns = ['_'.join(col).strip() for col in summary_data.columns.values]
    return summary_data

def remove_outliers(summary_data, threshold=3):
    """
    Remove outliers from the data using Z-score.
    
    Parameters:
    summary_data (DataFrame): Aggregated feature data.
    threshold (float): Z-score threshold for identifying outliers.
    
    Returns:
    DataFrame: Data with outliers removed.
    """
    z_scores = np.abs(stats.zscore(summary_data))
    outliers = (z_scores > threshold).any(axis=1)
    return summary_data[~outliers]

def normalize_features(summary_data):
    """
    Normalize the features using StandardScaler.
    
    Parameters:
    summary_data (DataFrame): Data to be normalized.
    
    Returns:
    ndarray: Normalized data.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(summary_data)

def apply_pca(normalized_data, n_components=2):
    """
    Apply Principal Component Analysis (PCA) to the data.
    
    Parameters:
    normalized_data (ndarray): Normalized data.
    n_components (int): Number of principal components.
    
    Returns:
    tuple: Principal components and explained variance ratio.
    """
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(normalized_data)
    return principal_components, pca.explained_variance_ratio_

def plot_pca(principal_components, explained_variance_ratio, plot_path):
    """
    Plot the principal components and save the plot.
    
    Parameters:
    principal_components (ndarray): Principal components data.
    explained_variance_ratio (ndarray): Explained variance ratio of the components.
    plot_path (str): Path to save the plot.
    """
    pca_data = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
    plt.figure(figsize=(10, 7))
    plt.scatter(pca_data['Principal Component 1'], pca_data['Principal Component 2'])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Stock Performance Data')
    plt.savefig(plot_path)
    plt.show()
    print(f'PCA plot saved as {plot_path}')
    print('Explained variance ratio:', explained_variance_ratio)

def save_to_csv(data, file_path):
    """
    Save data to a CSV file.
    
    Parameters:
    data (DataFrame): Data to be saved.
    file_path (str): Path to the CSV file.
    """
    data.to_csv(file_path)
    print(f'Data saved to {file_path}')

def main():
    """
    Main function to preprocess stock data, apply PCA, and save the results.
    """
    stock_data = load_stock_data(FILE_PATH)
    stock_data = preprocess_data(stock_data)
    summary_data = aggregate_features(stock_data)
    summary_data = remove_outliers(summary_data)
    normalized_data = normalize_features(summary_data)
    principal_components, explained_variance_ratio = apply_pca(normalized_data)
    
    pca_data = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'], index=summary_data.index)
    final_data = pd.concat([summary_data, pca_data], axis=1)
    
    save_to_csv(final_data, FINAL_OUTPUT_FILE_PATH)
    plot_pca(principal_components, explained_variance_ratio, PCA_PLOT_PATH)

if __name__ == "__main__":
    main()
