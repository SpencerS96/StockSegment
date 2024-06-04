import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Constants
OUTPUT_FOLDER = 'Analysis'
PREPROCESSED_FILE_PATH = 'Data/sp500_final_2023.csv'  # Replace with your actual file path
CONSTITUENTS_FILE_PATH = 'Data/constituents.csv'  # Replace with your actual file path

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def load_and_merge_data(preprocessed_file_path, constituents_file_path):
    """
    Load and merge preprocessed data with constituents data.
    
    Parameters:
    preprocessed_file_path (str): Path to the preprocessed CSV file.
    constituents_file_path (str): Path to the constituents CSV file.
    
    Returns:
    DataFrame: Merged data.
    """
    data = pd.read_csv(preprocessed_file_path)
    constituents_data = pd.read_csv(constituents_file_path)
    data = pd.merge(data, constituents_data[['Symbol', 'GICS Sector', 'GICS Sub-Industry']], left_on='Ticker', right_on='Symbol', how='left')
    return data

def save_dataset_info(data, output_folder):
    """
    Save dataset information, first few rows, and summary statistics.
    
    Parameters:
    data (DataFrame): The dataset.
    output_folder (str): The folder to save the files.
    """
    with open(os.path.join(output_folder, 'dataset_info.txt'), 'w') as f:
        data.info(buf=f)
    data.head().to_csv(os.path.join(output_folder, 'first_few_rows.csv'), index=False)
    data.describe().to_csv(os.path.join(output_folder, 'summary_statistics.csv'))

def plot_histograms(data, columns, rows, cols, output_folder):
    """
    Plot histograms for specified columns.
    
    Parameters:
    data (DataFrame): The dataset.
    columns (list): List of columns to plot histograms for.
    rows (int): Number of rows in the subplot grid.
    cols (int): Number of columns in the subplot grid.
    output_folder (str): The folder to save the plots.
    """
    hist_folder = os.path.join(output_folder, 'Histograms')
    os.makedirs(hist_folder, exist_ok=True)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
    axes = axes.ravel()
    for idx, ax in enumerate(axes):
        if idx < len(columns):
            ax.hist(data[columns[idx]], bins=20, edgecolor='k')
            ax.set_title(columns[idx])
    plt.tight_layout()
    plt.savefig(os.path.join(hist_folder, 'histograms.png'))
    plt.close()

def plot_boxplots(data, columns, title, filename, output_folder):
    """
    Plot boxplots for specified columns.
    
    Parameters:
    data (DataFrame): The dataset.
    columns (list): List of columns to plot boxplots for.
    title (str): Title of the plot.
    filename (str): Filename to save the plot.
    output_folder (str): The folder to save the plot.
    """
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=data[columns], orient='h')
    plt.title(title)
    plt.savefig(os.path.join(output_folder, filename))
    plt.close()

def plot_correlation_matrix(data, columns, output_folder):
    """
    Plot and save the correlation matrix.
    
    Parameters:
    data (DataFrame): The dataset.
    columns (list): List of columns to include in the correlation matrix.
    output_folder (str): The folder to save the plot and matrix.
    """
    correlation_folder = os.path.join(output_folder, 'Correlation')
    os.makedirs(correlation_folder, exist_ok=True)
    correlation_matrix = data[columns].corr()
    plt.figure(figsize=(15, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title("Correlation Matrix")
    plt.savefig(os.path.join(correlation_folder, 'correlation_matrix.png'))
    plt.close()
    correlation_matrix.to_csv(os.path.join(correlation_folder, 'correlation_matrix.csv'))

def plot_pairplot(data, columns, output_folder):
    """
    Plot and save pairplot of specified columns.
    
    Parameters:
    data (DataFrame): The dataset.
    columns (list): List of columns to include in the pairplot.
    output_folder (str): The folder to save the plot.
    """
    pairplot_folder = os.path.join(output_folder, 'Pairplot')
    os.makedirs(pairplot_folder, exist_ok=True)
    sns.pairplot(data, vars=columns)
    plt.suptitle("Pairplot of Features", y=1.02)
    plt.savefig(os.path.join(pairplot_folder, 'pairplot.png'))
    plt.close()

def plot_combined_overall_and_sector_analysis(data, output_folder):
    """
    Plot combined analysis of overall and sector-specific data.
    
    Parameters:
    data (DataFrame): The dataset.
    output_folder (str): The folder to save the plots.
    """
    combined_folder = os.path.join(output_folder, 'Combined_Analysis')
    os.makedirs(combined_folder, exist_ok=True)

    numeric_columns = data.select_dtypes(include=[np.number]).columns
    sector_means = data.groupby('GICS Sector')[numeric_columns].mean().reset_index()
    overall_means = pd.DataFrame(data[numeric_columns].mean()).transpose()
    overall_means['GICS Sector'] = 'Overall'
    combined_data = pd.concat([sector_means, overall_means], ignore_index=True)

    plots = [
        ('Cumulative Return_mean', 'Cumulative Return'),
        ('Daily Return_mean', 'Daily Return'),
        ('Daily Return_std', 'Daily Return Std'),
        ('RSI_mean', 'RSI'),
        ('30 Day MA_mean', '30 Day MA'),
        ('30 Day EMA_mean', '30 Day EMA'),
        ('30 Day STD_mean', '30 Day Std'),
        ('Volume_mean', 'Volume')
    ]

    for column, title in plots:
        plt.figure(figsize=(15, 7))
        sns.boxplot(x='GICS Sector', y=column, data=data)
        sns.stripplot(x='GICS Sector', y=column, data=combined_data, color='red', marker='D', size=7)
        overall_data = pd.DataFrame({column: data[column].values})
        overall_data['GICS Sector'] = 'Overall'
        sns.boxplot(x='GICS Sector', y=column, data=overall_data, boxprops=dict(alpha=.3, color='red'))

        plt.xticks(rotation=90)
        plt.title(f"{title} by Sector and Overall")
        plt.savefig(os.path.join(combined_folder, f'{title.lower().replace(" ", "_")}_by_sector_and_overall.png'))
        plt.close()

def plot_sub_sector_analysis(data, output_folder):
    """
    Plot analysis for each sub-sector within each industry.
    
    Parameters:
    data (DataFrame): The dataset.
    output_folder (str): The folder to save the plots.
    """
    sub_sector_folder = os.path.join(output_folder, 'Sub_Sector_Analysis')
    os.makedirs(sub_sector_folder, exist_ok=True)

    industries = data['GICS Sector'].unique()
    for industry in industries:
        subset = data[data['GICS Sector'] == industry]
        industry_folder = os.path.join(sub_sector_folder, industry.replace(" ", "_"))
        os.makedirs(industry_folder, exist_ok=True)

        plots = [
            ('Cumulative Return_mean', 'Cumulative Return by Sub-Industry for'),
            ('Daily Return_mean', 'Daily Return by Sub-Industry for'),
            ('Daily Return_std', 'Daily Return Std by Sub-Industry for'),
            ('RSI_mean', 'RSI by Sub-Industry for'),
            ('30 Day MA_mean', '30 Day MA by Sub-Industry for'),
            ('30 Day EMA_mean', '30 Day EMA by Sub-Industry for'),
            ('30 Day STD_mean', '30 Day Std by Sub-Industry for'),
            ('Volume_mean', 'Volume by Sub-Industry for')
        ]

        for column, title in plots:
            plt.figure(figsize=(15, 7))
            sns.boxplot(x='GICS Sub-Industry', y=column, data=subset)
            plt.xticks(rotation=90)
            plt.title(f"{title} {industry}")
            plt.savefig(os.path.join(industry_folder, f'{column.lower().replace(" ", "_")}_by_sub_industry_{industry.replace(" ", "_")}.png'))
            plt.close()

        # Combined Bollinger Bands by Sub-Industry for each Industry
        subset_melted = subset.melt(id_vars=['GICS Sub-Industry'], 
                                    value_vars=['Bollinger High_mean', 'Bollinger Low_mean'],
                                    var_name='Bollinger Band', value_name='Value')
        plt.figure(figsize=(15, 7))
        sns.boxplot(x='GICS Sub-Industry', y='Value', hue='Bollinger Band', data=subset_melted)
        plt.xticks(rotation=90)
        plt.title(f"Bollinger Bands by Sub-Industry for {industry}")
        plt.savefig(os.path.join(industry_folder, f'bollinger_bands_by_sub_industry_{industry.replace(" ", "_")}.png'))
        plt.close()

def main():
    """
    Main function to load data, perform EDA, and save plots.
    """
    # Load and merge data
    data = load_and_merge_data(PREPROCESSED_FILE_PATH, CONSTITUENTS_FILE_PATH)

    # Save dataset information
    save_dataset_info(data, OUTPUT_FOLDER)

    # Features to plot histograms for
    hist_columns = data.columns.drop(['Ticker', 'Symbol', 'Principal Component 1', 'Principal Component 2', 'GICS Sector', 'GICS Sub-Industry'])
    plot_histograms(data, hist_columns, rows=3, cols=3, output_folder=OUTPUT_FOLDER)

    # Plot boxplots for different feature categories
    plot_boxplots(data, ['Daily Return_mean', 'Daily Return_std'], "Boxplot of Daily Returns", 'boxplot_daily_returns.png', OUTPUT_FOLDER)
    plot_boxplots(data, ['Cumulative Return_mean'], "Boxplot of Cumulative Returns", 'boxplot_cumulative_returns.png', OUTPUT_FOLDER)
    plot_boxplots(data, ['30 Day MA_mean', '30 Day EMA_mean'], "Boxplot of Moving Averages", 'boxplot_moving_averages.png', OUTPUT_FOLDER)
    plot_boxplots(data, ['30 Day STD_mean', 'Bollinger High_mean', 'Bollinger Low_mean'], "Boxplot of Volatility", 'boxplot_volatility.png', OUTPUT_FOLDER)
    plot_boxplots(data, ['RSI_mean'], "Boxplot of Momentum", 'boxplot_momentum.png', OUTPUT_FOLDER)
    plot_boxplots(data, ['Volume_mean'], "Boxplot of Volume_mean", 'boxplot_volume.png', OUTPUT_FOLDER)

    # Plot correlation matrix and pairplot
    plot_correlation_matrix(data, hist_columns, OUTPUT_FOLDER)
    plot_pairplot(data, hist_columns, OUTPUT_FOLDER)

    # Plot combined overall and sector analysis
    plot_combined_overall_and_sector_analysis(data, OUTPUT_FOLDER)

    # Plot sub-sector analysis
    plot_sub_sector_analysis(data, OUTPUT_FOLDER)

    print(f'All EDA outputs have been saved to the {OUTPUT_FOLDER} folder.')

if __name__ == "__main__":
    main()
