# Stock Cluster Classification
This project classifies clusters of S&P 500 stocks* based on various features. It was created to demonstrate my knowledge of cluster classification techniques.

## Collect Data
The first step in this project is to collect data. Specifically, get a list of the S&P 500 companies*. The list includes the company's symbol, name (security), sector, sub-sector, headquarters location, date added to S&P, and date founded.

Then, the symbol is looked up using the yfinance library. Data between the START_DATE and END_DATE is accessed. The following columns are created: Date, Open, High, Low, Close, Adj Close, Volume, and Ticker. This data performs each indicated stock on a given day between the START_DATE and END_DATE.

## PreProcessing
Once the required data has been collected, preprocessing is needed before EDA (exploratory data analysis) and clustering is performed.

### Missing values, Returns, moving averages, and Indicators
A stock with a missing value is filled  with similar information. Then, the following indicators are calculated:
1. Daily Return - percentage change in a stock's daily price
2. Cumulative Return - the total percentage change in a stock's price over a specific period, reflective of overall gain or loss during that time.
3. 30 Day MA (moving averge) - smooths out short-term fluctuation by averaging clsing price over the last 30 days
4. 30 Day STD (standard deviation) - measures the volitility of a stock's price over the last 30 days
5. 30 Day EMA (exponential moving average) - similar to 30 day MA but gives more weight to recent prices, making it more responsive to new information
6. RSI (Relative Strength Index) - show the momentum oscillator that measures the speed/change of price movements, shows if stock os over bought or sold
7. Bollinger High, Bollinger Low - calculated at certain number of STD above or below moving averge. Indicating overbought or oversold.

### Aggregate Indicators
The indicators listed above are then summarized over the period in the data. That is the period given when collecting data. This gives a summary of the information given above.

### Remove Outliers
To ensure the reliability of the analysis, the summary data for each stock is evaluated against a z-score to identify and eliminate outliers. Stocks with performance metrics that are significantly different from the rest of the S&P 500 are considered outliers and removed from the dataset. This step is crucial in maintaining the consistency of the clustering results and avoiding the influence of extreme values on the analysis.

### Normalize features
After removing outliers, the summary data for each remaining stock is normalized. Normalization scales the features to have a mean of zero and a standard deviation of one, ensuring that all performance metrics contribute equally to the clustering process. This step prevents features with larger numerical ranges from dominating the clustering results and allows for more accurate comparisons between stocks.

### Principal component analysis
The normalized data is then processed using Principal Component Analysis (PCA) from Sklearn to reduce dimensionality and capture the most significant variations in the data. PCA transforms the data into a set of principal components, which are orthogonal and explain the maximum variance in the dataset. The explained variance ratio of these components is calculated to determine their significance. The principal components are then plotted against each other, providing a visual representation of the data structure and aiding in the identification of distinct clusters.


## EDA
An exploratory data analysis is performed to determine the appropriate features in the cluster classification. The following EDA are used:
1. Histogram
2. box plot
3. correlation matrix
4. pair plot

### Sector Analysis
The indicators defined in preprocessing are then looked at by sector. A boxplot for each indicator and a boxplot for each sector are created overall. This allows for evaluating indicators across sectors if there are differences between sectors.

### Sub-Sector Analysis
Multiple subsectors exist within each sector. These subsector indicators vary within a single sector, leading to additional analysis. Similar to the Secor analysis for each indicator mentioned earlier, a boxplot is created. Note that only sub-sectors in the same sector are compared, as comparing sub-sectors from other sectors would lead to inaccurate and misguided comparisons.

## Cluster Classification
Once EDA has been completed, we should understand the data structure we are working with.

### Feature Selection
One issue that always plagues data analysis is feature selection (which variables to include). To eliminate the guesswork, I took a two-step approach. First, I applied a variance threshold, eliminating features that do not cross the threshold (threshold = 1%). The second step is to apply a Random Forest Classifier (RFE). An RFE was selected because we are working with unsupervised data, and RFE allows for feature selection of unsupervised data.

### Cluster Classification Algorithms
The following algorithms were used to classify cluster of stocks
1. SOM
2. k-means
3. dscan
4. agglomerative
5. gmm
6. hierarchical
7. spectral

#### Self-Organizing Map (SOM)
A SOM is an unsupervised learnign algorithm that is used to produce low-dimentional representation of high-dimensional data. SOMs are useful to visualize and understand complex datasets by clustering similar data points. In this project a SOM is used to cluste rS&P 500 stocks based on their performance metrics.
The algorithm creates a grid of neurons where each neuron represents a cluster of similar stocks. the position of each neuron on the grid reflects similarity between stock based on performance metrics.
I included several possible options of parameters. The reason for this is to find the optimal set of parameters.
- x range, y range: the possible grid dimensions
- sigma range: controls the radius of the neighbourhood function, influences how much the weights of surrounding neurons are updated during training
- learning rate: affect the magnitude of weights during training
To determine the optimal set of parameters the algorithm runs through all combination and picks the set that yields the highest silhouette score.

#### k-means
K-Means clustering is an unsupervised learnign algorithm that paritions data into K distinct clusters based on feature similarity. In this project K-means is aplied to cluster S&P 500 stocks based on their performance metrics.
The algorithm initializes K centroids, assigning data points to the nearest centroid. Then updating the centroids based on assigned points. Then is repeated until convergence.
The elbow method is used to determine the optimal number of clusters (K). The elbow method is calculated by using within-cluster sum of squres (WCSS) for different values of K. Where the rate of decrease sharply changes (elbow point) that is the optimal number of clusters.
For each number of clusters, the silhouette score is calculated. This score helps to validate the consistency within clusters and helps identify the optimal number of clusters by identifying the highest silhouette score.

#### Density-Based Spatial Clustering of Applications with Noise (DSCAN)
DBSCAN is an unsupervised algorithm that identifies clusters based on the density of data points. Unlike kmeans DBSCAN does not need to know the numer of clusters as it identifies clusters itself by varying shape and size.
The DBSCAN works by identifying clusters of areas that have high density seperated by areas of low density. To do this it uses the maximum distance between two sample points to be considered neighborhood to the other (eps) and the number of samples in a neighborhood for a point to be considered as a core point (min_samples).
To find the optimal eps the distance between nearest enighbors is analyzed. The distance at which the rate of increase in these distances changes the most sharply is selected as the optimal eps.
To find the optimal value for min_samples the algorithm iterates though possible values and selects the one that produces the highest silhouette score (best clustering quality).
Furthermore, the algorithm iterates through various compinations of eps and min_samples to find the optimal parameter that produce the highest silhouette score

#### Agglomerative Clustering
Agglomerative Clustering is a hierarchical clustering algorithm that builds nested clusters by successively merging pairs of clusters.
The algorithm starts with each data point as a single cluster and merges pairs of clusters iteratively until all points are clustered into the specified number of clusters.
To find the optimal number of clusters the algorithm evaluestes the silhouette score for different number of clusters (2 to 20) in this case. The number of clusteres with the highest silhouette score is selected as the optimal.

#### Gaussian Mixture Model (GMM)
GMM is a probablistic clusterin algorithm that assumes the data is generated from a mixture of several gaussian distribitions.
The algorithm estimates parameters of gaussian distributions for each cluster and assigning data points to clusters based on the highest probability of belonging to a Gaussian distribution.
To find the optimal number of clusters two criteria are used Bayesian Information Criterion (BIC) and Akaike Information Criterion (AIC). Both evaluate the goodness of fit of the model, and penalizing complexity as an effort to avoid overfitting. The optimal number of cluster is when BIC or AIC is minimized.

#### hierarchical
Hierarchical Clustering is a method which tries to build a hierarchy of clusters. Starts with each data point as a single cluster and merges pairs of clusters until the prefined number of clusters is reached. Creating a tree-based representation of the data, called a dendrogram.
In this project the number of clusters is determined by analyzing the linkage matrix. Where the distance at which the rate of increase in linkage distances changes the most sharply is selected as the optimal cutting point for the dendrogram

#### spectral
Spectral Clustering uses graph theory to cluster data points.
Constructs a similarity graph from the data then partitioning the grapph into clusters using eigenvalues and eigenvector of the graph laplacia matrix.
To find the optimal number of cluster the silhouette score for different number of clusters is considered. Whichever cluster has the highest silhouette score is selected as the optimal.

### Error Metrics
The following three scores are used to compare and contrast the performance of the above-mentioned algorithms:
1. Silhouette score
2. Davies-Bouldin index
3. Calinski-Harabasz index

#### Silhouette score
Measures how similar an entry is to its own cluster compared to the other clusters. The range of this score is from -1 to 1. Closer to 1 indicates that the data are well-clustered, with clear seperation. Closer to 0 indicates that clusters overlap. And if the score is less than 0 (negative) than the data points are assigned to the wrong cluster.
#### Davies-Bouldin index
Measures the average similarity ration of each cluster with the most similar cluster. lower the better.
#### Calinski-Harabasz index (variance ratio criterion)
Measures the ratio of the sum of between-cluster dispersion and within-cluster dispersion. Higher values signify well-defined clusters. 

## Next Steps
1. create results file that will store the error metrics for all the methods tried
2. create a dashboard using apache superset
3. allow stock/industry/sub-industry selection for custom analysis
