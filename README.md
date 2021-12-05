# Happiness-Data-Analysis
The World Happiness Report is a landmark survey of the state of global happiness, which ranks 150 countries by their happiness levels and was released by the United Nations. Data like this are used by experts across fields such as economics, psychology, survey analysis, national statistics, health, public policy etc. to inform their policy-making decisions.
The data contains 150 rows corresponding to countries and ten factors: Region, Religion, level of religiousness, Economy in GDP per capita, Family (family values), Health (Life expectancy), Freedom, Government trust, Generosity and Dystopia residual. Here Region, Religion and level of religiousness are categorical variables whereas all others are numerical.
The objective of our study was to reflect the use of effective and novel data mining and machine learning techniques. And determine which technique was best suitable for the data provided. After that we have to justify the use of the algorithms we used for our data resulting in appropriate and justifiable conclusions.
According to our study, we figured out that the Global happiness characteristics here are income, healthy life expectancy, having someone to count on in times of trouble, generosity, freedom and trust. And two sets of data, explained and unexplained data, were present. We worked with the unexplained data.
Inorder to find the main groups of countries and their global happiness characteristics, several novel machine learning techniques were considered. At first we had to preprocess the given data-set as how we wanted it to be like. Dealing with Missing Values:
At first, we read the data into a DataFrame using python’s package called Pandas.Then we preprocessed the data. For that, we looked for any empty/NULL values in the dataset and found none. Dealing with Outliers:
After that, the outliers were identified. There are several techniques like IQR, Hampel, and DBSCAN that are used to detect the outliers. In this case to visualize the data, BoxPlot was
used. Boxplot is a type of a chart used for explanatory data analysis. By displaying the data quartiles and averages, box plots visually depict the distribution of numerical data and skewness. We performed a box plot on all the 6 happiness characteristics : income, healthy life expectancy, having someone to count on in times of trouble, generosity, freedom and trust to figure out where there are any outliers present.
Outliers increase the variability in your data, which decreases statistical power thus they are removed. Outliers were detected as soons a data point was found outside the box plot’s whiskers. After detection of the outliers they are removed, there are several outliers removal techniques like Standard Deviation Method, Interquartile Range Method, One-class classification. We used IQR to remove the outliers because it is the best measure of variability for skewed distributions or data sets with outliers. And it’s based on values that come from the middle half of the distribution, it’s unlikely to be influenced by outliers.

For each time we detected an outlier, we removed them to get more conscious data. The final
shape of our data showed a decrease of 11 rows:
Old Shape:  (149, 20)
New Shape:  (138, 20)
Data Normalization: The production of clean data is generally referred to as data normalisation. Data normalisation is the process of organising data such that it seems consistent across all records and fields. It improves the cohesion of entry types, resulting in better data cleansing, lead creation, segmentation, and segmentation. This basically is to get rid of any duplicate values. It goes around and eliminates all the data redundancies. Data normalisation removes a variety of irregularities that might make data analysis more difficult.
We have normalized tha data using the mean normalization technique
normalized_happy_data=(happy_data-happy_data.mean())/happy_data.std()
so that there are no duplicates or anomalies in the and we can proceed smoothly with a clean data set. We again replaced the NaN values with 0 using the fillna() function.
Determining the number of clusters:
The optimal number of clusters into which the data can be grouped is a crucial stage in any unsupervised technique. One of the most prominent approaches for determining the ideal value of k is the Elbow Method. In a cluster, WCSS is the sum of squared distances between each point and the centroid. The plot appears like an Elbow when we plot the WCSS with the K value. The WCSS value will begin to fall as the number of clusters grows. When we examine the graph, we can see that it will shift rapidly and  form an elbow shape. The graph tilts practically parallel to the X-axis at this point. The ideal K value, or the number of clusters is the corresponding value of K at this point.
In our graph we can observe that the shift of the curve takes place after 3(number of clusters). This results in assuming that the value of k will be 3.
We used another technique to determine the number of clusters called the Silhouette Coefficient or Silhouette score. It is a metric used to calculate the goodness of a clustering technique. From this, we also found out that the number of clusters is 3.


Dimensionality Reduction:
After obtaining the number of clusters, we tried to reduce the noise and complexity in the data by reducing its dimensionalities. We used an unsupervised linear dimensionality reduction algorithm called Principal Component Analysis(PCA)
We have used the sklearn library to import the PCA() function.
pca = PCA(3) data = pca.fit_transform(normalized_happy_data) plt.figure(figsize=(15,10)) var = np.round(pca.explained_variance_ratio_*100, decimals = 1) lbls = [str(normalized_happy_data) for x in range(1,len(var)+1)] plt.bar(x=range(1,len(var)+1), height = var, tick_label = lbls) plt.show()


KMeans Clustering:
Our next step after determining the number of clusters and reducing the dataset is clustering. For that, there are many clustering algorithms available like: K-Means Clustering(iterative clustering), Dendrogram(Hierarchical Clustering)
K-means is a centroid based technique, to represent the cluster. A centroid point is the centre point of a cluster. So here we have found the centroids for our clusters using k-means using
centers = np.array(model.cluster_centers_).


Then I have used '#39C8C6' and  '#D3500C' to each feature as colours to plot my cluster components. Also used scatter plots to show them.

And plotted the centroids.

After plotting the centroids, we found the metric of each feature according to its weight.


Another technique we used was Dendrogram. It shows a diagram that represents the hierarchical relationship between objects. It is most commonly created as an output from hierarchical clustering. The main use of a dendrogram is to work out the best way to allocate objects to clusters.
import scipy.cluster.hierarchy as shc plt.figure(figsize=(10, 7)) plt.title("Global Happiness Dendrograms") dend = shc.dendrogram(shc.linkage(data, method='ward'))


If we draw a horizontal line that passes through the longest distance without crossing a horizontal line, we get 5 clusters.
We then further implemented Agglomerative Clustering using Sklearn. The technique assumes that each data point is similar enough to the other data points that the data at the starting can be assumed to be clustered in 1 cluster.
from sklearn.cluster import AgglomerativeClustering cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward') cluster.fit_predict(data)
plt.figure(figsize=(10, 7)) plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')

The visualization represents how closely the values are located in each cluster.
