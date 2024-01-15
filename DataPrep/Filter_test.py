import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

file = "/home/theresa/Schreibtisch/Theresa/STUDIUM/Master Statistics and Data Science/Padova/Network Science/Project/Data/filtered_data/filtered_tweets_2020_final.csv"
filtering = 0.1  # % of the dataset

# Reading the CSV file

df = pd.read_csv(file)

# Fitting PCA on the specific columns without limiting the number of components
# Standardizing the Data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[["retweet_count","reply_count","like_count","quote_count"]])

# Applying PCA

pca_full = PCA()

df['pca_score'] = pca_full.fit_transform(df_scaled)[:, 0]
# Extracting the PCA components (loadings) for the heatmap
pca_components = pca_full.components_

# Plotting the heatmap of PCA components
plt.figure(figsize=(10, 6))
sns.heatmap(pca_components, annot=True, cmap='coolwarm',
            xticklabels=["retweet count", "reply count", "like count", "quote count"],
            yticklabels=[f"PC{i+1}" for i in range(len(pca_components))])
plt.title("PCA Component Loadings")
plt.show()

# Extracting and plotting the explained variance ratio for each principal component
explained_variance = pca_full.explained_variance_ratio_
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, tick_label=[f"PC{i}" for i in range(1, len(explained_variance) + 1)])
plt.xlabel('Principal Components')
plt.ylabel('Variance Explained')
plt.title('Explained Variance by Each Principal Component')
# Saving the plot
plt.savefig('/home/theresa/Schreibtisch/Theresa/STUDIUM/Master Statistics and Data Science/Padova/Network Science/Project/Data/filtered_data/PCA_heatMap.pdf')

plt.show()
