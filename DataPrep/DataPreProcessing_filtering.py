
# Network Science - Project #BLM
# Data Pre-Processing: Filter Data

# Import libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

#### 2020 ####
file = "/home/theresa/Schreibtisch/Theresa/STUDIUM/Master Statistics and Data Science/Padova/Network Science/Project/Data/filtered_data/filtered_tweets_2020_all.csv"
filtering = 0.1  # % of the dataset

# Reading the CSV file
df = pd.read_csv(file)
# Remove rows with duplicate values in the 'tweet_text' column
df.drop_duplicates(subset='tweet_text', keep='first', inplace=True)

# Standardizing the Data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[["retweet_count","reply_count","like_count","quote_count"]])

# Applying PCA
pca_full = PCA()

result_pca = pca_full.fit_transform(df_scaled)
explained_variance = pca_full.explained_variance_ratio_

# Formula for the PCA score to explain the most variance
df['pca_score'] = result_pca[:, 0]*explained_variance[0] + result_pca[:, 1]*explained_variance[1]

# Filtering the top specified percentage of the dataset based on the PCA score
top_percentile_cutoff = np.percentile(df['pca_score'], 100 - filtering )
filtered_df = df[df['pca_score'] >= top_percentile_cutoff]

# Printing the score where we cut
print("Score Cutoff for Top", filtering , "% of the Dataset:", top_percentile_cutoff)

# Deleting the 'pca_score' column
filtered_df.drop(columns=['pca_score'], inplace=True)

# Writing the filtered DataFrame to a new CSV file
filtered_df.to_csv('/home/theresa/Schreibtisch/Theresa/STUDIUM/Master Statistics and Data Science/Padova/Network Science/Project/Data/filtered_data/filtered_tweets_2020_final.csv', index=False)


# Extracting the PCA components (loadings) for the heatmap
pca_components = pca_full.components_

# Plotting the heatmap of PCA components
plt.figure(figsize=(10, 6))
sns.heatmap(pca_components, annot=True, cmap='coolwarm',
            xticklabels=["retweet count", "reply count", "like count", "quote count"],
            yticklabels=[f"PC{i+1}" for i in range(len(pca_components))])
plt.title("PCA Component Loadings")
plt.savefig('/home/theresa/Schreibtisch/Theresa/STUDIUM/Master Statistics and Data Science/Padova/Network Science/Project/Data/filtered_data/PCA Component Loadings.pdf')
plt.show()


# Extracting and plotting the explained variance ratio for each principal component

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, tick_label=[f"PC{i}" for i in range(1, len(explained_variance) + 1)])
plt.xlabel('Principal Components')
plt.ylabel('Variance Explained')
plt.title('Explained Variance by Each Principal Component')
plt.savefig('/home/theresa/Schreibtisch/Theresa/STUDIUM/Master Statistics and Data Science/Padova/Network Science/Project/Data/filtered_data/Explained Variance by Each Principal Component.pdf')
plt.show()


# Plotting the feature importances for the first principal component
plt.figure(figsize=(10, 6))
plt.bar(range(len(pca_components[0])), pca_components[0]*explained_variance[0] + pca_components[1]*explained_variance[1], tick_label=["retweet count","reply count","like count","quote count"])
plt.xlabel('Features')
plt.ylabel('PCA Feature Importance')
plt.title('Feature Importances for the Final score')
plt.savefig('/home/theresa/Schreibtisch/Theresa/STUDIUM/Master Statistics and Data Science/Padova/Network Science/Project/Data/filtered_data/Feature Importances for the Final score.pdf')
plt.show()

#### 2013 ####
file = "/home/theresa/Schreibtisch/Theresa/STUDIUM/Master Statistics and Data Science/Padova/Network Science/Project/Data/filtered_data/filtered_tweets_2013_all.csv"
df_2013 = pd.read_csv(file)

# Remove rows with duplicate values in the 'tweet_text' column
df_2013.drop_duplicates(subset='tweet_text', keep='first', inplace=True)

df_2013.to_csv('/home/theresa/Schreibtisch/Theresa/STUDIUM/Master Statistics and Data Science/Padova/Network Science/Project/Data/filtered_data/filtered_tweets_2013_final.csv', index=False)


#### Combine both data sets ####
filtered_df['year'] = 2020
df_2013['year'] = 2013

full_data = pd.concat([df_2013,filtered_df], ignore_index=True)
full_data.to_csv('/home/theresa/Schreibtisch/Theresa/STUDIUM/Master Statistics and Data Science/Padova/Network Science/Project/Data/filtered_data/filtered_tweets_final.csv', index=False)
