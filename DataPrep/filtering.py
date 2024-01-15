import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

file = "/home/theresa/Schreibtisch/Theresa/STUDIUM/Master Statistics and Data Science/Padova/Network Science/Project/Data/filtered_data/filtered_tweets_2020_final.csv"
filtering = 0.1  # % of the dataset

# Reading the CSV file
dff = pd.read_csv(file)
df = dff.copy()

# Standardizing the Data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[["retweet_count", "reply_count", "like_count", "quote_count"]])

# Applying PCA
pca = PCA(n_components=1)  # We only need the first principal component
df['pca_score'] = pca.fit_transform(df_scaled)[:, 0]

# Filtering the top specified percentage of the dataset based on the PCA score
top_percentile_cutoff = np.percentile(df['pca_score'], 100 - filtering )
filtered_df = df[df['pca_score'] >= top_percentile_cutoff]

# Printing the score where we cut
print("Score Cutoff for Top", filtering , "% of the Dataset:", top_percentile_cutoff)

# Deleting the 'pca_score' column
filtered_df.drop(columns=['pca_score'], inplace=True)

# Writing the filtered DataFrame to a new CSV file
filtered_df.to_csv('/home/theresa/Schreibtisch/Theresa/STUDIUM/Master Statistics and Data Science/Padova/Network Science/Project/Data/filtered_data/filtered_tweeds_2020_final_true.csv', index=False)

# Applying PCA again to get the components
pca_components = pca.components_

# Plotting the feature importances for the first principal component
plt.figure(figsize=(10, 6))
plt.bar(range(len(pca_components[0])), pca_components[0], tick_label=["retweet_count", "reply_count", "like_count", "quote_count"])
plt.xlabel('Features')
plt.ylabel('PCA Feature Importance')
plt.title('Feature Importances for the First Principal Component')
plt.show()

# Saving the plot
plt.savefig('/home/theresa/Schreibtisch/Theresa/STUDIUM/Master Statistics and Data Science/Padova/Network Science/Project/Data/filtered_data/PCA_firstComponent.pdf')

pca_full = PCA()

pca_full.fit(dff[["retweet_count","reply_count","like_count","quote_count"]])

# Extracting the explained variance ratio of each component

explained_variance = pca_full.explained_variance_ratio_


# Extracting the explained variance ratio of each component

explained_variance = pca_full.explained_variance_ratio_



# Plotting the explained variance ratio for each principal component

plt.figure(figsize=(10, 6))

plt.bar(range(1, len(explained_variance) + 1), explained_variance, tick_label=[f"PC{i}" for i in range(1, len(explained_variance) + 1)])

plt.xlabel('Principal Components')

plt.ylabel('Variance Explained')

plt.title('Explained Variance by Each Principal Component')

plt.show()
plt.savefig('/home/theresa/Schreibtisch/Theresa/STUDIUM/Master Statistics and Data Science/Padova/Network Science/Project/Data/filtered_data/PCA_components.pdf')
