import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

file = "tdvsccj.csv"
filtering = 0.1  # % of the dataset

# Reading the CSV file
df = pd.read_csv(file)

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
filtered_df.to_csv('filtered_data.csv', index=False)

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
plt.savefig('resultpca.png')
