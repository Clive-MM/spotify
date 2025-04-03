import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  

# Set Streamlit page title
st.title("Song Clustering with K-Means and PCA")

# Define file paths
main_data_path = 'data.csv'

# Load dataset
try:
    df = pd.read_csv(main_data_path)
    st.write("Song data loaded successfully.")
except FileNotFoundError:
    st.error(f"Error: File '{main_data_path}' not found.")
    st.stop()

# Display dataset preview
st.subheader("🎵 Song Data Preview")
st.dataframe(df.head())

# Select numeric columns for clustering
numeric_cols = df.select_dtypes(include=['number']).columns
numeric_df = df[numeric_cols].dropna()

# Standardize the numeric data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)

st.write("Data preprocessing completed. Standardized numerical features.")

# Fit K-Means with 25 clusters
kmeans = KMeans(n_clusters=25, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_data)

st.write("K-Means clustering applied. 25 clusters assigned.")


# Apply PCA to reduce to 2D for visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Store PCA results in DataFrame
df['PCA1'] = pca_data[:, 0]
df['PCA2'] = pca_data[:, 1]

st.write("PCA applied: Data reduced to 2D for visualization.")

# Create a scatter plot with seaborn
st.subheader("🎨 Clustering Visualization")

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=df["PCA1"], y=df["PCA2"], hue=df["Cluster"], palette="tab10", s=50, alpha=0.8)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-Means Clustering of Songs (PCA Reduced)")
plt.legend(title="Cluster")

# Show plot in Streamlit
st.pyplot(fig)

import plotly.express as px

# Create interactive scatter plot
fig = px.scatter(df, x="PCA1", y="PCA2", color=df["Cluster"].astype(str),
                 hover_data=["song_name", "artist_name"] if "song_name" in df.columns else None,
                 title="K-Means Clustering of Songs (PCA Reduced)")

# Show interactive plot
st.plotly_chart(fig)


