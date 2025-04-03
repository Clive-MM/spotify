import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE  

# Set Streamlit page title
st.title("Spotify Genre Clustering with K-Means and t-SNE")

# Load dataset
genre_data_path = 'data_by_genres.csv'
df = pd.read_csv(genre_data_path)

try:
    df = pd.read_csv(genre_data_path)
    st.write("Genre data loaded successfully from CSV.")
except FileNotFoundError:
    st.error(f"Error: File '{genre_data_path}' not found.")
    st.stop()

# Display dataset preview
st.subheader("🎵 Genre Data Preview")
st.dataframe(df.head())

# Select numeric columns for clustering
numeric_cols = df.select_dtypes(include=['number']).columns
numeric_df = df[numeric_cols].dropna()

# Standardize the numeric data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)

st.write("Data preprocessing completed. Standardized numerical features.")

# Fit K-Means with 12 clusters
kmeans = KMeans(n_clusters=12, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_data)

st.write("K-Means clustering applied. 12 clusters assigned.")

# Apply t-SNE to reduce to 2D for visualization
tsne = TSNE(n_components=2, perplexity=10, random_state=42)
tsne_data = tsne.fit_transform(scaled_data)

# Store t-SNE results in DataFrame
df['TSNE1'] = tsne_data[:, 0]
df['TSNE2'] = tsne_data[:, 1]

st.write("t-SNE applied: Data reduced to 2D for visualization.")

# Matplotlib scatter plot
st.subheader("🎨 Clustering Visualization (Matplotlib)")

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=df["TSNE1"], y=df["TSNE2"], hue=df["Cluster"], palette="tab10", s=50, alpha=0.8)
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("K-Means Clustering of Genres (t-SNE Reduced)")
plt.legend(title="Cluster")

# Show plot in Streamlit
st.pyplot(fig)

# Plotly Interactive Scatter Plot
st.subheader("🎨 Interactive Clustering Visualization (Plotly)")

fig = px.scatter(df, x="TSNE1", y="TSNE2", color=df["Cluster"].astype(str),
                 hover_data=["genres"] if "genres" in df.columns else None,
                 title="K-Means Clustering of Genres (t-SNE Reduced)")

# Show interactive plot
st.plotly_chart(fig)

# Display cluster summary
st.subheader("🔍 Cluster Analysis Summary")
cluster_summary = df.groupby("Cluster").agg({
    "popularity": "mean",
    "energy": "mean",
    "danceability": "mean",
    "acousticness": "mean",
    "valence": "mean"
}).reset_index()

st.dataframe(cluster_summary)

# --- Genres per cluster ---
st.subheader("📜 Genres Grouped by Cluster")

# Group genres by cluster
genres_by_cluster = df.groupby('Cluster')['genres'].apply(list).reset_index(name='Genres')
st.dataframe(genres_by_cluster)

# --- Genre counts per cluster ---
st.subheader("🔢 Genre Count per Cluster")

# Count number of genres in each cluster
genre_counts = df['Cluster'].value_counts().sort_index().reset_index()
genre_counts.columns = ['Cluster', 'Genre Count']
st.dataframe(genre_counts)

# --- Most representative genre per cluster ---
st.subheader("⭐ Most Representative Genre per Cluster")

# Pick the first genre (or most frequent one if duplicates exist) in each cluster
most_representative = df.groupby('Cluster')['genres'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]).reset_index()
most_representative.columns = ['Cluster', 'Representative Genre']
st.dataframe(most_representative)

