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

import plotly.express as px

# Standardize the numeric data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)

st.write("Data preprocessing completed. Standardized numerical features.")

# Fit K-Means with 25 clusters
kmeans = KMeans(n_clusters=25, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_data)

st.write("K-Means clustering applied. 25 clusters assigned to songs.")

# Apply PCA to reduce to 2D for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

st.write("PCA applied: Data reduced to 2D for visualization.")

# Visualize clusters with Plotly
st.subheader("🎨 Interactive Clustering Visualization")

# Ensure relevant song info is present for hover
hover_columns = []
if 'name' in df.columns:
    hover_columns.append('name')
if 'artists' in df.columns:
    hover_columns.append('artists')
if 'release_date' in df.columns:
    hover_columns.append('release_date')
if 'popularity' in df.columns:
    hover_columns.append('popularity')

# Create interactive scatter plot
fig = px.scatter(
    df,
    x="PCA1",
    y="PCA2",
    color=df["Cluster"].astype(str),
    hover_data=hover_columns,
    title="🎵 Song Clusters (25 Clusters) using PCA"
)

st.plotly_chart(fig)

# Optional: show cluster summaries
st.subheader("📊 Cluster Summary")
cluster_summary = df.groupby("Cluster").agg({
    "popularity": "mean",
    "energy": "mean" if "energy" in df.columns else "mean",
    "danceability": "mean" if "danceability" in df.columns else "mean",
    "acousticness": "mean" if "acousticness" in df.columns else "mean",
    "valence": "mean" if "valence" in df.columns else "mean"
}).reset_index()

st.dataframe(cluster_summary)

# --- 🎯 Song Recommendation System ---
st.subheader("🎧 Get Song Recommendations")

# Let user enter a song name
song_name_input = st.text_input("Enter a song name to get similar recommendations:")

if song_name_input:
    # Search for the song (case-insensitive)
    matched_song = df[df['name'].str.lower() == song_name_input.lower()]
    
    if matched_song.empty:
        st.warning("Song not found. Please check the spelling or try another.")
    else:
        # Get cluster of the selected song
        selected_cluster = matched_song['Cluster'].values[0]

        # Recommend other songs from the same cluster (excluding the selected song)
        recommendations = df[(df['Cluster'] == selected_cluster) & (df['name'].str.lower() != song_name_input.lower())]

        # Display top 10 recommendations
        st.success(f"Showing songs from the same cluster as '{song_name_input}':")
        st.dataframe(recommendations[['name', 'artists', 'release_date', 'popularity']].head(10))



