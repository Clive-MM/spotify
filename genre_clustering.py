import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE  # For t-SNE

# Set Streamlit page title
st.title("Spotify Data Analysis")

# Load dataset
genre_data_path = 'data_by_genres.csv'
df = pd.read_csv(genre_data_path)

# Display dataset preview
st.subheader("🎵 Genre Data Preview")
st.dataframe(df.head())

# 1. Select Numerical Features
numerical_features = df.select_dtypes(include=['number'])

# 2. Scale the Features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(numerical_features)

# 3. Fit the K-Means Model
kmeans = KMeans(n_clusters=12, random_state=42)
kmeans.fit(scaled_features)

# 4. Assign Cluster Labels
df['cluster'] = kmeans.labels_

# 5. t-SNE Dimensionality Reduction
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(scaled_features)

# Add t-SNE results to DataFrame
df['tsne_1'] = tsne_results[:, 0]
df['tsne_2'] = tsne_results[:, 1]

# 6. Visualize Clusters with t-SNE
st.subheader("Genre Clusters Visualization (t-SNE)")

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(
    x='tsne_1',
    y='tsne_2',
    c='cluster',  # Color by cluster
    data=df,
    cmap='Set3'  
)

# Add hover annotations
annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="white"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):
    x, y = scatter.get_offsets()[ind["ind"][0]]
    annot.xy = (x, y)
    genre = df['genres'][ind["ind"][0]]
    cluster = df['cluster'][ind["ind"][0]]
    text = f"Genre: {genre}\nCluster: {cluster}"
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = scatter.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

st.pyplot(fig)