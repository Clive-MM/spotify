import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import sys
import io
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS


# Set Streamlit page title
st.title("Spotify Data Analysis")

# Define file paths
main_data_path = 'data.csv'
genre_data_path = 'data_by_genres.csv'
year_data_path = 'data_by_year.csv'
artist_data_path = 'data_by_artist.csv'



# Read datasets
data = pd.read_csv(main_data_path)
genre_data = pd.read_csv(genre_data_path)
year_data = pd.read_csv(year_data_path)
artist_data = pd.read_csv(artist_data_path)

# Display the first  rows of each dataset
st.subheader("First Rows of Each Dataset")

st.write("**Main Data:**")
st.dataframe(data.head(10))

st.write("**Genre Data:**")
st.dataframe(genre_data.head(10))

st.write("**Year Data:**")
st.dataframe(year_data.head(10))

st.write("**Artist Data:**")
st.dataframe(artist_data.head(10))

# Function to capture and display DataFrame 
def get_dataframe_info(df):
    # Capture the output of df.info() method into a string buffer
    buf = io.StringIO()
    sys.stdout = buf
    df.info()
    sys.stdout = sys.__stdout__
    return buf.getvalue()

# Display Dataset Info 
st.subheader("Dataset Information")

st.write("### Main Data Info")
st.text(get_dataframe_info(data))

st.write("### Genre Data Info")
st.text(get_dataframe_info(genre_data))

st.write("### Year Data Info")
st.text(get_dataframe_info(year_data))

st.write("### Artist Data Info")
st.text(get_dataframe_info(artist_data))

# Convert 'year' column to numeric and clean data
data['year'] = pd.to_numeric(data['year'], errors='coerce')
data.dropna(subset=['year'], inplace=True)
data['year'] = data['year'].astype(int)

# Create 'decade' column
data['decade'] = data['year'].apply(lambda x: (x // 10) * 10)


# Display the created 'decade' column
st.write("### Decade Column Created Successfully")

# Show unique decades
st.write("**Unique Decades:**")
st.write(data['decade'].unique())

# Describe the statistics of the decade column
st.write("**Decade Column Statistics:**")
st.write(data['decade'].describe())

# Display the first 10 rows of the main data
st.write("\n**First 10 rows of Main Data:**")
st.write(data.head(10))

# Visualize the distribution of tracks across different decades using a count plot
st.subheader("Distribution of Tracks Across Decades")

# Set figure size
plt.figure(figsize=(12, 6))

# Create count plot
sns.countplot(
    x=data['decade'], 
    hue=data['decade'],  
    palette='viridis', 
    order=sorted(data['decade'].unique()),
    legend=False  
)

# Set the title and labels for the plot
plt.title("Number of Tracks per Decade", fontsize=14)
plt.xlabel("Decade", fontsize=12)
plt.ylabel("Number of Tracks", fontsize=12)

# Display the plot in Streamlit
st.pyplot(plt)

# Convert 'year' to 'decade'
year_data['decade'] = (year_data['year'] // 10) * 10

# Define sound features
sound_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']

# Streamlit Section for Trends Visualization
st.subheader("Trends of Various Sound Features Over Decades")

# Create line plot using Plotly Express
fig = px.line(
    year_data, 
    x='decade',   
    y=sound_features, 
    title='Trend of Various Sound Features Over Decades'
)

# Display the plot in Streamlit
st.plotly_chart(fig)

# Streamlit Section for Loudness Trend
st.subheader("Trend of Loudness Over Decades")

# Create line plot using Plotly Express
fig = px.line(
    year_data, 
    x='decade',   
    y='loudness',  
    title='📊 Evolution of Loudness in Music Over the Decades',
    markers=True,  
    template='plotly_dark'  
)

# Customize axis labels
fig.update_layout(
    xaxis_title="Decade",
    yaxis_title="Average Loudness (dB)",  
    font=dict(size=12),  
)

# Display the plot in Streamlit
st.plotly_chart(fig)

# Get the top 10 genres based on popularity
top10_genres = (
    genre_data.groupby("genres", as_index=False)  # Group by genres
    .mean(numeric_only=True)  # Compute mean for numeric columns
    .sort_values(by="popularity", ascending=False)  # Sort by popularity (descending)
    .head(10)  # Get top 10 genres
)

# Display the top 10 genres with their popularity in Streamlit
st.subheader("🎵 Top 10 Music Genres by Popularity")
st.dataframe(top10_genres[["genres", "popularity"]])  # Show genres & popularity

# Streamlit Section for the grouped bar chart
st.subheader("📊 Trend of Various Sound Features Over Top 10 Genres")

# Create a grouped bar chart
fig = px.bar(
    top10_genres,
    x="genres",
    y=["valence", "energy", "danceability", "acousticness"],
    barmode="group",
    title="Trend of Various Sound Features Over Top 10 Genres",
    labels={"value": "Feature Value", "variable": "Sound Features"},
    template="plotly_dark",
)

# Customize layout
fig.update_layout(
    xaxis_title="Music Genres",
    yaxis_title="Average Feature Value",
    legend_title="Sound Features",
    font=dict(size=12),
)

# Display the plot in Streamlit
st.plotly_chart(fig)

# Combine all genres into a single string for the word cloud
comment_words = " ".join(genre_data["genres"].astype(str))

# Define stopwords (common words to be ignored in visualization)
stopwords = set(STOPWORDS)

# Generate the word cloud
wordcloud = WordCloud(
    width=800, height=800,  # Set the size of the image
    background_color='white',  # Background color of the word cloud
    stopwords=stopwords,  # Remove common words
    max_words=40,  # Maximum words to display
    min_font_size=10  # Minimum font size
).generate(comment_words)

# Display the word cloud in Streamlit
st.subheader("🎨 Word Cloud of Music Genres")
fig, ax = plt.subplots(figsize=(8, 8))  # Create figure
ax.imshow(wordcloud, interpolation='bilinear')  # Display word cloud
ax.axis("off")  # Hide axes

st.pyplot(fig)  # Render the plot in Streamlit