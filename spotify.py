import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import sys
import io


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