import os
import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Load environment variables
load_dotenv()
client_id = os.getenv("SPOTIPY_CLIENT_ID")
client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")

# Validate and authenticate Spotify client 
if not client_id or not client_secret:
    st.error("Spotify credentials not found in .env file.")
    st.stop()

auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

# Load local dataset
st.title("🎧 Spotify Song Data Finder & Recommender")

try:
    dataset = pd.read_csv("data.csv")
    st.success("✅ Local dataset loaded.")
except FileNotFoundError:
    st.error("Dataset file not found. Please ensure 'data.csv' exists.")
    st.stop()

# Define feature columns for audio characteristics
feature_cols = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'duration_ms'
]

#Find a song via Spotify API
def find_song(name, artist):
    try:
        query = f'track:{name} artist:{artist}'
        result = sp.search(q=query, limit=1, type='track')
        tracks = result['tracks']['items']
        if not tracks:
            return None
        track = tracks[0]
        return {
            'name': track['name'],
            'artist': track['artists'][0]['name'],
            'id': track['id'],
            'uri': track['uri'],
            'duration_ms': track['duration_ms'],
            'explicit': track['explicit'],
            'popularity': track['popularity'],
            'album': track['album']['name'],
            'release_date': track['album']['release_date']
        }
    except Exception as e:
        st.error(f"Error finding song: {e}")
        return None

# Get song data from local dataset or Spotify
def get_song_data(song, dataset):
    name, artist = song.get('name'), song.get('artist')
    match = dataset[
        (dataset['name'].str.lower() == name.lower()) &
        (dataset['artists'].str.lower().str.contains(artist.lower()))
    ]
    if not match.empty:
        row = match.iloc[0]
        return {
            'name': row['name'],
            'artist': row['artists'],
            'id': row.get('id'),
            'uri': row.get('uri'),
            'duration_ms': row.get('duration_ms'),
            'explicit': row.get('explicit'),
            'popularity': row.get('popularity'),
            'album': row.get('album'),
            'release_date': row.get('release_date')
        }
    return find_song(name, artist)

#Compute mean vector of audio features
def get_mean_vector(song_list, dataset, feature_cols):
    vectors = []
    for song in song_list:
        data = get_song_data(song, dataset)
        if data is None:
            continue
        match = dataset[
            (dataset['name'].str.lower() == data['name'].lower()) &
            (dataset['artists'].str.lower().str.contains(data['artist'].lower()))
        ]
        if not match.empty:
            vector = match.iloc[0][feature_cols].values
            vectors.append(vector)
    return np.mean(np.array(vectors), axis=0) if vectors else None

# Flatten a list of dictionaries into a dict of lists
def flatten_dict_list(dict_list):
    result = defaultdict(list)
    for d in dict_list:
        for key, value in d.items():
            result[key].append(value)
    return dict(result)

# Recommend similar songs
def recommend_songs(song_list, dataset, feature_cols, n_recommendations=5):
    mean_vec = get_mean_vector(song_list, dataset, feature_cols)
    if mean_vec is None:
        return pd.DataFrame()

    song_features = dataset[feature_cols].values
    similarities = cosine_similarity([mean_vec], song_features)[0]

    dataset = dataset.copy()
    dataset['similarity'] = similarities

    # Remove input songs from recommendations
    input_names = [s['name'].lower() for s in song_list]
    input_artists = [s['artist'].lower() for s in song_list]
    mask = dataset.apply(
        lambda row: row['name'].lower() in input_names and any(artist in row['artists'].lower() for artist in input_artists),
        axis=1
    )
    filtered_df = dataset[~mask]

    recommendations = filtered_df.sort_values(by='similarity', ascending=False).head(n_recommendations)
    columns_to_show = ['name', 'artists', 'album', 'popularity', 'similarity'] if 'album' in recommendations.columns else ['name', 'artists', 'popularity', 'similarity']
    return recommendations[columns_to_show]

#  Search for a single song
st.subheader("🔍 Search Song Details")
song_name = st.text_input("Enter song name:")
artist_name = st.text_input("Enter artist name:")

if st.button("Search for Song"):
    if song_name and artist_name:
        song_input = {"name": song_name, "artist": artist_name}
        data = get_song_data(song_input, dataset)
        if data:
            st.success(f"✅ Found: {data['name']} by {data['artist']}")
            table = pd.DataFrame.from_dict(data, orient='index', columns=['Value'])
            table['Value'] = table['Value'].astype(str)
            st.table(table)
        else:
            st.error("🚫 Song not found.")
    else:
        st.warning("Please enter both song name and artist.")

# Recommend Similar Songs
st.markdown("---")
st.header("🎯 Recommend Similar Songs")

song1 = st.text_input("Song 1 - Name")
artist1 = st.text_input("Song 1 - Artist")

song2 = st.text_input("Song 2 - Name (optional)")
artist2 = st.text_input("Song 2 - Artist (optional)")

if st.button("Recommend Songs"):
    song_list = []
    if song1 and artist1:
        song_list.append({'name': song1, 'artist': artist1})
    if song2 and artist2:
        song_list.append({'name': song2, 'artist': artist2})

    if song_list:
        recs = recommend_songs(song_list, dataset, feature_cols, n_recommendations=5)
        if not recs.empty:
            st.success("🎶 Recommended Songs:")
            st.dataframe(recs.reset_index(drop=True))
        else:
            st.warning("No recommendations found.")
    else:
        st.warning("Please enter at least one song and artist.")
