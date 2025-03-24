import pandas as pd
import os

# Define file paths
main_data_path = 'data.csv'
genre_data_path = 'data_by_genres.csv'
year_data_path = 'data_by_year.csv'
artist_data_path = 'data_by_artist.csv'

# Check if files exist before reading
for path in [main_data_path, genre_data_path, year_data_path, artist_data_path]:
    if not os.path.exists(path):
        print(f"Error: File {path} not found!")

# Read datasets
data = pd.read_csv(main_data_path)
genre_data = pd.read_csv(genre_data_path)
year_data = pd.read_csv(year_data_path)
artist_data = pd.read_csv(artist_data_path)

# Display the first two rows of each dataset
print("\nMain Data:\n", data.head(2))
print("\nGenre Data:\n", genre_data.head(2))
print("\nYear Data:\n", year_data.head(2))
print("\nArtist Data:\n", artist_data.head(2))

# Retrieve dataset info
print("\n" + "="*40 + " MAIN DATA INFO " + "="*40)
data.info()

print("\n" + "="*40 + " GENRE DATA INFO " + "="*40)
genre_data.info()

print("\n" + "="*40 + " YEAR DATA INFO " + "="*40)
year_data.info()

print("\n" + "="*40 + " ARTIST DATA INFO " + "="*40)
artist_data.info()

# Ensure 'year' is numeric and clean the data
data['year'] = pd.to_numeric(data['year'], errors='coerce')  # Convert to numeric
data.dropna(subset=['year'], inplace=True)  # Drop rows with NaN in 'year'
data['year'] = data['year'].astype(int)  # Convert year to integer

# Creating a Decade column
data['decade'] = data['year'].apply(lambda x: (x // 10) * 10)

# Confirm the decade column has been created
print("\nFirst 10 rows with Decade column:")
print(data[['year', 'decade']].head(10))

# Confirming column creation
print("\nUpdated Data Info:")
print(data.info())

# Check unique decades
print("\nUnique Decades:")
print(data['decade'].unique())  # Corrected method call

# Describe the statistics of the decade column
print("\nDecade Column Statistics:")
print(data['decade'].describe())
