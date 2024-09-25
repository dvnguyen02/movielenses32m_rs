import pandas as pd

def merge_data(ratings, movies):
    return pd.merge(ratings, movies, on='movieId')

def process_timestamps(data):
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
    data['year'] = data['timestamp'].dt.year
    return data

def get_genre_dummies(data):
    return data['genres'].str.get_dummies(sep = '|')