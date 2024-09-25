import pandas as pd
from surprise import Dataset, Reader

def load_data(rating_path, movie_path):
    ratings = pd.read_csv(rating_path)
    movies = pd.read_csv(movie_path)
    return ratings, movies

def prepare_surprise_data(ratings):
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    return data