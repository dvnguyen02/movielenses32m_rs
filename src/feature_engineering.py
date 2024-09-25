def add_user_movie_averages(df):
    df['user_avg_rating'] = df.groupby('userId')['rating'].transform('mean')
    df['movie_avg_rating'] = df.groupby('movieId')['rating'].transform('mean')
    return df

def get_genre_distribution(movies, movie_ids):
    genre_list = movies[movies['movieId'].isin(movie_ids)]['genres'].str.split('|', expand=True).stack()
    return genre_list.value_counts(normalize=True)