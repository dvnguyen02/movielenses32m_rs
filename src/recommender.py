def get_top_n_recommendations(model, movies, ratings, user_id, n=10):
    all_movie_ids = movies['movieId'].unique()
    user_ratings = ratings[ratings['userId'] == user_id]['movieId']
    movies_to_predict = list(set(all_movie_ids) - set(user_ratings))
    
    predictions = [model.predict(user_id, movie_id) for movie_id in movies_to_predict]
    top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    
    top_n_titles = [(movies[movies['movieId'] == pred.iid]['title'].iloc[0], pred.est) for pred in top_n]
    return top_n_titles