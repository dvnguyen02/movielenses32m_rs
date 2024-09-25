from src.data_loader import load_data, prepare_surprise_data
from src.preprocessor import merge_ratings_movies, process_timestamps
from src.feature_engineering import add_user_movie_averages
from src.model_trainer import create_svd_model, train_model
from src.recommender import get_top_n_recommendations
from surprise import accuracy

def main():
    # Load and prepare data
    ratings, movies = load_data('data/ratings.csv', 'data/movies.csv')
    data = prepare_surprise_data(ratings)

    # Preprocess
    df = merge_ratings_movies(ratings, movies)
    df = process_timestamps(df)
    df = add_user_movie_averages(df)

    # Train model
    model = create_svd_model()
    trained_model, testset = train_model(data, model)

    # Evaluate model
    predictions = trained_model.test(testset)
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")

    # Generate recommendations
    user_id = 1
    recommendations = get_top_n_recommendations(trained_model, movies, ratings, user_id)
    print(f"Top 10 recommendations for user {user_id}:")
    for title, est in recommendations:
        print(f"{title}: Estimated rating = {est:.2f}")

if __name__ == "__main__":
    main()