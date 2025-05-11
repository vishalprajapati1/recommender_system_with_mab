from sklearn.model_selection import train_test_split
import pandas as pd


def load_data(file_path):
    ratings = pd.read_csv(file_path + 'ratings.dat', sep='::', engine='python',
                          names=['userId', 'movieId', 'rating', 'timestamp'])
    movies = pd.read_csv(file_path + 'movies.dat', sep='::', engine='python',
                         names=['movieId', 'title', 'genres'], encoding='latin-1')
    users = pd.read_csv(file_path + 'users.dat', sep='::', engine='python',
                        names=['userId', 'gender', 'age', 'occupation', 'zip'])
    return ratings, movies, users


def preprocess_data(ratings):
    user_ids = ratings['userId'].unique()
    movie_ids = ratings['movieId'].unique()
    user_to_idx = {userId: idx for idx, userId in enumerate(user_ids)}
    movie_to_idx = {movieId: idx for idx, movieId in enumerate(movie_ids)}
    
    ratings['userId'] = ratings['userId'].map(user_to_idx)
    ratings['movieId'] = ratings['movieId'].map(movie_to_idx)
    
    ratings['rating'] = (ratings['rating'] >= 4).astype(float)
    train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=23)
    
    train_data = train_data[['userId', 'movieId', 'rating']].values
    test_data = test_data[['userId', 'movieId', 'rating']].values
    
    return train_data, test_data, len(user_ids), len(movie_ids)
