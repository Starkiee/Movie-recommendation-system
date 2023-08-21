import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error

# Load the datasets
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Remove duplicates and missing values from ratings dataset
ratings = ratings.drop_duplicates()
ratings = ratings.dropna()

# Convert data into matrix format (user-item matrix)
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Split the dataset into training and testing sets
train_matrix, test_matrix = train_test_split(user_item_matrix, test_size=0.2, random_state=42)

# Implement collaborative filtering algorithm (using Singular Value Decomposition)
n_components = 50
svd = TruncatedSVD(n_components=n_components)
svd.fit(train_matrix)

# Transform the training matrix
train_matrix_svd = svd.transform(train_matrix)

# Inverse transform to get predicted ratings
predicted_ratings = svd.inverse_transform(train_matrix_svd)

# Calculate Mean Absolute Error
mae = mean_absolute_error(train_matrix, predicted_ratings)
print(f"Mean Absolute Error: {mae}")

# Define the number of recommendations to provide
num_recommendations = 5

# Define a function to recommend movies
def recommend_movies(user_id):
    user_predicted_ratings = predicted_ratings[user_id]
    top_movie_indices = user_predicted_ratings.argsort()[::-1][:num_recommendations]
    recommended_movie_ids = train_matrix.columns.to_numpy()[top_movie_indices]
    
    # Get movie titles from movies dataframe
    recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids.flatten())]['title']
    return recommended_movies

# Get user input for user_id
user_id = int(input("Enter the user ID: "))

# Generate recommended movies for the specified user
recommended_movies = recommend_movies(user_id)
print("Recommended movies:")
print(recommended_movies)


# Testing the model with new user ratings
"""new_user_ratings = {
    101: 5,
    205: 4,
}

new_user_vector = np.zeros(len(train_matrix.columns))
for movie_id, rating in new_user_ratings.items():
    new_user_vector[movie_id - 1] = rating  # Adjust movie_id to start from 0

new_user_vector_svd = svd.transform(new_user_vector.reshape(1, -1))
predicted_new_ratings = svd.inverse_transform(new_user_vector_svd)
top_movie_indices_new = predicted_new_ratings.argsort()[::-1][:num_recommendations]
recommended_movie_ids_new = train_matrix.columns.to_numpy()[top_movie_indices_new]

# Get movie titles from movies dataframe
recommended_movies_new = movies[movies['movieId'].isin(recommended_movie_ids_new.flatten())]['title']

print("Recommended movies for the new user:")
print(recommended_movies_new)"""