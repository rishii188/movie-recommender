import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from surprise import Dataset, Reader

#Load the rating, movie and user files
ratings = pd.read_csv('ratings.dat', sep = '::', engine = 'python', encoding = 'latin-1', names = ['UserID', 'MovieID', 'Rating', 'Timestamp'])
movies = pd.read_csv('movies.dat', sep = '::', engine = 'python', encoding = 'latin-1', names = ['MovieID', 'Title', 'Genres'])
users = pd.read_csv('users.dat', sep = '::', engine = 'python', encoding = 'latin-1', names = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])

# Display the structure of each DataFrame
print("\t\t\t--- DataFrame Structures: ---")
print('\n\n',ratings.info())
print('\n\n',movies.info())
print('\n\n',users.info())

# Display the first few rows of each DataFrame
print("\n\t\t\t--- First few rows of each DataFrame: ---")
print(ratings.head())
print(movies.head())
print(users.head())

# Check for missing values in each DataFrame
print("\n\t\t\t--- Missing values in each DataFrame: ---")
print(ratings.isnull().sum())
print(movies.isnull().sum())
print(users.isnull().sum())

# Basic statistics for numerical columns
print("\n\t\t\t--- Basic statistics for numerical columns: ---")
print(ratings.describe())
print(users['Age'].value_counts())

# Unique counts
print("\n\t\t\t--- Unique counts in each DataFrame: ---")
print(f"Unique Users: {ratings['UserID'].nunique()}")
print(f"Unique Movies: {ratings['MovieID'].nunique()}")

# Users giving consistent high/low ratings
print("\n\t\t\t--- User ratings: ---")
user_ratings = ratings.groupby('UserID')['Rating'].agg(['mean', 'std'])
user_ratings = user_ratings[user_ratings['mean'] > 4.7][user_ratings['std'] < 0.5].sort_values(by='mean', ascending=False)
print("Users with consistently high ratings:")
print(user_ratings)
print("\nUsers with consistently low ratings:")
user_ratings = ratings.groupby('UserID')['Rating'].agg(['mean', 'std'])
user_ratings = user_ratings[user_ratings['mean'] < 2.0][user_ratings['std'] < 0.5].sort_values(by='mean', ascending=True)
print(user_ratings)

# How many ratings per user?
user_counts = ratings['UserID'].value_counts()
sns.histplot(user_counts, bins=50, kde=True)
plt.title('Number of Ratings per User')
plt.xlabel('Ratings per user')
plt.ylabel('Number of Users')
plt.show()

# How many ratings per movie?
movie_counts = ratings['MovieID'].value_counts()
sns.histplot(movie_counts, bins=50, kde=True)
plt.title('Number of Ratings per Movie')
plt.xlabel('Ratings per Movie')
plt.ylabel('Number of Movies')
plt.show()

# Merge ratings with movie titles
ratings_movies = pd.merge(ratings, movies, on='MovieID')
ratings_movies.head()

# Top rated movies (by count)
top_movies = ratings_movies['Title'].value_counts().head(10)
top_movies.plot(kind='barh')
plt.title('Most Rated Movies')
plt.xlabel('Number of Ratings')
plt.ylabel('Movie Title')
plt.gca().invert_yaxis()
plt.show()

# Average rating per movie
movie_avg = ratings_movies.groupby('Title')['Rating'].agg(['mean', 'count'])
filtered = movie_avg[movie_avg['count'] > 100].sort_values('mean', ascending = False).head(10)
filtered['mean'].plot(kind='barh')
plt.title('Top Rated Movies (min 100 ratings)')
plt.xlabel('Average Rating')
plt.ylabel('Movie Title')
plt.gca().invert_yaxis()
plt.show()

# Genre breakdown
genre_counts = Counter()
for genre_list in movies['Genres']:
    for genre in genre_list.split('|'):
        genre_counts[genre] += 1

pd.Series(genre_counts).sort_values(ascending=True).plot(kind='barh', figsize=(10, 6))
plt.title('Genre Distribution')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')
plt.show()

# User demographics
sns.countplot(data=users, x='Gender')
plt.title('Gender Distribution of Users')
plt.show()

sns.histplot(users['Age'], bins=10, kde=True)
plt.title('User Age Distribution')
plt.xlabel('Age')
plt.ylabel('Number of Users')
plt.show()

# Scatter plot: Number of ratings vs average rating
plt.figure(figsize=(10, 6))
sns.scatterplot(data=movie_avg, x='count', y='mean')
plt.title('Number of Ratings vs Average Rating')
plt.xlabel('Number of Ratings')
plt.ylabel('Average Rating')
plt.show()

# TF-IDF for movie titles
movies['Title'] = movies['Title'].fillna('')
tfidf = TfidfVectorizer(stop_words='english')
# Turns movie titles into TF-IDF vectors
tfidf_matrix = tfidf.fit_transform(movies['Title'])
user_id = 1
# Filtering table for user_id = 1
user_data = ratings_movies[ratings_movies['UserID'] == user_id]
# Binary label: 1 if rating >= 4, else 0
user_data['Liked'] = (user_data['Rating'] >= 4).astype(int)
# Only get the TF-IDF rows for the movie this user rated
user_tfidf = tfidf_matrix[user_data['MovieID']]
# Binary classification model to predict if the user would like a movie
model = LogisticRegression()
# Fit the model using the user's TF-IDF vectors and their ratings
model.fit(user_tfidf, user_data['Liked']) #Train the model
# Predict probabilities for all movies
all_preds = model.predict_proba(tfidf_matrix)[:, 1]
# Get indices of movies sorted by predicted like probability
top_movie_indices = all_preds.argsort()[::-1]

# Create a DataFrame of predictions
movies['predicted_like_prob'] = all_preds
recommendations = movies.sort_values('predicted_like_prob', ascending=False)

# Filter out movies the user has already rated
rated_movies = set(user_data['MovieID'])
recommendations = recommendations[~recommendations['MovieID'].isin(rated_movies)] # Remove rated movies

# Show top recommended movies
print("\n\t\t\t--- Top Recommended Movies for User ID 1: ---")
print(recommendations[['Title', 'predicted_like_prob']].head(10))
