import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split, cross_validate
import os

# Create visuals folder if not exists
if not os.path.exists("visuals"):
    os.makedirs("visuals")

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
plt.savefig("visuals/ratings_per_user.png", dpi=300, bbox_inches='tight')
plt.show()

# How many ratings per movie?
movie_counts = ratings['MovieID'].value_counts()
sns.histplot(movie_counts, bins=50, kde=True)
plt.title('Number of Ratings per Movie')
plt.xlabel('Ratings per Movie')
plt.ylabel('Number of Movies')
plt.savefig("visuals/ratings_per_movie.png", dpi=300, bbox_inches='tight')
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
plt.savefig("visuals/top_rated_movies.png", dpi=300, bbox_inches='tight')
plt.show()

# Average rating per movie
movie_avg = ratings_movies.groupby('Title')['Rating'].agg(['mean', 'count'])
filtered = movie_avg[movie_avg['count'] > 100].sort_values('mean', ascending = False).head(10)
filtered['mean'].plot(kind='barh')
plt.title('Top Rated Movies (min 100 ratings)')
plt.xlabel('Average Rating')
plt.ylabel('Movie Title')
plt.gca().invert_yaxis()
plt.savefig("visuals/top_rated_movies_avg.png", dpi=300, bbox_inches='tight')
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
plt.savefig("visuals/genre_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

# User demographics
sns.countplot(data=users, x='Gender')
plt.title('Gender Distribution of Users')
plt.savefig("visuals/gender_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

sns.histplot(users['Age'], bins=10, kde=True)
plt.title('User Age Distribution')
plt.xlabel('Age')
plt.ylabel('Number of Users')
plt.savefig("visuals/age_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

# Scatter plot: Number of ratings vs average rating
plt.figure(figsize=(10, 6))
sns.scatterplot(data=movie_avg, x='count', y='mean')
plt.title('Number of Ratings vs Average Rating')
plt.xlabel('Number of Ratings')
plt.ylabel('Average Rating')
plt.savefig("visuals/ratings_vs_avg.png", dpi=300, bbox_inches='tight')
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

# Save bar chart of top 10 recommendations
top_recs = recommendations.head(10)
plt.figure(figsize=(10, 6))
plt.barh(top_recs['Title'], top_recs['predicted_like_prob'])
plt.xlabel('Predicted Like Probability')
plt.ylabel('Movie Title')
plt.title(f'Top Recommended Movies for User {user_id}')
plt.gca().invert_yaxis()
plt.savefig(f"visuals/top_recommendations_user_{user_id}.png", dpi=300, bbox_inches='tight')
plt.show()

# Split user data into train/test sets
train_data, test_data = train_test_split(user_data, test_size=0.2, random_state=42)

model.fit(tfidf_matrix[train_data['MovieID']], train_data['Liked'])

# Predictions on the test set
y_true = test_data['Liked']
y_pred_proba = model.predict_proba(tfidf_matrix[test_data['MovieID']])[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

# Calculate accuracy
acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred_proba)

# Print metrics
print("\n\t\t\t--- Model Evaluation Metrics: ---")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Check to see if visuals folder exists
os.makedirs("visuals", exist_ok=True)

# Preparing Surprise Dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['UserID', 'MovieID', 'Rating']], reader)

# 5-fold cross-validation
svd_cv = SVD(random_state=42, n_factors=100, reg_all=0.02, n_epochs=20)
cv_results = cross_validate(svd_cv, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)

cv_rmse_mean = cv_results['test_rmse'].mean()
cv_rmse_std = cv_results['test_rmse'].std()
cv_mae_mean = cv_results['test_mae'].mean()
cv_mae_std = cv_results['test_mae'].std()

print("\n--- 5-fold CV (SVD) ---")
print(f"RMSE: {cv_rmse_mean:.4f} ± {cv_rmse_std:.4f}")
print(f"MAE : {cv_mae_mean:.4f} ± {cv_mae_std:.4f}")

# Hold-out split for ranking metrics and top-N recs
trainset, testset = surprise_train_test_split(data, test_size=0.2, random_state=42)

svd = SVD(random_state=42, n_factors=100, reg_all=0.02, n_epochs=20)
svd.fit(trainset)

# Predict on hold-out test for RMSE/MAE
predictions = svd.test(testset)
holdout_rmse = accuracy.rmse(predictions, verbose=False)
holdout_mae  = accuracy.mae(predictions, verbose=False)
print("\n--- Hold-out (SVD) ---")
print(f"RMSE: {holdout_rmse:.4f}")
print(f"MAE : {holdout_mae:.4f}")

def precision_recall_at_k(preds, k=10, threshold=4.0):
    # Map: user -> list of (est, true_r)
    user_est_true = {}
    for p in preds:
        user_est_true.setdefault(p.uid, []).append((p.est, p.r_ui))
    precisions, recalls = {}, {}

    for uid, user_ratings in user_est_true.items():
        # Sort by estimated rating desc
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Top-K predicted relevant
        top_k = user_ratings[:k]

        # True relevant items in all of user's test items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        # True relevant among top-K
        n_rel_and_rec_k = sum((true_r >= threshold) for (_, true_r) in top_k)

        precisions[uid] = n_rel_and_rec_k / k if k else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    # Macro-average over users
    precision = float(np.mean(list(precisions.values()))) if precisions else 0.0
    recall = float(np.mean(list(recalls.values()))) if recalls else 0.0
    return precision, recall

p_at_10, r_at_10 = precision_recall_at_k(predictions, k=10, threshold=4.0)
print("\n--- Ranking Metrics (hold-out) ---")
print(f"Precision@10: {p_at_10:.4f}")
print(f"Recall@10   : {r_at_10:.4f}")

# Top-N recommendations for a specific user excluding seen
def get_top_n_for_user(algo, trainset, raw_user_id, movies_df, n=10):
    try:
        inner_uid = trainset.to_inner_uid(raw_user_id)
    except ValueError:
        raise ValueError(f"User {raw_user_id} not in training set.")

    # Items the user has already rated
    user_rated_inner = set(j for (j, _) in trainset.ur[inner_uid])

    # Predict for all items not yet rated
    candidates = []
    for raw_iid in trainset._raw2inner_id_items.keys():
        inner_iid = trainset.to_inner_iid(raw_iid)
        if inner_iid in user_rated_inner:
            continue
        est = algo.predict(raw_user_id, raw_iid, verbose=False).est
        candidates.append((raw_iid, est))

    # Sort by predicted rating desc
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_n = candidates[:n]

    # Join with titles for display
    top_movie_ids = [mid for (mid, _) in top_n]
    top_df = movies_df[movies_df['MovieID'].isin(top_movie_ids)].copy()
    score_map = dict(top_n)
    top_df['predicted_rating'] = top_df['MovieID'].map(score_map)
    return top_df.sort_values('predicted_rating', ascending=False)

# choose a user present in training set
example_user = int(ratings['UserID'].iloc[0])  # or set to 1 if exists
top_recs = get_top_n_for_user(svd, trainset, example_user, movies, n=10)
print(f"\n--- Top-10 Recommendations for User {example_user} (SVD) ---")
print(top_recs[['Title', 'predicted_rating']].head(10))

# Save metrics and a small bar chart
metrics_df = pd.DataFrame({
    'Metric': ['CV_RMSE_mean', 'CV_RMSE_std', 'CV_MAE_mean', 'CV_MAE_std',
               'Holdout_RMSE', 'Holdout_MAE', 'Precision@10', 'Recall@10'],
    'Value':  [cv_rmse_mean,   cv_rmse_std,   cv_mae_mean,   cv_mae_std,
               holdout_rmse,   holdout_mae,   p_at_10,       r_at_10]
})
metrics_df.to_csv('svd_metrics.csv', index=False)
print("\nSaved: svd_metrics.csv")

# bar chart of top-10 recs
plt.figure(figsize=(10,6))
plt.barh(top_recs['Title'].head(10), top_recs['predicted_rating'].head(10))
plt.xlabel('Predicted Rating')
plt.ylabel('Movie Title')
plt.title(f'Top-10 Recommendations (SVD) for User {example_user}')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"visuals/svd_top10_user_{example_user}.png", dpi=300, bbox_inches='tight')
plt.show()
