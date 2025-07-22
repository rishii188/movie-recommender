import pandas as pd

#Load the rating files
ratings = pd.read_csv('ratings.dat', sep = '::', engine = 'python', names = ['UserID', 'MovieID', 'Rating', 'Timestamp'])

print(ratings.head())
print(ratings.info())
