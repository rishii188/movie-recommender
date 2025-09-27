# Movie Recommender System

This project builds a movie recommendation system using the **MovieLens dataset**. It combines exploratory data analysis, classical ML baselines, and collaborative filtering (SVD) to generate and evaluate personalised movie recommendations.

## Project structure
- `main.py`: main script containing EDA, baselines, and recommender models.  
- `movies.dat`, `ratings.dat`, `users.dat`: MovieLens 1M dataset files.  
- `visuals/`: folder for generated plots and charts.  
- `svd_metrics.csv`: saved evaluation metrics for the SVD model.  

## Methods
1. **Exploratory Data Analysis (EDA)**  
   - Distribution of ratings per user and per movie.  
   - Top-rated and most popular movies.  
   - Genre breakdown.  
   - User demographics (gender, age).  

2. **Content-based filtering (baseline)**  
   - Used **TF-IDF vectorisation** on movie titles.  
   - Logistic Regression to predict whether a user will “like” a movie (rating ≥ 4).  

3. **Collaborative filtering**  
   - Implemented **SVD (Singular Value Decomposition)** from the Surprise library.  
   - Evaluated using 5-fold cross-validation and a hold-out split.  
   - Calculated metrics: RMSE, MAE, Precision@10, Recall@10.  

## Example results
- Baseline Logistic Regression achieved reasonable accuracy for content-based predictions.  
- SVD model performed best with:  
  - CV RMSE ≈ *[fill in]*  
  - Precision@10 ≈ *[fill in]*  
  - Recall@10 ≈ *[fill in]*  

*(See `svd_metrics.csv` for detailed metrics.)*

## Visuals
The project generates plots such as:  
- Ratings per user / per movie  
- Top rated movies  
- Genre distribution  
- User demographics  
- Recommendation bar charts (per user)  

All saved to the `visuals/` directory.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

Minimal `requirements.txt`:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
scikit-surprise
```

## Usage
Run the main script:
```bash
python main.py
```

This will:
- Perform exploratory analysis.  
- Train baseline models.  
- Train/evaluate an SVD recommender.  
- Save metrics (`svd_metrics.csv`) and plots in `visuals/`.  

## Status
The system currently supports:
- EDA and visualisation of the MovieLens dataset.  
- Baseline content-based model using TF-IDF + Logistic Regression.  
- Collaborative filtering with SVD (evaluated with multiple metrics).  

Future improvements could include:
- Adding user/item-based kNN models.  
- Hyperparameter tuning for SVD.  
- Streamlit dashboard for interactive recommendations.
