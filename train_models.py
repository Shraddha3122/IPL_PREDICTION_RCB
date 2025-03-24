# Import libraries
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, r2_score
import warnings

# Suppress XGBoost warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Create models directory if not exists
if not os.path.exists('models'):
    os.makedirs('models')

# Load datasets with low_memory=False
matches_df = pd.read_csv('D:/WebiSoftTech/ML_MINI_PROJECTS/IPL PREDICTION(RCB)/IPL Matches 2008-2023.csv', low_memory=False)
ball_df = pd.read_csv('D:/WebiSoftTech/ML_MINI_PROJECTS/IPL PREDICTION(RCB)/IPL Ball-by-Ball 2008-2023.csv', low_memory=False)

# Check column names to confirm merging readiness
print(f"✅ Columns in matches_df: {matches_df.columns}")
print(f"✅ Columns in ball_df: {ball_df.columns}")

# Merge first_innings_score from ball_df with matches_df
first_innings_df = ball_df.groupby('id')['total_runs'].sum().reset_index(name='first_innings_score')
matches_df = matches_df.merge(first_innings_df, on='id', how='left')

# Handle NaN values after merging
if matches_df['first_innings_score'].isnull().sum() > 0:
    print(f"❌ Found {matches_df['first_innings_score'].isnull().sum()} NaN values. Dropping NaN rows.")
    matches_df = matches_df.dropna(subset=['first_innings_score'])

# Confirm merge success
if 'first_innings_score' not in matches_df.columns:
    raise KeyError("❌ Column 'first_innings_score' not found after merging. Check column names in Ball-by-Ball dataset.")

# Preprocessing
matches_df = matches_df.dropna(subset=['winner'])
matches_df['winner'] = np.where(matches_df['winner'] == 'Royal Challengers Bangalore', 1, 0)

# Encode categorical columns
encoder = LabelEncoder()
for col in ['venue', 'toss_decision', 'team1', 'team2']:
    matches_df[col] = encoder.fit_transform(matches_df[col])

# Save encoders
with open('models/encoders.pkl', 'wb') as f:
    pickle.dump(encoder, f)

# Feature and target selection
X_win = matches_df[['venue', 'toss_decision', 'team1', 'team2']]
y_win = matches_df['winner']
X_score = matches_df[['venue', 'team1', 'team2']]
y_score = matches_df['first_innings_score']

# Apply SMOTE for class imbalance
smote = SMOTE(random_state=42)
X_win, y_win = smote.fit_resample(X_win, y_win)

# Standardize numerical features for regression models
scaler = StandardScaler()
X_score = scaler.fit_transform(X_score)
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))

# Define models and parameter grid
models_classification = {
    'RandomForest': (RandomForestClassifier(), {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}),
    'XGBoost': (XGBClassifier(eval_metric='logloss'), {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}),
    'LogisticRegression': (LogisticRegression(max_iter=1000), {'C': [0.1, 1, 10]}),
    'GradientBoosting': (GradientBoostingClassifier(), {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]})
}

models_regression = {
    'RandomForest': (RandomForestRegressor(), {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}),
    'XGBoost': (XGBRegressor(), {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}),
    'GradientBoosting': (GradientBoostingRegressor(), {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]})
}

# Function to find the best classification model using GridSearchCV
def get_best_classification_model(X, y, models):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model, best_score = None, 0

    # StratifiedKFold for better validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for name, (model, params) in models.items():
        grid = GridSearchCV(model, params, cv=skf, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        y_pred = grid.best_estimator_.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        print(f"✅ {name} Best Accuracy: {score:.4f} with Params: {grid.best_params_}")

        if score > best_score:
            best_score = score
            best_model = grid.best_estimator_

    return best_model

# Function to find the best regression model using GridSearchCV
def get_best_regression_model(X, y, models):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check for NaN values in y before fitting models
    if np.isnan(y).sum() > 0:
        raise ValueError("❌ y_score contains NaN values. Check preprocessing or data merge.")

    best_model, best_r2 = None, -1

    for name, (model, params) in models.items():
        grid = GridSearchCV(model, params, cv=10, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)
        y_pred = grid.best_estimator_.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(f"📊 {name} Best R² Score: {r2:.4f} with Params: {grid.best_params_}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = grid.best_estimator_

    return best_model

# Train and save the best models
print("🎯 Training Models for Match Win Prediction...")
best_win_model = get_best_classification_model(X_win, y_win, models_classification)
pickle.dump(best_win_model, open('models/best_win_model.pkl', 'wb'))

print("📊 Training Models for Score Prediction...")
best_score_model = get_best_regression_model(X_score, y_score, models_regression)
pickle.dump(best_score_model, open('models/best_score_model.pkl', 'wb'))

print("✅ Models optimized and saved successfully!")
