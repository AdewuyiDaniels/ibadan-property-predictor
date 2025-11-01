import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from xgboost import XGBRegressor
import warnings
import json
import joblib
from datetime import datetime

warnings.filterwarnings("ignore")

class Config:
    RANDOM_STATE = 42
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    CV_FOLDS = 5
    N_JOBS = -1
    EXPERIMENT_NAME = f"housing_price_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    MODEL_DIR = 'models'
    RESULTS_DIR = 'results'

config = Config()

def load_data():
    print("Loading cleaned Ibadan dataset...")
    df = pd.read_csv("cleaned_ibadan_properties.csv")
    return df

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def get_param_grids():
    return {
        'RandomForest': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [10, 20, None],
            'model__min_samples_split': [2, 5],
            'model__min_samples_leaf': [1, 2]
        },
        'GradientBoosting': {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.05, 0.1, 0.2],
            'model__max_depth': [3, 5]
        },
        'XGBoost': {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.05, 0.1, 0.2],
            'model__max_depth': [3, 5],
            'model__subsample': [0.8, 1.0]
        },
        'Ridge': {'model__alpha': [0.1, 1.0, 10.0]},
        'Lasso': {'model__alpha': [0.001, 0.01, 0.1]}
    }

def calculate_metrics(y_true, y_pred, prefix=''):
    y_true_actual = np.expm1(y_true)
    y_pred_actual = np.expm1(y_pred)

    metrics = {
        f'{prefix}_r2': r2_score(y_true, y_pred),
        f'{prefix}_mae': mean_absolute_error(y_true, y_pred),
        f'{prefix}_mae_actual': mean_absolute_error(y_true_actual, y_pred_actual),
        f'{prefix}_mape': mean_absolute_percentage_error(y_true_actual, y_pred_actual) * 100
    }
    return metrics

def train_models():
    df = load_data()
    df = remove_outliers(df, 'Price')

    X = df.drop(columns=['Price'])
    y = np.log1p(df['Price'])

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=config.VAL_SIZE / (1 - config.TEST_SIZE), random_state=config.RANDOM_STATE)

    numeric_transformer = Pipeline(steps=[('scaler', RobustScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)],
        remainder='passthrough'
    )

    models = {
        'RandomForest': RandomForestRegressor(random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS),
        'GradientBoosting': GradientBoostingRegressor(random_state=config.RANDOM_STATE),
        'XGBoost': XGBRegressor(random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS, verbosity=0),
        'Ridge': Ridge(),
        'Lasso': Lasso(max_iter=5000)
    }

    param_grids = get_param_grids()
    results = []

    print(f"Starting model training for experiment: {config.EXPERIMENT_NAME}")
    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

        search = RandomizedSearchCV(pipeline, param_grids[name], n_iter=10, cv=config.CV_FOLDS, random_state=config.RANDOM_STATE, scoring='r2', n_jobs=config.N_JOBS, error_score='raise')
        search.fit(X_train, y_train)

        best_model_pipeline = search.best_estimator_

        y_train_pred = best_model_pipeline.predict(X_train)
        y_val_pred = best_model_pipeline.predict(X_val)
        y_test_pred = best_model_pipeline.predict(X_test)

        train_metrics = calculate_metrics(y_train, y_train_pred, 'train')
        val_metrics = calculate_metrics(y_val, y_val_pred, 'val')
        test_metrics = calculate_metrics(y_test, y_test_pred, 'test')

        result = {
            'model_name': name,
            'best_params': search.best_params_,
            'cv_r2_mean': search.best_score_,
            **train_metrics,
            **val_metrics,
            **test_metrics,
            'pipeline_object': best_model_pipeline
        }
        results.append(result)

        print(f"{name} validation R2: {result['val_r2']:.4f}, Test R2: {result['test_r2']:.4f}")

    results_df = pd.DataFrame(results).sort_values('test_r2', ascending=False)

    best_model_name = results_df.iloc[0]['model_name']
    best_pipeline = [r['pipeline_object'] for r in results if r['model_name'] == best_model_name][0]

    print("\nSaving results...")
    results_df_serializable = results_df.drop(columns=['pipeline_object'], errors='ignore')
    results_df_serializable['best_params'] = results_df_serializable['best_params'].astype(str)
    results_df_serializable.to_json(f"{config.RESULTS_DIR}/experiment_results.json", orient='records', indent=4)

    joblib.dump(best_pipeline, f"{config.MODEL_DIR}/best_model.pkl")
    joblib.dump(preprocessor, f"{config.MODEL_DIR}/preprocessor.pkl")

    print("Model training complete.")

if __name__ == '__main__':
    train_models()
