# gridsearch.py
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from config import HYPERPARAMS_GRID

def get_best_hyperparameters(X_train, y_train, model_type):
    """Perform grid search and return best hyperparameters[1][2][3]"""
    
    if model_type == 'random_forest':
        estimator = RandomForestRegressor(random_state=42)
        param_grid = HYPERPARAMS_GRID['random_forest']
        
    elif model_type == 'ridge':
        estimator = Ridge(random_state=42)
        param_grid = HYPERPARAMS_GRID['ridge']
        
    elif model_type == 'gradient_boosting':
        estimator = GradientBoostingRegressor(random_state=42)
        param_grid = HYPERPARAMS_GRID['gradient_boosting']
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create GridSearchCV object[1][2][3]
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=5,  # Using 5-fold cross-validation
        scoring='neg_mean_squared_error',  # Optimize for lower MSE (negated)
        n_jobs=-1  # Use all available cores
    )
    
    # Perform the grid search on training data[1][2][3]
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and best score[1][2][3]
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # Convert back to positive MSE
    
    print(f"\nBest parameters for {model_type}:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best cross-validation MSE: {best_score:.6f}")
    
    return best_params, best_score

def run_all_grid_searches(X_train, y_train):
    """Run grid search for all models and return best parameters"""
    best_params_all = {}
    best_scores_all = {}
    
    models = ['random_forest', 'ridge', 'gradient_boosting']
    
    for model_type in models:
        print(f"\n{'='*50}")
        print(f"Running grid search for {model_type.upper()}")
        print(f"{'='*50}")
        
        best_params, best_score = get_best_hyperparameters(X_train, y_train, model_type)
        best_params_all[model_type] = best_params
        best_scores_all[model_type] = best_score
    
    return best_params_all, best_scores_all
