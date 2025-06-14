# models.py
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from config import DEFAULT_HYPERPARAMS

# def get_model(model_type, hyperparams=None):
#     """Initialize model with best hyperparameters from grid search or defaults"""
    
#     if hyperparams is None:
#         hyperparams = DEFAULT_HYPERPARAMS[model_type]
    
#     # Add random_state for reproducibility
#     if 'random_state' not in hyperparams:
#         hyperparams['random_state'] = 42
    
#     models = {
#         "gradient_boosting": GradientBoostingRegressor(**hyperparams),
#         "ridge": Ridge(**hyperparams),
#         "random_forest": RandomForestRegressor(**hyperparams)
#     }
    
#     return models[model_type]

def get_model(model_type, hyperparams=None):
    """Initialize model with best hyperparameters from grid search"""
    if hyperparams is None:
        hyperparams = DEFAULT_HYPERPARAMS[model_type]
    
    # Remove incompatible parameters for Ridge
    if model_type == "ridge":
        hyperparams = {k: v for k, v in hyperparams.items() 
                      if k in ['alpha', 'random_state']}
    
    # Add random_state for reproducibility
    if 'random_state' not in hyperparams and model_type != "ridge":
        hyperparams['random_state'] = 42

    if model_type == "gradient_boosting":
        return GradientBoostingRegressor(**hyperparams)
    elif model_type == "ridge":
        return Ridge(**hyperparams)
    elif model_type == "random_forest":
        return RandomForestRegressor(**hyperparams)
    else:
        raise ValueError(f"Unknown model type: {model_type}")



def train_and_predict(model, X_train, y_train, X_test):
    """Train model and make predictions"""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, predictions


