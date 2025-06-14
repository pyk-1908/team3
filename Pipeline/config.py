# config.py
HYPERPARAMS_GRID = {
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'ridge': {
        'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
    },
    'gradient_boosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6]
    }
}

# Default hyperparameters (fallback)
DEFAULT_HYPERPARAMS = {
    'gradient_boosting': {
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 6,
        'random_state': 42
    },
    'ridge': {
        'alpha': 100,
        'random_state': 42
    },
    'random_forest': {
        'n_estimators': 100,
        'min_samples_leaf': 2,
        'min_samples_split': 5,
        'max_depth': None,
        'random_state': 42
    }
}
