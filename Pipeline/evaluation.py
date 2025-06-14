# evaluation.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def evaluate_model(y_true, y_pred, n_features, model_name):
    """Calculate evaluation metrics"""
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'Adjusted R2' : 1 - (1 - r2_score(y_true, y_pred)) * (len(y_true) - 1) / (len(y_true) - n_features - 1)
    }
    
    print(f"\n{model_name} Performance:")
    print("-" * 30)
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
    
    return metrics

def plot_feature_importance(model, feature_names, model_name):
    """Generate feature importance plot for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 8))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.title(f"Feature Importances - {model_name}")
        plt.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.gca().invert_yaxis()
        plt.xlabel('Importance')
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'feature_importance_{model_name}.png')  # Save the plot
        plt.close()
    else:
        print("-" * 30)
        print(f"\n\nFeature importance plot is not available for {model_name}")
        print("-" * 30)
        print("\n\n")

def plot_predictions_vs_actual(y_true, y_pred, model_name):
    """Plot predictions vs actual values"""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predictions vs Actual - {model_name}')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'pred_vs_actual_{model_name}.png')  # Save the plot
    plt.close()


# evaluation.py (updated)
def compare_models(results_dict):
    """Compare model performance using a formatted table"""
    # Create metric headers
    metrics = ['MSE','RMSE', 'MAE', 'R2', 'Adjusted R2']
    header = f"| {'Model':<20} | {' | '.join([f'{m:<10}' for m in metrics])} |"
    separator = f"|{'-'*21}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|"
    
    # Find best values for each metric
    best_values = {
        'MSE': min(v['MSE'] for v in results_dict.values()),
        'RMSE': min(v['RMSE'] for v in results_dict.values()),
        'MAE': min(v['MAE'] for v in results_dict.values()),
        'R2': max(v['R2'] for v in results_dict.values()),
        'Adjusted R2': max(v['Adjusted R2'] for v in results_dict.values())

    }
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    print(f"\n{header}")
    print(separator)
    
    # Build table rows with highlighting
    for model_name, metrics_dict in results_dict.items():
        cells = []
        for metric in metrics:
            value = metrics_dict[metric]
            # Highlight best performance
            if (metric in ['MSE','RMSE', 'MAE'] and value == best_values[metric]) or \
               (metric in ['R2', 'Adjusted R2'] and value == best_values[metric]):
                cells.append(f"**{value:.6f}**")
            else:
                cells.append(f"{value:.6f}")
        
        row = f"| {model_name:<20} | {' | '.join([f'{cell:<10}' for cell in cells])} |"
        print(row)
    
    # Find and display best overall model
    best_model = max(results_dict.items(), key=lambda x: x[1]['R2'])
    print("\n" + "-"*60)
    print(f"Best Overall Model: {best_model[0]} (R2 = {best_model[1]['R2']:.6f})")
    print("="*60)

