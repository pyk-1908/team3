# main.py
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_data
from gridsearch import run_all_grid_searches
from models import get_model, train_and_predict
from evaluation import evaluate_model, plot_feature_importance, plot_predictions_vs_actual, compare_models

def main():
    """Main pipeline to run the complete ML workflow"""
    print("="*60)
    print("MACHINE LEARNING PIPELINE")
    print("="*60)
    
    # Step 1: Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    
    # Replace 'your_data.csv' with actual data file
    df = pd.read_csv('data/Cate_added_data.csv')  
    X, y = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Step 2: Run grid search for all models
    print("\n2. Running hyperparameter optimization...")
    best_params_all, best_scores_all = run_all_grid_searches(X_train, y_train)
    
    # Step 3: Train models with best parameters and evaluate
    print("\n3. Training final models and evaluation...")
    print("="*60)
    
    model_types = ['random_forest', 'ridge', 'gradient_boosting']
    results = {}
    trained_models = {}
    
    for model_type in model_types:
        print(f"\nTraining {model_type.upper()}...")
        
        # Get model with best hyperparameters
        model = get_model(model_type, best_params_all[model_type])
        
        # Train and predict
        trained_model, predictions = train_and_predict(model, X_train, y_train, X_test)
        trained_models[model_type] = trained_model
        
        # Evaluate
        metrics = evaluate_model(y_test, predictions, X_test.shape[1],model_type.upper())
        results[model_type] = metrics
        
        # Generate plots
        plot_predictions_vs_actual(y_test, predictions, model_type.upper())
        plot_feature_importance(trained_model, X.columns, model_type.upper())
    
    # Step 4: Compare all models
    print("\n4. Model comparison...")
    compare_models(results)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return trained_models, results, best_params_all

if __name__ == "__main__":
    trained_models, results, best_params = main()
