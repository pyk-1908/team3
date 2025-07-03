from CausalNex.DataLoader import DataLoader
from CausalNex.Model import CausalNexModel, BayesianNetworkModel
from CausalNex.Preprocessing import Preprocessing, FeatureDiscretizationAnalyzer
from CausalNex.Logger import Logger
from CausalNex.evaluation import placebo_test
import os
import argparse
import pandas as pd
from lazypredict.Supervised import LazyClassifier

def run_experiment(is_new_experiment=True, is_classification=False, model_path=None):
    """
    Run the causal analysis experiment.
    
    Args:
        is_new_experiment (bool): If True, train and save new models. If False, load existing ones.
        model_path (str): Path to saved model if is_new_experiment is False
    """
    # Initialize logger
    logger = Logger(output_dir="output")
    
    dataset_path = 'Dataset/merged_data.csv'
    visualization_folder = 'visualizations'
    NOTEARS_checkpoint_folder = os.path.join('checkpoints', 'NOTEARS_DAG_definition')
    model_save_path = os.path.join('models', 'bayesian_network.pkl')
    preprocessor_save_path = os.path.join('models', 'preprocessor.pkl')

    # Create necessary directories
    for directory in [visualization_folder, NOTEARS_checkpoint_folder, 'models']:
        os.makedirs(directory, exist_ok=True)

    #################### Load the data  ######################
    data_loader = DataLoader()
    data_loader.load_data(dataset_path, 'merged_dataset')
    merged_dataset = data_loader.data
    logger.log("Data loaded successfully.")

    ##################################################################
    ################# Learning the Confounders #######################
    ##################################################################

    ################# preprocessing #######################

    preprocessor = Preprocessing(logger=logger)

    # add average ACR variable
    merged_dataset = preprocessor.calculate_other_provider_avg_acr(merged_dataset)

    # add negative Treatment variable
    # drop Rate_Lag and ACR columns as we will depend only on Treatment increase, decrease, or no change
    merged_dataset = preprocessor.calculate_treatment(merged_dataset)

    # add churn variable
    # drop Members and Members_Lag columns as we will depend only on Chrun existence or not
    merged_dataset = preprocessor.calculate_churn(merged_dataset)

    # Split data BEFORE preprocessing to prevent data leakage
    # We'll use a simple train/test split for preprocessing consistency
    from sklearn.model_selection import train_test_split
    
    # Split for preprocessing (we'll do another split later for model training)
    train_data, test_data = train_test_split(merged_dataset, test_size=0.2, random_state=42, stratify=merged_dataset.get('Churn'))
    logger.log(f"Data split into train ({len(train_data)}) and test ({len(test_data)}) sets.")

    # learn the network structure automatically from the data using the NOTEARS algorithm. 
    # (NOTEARS is a recently published algorithm for learning DAGs from data, framed as a continuous optimisation problem. It allowed us to overcome the challenges of combinatorial optimisation, giving a new impetus to the usage of BNs in machine learning applications.)
    #  we want to make our data numeric, since this is what the NOTEARS expects.
    #  We can do this by label encoding non-numeric variables.
    
    if is_new_experiment:
        # Fit label encoders on training data only
        preprocessor.fit_label_encode_non_numeric(train_data)
        # Transform both train and test data
        train_encoded = preprocessor.transform_label_encode_non_numeric(train_data)
        test_encoded = preprocessor.transform_label_encode_non_numeric(test_data)
        
        # Combine for structure learning (this is acceptable since we're learning the causal structure)
        merged_dataset_encoded = pd.concat([train_encoded, test_encoded], ignore_index=True)
        
        logger.log("Label encoders fitted on training data and applied to both sets.")
        
        # Save the fitted preprocessor for future use
        import pickle
        with open(preprocessor_save_path, 'wb') as f:
            pickle.dump(preprocessor, f)
        logger.log("Preprocessor saved successfully.")
        
    else:
        # Load existing preprocessor
        import pickle
        with open(preprocessor_save_path, 'rb') as f:
            preprocessor = pickle.load(f)
        
        # Transform both datasets using loaded preprocessor
        train_encoded = preprocessor.transform_label_encode_non_numeric(train_data)
        test_encoded = preprocessor.transform_label_encode_non_numeric(test_data)
        merged_dataset_encoded = pd.concat([train_encoded, test_encoded], ignore_index=True)
        
        logger.log("Existing preprocessor loaded and applied to data.")

    if is_new_experiment:
        # Save encoded dataset
        logger.save_dataframe(merged_dataset_encoded, "preprocessed_dataset")

    ############## Defining the DAG with StructureModel #######################
    CausalNex = CausalNexModel()

    # Create and save new structure model
    structure_model_path = os.path.join(NOTEARS_checkpoint_folder, 'NOTEARS_DAG_causal_structure_model.dot')
    
    if is_new_experiment:
        # Use full encoded dataset for structure learning
        CausalNex.create_structure(merged_dataset_encoded, max_iter=10000, w_threshold=0.1)
        CausalNex.save_structure(structure_model_path)
        logger.log("New structure model created and saved.")
    else:
        # Load existing structure model
        CausalNex.load_structure(structure_model_path)
        logger.log("Existing structure model loaded.")

    ################# adjusting the structure model #######################
    # Remove edges with smallest weight until the graph is a DAG.
    CausalNex.structure_model.threshold_till_dag()
    CausalNex.structure_model.remove_node('Year')
    CausalNex.structure_model.remove_node('Provider')
    DAG = CausalNex.adjacency_dict()
    print("Structure model adjusted to ensure it is a DAG.")
    logger.log("Structure model adjusted to ensure it is a DAG.")
    
    if is_new_experiment:
        # Plot and save the adjusted structure model
        adjusted_plot_title = f"DAG_Causal_Structure_Model_{logger.timestamp}"
        CausalNex.plot_structure(plot_title=adjusted_plot_title, path=visualization_folder)
        print(f"Adjusted structure model plotted successfully as {adjusted_plot_title}")
        logger.log(f"Adjusted structure model plotted successfully as {adjusted_plot_title}")

        # Save the adjusted structure model as a DOT file
        adjusted_dot_file = os.path.join(NOTEARS_checkpoint_folder, f'NOTEARS_DAG_causal_structure_model.dot')
        CausalNex.save_structure(adjusted_dot_file)
        logger.log(f"Adjusted structure model saved successfully as {adjusted_dot_file}")

        # Save adjusted model configuration
        adjusted_model_info = {
            "max_iter": 10000,
            "w_threshold": 0.1,
            "dataset_path": dataset_path,
            "encoding_info": {col: list(preprocessor.label_encoders[col].classes_) for col in preprocessor.label_encoders.keys()},
            "adjusted": True
        }
        logger.save_model_results(adjusted_model_info, "adjusted_model_configuration")

        # save the DAG structure as a JSON file
        logger.save_model_results(DAG, "DAG_structure")

    #################### preprocessing for Bayesian Network ######################
    # define the causal variables from the structure model
    causal_variables = list(CausalNex.structure_model.nodes())
    
    # Extract causal variables from both train and test sets
    train_causal = train_encoded[causal_variables].copy()
    test_causal = test_encoded[causal_variables].copy()
    
    # analyze the causal variables to check for any discretization needs
    # Use only training data for analysis to prevent data leakage
    analyzer = FeatureDiscretizationAnalyzer(train_causal, logger=logger)
    # Inspect all features
    analyzer.inspect_features()
    # Create visualizations
    analyzer.create_visualizations(causal_variables, save=is_new_experiment)

    # Discretize the features based on the analysis
    # config (other features do not need discretization)
    discretization_config = {
        'RiskFactor': {'method': 'equal_width', 'n_bins': 20},
        'Avg_ACR_Other_Providers': {'method': 'equal_frequency', 'n_bins': 18},
    }
    
    if is_new_experiment:
        # Fit discretizers on training data only
        preprocessor.fit_batch_discretize(train_causal, discretization_config)
        # Transform both datasets
        train_discretized = preprocessor.transform_batch_discretize(train_causal)
        test_discretized = preprocessor.transform_batch_discretize(test_causal)
        
        # Update the saved preprocessor with discretization parameters
        with open(preprocessor_save_path, 'wb') as f:
            pickle.dump(preprocessor, f)
        logger.log("Discretizers fitted on training data and preprocessor updated.")
        
    else:
        # Use existing fitted discretizers
        train_discretized = preprocessor.transform_batch_discretize(train_causal)
        test_discretized = preprocessor.transform_batch_discretize(test_causal)
        logger.log("Existing discretizers applied to data.")

    # Combine discretized data for Bayesian Network training (if needed)
    discretized_dataset = pd.concat([train_discretized, test_discretized], ignore_index=True)
    
    # Update data_loader with discretized dataset
    data_loader.data = discretized_dataset

    # Treatment is the treatment variable, and Churn is the outcome variable.
    classification_report = None
    auc = None
    roc = None
    if not is_classification:

        ###################### Learning the Bayesian Network ######################
        if is_new_experiment:
            # Create and train new Bayesian Network model
            bayesian_network_model = BayesianNetworkModel(CausalNex.structure_model)
            
            # Use the properly preprocessed training data for model fitting
            bayesian_network_model.fit(train_discretized, train=train_discretized)
            bayesian_network_model.save_cpds_with_logger(cpds=['Quarter', 'RiskFactor', 'Churn', 'Regionality', 'Treatment'], logger=logger)
            bayesian_network_model.save_model(model_save_path)
            logger.log("New Bayesian Network model trained and saved.")
        else:
            # Load existing Bayesian Network model
            bayesian_network_model = BayesianNetworkModel(model_path=model_save_path)
            logger.log("Existing Bayesian Network model loaded.")

        # Evaluate model on properly preprocessed test data
        classification_report, roc, auc = bayesian_network_model.classification_report(test_discretized, 'Churn')
        
        # Save results if it's a new experiment
        if is_new_experiment:
            logger.save_roc_plot(roc, auc, filename="bayesian_network_roc", folder=visualization_folder)
            logger.log(f"Results saved for new experiment. AUC: {auc:.3f}")
    
    

    ####################### Classification Experiment ########################
    if is_classification:
        # drop the 'Chrun' column from the dataset
        # Initial split into training, validation, and test sets
        X = merged_dataset.drop(columns=['Churn'])
        y = merged_dataset['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        # Further split the training data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

        # apply fitted preprocessor to train val, and test sets
        X_train = preprocessor.transform_label_encode_non_numeric(X_train)
        X_val = preprocessor.transform_label_encode_non_numeric(X_val)
        X_test = preprocessor.transform_label_encode_non_numeric(X_test)

        # Initialize LazyClassifier
        lazy_classifier = LazyClassifier(
            ignore_warnings=True,
            custom_metric=None,
            random_state=42,
            verbose=0
        )
        # Fit LazyClassifier on training data
        models_class, predictions_class = lazy_classifier.fit(X_train, X_val, y_train, y_val)
        # add models_class to logger
        logger.save_dataframe(models_class, "lazy_classifier_models", format='excel')

        # use the best model from LazyClassifier to predict on the test set
        from lightgbm import LGBMClassifier
        from sklearn.metrics import f1_score

        classifier = LGBMClassifier()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        from sklearn.metrics import classification_report, roc_auc_score, roc_curve


        classification_report = classification_report(y_test, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y_test, classifier.predict_proba(X_test)[:, 1])
        fpr, tpr, _ = roc_curve(y_test, classifier.predict_proba(X_test)[:, 1])
        roc = list(zip(fpr, tpr))
        auc = roc_auc
        logger.log(f"Classification Report: {classification_report}")
        logger.log(f"AUC: {auc:.3f}")   



        # save roc plot
        logger.save_roc_plot(roc, auc, filename="LGBMClassifier", folder=visualization_folder)




    ####################### Do Calculus ########################
    # Perform do-calculus to estimate the effect of treatment on churn
    #  First let’s update our model using the complete dataset
    # removed Year as it has a lot of cpds
    # CausalNex.structure_model.remove_node('Quarter')
    # # CausalNex.structure_model.remove_node('Regionality')
    # CausalNex.structure_model.remove_node('RiskFactor')


    # cropped_data_loader = DataLoader()
    # cropped_data_loader.data = data_loader.data.copy().drop('Year', axis=1)
    # cropped_data_loader.data = cropped_data_loader.data.drop('Provider', axis=1)
    # # cropped_data_loader.data = cropped_data_loader.data.drop('Quarter', axis=1)
    # # # cropped_data_loader.data = cropped_data_loader.data.drop('Regionality', axis=1)
    # # cropped_data_loader.data = cropped_data_loader.data.drop('RiskFactor', axis=1)
    average_ate = None
    placebo_results = None
    if not is_classification:

        new_bayesian_network = BayesianNetworkModel(structure_model=CausalNex.structure_model)
        bn = new_bayesian_network.fit_cpds(discretized_dataset)


        # Average Treatment Effect (ATE)
        ate_results, average_ate = bayesian_network_model.estimate_ate(bn=bn, treatment='Treatment', outcome='Churn')
        logger.log(f"Average Treatment Effect (ATE): {ate_results}, Average ATE: {average_ate}")

        # Conditional Average Treatment Effect (CATE)
        data_with_CATE = data_loader.add_CATE(bayesian_network_model, bn, treatment='Treatment', outcome='Churn')

        # Save the dataset with CATE
        logger.save_dataframe(data_with_CATE, "dataset_with_CATE", format='csv')

        # Perform placebo test
        placebo_results = placebo_test(bayesian_network_model= bayesian_network_model, bn = new_bayesian_network, data_with_CATE=data_with_CATE,
                                                            treatment='Treatment', outcome='Churn', logger=logger, save=is_new_experiment)


    model_results = {
        "classification_report": classification_report,
        "auc": auc,
        "ATE": average_ate,
        "placebo_test": placebo_results,
    }
    if is_new_experiment:
        logger.save_model_results(model_results, "model_results")
        logger.log("all model_results saved.")
    
    return classification_report, roc, auc




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run causal analysis experiment')
    parser.add_argument('--new-experiment', action='store_true', default=False,
                      help='If set, train and save new models. Otherwise, load existing ones.')
    parser.add_argument('--model-path', type=str, default=None,
                      help='Path to saved model for loading (ignored if --new-experiment is set)')
    parser.add_argument('--classification-experiment', action='store_true', default=False,
                      help='If set, execute the classification experiment.')
    
    args = parser.parse_args()
    
    report, roc, auc = run_experiment(
        is_new_experiment=args.new_experiment,
        is_classification=args.classification_experiment,
        model_path=args.model_path
    )
    print(f"Experiment completed. Final AUC: {auc:.3f}")

