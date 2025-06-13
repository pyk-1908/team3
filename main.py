from CausalNex.DataLoader import DataLoader
from CausalNex.Model import CausalNexModel, BayesianNetworkModel
from CausalNex.Preprocessing import Preprocessing, FeatureDiscretizationAnalyzer
from CausalNex.Logger import Logger
from CausalNex.evaluation import placebo_test
import os
import argparse

def run_experiment(is_new_experiment=True, model_path=None):
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

    # add negative Treatment variable
    # drop Rate_Lag and ACR columns as we will depend only on Treatment increase, decrease, or no change
    merged_dataset = preprocessor.calculate_treatment(merged_dataset)

    # add churn variable
    # drop Members and Members_Lag columns as we will depend only on Chrun existence or not
    merged_dataset = preprocessor.calculate_churn(merged_dataset)

    # learn the network structure automatically from the data using the NOTEARS algorithm. 
    # (NOTEARS is a recently published algorithm for learning DAGs from data, framed as a continuous optimisation problem. It allowed us to overcome the challenges of combinatorial optimisation, giving a new impetus to the usage of BNs in machine learning applications.)
    #  we want to make our data numeric, since this is what the NOTEARS expects.
    #  We can do this by label encoding non-numeric variables.
    merged_dataset, label_encoders = preprocessor.label_encode_non_numeric(merged_dataset)
    logger.log("Data label encoded successfully.")


    if is_new_experiment:
        # Save encoded dataset
        logger.save_dataframe(merged_dataset, "preprocessed_dataset")

    ############## Defining the DAG with StructureModel #######################
    CausalNex = CausalNexModel()

    # Create and save new structure model
    structure_model_path = os.path.join(NOTEARS_checkpoint_folder, 'NOTEARS_DAG_causal_structure_model.dot')
    CausalNex.create_structure(merged_dataset, max_iter=10000, w_threshold=0.1)
    if is_new_experiment:
        CausalNex.save_structure(structure_model_path)
        logger.log("New structure model created and saved.")

    ################# adjusting the structure model #######################
    # Remove edges with smallest weight until the graph is a DAG.
    CausalNex.structure_model.threshold_till_dag()
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
            "encoding_info": {col: list(le.classes_) for col, le in label_encoders.items()},
            "adjusted": True
        }
        logger.save_model_results(adjusted_model_info, "adjusted_model_configuration")


    #################### preprocessing for Baysian Network ######################
    # define the causal variables from the structure model
    causal_variables = list(CausalNex.structure_model.nodes())
    data = merged_dataset[causal_variables].copy()
    # analyze the causal variables to check for any discretization needs
    analyzer = FeatureDiscretizationAnalyzer(data,logger=logger)
    # Inspect all features
    analyzer.inspect_features()
    # Create visualizations
    analyzer.create_visualizations(causal_variables, save=is_new_experiment)


    # Discretize the features based on the analysis
    # config ( other features do not need discretization)
    discretization_config = {
        'RiskFactor': {'method': 'equal_width', 'n_bins': 20},
    }
    # Discretize the dataset
    discretized_dataset = preprocessor.batch_discretize(data, discretization_config)

    data_loader.data = discretized_dataset


    # Treatment is the treatment variable, and Churn is the outcome variable.



    ###################### Learning the Bayesian Network ######################
    if is_new_experiment:
        # Create and train new Bayesian Network model
        bayesian_network_model = BayesianNetworkModel(CausalNex.structure_model)
        train, test = data_loader.split_data(test_size=0.2)
        bayesian_network_model.fit(discretized_dataset, train=train)
        bayesian_network_model.save_cpds_with_logger(cpds=['Year', 'Provider','Quarter', 'RiskFactor', 'Churn', 'Regionality', 'Treatment'] ,logger= logger)
        bayesian_network_model.save_model(model_save_path)
        logger.log("New Bayesian Network model trained and saved.")
    else:
        # Load existing Bayesian Network model
        bayesian_network_model = BayesianNetworkModel(model_path=model_save_path)
        train, test = data_loader.split_data(test_size=0.2)
        logger.log("Existing Bayesian Network model loaded.")

    # Evaluate model
    classification_report, roc, auc = bayesian_network_model.classification_report(test, 'Churn')
    
    # Save results if it's a new experiment
    if is_new_experiment:
        logger.save_roc_plot(roc, auc, filename="bayesian_network_roc", folder=visualization_folder)
        logger.log(f"Results saved for new experiment. AUC: {auc:.3f}")
    
    
    ####################### Do Calculus ########################
    # Perform do-calculus to estimate the effect of treatment on churn
    #  First let’s update our model using the complete dataset
    # removed Year as it has a lot of cpds
    CausalNex.structure_model.remove_node('Year')
    CausalNex.structure_model.remove_node('Provider')
    # CausalNex.structure_model.remove_node('Quarter')
    # # CausalNex.structure_model.remove_node('Regionality')
    # CausalNex.structure_model.remove_node('RiskFactor')


    # cropped_data_loader = DataLoader()
    # cropped_data_loader.data = data_loader.data.copy().drop('Year', axis=1)
    # cropped_data_loader.data = cropped_data_loader.data.drop('Provider', axis=1)
    # # cropped_data_loader.data = cropped_data_loader.data.drop('Quarter', axis=1)
    # # # cropped_data_loader.data = cropped_data_loader.data.drop('Regionality', axis=1)
    # # cropped_data_loader.data = cropped_data_loader.data.drop('RiskFactor', axis=1)
   

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
        "roc": roc,
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
    
    args = parser.parse_args()
    
    report, roc, auc = run_experiment(
        is_new_experiment=args.new_experiment,
        model_path=args.model_path
    )
    print(f"Experiment completed. Final AUC: {auc:.3f}")


