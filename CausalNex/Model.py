from causalnex.structure.notears import from_pandas
from causalnex.structure import StructureModel
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
import networkx as nx
from causalnex.network import BayesianNetwork
from causalnex.evaluation import classification_report, roc_auc
from causalnex.inference import InferenceEngine
import pandas as pd
import os
import pickle

class CausalNexModel:
    def __init__(self):
        self.structure_model = None

    def create_structure(self, df=None, *args, **kwargs):
        """
        Create a structure model from the given DataFrame. If not given then create an empty structure.
        
        Args:
            df: DataFrame to create structure from
            *args: Additional positional arguments to pass to from_pandas()
            **kwargs: Additional keyword arguments to pass to from_pandas()
            
        Returns:
            self: Returns the instance for method chaining
        """
        if df is None or df.empty:
            self.structure_model = StructureModel()
        else:
            self.structure_model = from_pandas(df, *args, **kwargs)
        return self

    def plot_structure(self, path, node_style=NODE_STYLE.WEAK, edge_style=EDGE_STYLE.WEAK, plot_title="Causal Structure Model"):
        """
        Plot the causal structure model.

        Args:
            node_style (dict, optional): Custom styles for nodes.
            edge_style (dict, optional): Custom styles for edges.
            plot_title (str, optional): Title for the plot.

        Returns:
            self: Returns the instance for method chaining
        """
        if self.structure_model is None:
            raise ValueError("No structure model exists. Create one first using create_structure()")

        viz = plot_structure(
            self.structure_model, 
            all_node_attributes=node_style, 
            all_edge_attributes=edge_style
        )
        viz.toggle_physics(False)
        viz.show(f'{path}/{plot_title}.html')
        return self

    def save_structure(self, file_path):
        """
        Save the causal structure model to a file.

        Args:
            file_path (str): The path where the model will be saved.

        Returns:
            self: Returns the instance for method chaining
        """
        if self.structure_model is None:
            raise ValueError("No structure model exists. Create one first using create_structure()")

        nx.drawing.nx_pydot.write_dot(self.structure_model, file_path)
        return self
    

    def adjacency_dict(self):
        """
        Get the adjacency dictionary of the structure model.

        Returns:
            dict: Adjacency dictionary of the structure model.
        """
        if self.structure_model is None:
            raise ValueError("No structure model exists. Create one first using create_structure()")
        
        adjacency_dict = {}
        for node, adjacencies in self.structure_model.adjacency():
            adjacency_dict[node] = list(adjacencies)

        return adjacency_dict
    

class BayesianNetworkModel:
    def __init__(self, structure_model=None, model_path=None):
        """
        Initialize the Bayesian Network Model with either a structure model or load from file.
        
        Args:
            structure_model (StructureModel, optional): A pre-defined structure model
            model_path (str, optional): Path to a saved model file
        """
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.bayesian_network = BayesianNetwork(structure_model) if structure_model else None

        self.ie = None  # Inference Engine instance

    def fit(self, df, train):
        """
        specifying all of the states that each node can take

        Args:
            df (pd.DataFrame): DataFrame containing the discretised data to fit the node states. (all dataset)
            train list: List of columns to be used for training the Bayesian Network.

        Returns:
            self: Returns the instance for method chaining
        """
        if self.bayesian_network is None:
            raise ValueError("No Bayesian Network structure defined. Create one first.")
        
        self.bayesian_network.fit_node_states(df) # need to earn all the possible states of the nodes using the whole dataset
        self.bayesian_network.fit_cpds(train, method="BayesianEstimator", bayes_prior="K2")
        return self
    
    def fit_cpds(self, df):
        """
        Fit the Conditional Probability Distributions (CPDs) of the Bayesian Network.

        Args:
            df (pd.DataFrame): DataFrame containing the data to fit CPDs.

        Returns:
            self: Returns the instance for method chaining
        """
        if self.bayesian_network is None:
            raise ValueError("No Bayesian Network structure defined. Create one first.")
        
        bn = self.bayesian_network.fit_node_states_and_cpds(df, method="BayesianEstimator", bayes_prior="K2")
        return bn

    
    def predict(self, test_df, target:str=None):
        """
        Predict using the Bayesian Network.

        Args:
            df (pd.DataFrame): DataFrame containing the data to predict.

        Returns:
            pd.DataFrame: DataFrame with predictions.
        """
        if self.bayesian_network is None:
            raise ValueError("No Bayesian Network structure defined. Create one first.")
    
        return self.bayesian_network.predict(test_df,target)
    

    def get_cpd(self, cpd_name):
        """
        Convert CPDs to a structured DataFrame format.
        
        Returns:
            pd.DataFrame: DataFrame containing CPD information
        """
        if self.bayesian_network is None:
            raise ValueError("No Bayesian Network structure defined.")
        return self.bayesian_network.cpds[cpd_name]
        

    def save_cpds_with_logger(self, cpds, logger, file_name_base="bayesian_cpds"):
        """
        Save CPDs using the logger.
        
        Args:
            logger (Logger): Logger instance to use for saving
            filename (str): Base filename for the saved CPDs
        """
        try:
            for cpd in cpds:
                cpd_df = self.get_cpd(cpd)
                filename = f"{file_name_base}_{cpd}"
                # Save the DataFrame using the logger
                logger.save_dataframe(cpd_df, filename, format='csv')
                logger.log(f"Successfully saved CPDs to {filename}")
                print(f"Successfully saved CPDs to {filename}")
        except Exception as e:
            logger.log(f"Error saving CPDs: {str(e)}")
    
    def classification_report(self, test_df, target):
        """
        Generate a classification report for the Bayesian Network predictions.
        
        Args:
            test_df (pd.DataFrame): DataFrame containing the test data
            target (str): The target variable for classification
        
        Returns:
            str: Classification report as a string
        """
        if self.bayesian_network is None:
            raise ValueError("No Bayesian Network structure defined.")
        roc, auc = roc_auc(self.bayesian_network, test_df, target)
        report = classification_report(self.bayesian_network, test_df, target)
        report['auc'] = auc
        return report, roc, auc
    
    def save_model(self, filepath):
        """
        Save the Bayesian Network model to a file.
        
        Args:
            filepath (str): Path where to save the model
            
        Returns:
            self: Returns the instance for method chaining
        """
        if self.bayesian_network is None:
            raise ValueError("No Bayesian Network model to save.")
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.bayesian_network, f)
        return self

    def load_model(self, filepath):
        """
        Load a Bayesian Network model from a file.
        
        Args:
            filepath (str): Path to the saved model file
            
        Returns:
            self: Returns the instance for method chaining
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
            
        with open(filepath, 'rb') as f:
            self.bayesian_network = pickle.load(f)
        return self
    
    def estimate_ate(self, bn = None, treatment='Treatment', outcome='Churn', treatment_values=None):
        """
        Estimate the Average Treatment Effect (ATE) using the Bayesian Network for all pairs of treatment values.
        
        Args:
            treatment (str): The treatment variable
            outcome (str): The outcome variable
            treatment_values (list): List of possible treatment values. If None, uses [-1, 0, 1]
        
        Returns:
            tuple: (Dictionary of ATE comparisons between treatment value pairs, average ATE)
        """
        if bn is None and self.bayesian_network is None: 
            raise ValueError("No Bayesian Network model defined.")
        

        # Default treatment values if none provided
        if treatment_values is None:
            treatment_values = [-1, 0, 1]
        
        if self.ie is None:
            # Create inference engine if not already created
            if bn is None:
                self.ie = InferenceEngine(self.bayesian_network)
            else:
                self.ie = InferenceEngine(bn)
      
        expected_outcomes = {}
        for t in treatment_values:
            # Perform intervention for the treatment value
            self.ie.do_intervention(treatment, t)
            
            # Query the outcome variable
            outcome_probabilities = self.ie.query({})[outcome]
            
            # Reset intervention
            self.ie.reset_do(treatment)
            
            # Calculate expected outcome
            expected_outcome = sum([p * v for v, p in outcome_probabilities.items()])
            
            # Store expected outcome for this treatment value
            expected_outcomes[t] = expected_outcome
        # Compute pairwise ATEs
        ate = {
            "E[Y|do(T=-1)]": expected_outcomes[-1],
            "E[Y|do(T=0)]": expected_outcomes[0],
            "E[Y|do(T=1)]": expected_outcomes[1],
            "ATE(1 vs 0)": expected_outcomes[1] - expected_outcomes[0],
            "ATE(1 vs -1)": expected_outcomes[1] - expected_outcomes[-1],
            "ATE(0 vs -1)": expected_outcomes[0] - expected_outcomes[-1],
        }
        # Calculate average ATE
        average_ate = sum(ate.values()) / len(ate)
        return ate, average_ate
    
    def estimate_cate(self, bn = None, treatment='Treatment', outcome='Churn', treatment_values=None, x=None):
       
        if self.ie is None:
            # Create inference engine if not already created
            if bn is None:
                self.ie = InferenceEngine(self.bayesian_network)
            else:
                self.ie = InferenceEngine(bn)
        

        expected_outcomes = {}
       
        # Default treatment values if none provided
        if treatment_values is None:
            treatment_values = [-1, 0, 1]
        
        for t in treatment_values:
            # Perform intervention for the treatment value
            self.ie.do_intervention(treatment, t)
            
            # Query the outcome variable
            outcome_probabilities = self.ie.query(x)[outcome]
            
            # Reset intervention
            self.ie.reset_do(treatment)
            
            # Calculate expected outcome
            expected_outcome = sum([p * v for v, p in outcome_probabilities.items()])
            
            # Store expected outcome for this treatment value
            expected_outcomes[t] = expected_outcome

            # reset intervention
            self.ie.reset_do(treatment)

        # Compute pairwise CATEs
        cate_row = {
        "E[Y|do(T=-1)]": expected_outcomes[-1],
        "E[Y|do(T=0)]": expected_outcomes[0],
        "E[Y|do(T=1)]": expected_outcomes[1],
        "CATE(1 vs 0)": expected_outcomes[1] - expected_outcomes[0],
        "CATE(1 vs -1)": expected_outcomes[1] - expected_outcomes[-1],
        "CATE(0 vs -1)": expected_outcomes[0] - expected_outcomes[-1],
        }

        return cate_row


        