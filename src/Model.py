from causalnex.structure.notears import from_pandas, StructureModel
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
import networkx as nx
from causalnex.network import BayesianNetwork

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
    

class BayesianNetworkModel:
    def __init__(self, structure_model=None):
        """
        Initialize the Bayesian Network Model with an optional structure model.
        
        Args:
            structure_model (StructureModel, optional): A pre-defined structure model.
        """
        self.bayesian_network = BayesianNetwork(structure_model) if structure_model else None

    def fit(self, df, train):
        """
        specifying all of the states that each node can take

        Args:
            df (pd.DataFrame): DataFrame containing the discretised data to fit the node states.
            train list: List of columns to be used for training the Bayesian Network.

        Returns:
            self: Returns the instance for method chaining
        """
        if self.bayesian_network is None:
            raise ValueError("No Bayesian Network structure defined. Create one first.")
        
        self.bayesian_network.fit_node_states(df)
        self.bayesian_network.fit_cpds(train, method="BayesianEstimator", bayes_prior="K2")
        return self