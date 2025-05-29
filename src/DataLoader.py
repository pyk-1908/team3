import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Dict, Union


class DataLoader:
    def __init__(self):
        self.data = None
        self.loaded_files = {}  # Store multiple loaded DataFrames
        
    def load_data(self, file_path: str, name: str = None):
        """
        Load data from an Excel file and handle missing values.
        
        Args:
            file_path (str): Path to the Excel file.
            name (str, optional): Name to reference this dataset. Defaults to filename.
            
        Returns:
            self: Returns the instance for method chaining
        """
        # Use filename as default name if none provided
        if name is None:
            name = file_path.split('/')[-1].split('.')[0]
            
        # Load the data
        data = pd.read_excel(file_path)
        data.fillna(method='ffill', inplace=True)
        
        # Store in loaded_files dictionary
        self.loaded_files[name] = data
        
        # Set as current data if it's the first file
        if self.data is None:
            self.data = data
            
        return self

    def load_multiple_files(self, file_dict: Dict[str, str]):
        """
        Load multiple files at once.
        
        Args:
            file_dict (Dict[str, str]): Dictionary of {name: file_path} pairs
            
        Returns:
            self: Returns the instance for method chaining
        """
        for name, file_path in file_dict.items():
            self.load_data(file_path, name)
        return self

    def merge_loaded_files(self, merge_config: List[Dict]):
        """
        Merge multiple loaded files based on configuration.
        
        Args:
            merge_config (List[Dict]): List of merge configurations.
                Each dict should contain:
                - 'left': name of left dataset
                - 'right': name of right dataset
                - 'left_on': column(s) from left dataset
                - 'right_on': column(s) from right dataset
                
        Returns:
            self: Returns the instance for method chaining
        """
        if not self.loaded_files:
            raise ValueError("No files loaded. Load files first using load_data() or load_multiple_files()")
        
        result = self.loaded_files[merge_config[0]['left']]
        
        for config in merge_config:
            right_df = self.loaded_files[config['right']]
            result = pd.merge(
                result,
                right_df,
                left_on=config['left_on'],
                right_on=config['right_on']
            )
            
        self.data = result
        return self

    def split_data(self, target_column, test_size=0.2):
        """
        Split the data into training and testing sets.
        
        Args:
            target_column (str): Name of the target column
            test_size (float): Proportion of dataset to include in the test split
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data first.")
            
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        return train_test_split(X, y, test_size=test_size, random_state=42)

    def get_data(self):
        """
        Get the current DataFrame.
        
        Returns:
            pd.DataFrame: The current DataFrame
        """
        return self.data


