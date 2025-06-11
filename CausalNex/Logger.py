import os
from datetime import datetime
import json
import pandas as pd
from typing import Any, Dict, Union, List

class Logger:
    def __init__(self, output_dir: str = "output"):
        """
        Initialize Logger with output directory.
        
        Args:
            output_dir (str): Directory where outputs will be saved
        """
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._create_output_dir()
        self.log_file = os.path.join(self.output_dir, f"log_{self.timestamp}.txt")
        
    def _create_output_dir(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def log(self, message: str):
        """
        Log a message with timestamp.
        
        Args:
            message (str): Message to log
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_message)
            
    def save_model_results(self, results: Dict[str, Any], filename: str):
        """
        Save model results as JSON.
        
        Args:
            results (Dict): Dictionary containing model results
            filename (str): Name for the output file
        """
        output_path = os.path.join(self.output_dir, f"{filename}_{self.timestamp}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
            
    def save_dataframe(self, df: pd.DataFrame, filename: str, format: str = 'csv'):
        """
        Save DataFrame in specified format.
        
        Args:
            df (pd.DataFrame): DataFrame to save
            filename (str): Name for the output file
            format (str): Format to save ('csv' or 'excel')
        """
        output_path = os.path.join(self.output_dir, f"{filename}_{self.timestamp}")
        
        if format.lower() == 'csv':
            df.to_csv(f"{output_path}.csv", index=False)
        elif format.lower() == 'excel':
            df.to_excel(f"{output_path}.xlsx", index=False)
        else:
            raise ValueError("Unsupported format. Use 'csv' or 'excel'")
    
    def save_roc_plot(self, roc_data: List[tuple], auc: float, filename: str = "roc_curve", folder: str = None):
        """
        Create and save ROC curve plot from list of (fpr, tpr) tuples.
        
        Args:
            roc_data (List[tuple]): List of (fpr, tpr) tuples
            auc (float): Area Under the Curve value
            filename (str): Base name for the output file
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Unzip the tuples into separate lists
            fpr, tpr = zip(*roc_data)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            
            # Save plot
            if folder is None:
                folder = self.output_dir
            output_path = os.path.join(folder, f"{filename}_{self.timestamp}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log(f"ROC curve plot saved to {output_path}")
            
        except Exception as e:
            self.log(f"Error saving ROC plot: {str(e)}")
            raise