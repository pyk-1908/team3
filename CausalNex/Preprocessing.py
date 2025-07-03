import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
import os
import json
from datetime import datetime

class FeatureDiscretizationAnalyzer:
    def __init__(self, data, logger=None, output_dir="output/feature_analysis"):
        """
        Initialize the analyzer.
        
        Args:
            data (pd.DataFrame): Input data
            logger (Logger, optional): Logger instance for logging
            output_dir (str): Directory to save visualizations and results
        """
        self.data = data
        self.logger = logger
        self.output_dir = output_dir
        self.numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        self.analysis_results = {}
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def inspect_features(self):
        """Comprehensive feature inspection for discretization planning"""
        self.logger.log("="*60)
        self.logger.log("FEATURE INSPECTION FOR DISCRETIZATION")
        self.logger.log("="*60)
        
        for feature in self.numeric_features:
            self.logger.log(f"\n FEATURE: {feature}")
            self.logger.log("-"*40)
            
            # Basic statistics
            self.analyze_distribution(feature)
            
            # Determine discretization need and method
            self.recommend_discretization(feature)
            
            self.logger.log("\n" + "="*40)
    
    def analyze_distribution(self, feature):
        """Analyze feature distribution and characteristics"""
        data_col = self.data[feature].dropna()
        
        # Basic stats
        stats_dict = {
            'count': len(data_col),
            'mean': data_col.mean(),
            'std': data_col.std(),
            'min': data_col.min(),
            'max': data_col.max(),
            'range': data_col.max() - data_col.min(),
            'unique_values': data_col.nunique(),
            'missing_values': self.data[feature].isnull().sum()
        }
        
        self.logger.log(f"Count: {stats_dict['count']:,}")
        self.logger.log(f"Mean: {stats_dict['mean']:.4f}")
        self.logger.log(f"Std: {stats_dict['std']:.4f}")
        self.logger.log(f"Min: {stats_dict['min']:.4f}")
        self.logger.log(f"Max: {stats_dict['max']:.4f}")
        self.logger.log(f"Range: {stats_dict['range']:.4f}")
        self.logger.log(f"Unique values: {stats_dict['unique_values']:,}")
        self.logger.log(f"Missing values: {stats_dict['missing_values']:,}")
        
        # Distribution characteristics
        skewness = stats.skew(data_col)
        kurtosis = stats.kurtosis(data_col)
        
        self.logger.log(f"Skewness: {skewness:.4f}")
        self.logger.log(f"Kurtosis: {kurtosis:.4f}")
        
        # Store analysis results
        self.analysis_results[feature] = {
            'stats': stats_dict,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'data': data_col
        }
    
    def recommend_discretization(self, feature):
        """Recommend discretization method and parameters"""
        stats_dict = self.analysis_results[feature]['stats']
        skewness = self.analysis_results[feature]['skewness']
        data_col = self.analysis_results[feature]['data']
        
        self.logger.log(f"\n DISCRETIZATION RECOMMENDATIONS:")
        
        # Check if discretization is needed
        unique_ratio = stats_dict['unique_values'] / stats_dict['count']
        
        if stats_dict['unique_values'] <= 10:
            self.logger.log(" DISCRETIZATION NOT NEEDED - Already has few unique values")
            return
        
        if unique_ratio > 0.5:
            self.logger.log("HIGH CARDINALITY - Consider discretization")
        
        # Recommend number of bins
        recommended_bins = self.suggest_bin_count(data_col)
        self.logger.log(f" Recommended bins: {recommended_bins}")
        
        # Recommend discretization method
        self.logger.log(f"\nRECOMMENDED METHODS:")
        
        # Equal-width binning
        self.logger.log(f"1. EQUAL-WIDTH BINNING")
        self.logger.log(f"   - Good for: Uniform distribution")
        self.logger.log(f"   - Bins: {recommended_bins}")
        self.show_equal_width_ranges(data_col, recommended_bins)
        
        # Equal-frequency binning
        self.logger.log(f"\n2. EQUAL-FREQUENCY BINNING (Quantile-based)")
        self.logger.log(f"   - Good for: Skewed distributions")
        self.logger.log(f"   - Bins: {recommended_bins}")
        self.show_quantile_ranges(data_col, recommended_bins)
        
        # K-means binning
        self.logger.log(f"\n3. K-MEANS BINNING")
        self.logger.log(f"   - Good for: Clustering similar values")
        self.logger.log(f"   - Bins: {recommended_bins}")
        
        # Custom recommendations based on distribution
        if abs(skewness) > 1:
            self.logger.log(f"\n SPECIAL RECOMMENDATION:")
            self.logger.log(f"   - Distribution is highly skewed ({skewness:.2f})")
            self.logger.log(f"   - Consider: Quantile-based or log-transformation first")
        
        if stats_dict['range'] > 1000:
            self.logger.log(f"   - Large range detected ({stats_dict['range']:.0f})")
            self.logger.log(f"   - Consider: Log-transformation or robust scaling")
    
    def suggest_bin_count(self, data_col):
        """Suggest optimal number of bins using multiple methods"""
        n = len(data_col)
        
        # Sturges' rule
        sturges = int(np.ceil(np.log2(n) + 1))
        
        # Scott's rule
        h_scott = 3.5 * data_col.std() / (n ** (1/3))
        scott_bins = int(np.ceil((data_col.max() - data_col.min()) / h_scott))
        
        # Freedman-Diaconis rule
        q75, q25 = np.percentile(data_col, [75, 25])
        iqr = q75 - q25
        h_fd = 2 * iqr / (n ** (1/3))
        fd_bins = int(np.ceil((data_col.max() - data_col.min()) / h_fd)) if h_fd > 0 else sturges
        
        # Square root rule
        sqrt_bins = int(np.ceil(np.sqrt(n)))
        
        # Take median of suggestions, bounded between 3 and 20
        suggestions = [sturges, scott_bins, fd_bins, sqrt_bins]
        median_suggestion = int(np.median([x for x in suggestions if 3 <= x <= 50]))
        
        return max(3, min(20, median_suggestion))
    
    def show_equal_width_ranges(self, data_col, n_bins):
        """Show equal-width bin ranges"""
        min_val, max_val = data_col.min(), data_col.max()
        bin_width = (max_val - min_val) / n_bins
        
        self.logger.log(f"   Bin width: {bin_width:.4f}")
        for i in range(n_bins):
            start = min_val + i * bin_width
            end = min_val + (i + 1) * bin_width
            self.logger.log(f"   Bin {i+1}: [{start:.4f}, {end:.4f})")
    
    def show_quantile_ranges(self, data_col, n_bins):
        """Show quantile-based bin ranges"""
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = data_col.quantile(quantiles).values
        
        for i in range(n_bins):
            self.logger.log(f"   Bin {i+1}: [{bin_edges[i]:.4f}, {bin_edges[i+1]:.4f})")
    
    def create_visualizations(self, feature_list=None, save=True):
        """
        Create visualizations for feature distributions.
        
        Args:
            feature_list (List[str], optional): List of features to visualize
            save (bool): Whether to save the plot to file
        """
        if feature_list is None:
            feature_list = self.numeric_features[:6]
            
        n_features = len(feature_list)
        fig, axes = plt.subplots(2, n_features, figsize=(4*n_features, 8))
        
        if n_features == 1:
            axes = axes.reshape(2, 1)
        
        for i, feature in enumerate(feature_list):
            data_col = self.data[feature].dropna()
            
            # Histogram
            axes[0, i].hist(data_col, bins=30, alpha=0.7, edgecolor='black')
            axes[0, i].set_title(f'{feature} - Distribution')
            axes[0, i].set_xlabel(feature)
            axes[0, i].set_ylabel('Frequency')
            
            # Box plot
            axes[1, i].boxplot(data_col)
            axes[1, i].set_title(f'{feature} - Box Plot')
            axes[1, i].set_ylabel(feature)
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f'feature_distributions_{timestamp}.png')
            plt.savefig(filename)
            if self.logger:
                self.logger.log(f"Feature distributions plot saved to {filename}")

    def demonstrate_discretization_methods(self, feature, n_bins=5):
        """Demonstrate different discretization methods on a specific feature"""
        data_col = self.data[feature].dropna()
        
        self.logger.log(f"\nDISCRETIZATION DEMONSTRATION FOR: {feature}")
        self.logger.log("="*50)
        
        # Original data sample
        self.logger.log(f"Original values (first 10): {data_col.head(10).tolist()}")
        
        # Method 1: Equal-width binning
        equal_width = pd.cut(data_col, bins=n_bins, labels=False)
        self.logger.log(f"\nEqual-width binning: {equal_width.head(10).tolist()}")
        
        # Method 2: Equal-frequency binning
        equal_freq = pd.qcut(data_col, q=n_bins, labels=False, duplicates='drop')
        self.logger.log(f"Equal-frequency binning: {equal_freq.head(10).tolist()}")
        
        # Method 3: K-means binning
        kmeans_discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans')
        kmeans_result = kmeans_discretizer.fit_transform(data_col.values.reshape(-1, 1)).flatten()
        self.logger.log(f"K-means binning: {kmeans_result[:10].astype(int).tolist()}")
        
        # Show bin boundaries
        self.logger.log(f"\nBIN BOUNDARIES:")
        self.logger.log(f"Equal-width: {pd.cut(data_col, bins=n_bins).categories.tolist()}")
        self.logger.log(f"Equal-frequency: {pd.qcut(data_col, q=n_bins, duplicates='drop').categories.tolist()}")



class Preprocessing:
    def __init__(self, logger=None):
        """
        Initialize Preprocessing class.
        
        Args:
            logger (Logger, optional): Logger instance for logging
        """
        self.logger = logger
        self.label_encoders = {}
        self.discretizers = {}  # Store fitted discretizers
        self.discretization_configs = {}  # Store discretization configurations

    def fit_label_encode_non_numeric(self, df):
        """
        Fit label encoders on all non-numeric columns in the given DataFrame.
        Args:
            df (pd.DataFrame): Training DataFrame.
        Returns:
            self: Returns self for method chaining
        """
        
        # Get non-numeric columns using numpy
        non_numeric_columns = list(df.select_dtypes(exclude=[np.number]).columns)
        
        for col in non_numeric_columns:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.label_encoders[col] = le
            
        return self

    def transform_label_encode_non_numeric(self, df):
        """
        Transform non-numeric columns using fitted label encoders.
        Args:
            df (pd.DataFrame): DataFrame to transform.
        Returns:
            pd.DataFrame: DataFrame with non-numeric columns label encoded
        """
        df_encoded = df.copy()
        
        for col, encoder in self.label_encoders.items():
            if col in df_encoded.columns:
                # Handle unseen categories by replacing with a default value
                try:
                    df_encoded[col] = encoder.transform(df_encoded[col].astype(str))
                except ValueError as e:
                    if "unseen labels" in str(e):
                        # Handle unseen categories
                        if self.logger:
                            self.logger.log(f"Warning: Unseen categories in column '{col}'. Using most frequent category.")
                        # Replace unseen categories with the most frequent one from training
                        most_frequent = encoder.classes_[0]  # First class is often most frequent
                        df_temp = df_encoded[col].astype(str).copy()
                        mask = ~df_temp.isin(encoder.classes_)
                        df_temp[mask] = most_frequent
                        df_encoded[col] = encoder.transform(df_temp)
                    else:
                        raise e
                        
        return df_encoded
    
    def fit_transform_label_encode_non_numeric(self, df):
        """
        Fit and transform non-numeric columns using label encoding.
        
        Args:
            df (pd.DataFrame): DataFrame to fit and transform.
        
        Returns:
            pd.DataFrame: DataFrame with non-numeric columns label encoded
        """
        self.fit_label_encode_non_numeric(df)
        return self.transform_label_encode_non_numeric(df)

    def fit_discretize_feature(self, data, feature, method='equal_width', n_bins=5, custom_bins=None):
        """
        Fit discretizer for a single feature
        Args:
            data (pd.DataFrame): Training dataset
            feature (str): Feature name to discretize
            method (str): 'equal_width', 'equal_frequency', 'kmeans', or 'custom'
            n_bins (int): Number of bins
            custom_bins (list): Custom bin edges for 'custom' method
        Returns:
            self: Returns self for method chaining
        """
        config = {
            'method': method,
            'n_bins': n_bins,
            'custom_bins': custom_bins
        }
        self.discretization_configs[feature] = config
        
        if method == 'equal_width':
            # Store bin edges for equal width
            _, bin_edges = pd.cut(data[feature], bins=n_bins, labels=False, retbins=True)
            self.discretizers[feature] = {'type': 'equal_width', 'bin_edges': bin_edges}
            
        elif method == 'equal_frequency':
            # Store quantile edges for equal frequency
            _, bin_edges = pd.qcut(data[feature], q=n_bins, labels=False, duplicates='drop', retbins=True)
            self.discretizers[feature] = {'type': 'equal_frequency', 'bin_edges': bin_edges}
            
        elif method == 'kmeans':
            # Fit KBinsDiscretizer
            discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans')
            discretizer.fit(data[feature].values.reshape(-1, 1))
            self.discretizers[feature] = {'type': 'kmeans', 'discretizer': discretizer}
            
        elif method == 'custom' and custom_bins:
            # Store custom bin edges
            self.discretizers[feature] = {'type': 'custom', 'bin_edges': custom_bins}
            
        else:
            raise ValueError("Invalid method or missing custom_bins")
            
        return self

    def transform_discretize_feature(self, data, feature):
        """
        Transform a single feature using fitted discretizer
        Args:
            data (pd.DataFrame): Dataset to transform
            feature (str): Feature name to discretize
        Returns:
            pd.Series: Discretized feature
        """
        if feature not in self.discretizers:
            raise ValueError(f"Feature '{feature}' has not been fitted. Call fit_discretize_feature first.")
            
        discretizer_info = self.discretizers[feature]
        
        if discretizer_info['type'] in ['equal_width', 'equal_frequency', 'custom']:
            return pd.cut(data[feature], bins=discretizer_info['bin_edges'], labels=False, include_lowest=True)
            
        elif discretizer_info['type'] == 'kmeans':
            return discretizer_info['discretizer'].transform(data[feature].values.reshape(-1, 1)).flatten()
    
    def fit_batch_discretize(self, data, features_config):
        """
        Fit discretizers for multiple features at once.
        Args:
            data (pd.DataFrame): Training dataset
            features_config (dict): Configuration for each feature
        Returns:
            self: Returns self for method chaining
        """
        for feature, config in features_config.items():
            if feature in data.columns:
                self.fit_discretize_feature(data, feature, **config)
                if self.logger:
                    self.logger.log(f"Fitted discretizer for feature '{feature}' using method '{config.get('method', 'equal_width')}'")
        return self
    
    def transform_batch_discretize(self, data):
        """
        Transform multiple features using fitted discretizers.
        Args:
            data (pd.DataFrame): Dataset to transform
        Returns:
            pd.DataFrame: Dataset with discretized features
        """
        result_data = data.copy()
        
        for feature in self.discretizers.keys():
            if feature in data.columns:
                result_data[feature] = self.transform_discretize_feature(data, feature)
                if self.logger:
                    config = self.discretization_configs.get(feature, {})
                    self.logger.log(f"Transformed feature '{feature}' using method '{config.get('method', 'unknown')}'")
                    
        return result_data
    
    def fit_transform_batch_discretize(self, data, features_config):
        """
        Fit and transform multiple features in one step.
        Args:
            data (pd.DataFrame): Training dataset
            features_config (dict): Configuration for each feature
        Returns:
            pd.DataFrame: Dataset with discretized features
        """
        self.fit_batch_discretize(data, features_config)
        return self.transform_batch_discretize(data)
    

    def calculate_treatment(self, df):
        """
        Calculate Treatment value based on the difference between current ACR and Rate_Lag.
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'ACR' and 'Rate_Lag' columns
            
        Returns:
            pd.DataFrame: DataFrame with updated Treatment values
        """
        df_copy = df.copy()
        
        # Calculate the difference between current ACR and Rate_Lag
        df_copy["ACR_diff"] = df_copy["ACR"] - df_copy["Rate_Lag"]
        
        # 1: Increase Treatment 
        # -1: Decrease Treatment 
        # 0: No change
        # Update Treatment based on the difference
        df_copy["Treatment"] = df_copy["ACR_diff"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        
        
        return df_copy.drop(columns=["ACR_diff", "Rate_Lag", "ACR"])

    def calculate_churn(self, df):
        """
        Create a binary 'Churn' outcome based on the 'ChurnRate' column.
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'ChurnRate' column
            
        Returns:
            pd.DataFrame: DataFrame with updated 'Churn' column
        """
        df_copy = df.copy()
        # 1: Churned, 0: Not Churned

        df_copy["Churn"] = df_copy["ChurnRate"].apply(lambda x: 1 if x > 0 else 0)
        return df_copy.drop(columns=["ChurnRate", "Members", "Members_Lag"])

    
    def calculate_other_provider_avg_acr(self, df):
        # Create a copy to avoid modifying the original
        df_copy = df.copy()

        # Create a multi-index group (Year, Quarter)
        group = df_copy.groupby(['Year', 'Quarter'])

        # Compute total ACR sum and count per (Year, Quarter)
        total_acr = group['ACR'].transform('sum')
        count = group['ACR'].transform('count')

        # Compute the sum and count for each (Year, Quarter, Provider)
        df_copy['provider_acr_sum'] = df_copy.groupby(['Year', 'Quarter', 'Provider'])['ACR'].transform('sum')
        df_copy['provider_count'] = df_copy.groupby(['Year', 'Quarter', 'Provider'])['ACR'].transform('count')

        # Calculate average ACR of *other* providers in the same quarter
        df_copy['Avg_ACR_Other_Providers'] = (
            (total_acr - df_copy['provider_acr_sum']) / (count - df_copy['provider_count'])
        )

        # Drop helper columns
        df_copy.drop(columns=['provider_acr_sum', 'provider_count'], inplace=True)

        return df_copy


    
    def get_label_encoders(self):
        """Get the dictionary of label encoders used in encoding."""
        return self.label_encoders
    
