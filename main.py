
from src.DataLoader import DataLoader
from src.Model import CausalNexModel, BayesianNetworkModel
from src.Preprocessing import label_encode_non_numeric, discretize_columns



if __name__ == '__main__':
    Additional_Contribution_Rate_path = 'Dataset/Additional Contribution Rate per Year and Quarter.xlsx'
    Morbidity_Region_path = 'Dataset/Morbidity_Region.xlsx'
    visualization_folder = 'visualizations'


    #################### Load the data  ######################
    data_loader = DataLoader()
    data_loader.load_multiple_files({
        'additional_contribution_rate': Additional_Contribution_Rate_path,
        'morbidity_region': Morbidity_Region_path
    })
    print("Data loaded successfully.")
    merged_dataset = data_loader.merge_loaded_files([
        {
            'left': 'additional_contribution_rate',
            'right': 'morbidity_region',
            'left_on': ['Insurance Provider', 'Year'],
            'right_on': ['Krankenkasse ( statutory insurance provider)', 'Jahr (year)']
        }
    ])
    print("Data merged successfully.")

    ################# Preprocess the data #######################
    #  we want to make our data numeric, since this is what the NOTEARS expects. We can do this by label encoding non-numeric variables.
    merged_dataset, label_encoders = label_encode_non_numeric(merged_dataset.data)
    print("Data label encoded successfully.")
 

    ############## Model #######################
    # crete a structure model
    structure_model = CausalNexModel()
    structure_model.create_structure(merged_dataset, max_iter=10000, w_threshold=0.1)
    print("Structure model created successfully.")
    # Plot the structure model
    structure_model.plot_structure(plot_title="Causal_Structure_Model", path=visualization_folder)
    print("Structure model plotted successfully.")
    # Save the structure model
    structure_model.save_structure('visualizations/causal_structure_model.dot')
    print("Structure model saved successfully.")

    