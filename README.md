# Data Science 2025



## Overview
- This is a project to study the Causal Effect on Additional Contribution Rate on Member Churn in Statutory Health
Insurance Funds

---

## Training Environment
The model was trained locally on a MacBook using only CPU resources, without the need for GPU acceleration. This makes the training process lightweight and easily reproducible on standard hardware.
- **Device:** MacBook Pro with Apple M4 Pro chip
- **CPU:** 12-core Apple Silicon (M4 Pro)
- **RAM:** 24 GB
- **OS:** macOS Sequoia 15.5


## Installation and Execution
1. Create a virtual environment with python version between 3.8 and 3.10
   ```bash
   python3.10 -m venv venv_name
   ```
2. Install the requirements 
    ```bash 
   source venv_name/bin/activate && pip install -r requirements.txt
    ```
3. Run the notebook
   ```bash
   venv_name//bin/jupyter nbconvert --to notebook --execute --inplace CauseHealPred.ipynb
   ```
**Note** For macOS Graphviz may be installed in a location that is not on the default search path. In this case, it may be necessary to manually specify the path to the graphviz include and/or library directories:
```bash
pip install --config-settings="--global-option=build_ext" \
            --config-settings="--global-option=-I$(brew --prefix graphviz)/include/" \
            --config-settings="--global-option=-L$(brew --prefix graphviz)/lib/" \
            pygraphviz
```

For CausalNex Model
   ```bash
   python3 main.py
   ```
For CausalML Model
   ```bash
         # Install Jupyter if needed
      pip install jupyter

         # Open and run interactively
      jupyter notebook CausalMLAdjusted.ipynb

         # Or execute all cells headlessly and save output
      jupyter nbconvert --to notebook --execute CausalMLAdjusted.ipynb --output CausalMLAdjusted_executed.ipynb

---
```
## Structure for Causal Nex analysis
- **CausalNex/DataLoader.py:**  Loads data from files.
- **CausalNex/Preprocessing.py:** Handles the preprocessing.
- **CausalNex/Model.py:** Contain the model.
- **main.py:** Main loop.

---


## Usage for end to end flow

1. **Navigate to the ```pipeline``` folder and follow the instructions in there**

---

## Data
### Dataset Description 
Additional Contribution Rate per Year and Quarter.xlsx contains data on different health insurance providers from 2016 to 2025 per quarter, including their additional contribution rate, showing the absolute values of insured members and insured people. 
Market Share per insurance provider.xlsx contains data on different health insurance providers from 2016 to 2025, including their respective market share and the absolute values of insured members and insured people.
Morbidity_Region.xlsx: contains data showing whether a health insurance provider is a regional provider, and the respective risk factor and its development from 2016 until 2025.

German statutory health insurance data from a major international consulting firm. The data ranges from 2013 to 2025 and was initially delivered in multiple Excel files, each containing a different structure and set of variables. 

***Additional Contribution Rate (ACR) data***
This dataset contains quarterly values of the additional contribution rates for all German statutory insurers. Insurers closed or inactive during the observation period were removed to ensure time consistency. This dataset includes 94 active providers across the whole period from 2013 to 2025.

***Morbidity Dataset***
The dataset includes annual morbidity scores and regional classifications at the insurer level. It only includes active insurers and excludes missing values and zero entries, resulting in a final sample of 91 providers. Notably, this data set only begins in 2016 and contains no data from 2013 to 2015.

***Preprocessing and merged data***
- Can be found under pipeline folder ```pipeline/data/Cate_added_data.csv```


---

## Project status
Ongoing 

---

## Citation

1. https://causalnex.readthedocs.io/en/latest/05_resources/05_faq.html#what-is-causalnex

---

## Acknowledgments

- Microsoft Research and PyWhy organization
- Healthcare data providers
- Open source community contributions


