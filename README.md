# Data Science 2025



## Overview
- This is a project to study the Causal Effect on Additional Contribution Rate on Member Churn in Statutory Health
Insurance Funds

---

## Installation and Execution for Causal Nex analysis
1. Create a virtual environment with python version between 3.8 and 3.10
   ```bash
   python3.10 -m venv venv_name
   ```
2. Install the requirements 
    ```bash 
    python3 -m venv venv_name && source venv_name/bin/activate && pip install -r requirements.txt
    ```
3. Run the script
   ```bash
   python3 main.py
   ```

---

## Structure for Causal Nex analysis
- **src/DataLoader.py:**  Loads data from files.
- **src/Preprocessing.py:** Handles the preprocessing.
- **src/Model.py:** Contain the model.
- **main.py:** Main loop.

---

## Usage for end to end flow

1. **Navigate to the ```pipeline``` folder and follow the instructions in there**

---
## Features for end to end flow

- **End-to-End Causal Analysis**: Complete pipeline from data loading to results interpretation

---

## Data
**Dataset Description**

German statutory health insurance data from a major international consulting firm. The data ranges from 2013 to 2025 and was initially delivered in multiple Excel files, each containing a different structure and set of variables. 

***Additional Contribution Rate (ACR) data***
This dataset contains quarterly values of the additional contribution rates for all German statutory insurers. Insurers closed or inactive during the observation period were removed to ensure time consistency. This dataset includes 94 active providers across the whole period from 2013 to 2025.

***Morbidity Dataset***
The dataset includes annual morbidity scores and regional classifications at the insurer level. It only includes active insurers and excludes missing values and zero entries, resulting in a final sample of 91 providers. Notably, this data set only begins in 2016 and contains no data from 2013 to 2015.

***Preprocessing and merged data***
- Can be found under pipeline folder ```pipeline/data/Cate_integrated_data.csv```


---

## Project status
Ongoing 



## Citation

1. https://causalnex.readthedocs.io/en/latest/05_resources/05_faq.html#what-is-causalnex


## Acknowledgments

- Microsoft Research and PyWhy organization
- Healthcare data providers
- Open source community contributions


