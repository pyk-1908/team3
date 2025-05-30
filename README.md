# Data Science 2025



## Overview
- This is a project to study the Causal Effect on Additional Contribution Rate on Member Churn in Statutory Health
Insurance Funds

---

## Installation and Execution
1. Create a virtual environment with python version between 3.8 and 3.10
   ```bash
   python3.10 -m venv venv_name
   ```
2. Install the requirements 
    ```bash 
    python3 -m venv venv_name && source venv_name/bin/activate && pip install -r requirements.txt
    ```
3. Run the notebook
   ```bash
   jupyter nbconvert --to notebook --execute --inplace CausalHealPred.ipynb
   ```

For CausalNex Model
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
## Features for end to end flow

- **End-to-End Causal Analysis**: Complete pipeline from data loading to results interpretation in CausalHealPred.ipynb

---

## Data
**Dataset Description**

---

## Project status
Ongoing 



## Citation

1. https://py-why.github.io/dowhy/
2. https://causalnex.readthedocs.io/en/latest/05_resources/05_faq.html#what-is-causalnex


## Acknowledgments

- DoWhy development team for the causal inference framework
- Microsoft Research and PyWhy organization
- Healthcare data providers
- Open source community contributions


