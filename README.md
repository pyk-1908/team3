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
    python3 -m venv venv_name && source venv_name/bin/activate && pip install -r requirements.txt
    ```
3. Run the notebook
   ```bash
   jupyter nbconvert --to notebook --execute --inplace CauseHealPred.ipynb
   ```

For CausalNex Model
   ```bash
   python3 main.py
   ```

---

## Structure for Causal Nex analysis
- **CausalNex/DataLoader.py:**  Loads data from files.
- **CausalNex/Preprocessing.py:** Handles the preprocessing.
- **CausalNex/Model.py:** Contain the model.
- **main.py:** Main loop.

---

## Features for end to end flow

- **End-to-End Causal Analysis**: Complete pipeline from data loading to results interpretation in CauseHealPred.ipynb

---

## Data
### Dataset Description 
Additional Contribution Rate per Year and Quarter.xlsx contains data on different health insurance providers from 2016 to 2025 per quarter, including their additional contribution rate, showing the absolute values of insured members and insured people. 
Market Share per insurance provider.xlsx contains data on different health insurance providers from 2016 to 2025, including their respective market share and the absolute values of insured members and insured people.
Morbidity_Region.xlsx: contains data showing whether a health insurance provider is a regional provider, and the respective risk factor and its development from 2016 until 2025.

---

## Project status
Ongoing 

---

## Citation

1. https://py-why.github.io/dowhy/
2. https://causalnex.readthedocs.io/en/latest/05_resources/05_faq.html#what-is-causalnex

---

## Acknowledgments

- DoWhy development team for the causal inference framework
- Microsoft Research and PyWhy organization
- Healthcare data providers
- Open source community contributions


