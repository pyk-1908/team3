# Churn Prediction Modular Pipeline

## Overview

This project implements a modular, reproducible machine learning pipeline for predicting churn rates, following the workflow and feature engineering. The pipeline includes configuration, preprocessing, hyperparameter optimization, model training, evaluation, and result visualization.

---

## Directory Structure

## Directory Structure
```
project-root/
├── config.py
├── preprocessing.py
├── gridsearch.py
├── models.py
├── evaluation.py
├── main.py
├── requirements.txt
├── README.md
└── data/
    └── Cate_added_data.csv  # <–  data file 
└── OutputPlots/
    └──feature_importance_GRADIENT_BOOSTING.png
    └──feature_importance_RANDOM_FOREST.png

```
---

## Setup Instructions

### 1. Clone the Repository

git clone
cd Pipeline 


### 2. Install Dependencies

It is recommended to use a virtual environment:

pip install -r requirements.txt


---

## Data Preparation

- Place your data file in the `data/` directory.
- The expected filename is `Cate_added_data.csv` (you can change this, but see "Changing Data Path" below).
- The file should contain the columns described in the Baseline-Churnpred notebook, including:
  - Year, Provider, Quarter, Members, ACR, RiskFactor, Regionality, Members_Lag, Rate_Lag, ChurnRate, Treatment, QuarterInt, ACR_next, treatment, CATE_DR, CATE_XL, Quarter_Since_Start, etc.

**If your file is named differently or located elsewhere:**
- Edit the line in `main.py`:
    df = pd.read_csv(‘data/Cate_added_data.csv’)

and update the path/filename as needed.

---

## Running the Pipeline

From the project root, run:

- python main.py


This will:
- Load and preprocess the data
- Run grid search to find the best hyperparameters for each model
- Train three models (Random Forest, Ridge Regression, Gradient Boosting)
- Evaluate the models and print results in a table
- Save all plots (feature importance and prediction vs actual) as `.png` files in the project directory

**Note:**  
If you run this pipeline from a Jupyter notebook, copy the code from `main.py` into notebook cells for inline plot display.  
If you run as a script, plots will be saved as images (see the project directory after running).

---

## Output

- **Console:** Model performance metrics and a summary table highlighting the best model
- **Files:**  
  - `feature_importance_<MODEL>.png` (for tree-based models)
  - `pred_vs_actual_<MODEL>.png` (for all models)

---

## Changing the Data Path

- To use a different data file or location, change the path in `main.py`:

- Ensure the file contains all required columns as described above.

---

## Code Structure

| File             | Purpose                                                                 |
|------------------|-------------------------------------------------------------------------|
| `config.py`      | Hyperparameter grids and defaults                                       |
| `preprocessing.py` | Data cleaning, feature engineering, encoding                          |
| `gridsearch.py`  | Grid search for best hyperparameters                                    |
| `models.py`      | Model initialization and training                                       |
| `evaluation.py`  | Model evaluation, metrics, and plotting                                 |
| `main.py`        | Orchestrates the pipeline                                               |

---

## Customization

- **Feature Engineering:**  
Edit `preprocessing.py` to adjust which columns are dropped, added, or encoded.
- **Hyperparameters:**  
Edit `config.py` to change search grids or default values.
- **Plots:**  
Edit `evaluation.py` to change or add new plots.

---

## Troubleshooting

- **Plots not showing?**  
If running as a script, plots are saved as PNG files. Open them from your file explorer.
- **Data errors?**  
Ensure your CSV matches the expected columns and format.
- **Dependencies?**  
Run `pip install -r requirements.txt` to install all dependencies.


---

## Acknowledgments

- open-source Python communities.


