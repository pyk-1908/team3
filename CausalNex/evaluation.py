import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import ttest_ind

def placebo_test(bayesian_network_model, bn, data_with_CATE, treatment, outcome, logger, save=False):
    
    df_placebo = data_with_CATE.copy().drop('CATE(1 vs 0)', axis=1)
    df_placebo = df_placebo.drop('CATE(1 vs -1)', axis=1)
    df_placebo = df_placebo.drop('CATE(0 vs -1)', axis=1)

    # break any true causal relationship between treatment and outcome.
    df_placebo[treatment] = np.random.permutation(df_placebo[treatment].values) 

    # fit the model to the placebo data
    bn = bn.fit_cpds(df_placebo)

    for _, row in df_placebo.iterrows():
        x = row.drop([treatment, outcome]).to_dict()
        cate_results = bayesian_network_model.estimate_cate(bn=bn, treatment=treatment, outcome=outcome, x=x)
        # Add PLACEBO CATE results to the DataFrame
        df_placebo.at[row.name, 'Placebo CATE(1 vs 0)'] = cate_results['CATE(1 vs 0)']
        df_placebo.at[row.name, 'Placebo CATE(1 vs -1)'] = cate_results['CATE(1 vs -1)']
        df_placebo.at[row.name, 'Placebo CATE(0 vs -1)'] = cate_results['CATE(0 vs -1)']
    
    plt.hist(data_with_CATE['CATE(1 vs 0)'], bins=30, alpha=0.5, label='Real CATE')
    plt.hist(df_placebo['Placebo CATE(1 vs 0)'], bins=30, alpha=0.5, label='Placebo CATE (1 vs 0)')
    plt.legend()
    plt.title("Real vs Placebo CATE (1 vs 0) Distribution")
    if save:
        logger.save_figure(plt, "placebo_cate_distribution")

    plt.hist(data_with_CATE['CATE(1 vs -1)'], bins=30, alpha=0.5, label='Real CATE')
    plt.hist(df_placebo['Placebo CATE(1 vs -1)'], bins=30, alpha=0.5, label='Placebo CATE (1 vs -1)')
    plt.legend()
    plt.title("Real vs Placebo CATE (1 vs -1) Distribution")
    if save:
        logger.save_figure(plt, "placebo_cate_distribution")

    plt.hist(data_with_CATE['CATE(0 vs -1)'], bins=30, alpha=0.5, label='Real CATE')
    plt.hist(df_placebo['Placebo CATE(0 vs -1)'], bins=30, alpha=0.5, label='Placebo CATE (0 vs -1)')
    plt.legend()
    plt.title("Real vs Placebo CATE (0 vs -1) Distribution")
    if save:
        logger.save_figure(plt, "placebo_cate_distribution")
    
    t_stat_1_0, p_val_1_0 = ttest_ind(data_with_CATE['CATE(1 vs 0)'], df_placebo['Placebo CATE(1 vs 0)'])
    logger.log(f"Placebo Test (1 vs 0): T-statistic = {t_stat_1_0}, P-value = {p_val_1_0}")

    t_stat_1_neg1, p_val_1_neg1 = ttest_ind(data_with_CATE['CATE(1 vs -1)'], df_placebo['Placebo CATE(1 vs -1)'])
    logger.log(f"Placebo Test (1 vs -1): T-statistic = {t_stat_1_neg1}, P-value = {p_val_1_neg1}")

    t_stat_0_neg1, p_val_0_neg1 = ttest_ind(data_with_CATE['CATE(0 vs -1)'], df_placebo['Placebo CATE(0 vs -1)'])
    logger.log(f"Placebo Test (0 vs -1): T-statistic = {t_stat_0_neg1}, P-value = {p_val_0_neg1}")

    # add statistics to a dictionary
    stats = {
        't_stat_1_0': t_stat_1_0,
        'p_val_1_0': p_val_1_0,
        't_stat_1_neg1': t_stat_1_neg1,
        'p_val_1_neg1': p_val_1_neg1,
        't_stat_0_neg1': t_stat_0_neg1,
        'p_val_0_neg1': p_val_0_neg1
    }
    return stats