
"""Simple biostatistics package using pandas, numpy, statsmodels and matplotlib."""
import math

from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

import scipy.stats as stats

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats import weightstats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.power import TTestIndPower


def display(content):
    """To be able to use the IPython display method while inside the Jupyter notebooks."""
    try:
        import IPython
        IPython.display.display(content)
    except NameError:
        print(content)


def markdown(content):
    """To be able to use the IPython display method with markdown while inside the Jupyter notebooks."""
    try:
        import IPython
        IPython.display.display(IPython.display.Markdown(content))
    except ImportError:
        print(content)


def summarize(data_df):
    """To get some info describing the data."""
    data_copy_df = data_df.copy()

    summary_df = data_copy_df.describe(percentiles=[0.025, 0.25, 0.75, 0.975]).T

    pd.set_option('display.expand_frame_repr', False)

    for variable, row in summary_df.iterrows():
        describe = stats.describe(data_copy_df[variable])
        summary_df.loc[variable, 'mean'] = round(summary_df.loc[variable, 'mean'], 3)
        summary_df.loc[variable, 'std'] = round(summary_df.loc[variable, 'std'], 3)
        summary_df.loc[variable, 'IQR'] = round(summary_df.loc[variable, '75%'] - summary_df.loc[variable, '25%'], 3)
        summary_df.loc[variable, 'se'] = round(summary_df.loc[variable, 'std'] / math.sqrt(int(summary_df.loc[variable, 'count'])), 3)
        summary_df.loc[variable, 'skew'] = round(describe.skewness, 3)
        summary_df.loc[variable, 'kurt'] = round(describe.kurtosis, 3)

    return summary_df


def check_normality(data_df, variable):
    """
    To plot graphs to check for the normality of the variable.

    It includes a boxplot, an histogram with a fitted normal distribution and a Q-Q plot
    """
    # We copy the data to avoid to change the original data unintentionally.
    data_copy_df = data_df.copy().dropna()
    fig, (ax_box, ax_hist, ax_qq) = plt.subplots(1, 3, figsize=[16, 6])

    describe_df = data_copy_df.describe()

    mean = describe_df.loc['mean', variable]
    std = describe_df.loc['std', variable]
    n = int(describe_df.loc['count', variable])

    # For the box plot
    ax_box.title.set_text('Box plot')
    boxplotted = ax_box.boxplot(data_copy_df[variable], patch_artist=True)

    # see https://stackoverflow.com/questions/41997493/python-matplotlib-boxplot-color
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(boxplotted[element], color='C7', linestyle='dashed')

    # For the histogram plot
    ax_hist.title.set_text('Histogram with fitted normal distribution')
    ax_hist.hist(data_copy_df[variable], bins=10, density=True, color='C0', label='Density')

    x = np.linspace(data_copy_df[variable].min(), data_copy_df[variable].max(), 1000)
    fitted_norm = stats.t.pdf(x, df=len(data_copy_df.index) - 1, loc=mean, scale=std)
    ax_hist.plot(x, fitted_norm, color='white')
    ax_hist.plot(x, fitted_norm, color='C7', linestyle='dashed', label='Fitted gaussian')

    ax_hist.legend()

    # For the QQ plot
    ax_qq.title.set_text('Q-Q plot of observations vs normal distribution')
    sorted_df = data_copy_df.sort_values(by=variable).reset_index(drop=True)
    for i in range(n):
        sorted_df.loc[i, 'norm_quantiles'] = stats.t.ppf((i + 1) / n, df=len(data_copy_df.index) - 1, loc=mean, scale=std)

    ax_qq.plot(sorted_df[variable], sorted_df[variable], color='C7', linewidth=1, linestyle='dashed')
    ax_qq.plot(sorted_df['norm_quantiles'], sorted_df[variable], marker='o', color='C0', markersize=4, linewidth=0)
    ax_qq.grid(True)

    plt.show()


def get_ci(data_df, variable, alpha=0.05):
    """To get the confidence interval of the mean for the variable assuming normality."""
    data_copy_df = data_df.copy()

    summary_df = summarize(data_copy_df)

    ci_df = pd.DataFrame(columns=['Mean', 'CI low', 'CI high'])

    mean = summary_df.loc[variable, 'mean']
    se = summary_df.loc[variable, 'se']

    ci_low = stats.t.ppf(alpha / 2, df=len(data_copy_df.index) - 1, loc=mean, scale=se)
    ci_high = stats.t.ppf(1 - alpha / 2, df=len(data_copy_df.index) - 1, loc=mean, scale=se)

    ci_df.loc[variable, 'Mean'] = mean
    ci_df.loc[variable, 'CI low'] = ci_low
    ci_df.loc[variable, 'CI high'] = ci_high

    return ci_df


def get_pi(data_df, variable, alpha=0.05):
    """To get the confidence interval of the prediction for the variable assuming normality."""
    data_copy_df = data_df.copy()

    summary_df = summarize(data_copy_df)

    pi_df = pd.DataFrame(columns=['Mean', 'PI low', 'PI high', 'Below PI low', 'Above PI high'])

    mean = summary_df.loc[variable, 'mean']
    std = summary_df.loc[variable, 'std']

    pi_low = stats.norm.ppf(alpha / 2, loc=mean, scale=std)
    pi_high = stats.norm.ppf(1 - alpha / 2, loc=mean, scale=std)

    count_obs = summary_df.loc[variable, 'count']
    count_obs_above_pred = len(data_copy_df.loc[data_copy_df[variable] >= pi_high].index)
    count_obs_below_pred = len(data_copy_df.loc[data_copy_df[variable] <= pi_low].index)

    pi_df.loc[variable, 'Mean'] = mean
    pi_df.loc[variable, 'PI low'] = pi_low
    pi_df.loc[variable, 'PI high'] = pi_high
    pi_df.loc[variable, 'Below PI low'] = count_obs_below_pred / count_obs
    pi_df.loc[variable, 'Above PI high'] = count_obs_above_pred / count_obs

    return pi_df


def ttest_1samp(data_df, variable, hypothesis=0):
    """Exact T-test testing if the mean of the variable is equal to the hypothesis."""
    data_copy_df = data_df.dropna()
    ttest_results = stats.ttest_1samp(data_copy_df[variable], hypothesis)

    ttest_results_df = pd.DataFrame(columns=['hypothesis', 'statistic', 'p_value'])
    ttest_results_df.loc[variable, 'hypothesis'] = hypothesis
    ttest_results_df.loc[variable, 'statistic'] = ttest_results.statistic
    ttest_results_df.loc[variable, 'p_value'] = ttest_results.pvalue

    display(get_ci(data_copy_df, variable).join(ttest_results_df))


def ttest_ind_by_group(data_df, group_var, variable):
    """T-test for 2 independant variables."""
    groups = data_df[group_var].value_counts()
    data1_df = data_df.loc[data_df[group_var] == groups.index[0]]
    data2_df = data_df.loc[data_df[group_var] == groups.index[1]]

    compare_means = weightstats.CompareMeans.from_data(data1_df[variable], data2_df[variable])

    ttest_ind_results_df = pd.DataFrame(columns=['Obs', 'Mean', 'CI low', 'CI high'])

    ttest_ind_results_df = pd.concat([ttest_ind_results_df, get_ci(data1_df, variable)], sort=False)
    ttest_ind_results_df.loc[variable, 'Obs'] = len(data1_df)
    ttest_ind_results_df.rename(index={variable: groups.index[0]}, inplace=True)

    ttest_ind_results_df = pd.concat([ttest_ind_results_df, get_ci(data2_df, variable)], sort=False)
    ttest_ind_results_df.loc[variable, 'Obs'] = len(data2_df)
    ttest_ind_results_df.rename(index={variable: groups.index[1]}, inplace=True)

    ttest_ind_results_df.loc['diff', 'Mean'] = \
        ttest_ind_results_df.loc[groups.index[0], 'Mean'] - ttest_ind_results_df.loc[groups.index[1], 'Mean']

    diff_ci_low, diff_ci_high = compare_means.tconfint_diff()
    ttest_ind_results_df.loc['diff', 'Obs'] = ''
    ttest_ind_results_df.loc['diff', 'CI low'] = diff_ci_low
    ttest_ind_results_df.loc['diff', 'CI high'] = diff_ci_high

    display(ttest_ind_results_df)

    ttest_results_df = pd.DataFrame(columns=['statistic', 'p_value', 'df'])
    ttest_results_df.loc[variable, :] = compare_means.ttest_ind()

    display(ttest_results_df)


def ftest_std_by_group(data_df, group_var, variable):
    """F-test to check the assumption of same standard deviation from 2 independant variables."""
    groups = data_df[group_var].value_counts()

    summary1_df = summarize(data_df.loc[data_df[group_var] == groups.index[0]])
    summary2_df = summarize(data_df.loc[data_df[group_var] == groups.index[1]])

    std1 = summary1_df.loc[variable, 'std']
    std2 = summary2_df.loc[variable, 'std']

    if std1 >= std2:
        std_large = std1
        std_small = std2
        n_large = summary1_df.loc[variable, 'count']
        n_small = summary2_df.loc[variable, 'count']
    else:
        std_large = std2
        std_small = std1
        n_large = summary2_df.loc[variable, 'count']
        n_small = summary1_df.loc[variable, 'count']

    f_score = math.pow(std_large / std_small, 2)

    p_value = 2 * (1 - stats.f.cdf(f_score, dfn=n_large - 1, dfd=n_small - 1))

    ztest_results_df = pd.DataFrame(columns=['statistic', 'p_value', 'df_num', 'df_den'])

    ztest_results_df.loc[variable, :] = (f_score, p_value, n_large - 1, n_small - 1)

    return ztest_results_df


def bootstrap_by_group(data_df, group_var, variable, n_iter):
    """Bootstrap analysis to estimate the distribution of the mean."""
    bootstrap_params_df = pd.DataFrame(columns=['diff']).astype('float64')

    groups = data_df[group_var].value_counts()

    for i in range(0, n_iter):
        bootstraped_data_df = data_df.sample(frac=1, replace=True)

        data1_df = bootstraped_data_df.loc[bootstraped_data_df[group_var] == groups.index[0]]
        data2_df = bootstraped_data_df.loc[bootstraped_data_df[group_var] == groups.index[1]]

        describe1_df = data1_df.describe()
        describe2_df = data2_df.describe()
        bootstrap_params_df.loc[i, 'diff'] = \
            describe1_df.loc['mean', variable] - describe2_df.loc['mean', variable]

    check_normality(bootstrap_params_df, 'diff')

    return get_pi(bootstrap_params_df, 'diff')


def ranksums_by_group(data_df, group_var, variable):
    """T-test for 2 independant variables."""
    groups = data_df[group_var].value_counts()
    data1_df = data_df.loc[data_df[group_var] == groups.index[0]]
    data2_df = data_df.loc[data_df[group_var] == groups.index[1]]

    ci_df = pd.DataFrame(columns=['Obs', 'Mean', 'CI low', 'CI high'])

    ci_df = pd.concat([ci_df, get_ci(data1_df, variable)], sort=False)
    ci_df.loc[variable, 'Obs'] = len(data1_df)
    ci_df.rename(index={variable: groups.index[0]}, inplace=True)

    ci_df = pd.concat([ci_df, get_ci(data2_df, variable)], sort=False)
    ci_df.loc[variable, 'Obs'] = len(data2_df)
    ci_df.rename(index={variable: groups.index[1]}, inplace=True)

    ci_df.loc['diff', 'Mean'] = \
        ci_df.loc[groups.index[0], 'Mean'] - ci_df.loc[groups.index[1], 'Mean']

    ci_df.loc['diff', 'Obs'] = ''
    ci_df.loc['diff', 'CI low'] = ''
    ci_df.loc['diff', 'CI high'] = ''

    display(ci_df)

    ranksums_results = stats.ranksums(data1_df[variable], data2_df[variable])

    ttest_results_df = pd.DataFrame(columns=['statistic', 'p_value'])
    ttest_results_df.loc[variable, 'statistic'] = ranksums_results.statistic
    ttest_results_df.loc[variable, 'p_value'] = ranksums_results.pvalue

    display(ttest_results_df)


def power_calculation_ttest(mean_diff, std, alpha=0.05, power=None, n_participants=None):
    """To calculate the number of participants needed to reach a specific power or the power for a specific number of participants."""
    power_test = TTestIndPower()

    if power is not None:
        n_participants = math.ceil(power_test.solve_power(effect_size=mean_diff / std, alpha=alpha, power=power))
    else:
        power = power_test.solve_power(effect_size=mean_diff / std, alpha=alpha, nobs1=n_participants)

    power_results = {'Power calculation': [alpha, (1 - power), mean_diff, std, n_participants]}

    return pd.DataFrame.from_dict(
        power_results, columns=['Type I errors', 'Type II errors', 'Difference', 'Standard deviation', 'Participants needed'],
        orient='index')


def plot_scatter_blandalt(input_df, var1, var2):
    """Plot figures to validate the "same distribution" assumption."""
    data_df = input_df.sort_values(by=var1).copy()

    fig, (ax_scatter, ax_blandalt) = plt.subplots(1, 2, figsize=[16, 6])

    #################
    # Scatter plot #
    #################
    min_var1 = data_df.iloc[0][var1]
    max_var1 = data_df.iloc[len(data_df) - 1][var1]

    data_df['diff'] = data_df[var2] - data_df[var1]
    mean_diff = data_df[['diff']].mean().values[0]

    # Plotting data
    ax_scatter.plot(
        data_df[var1], data_df[var2], marker='o', color='C0', markersize=4, linewidth=0)

    # Plotting helper lines
    ax_scatter.plot(
        [min_var1, max_var1], [min_var1, max_var1],
        linewidth=1, color='C7', label='Identity', linestyle='dashed', alpha=0.3)
    ax_scatter.plot(
        [min_var1, max_var1], [min_var1 + mean_diff, max_var1 + mean_diff],
        linewidth=1, color='C7', linestyle='dashed', label='Identity shifted by mean of difference')

    ax_scatter.title.set_text('Scatter plot of the paired distributions')
    ax_scatter.set_xlabel(var1)
    ax_scatter.set_ylabel(var2)
    ax_scatter.legend()

    #####################
    # Bland-Altman plot #
    #####################
    data_df['avg'] = (data_df[var1] + data_df[var2]) / 2
    min_avg = data_df[['avg']].min()
    max_avg = data_df[['avg']].max()

    ci = get_ci(data_df, 'diff').loc['diff']
    pi = get_pi(data_df, 'diff').loc['diff']

    # Plotting data
    ax_blandalt.plot(
        data_df['avg'], data_df['diff'],
        marker='o', color='C0', markersize=4, linewidth=0)

    # Plotting helper lines
    ax_blandalt.plot(
        [min_avg, max_avg], [pi['PI high'], pi['PI high']],
        color='C7', linewidth=1, linestyle='dashed', alpha=1, label='PI high')
    ax_blandalt.plot(
        [min_avg, max_avg], [ci['CI high'], ci['CI high']],
        color='C7', linewidth=1, linestyle='dashed', alpha=0.5, label='CI high')
    ax_blandalt.plot(
        [min_avg, max_avg], [mean_diff, mean_diff],
        color='C7', linewidth=1, linestyle='dashed', alpha=1, label='Mean')
    ax_blandalt.plot(
        [min_avg, max_avg], [ci['CI low'], ci['CI low']],
        color='C7', linewidth=1, linestyle='dashed', alpha=0.5, label='CI low')
    ax_blandalt.plot(
        [min_avg, max_avg], [pi['PI low'], pi['PI low']],
        color='C7', linewidth=1, linestyle='dashed', alpha=1, label='PI low')

    ax_blandalt.set_xlabel('Average of {} and {}'.format(var1, var2))
    ax_blandalt.set_ylabel('Difference between {} and {}'.format(var2, var1))
    ax_blandalt.title.set_text('Bland-Altman plot')
    ax_blandalt.legend()

    plt.show()


def ttest_paired(data_df, var1, var2):
    """T-test for paired data looking at the diff."""
    data_copy_df = data_df
    data_copy_df['diff'] = data_copy_df[var2] - data_copy_df[var1]

    test_results = stats.ttest_rel(data_df[var1], data_df[var2])

    ci_df = get_ci(data_df, 'diff')
    ci_df.loc['diff', 'statistic'] = test_results.statistic
    ci_df.loc['diff', 'p_value'] = test_results.pvalue

    display(ci_df)


def anova_by_group(data_df, resp_var, group_var):
    """One way anova."""
    model = ols(resp_var + ' ~ ' + group_var, data=data_df).fit()

    anova_df = sm.stats.anova_lm(model, typ=2)
    anova_df['mean_sq'] = anova_df['sum_sq'] / anova_df['df']

    args = []
    describe_df = pd.DataFrame()

    for group in data_df[group_var].unique():
        grouped_data_df = data_df.loc[data_df[group_var] == group]
        group_describe_df = grouped_data_df.describe().T.rename({resp_var: group})
        describe_df = pd.concat([describe_df, group_describe_df.loc[group_describe_df.index == group]])

        args.append(grouped_data_df[resp_var])

    markdown('#### Groups description')
    display(describe_df)

    markdown('#### ANOVA')
    display(anova_df[['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)']].replace({np.NaN: ''}))

    markdown('#### Bartlett\'s test of same variance')
    display(stats.bartlett(*args))


def kruskal_by_group(data_df, resp_var, group_var):
    """Kruskal-Wallis-test of 'same distribution'."""
    data_copy_df = data_df

    args = []

    for group in data_copy_df[group_var].unique():
        grouped_data_df = data_copy_df.loc[data_copy_df[group_var] == group]
        args.append(grouped_data_df[resp_var])

    display(stats.kruskal(*args))


class LinearRegression():
    """To perform a linear regression."""

    model = None

    def __init__(self, data_df, resp_var, cont_var=None, cat_var=None, with_interactions=False):
        """To initialize the LinearRegression object."""
        self.data_df = data_df
        self.resp_var = resp_var
        self.cont_var = cont_var
        self.cat_var = cat_var
        self.with_interactions = with_interactions

    def fit(self):
        """Fit the linear regression and return an ANOVA analysis and the fitted parameters."""
        formula = self.resp_var + ' ~ '

        if self.cont_var is not None:
            formula += self.cont_var

            if self.cat_var is not None:
                if self.with_interactions:
                    formula += ' * ' + self.cat_var
                else:
                    formula += ' + ' + self.cat_var
        else:
            if self.cat_var is not None:
                formula += self.cat_var
            else:
                formula += '1'

        self.model = ols(formula, data=self.data_df).fit()

        anova_df = sm.stats.anova_lm(self.model, typ=2)
        anova_df['mean_sq'] = anova_df['sum_sq'] / anova_df['df']

        markdown("#### ANOVA")
        display(anova_df.replace({np.NaN: ''}))

        markdown("#### Regression parameters")
        display(self.model.summary2().tables[1])

    def check_model(self):
        """Plot figures to validate the model."""
        if self.model is None:
            print('No model is defined, please run fit() on your regression first.')
            return

        markdown('#### Checking linearity and identically distributed errors')
        if self.cont_var is not None:
            expl_data = self.data_df[self.cont_var]

            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=[16, 8])

            ax1.plot([min(expl_data), max(expl_data)], [0, 0], color='C7', linestyle='dashed')
            ax1.title.set_text('Residuals vs. explanatory variable "' + self.cont_var + '"')
            ax1.set_ylabel('residuals')
            ax1.set_xlabel(self.cont_var)
            ax1.scatter(expl_data, self.model.resid)

        else:
            fig, ax0 = plt.subplots(figsize=[8, 8])

        predictions = self.model.predict()

        ax0.plot([min(predictions), max(predictions)], [0, 0], color='C7', linestyle='dashed')
        ax0.title.set_text('Residuals vs predictions')
        ax0.scatter(self.model.predict(), self.model.resid)
        ax0.set_ylabel('residuals')
        ax0.set_xlabel('predictions')

        plt.show()

        markdown('#### Checking normality of errors')
        resid_df = pd.DataFrame(self.model.resid, columns={'residuals'})
        check_normality(resid_df, 'residuals')

    def plot_predictions(self, alpha=0.05):
        """Plot the data vs the prediction using the continuous variable for the x axis."""
        fig, axis = plt.subplots(figsize=[16, 8])

        predictions_df = self.model.get_prediction().summary_frame(alpha)

        data_df = self.data_df.join(predictions_df)

        categories = ['all']

        if self.cat_var is not None:
            categories = data_df[self.cat_var].unique().tolist()

        for category in categories:
            if category == 'all':
                cat_data_df = data_df.copy()
                data_label = 'Data points'
            else:
                cat_data_df = data_df.loc[data_df[self.cat_var] == category].copy()
                data_label = 'Data points for ' + self.cat_var + ' ' + category

            if self.cont_var is not None:
                x = cat_data_df[self.cont_var]
                vs_var = self.cont_var
            else:
                x = [category] * len(cat_data_df)
                if category == 'all':
                    vs_var = 'nothing'
                else:
                    vs_var = self.cat_var

            axis.plot(
                x, cat_data_df[self.resp_var],
                marker='o', markersize=4, linewidth=0, label=data_label)

            axis.plot(
                x, cat_data_df['obs_ci_upper'],
                color='C7', linestyle='dashed', alpha=0.5)
            axis.plot(
                x, cat_data_df['mean'],
                color='C7', linestyle='dashed')
            axis.plot(
                x, cat_data_df['obs_ci_lower'],
                color='C7', linestyle='dashed', alpha=0.5)
            axis.title.set_text('Predictions vs ' + vs_var)

        axis.set_ylabel('predictions')
        axis.set_xlabel(vs_var)
        axis.legend()

        plt.show()

    def predict(self, data_df, alpha=0.05):
        """Compute the prediction for the data."""
        predictions_df = self.model.get_prediction(data_df).summary_frame(alpha=0.05)

        predictions_df.rename(columns={
            'mean': 'prediction', 'mean_se': 'se', 'mean_ci_lower': 'CI low', 'mean_ci_upper': 'CI high',
            'obs_ci_lower': 'PI low', 'obs_ci_upper': 'PI high'}, inplace=True)
        return data_df.join(predictions_df[['prediction', 'PI low', 'PI high']])


def mcnemar_test(data_df, var1, var2):
    """Test of difference between two paired binary variables."""
    data_copy_df = data_df.copy()

    # First we want to compute the contingency table
    values_list = data_copy_df[var1].value_counts().index.tolist()

    indexes = pd.MultiIndex.from_product([[var1], values_list])
    columns = pd.MultiIndex.from_product([[var2], values_list])

    contingency_table_df = pd.DataFrame(columns=columns, index=indexes)

    for value1 in values_list:
        for value2 in values_list:
            contingency_table_df.loc[(var1, value1), (var2, value2)] = len(
                data_copy_df.loc[(data_copy_df[var1] == value1) & (data_copy_df[var2] == value2)])

    display(contingency_table_df)

    # Then we use the McNemar test on it with the assumption that it's the same distribution
    mcnemar_results = mcnemar(contingency_table_df.values)
    mcnemar_results_df = pd.DataFrame(columns=['statistic', 'p_value'])

    mcnemar_results_df.loc[var1 + ' vs ' + var2, 'statistic'] = mcnemar_results.statistic
    mcnemar_results_df.loc[var1 + ' vs ' + var2, 'p_value'] = mcnemar_results.pvalue
    display(mcnemar_results_df)
