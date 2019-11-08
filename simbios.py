# +
from IPython.display import Markdown

import pandas as pd
import math
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats
from statsmodels.formula.api import ols


# -

def plot_box_hist_and_qq(input_df, variable):
    data_df = input_df.copy()
    fig, (ax_box, ax_hist, ax_qq) = plt.subplots(1, 3, figsize=[16,6])
    
    describe_df = data_df.describe()
    
    mean = describe_df.loc['mean', variable]
    std = describe_df.loc['std', variable]
    n = int(describe_df.loc['count', variable])
    
    # For the box plot
    ax_box.title.set_text('Box plot')
    boxplotted = ax_box.boxplot(data_df[variable], patch_artist=True)
    
    # see https://stackoverflow.com/questions/41997493/python-matplotlib-boxplot-color
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(boxplotted[element], color='C7', linestyle='dashed')

    # For the histogram plot
    ax_hist.title.set_text('Histogram with fitted normal distribution')
    ax_hist.hist(data_df[variable], bins=10, density=True, color='C0', label='Density')

    x = np.linspace(data_df[variable].min(), data_df[variable].max(), 1000)
    fitted_norm = stats.t.pdf(x, df=len(data_df.index)-1, loc=mean, scale=std)
    ax_hist.plot(x, fitted_norm, color='white')
    ax_hist.plot(x, fitted_norm, color='C7', linestyle='dashed', label='Fitted gaussian')
    
    ax_hist.legend()


    # For the QQ plot
    ax_qq.title.set_text('Q-Q plot of observations vs normal distribution')
    sorted_df = data_df.sort_values(by=variable).reset_index(drop=True)
    for i in range(n):
        sorted_df.loc[i, 'norm_quantiles'] = stats.t.ppf((i+1)/n, df=len(data_df.index)-1, loc=mean, scale=std)

    ax_qq.plot(sorted_df[variable], sorted_df[variable], color='C7', linewidth=1, linestyle='dashed')
    ax_qq.plot(sorted_df['norm_quantiles'], sorted_df[variable], marker='o', color='C0', markersize=4, linewidth=0)
    ax_qq.grid(True)

    plt.show()

def get_summary(input_df, display_results=True):
    data_df = input_df.copy()
    
    summary_df = data_df.describe(percentiles=[0.025, 0.25, 0.75, 0.975]).T
    
    pd.set_option('display.expand_frame_repr', False)
    
    for variable, row in summary_df.iterrows():
        describe = stats.describe(data_df[variable])
        summary_df.loc[variable, 'mean'] = round(summary_df.loc[variable, 'mean'], 3)
        summary_df.loc[variable, 'std'] = round(summary_df.loc[variable, 'std'], 3)
        summary_df.loc[variable, 'IQR'] = round(summary_df.loc[variable, '75%'] - summary_df.loc[variable, '25%'], 3)
        summary_df.loc[variable, 'se'] = round(summary_df.loc[variable, 'std']/math.sqrt(int(summary_df.loc[variable, 'count'])), 3)
        summary_df.loc[variable, 'skew'] = round(describe.skewness, 3)
        summary_df.loc[variable, 'kurt'] = round(describe.kurtosis, 3)
        
    if display_results:
        display(summary_df)

    return summary_df

def get_ci(input_df, alpha=0.95, display_results=True):
    data_df = input_df.copy()
    
    summary_df = get_summary(data_df, False)
    
    ci_df = pd.DataFrame(columns=['Mean', 'CI low', 'CI high'])

    for variable, row in summary_df.iterrows():
        mean = summary_df.loc[variable, 'mean']
        std = summary_df.loc[variable, 'std']
        se = summary_df.loc[variable, 'se']

        ci_low = stats.t.ppf((1-alpha)/2, df=len(data_df.index)-1, loc=mean, scale=se)
        ci_high = stats.t.ppf((1+alpha)/2, df=len(data_df.index)-1, loc=mean, scale=se)
        
        
        ci_df.loc[variable, 'Mean'] = mean
        ci_df.loc[variable, 'CI low'] = ci_low
        ci_df.loc[variable, 'CI high'] = ci_high
        
    if display_results:
        display(Markdown('#### {}% confidence interval'.format(alpha*100)))
        display(ci_df)
        
    return ci_df

def get_pi(input_df, alpha=0.95, display_results=True):
    data_df = input_df.copy()
    
    summary_df = get_summary(data_df, False)
    
    pi_df = pd.DataFrame(columns=['Mean', 'PI low', 'PI high', 'Below PI low', 'Above PI high'])
    
    for variable, row in summary_df.iterrows():
        mean = summary_df.loc[variable, 'mean']
        std = summary_df.loc[variable, 'std']

        pi_low = stats.norm.ppf((1-alpha)/2, loc=mean, scale=std)
        pi_high = stats.norm.ppf((1+alpha)/2, loc=mean, scale=std)
        
        count_obs = summary_df.loc[variable, 'count']
        count_obs_below_pred = len(data_df.loc[data_df[variable]>=pi_high].index)
        count_obs_above_pred = len(data_df.loc[data_df[variable]<=pi_low].index)
        
        pi_df.loc[variable, 'Mean'] = mean
        pi_df.loc[variable, 'PI low'] = pi_low
        pi_df.loc[variable, 'PI high'] = pi_high
        pi_df.loc[variable, 'Below PI low'] = count_obs_below_pred/count_obs
        pi_df.loc[variable, 'Above PI high'] = count_obs_above_pred/count_obs
        
    
    if display_results:
        display(Markdown('#### {}% prediction interval'.format(alpha*100)))
        display(pi_df)
        
    return pi_df

def print_exact_t_test(data_df, hypothesis):
    summary_df = get_summary(data_df)
    
    for variable, row in summary_df.iterrows():
        print('\nVariable:', variable)
        # print(stats.ttest_1samp(data_df[variable], hypothesis))
        
        mean = summary_df.loc[variable, 'mean']
        std = summary_df.loc[variable, 'std']
        se = summary_df.loc[variable, 'se']

        z = abs(mean-hypothesis)/se

        p_value = 2*(1-stats.t.cdf(z, df=len(data_df.index)-1))

        print('Hypothesis: the mean of', variable, 'is', hypothesis, '=> Z-score:', z, ', p_value:', p_value)
        print(stats.ttest_1samp(data_df[variable], hypothesis))


def print_approx_t_test_2_distribs(data1_df, data2_df, se_var = 'se'):
    summary1_df = get_summary(data1_df)
    summary2_df = get_summary(data2_df)
    for variable, row in summary1_df.iterrows():
        print('\nVariable:', variable)
        delta = abs(summary1_df.loc[variable, 'mean'] - summary2_df.loc[variable, 'mean'])
        se = math.sqrt(math.pow(summary1_df.loc[variable, se_var], 2) + math.pow(summary2_df.loc[variable, se_var], 2))
        ci_width = stats.norm.isf(0.025)*se*2
        print('Mean of delta of', variable,':', delta,
              ', Exact CI: [', delta-ci_width/2, ',', delta+ci_width/2, ']')

        z = delta/se

        p_value = 2*stats.norm.cdf(-z)

        print('Hypothesis: the two distributions of', variable, 'have the same mean => Approx. Z-score:',
              '{:f}'.format(z), ', p_value:', '{:f}'.format(p_value))


def print_f_test(data1_df, data2_df):
    summary1_df = get_summary(data1_df)
    summary2_df = get_summary(data2_df)
    for variable, row in summary1_df.iterrows():
        print('\nVariable:', variable)
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
        
        f_score = math.pow(std_large/std_small,2)
        
        p_value = 2*(1-stats.f.cdf(f_score, dfn=n_large-1, dfd=n_small-1))

        print('Hypothesis: the two standard deviations of', variable, 'are the same => Approx. F-score:',
              '{:f}'.format(f_score), ', p_value:', '{:f}'.format(p_value))


def print_exact_t_test_2_distribs(data1_df, data2_df):
    summary1_df = get_summary(data1_df)
    summary2_df = get_summary(data2_df)
    for variable, row in summary1_df.iterrows():
        print('\nVariable:', variable)
        delta = abs(summary1_df.loc[variable, 'mean'] - summary2_df.loc[variable, 'mean'])
        std1 = summary1_df.loc[variable, 'std']
        n1 = summary1_df.loc[variable, 'count']
        
        std2 = summary2_df.loc[variable, 'std']
        n2 = summary2_df.loc[variable, 'count']
        
        std_combined = math.sqrt((math.pow(std1, 2)*(n1-1)+math.pow(std2, 2)*(n2-1))/(n1+n2-2))
                                 
        se_combined = std_combined*math.sqrt(1/n1+1/n2)
        ci_width = stats.t.isf(0.025, n1+n2-2)*se_combined*2
        
        print('Mean of delta of', variable,':', delta,
              ', Exact CI: [', delta-ci_width/2, ',', delta+ci_width/2, ']')

        z = delta/se_combined

        p_value = 2*stats.t.cdf(-z, n1+n2-2)

        print('Hypothesis: the two distributions of', variable, 'have the same mean => Exact Z-score:',
              '{:f}'.format(z), ', p_value:', '{:f}'.format(p_value))


def plot_scatter_blandalt(input_df, var1, var2):
    data_df = input_df.sort_values(by=var1).copy()
    
    fig, (ax_scatter, ax_blandalt) = plt.subplots(1, 2, figsize=[16,6])
    
    #################
    # Scatter plot #
    #################
    min_var1 = data_df.iloc[0][var1]
    max_var1 = data_df.iloc[len(data_df)-1][var1]
    
    data_df['diff'] =  data_df[var2] - data_df[var1]
    mean_diff = data_df[['diff']].mean().values[0]
    
    # Plotting data
    ax_scatter.plot(data_df[var1], data_df[var2], 
        marker='o', color='C0', markersize=4, linewidth=0)
    
    # Plotting helper lines
    ax_scatter.plot([min_var1, max_var1], [min_var1, max_var1], 
        linewidth=1, color='C7', label='Identity', linestyle='dashed', alpha=0.3)
    ax_scatter.plot([min_var1, max_var1], [min_var1+mean_diff, max_var1+mean_diff], 
        linewidth=1, color='C7', linestyle='dashed', label='Identity shifted by mean of difference')
    
    ax_scatter.title.set_text('Scatter plot of the paired distributions')
    ax_scatter.set_xlabel(var1)
    ax_scatter.set_ylabel(var2)
    ax_scatter.legend()
    
    #####################
    # Bland-Altman plot #
    #####################
    data_df['avg'] = (data_df[var1] + data_df[var2])/2
    min_avg = data_df[['avg']].min()
    max_avg = data_df[['avg']].max()
    
    ci = get_ci(data_df[['diff']], 0.95).loc['diff']
    display(ci)
    pi = get_pi(data_df[['diff']], 0.95).loc['diff']
    
    # Plotting data
    ax_blandalt.plot(data_df['avg'], data_df['diff'],
        marker='o', color='C0', markersize=4, linewidth=0)
    
    # Plotting helper lines
    ax_blandalt.plot([min_avg, max_avg], [pi['PI high'], pi['PI high']],
        color='C7', linewidth=1, linestyle='dashed', alpha=1, label='PI high')
    ax_blandalt.plot([min_avg, max_avg], [ci['CI high'], ci['CI high']],
        color='C7', linewidth=1, linestyle='dashed', alpha=0.5, label='CI high')
    ax_blandalt.plot([min_avg, max_avg], [mean_diff, mean_diff],
        color='C7', linewidth=1, linestyle='dashed', alpha=1, label='Mean')
    ax_blandalt.plot([min_avg, max_avg], [ci['CI low'], ci['CI low']],
        color='C7', linewidth=1, linestyle='dashed', alpha=0.5, label='CI low')
    ax_blandalt.plot([min_avg, max_avg], [pi['PI low'], pi['PI low']],
        color='C7', linewidth=1, linestyle='dashed', alpha=1, label='PI low')
    
    ax_blandalt.set_xlabel('Average of {} and {}'.format(var1, var2))
    ax_blandalt.set_ylabel('Difference between {} and {}'.format(var2, var1))
    ax_blandalt.title.set_text('Bland-Altman plot')
    ax_blandalt.legend()
    
    plt.show()


class linear_regression():
    model = None
    
    def __init__(self, data_df, resp_var, cont_var=None, cat_var=None, with_interactions=False):
        self.data_df = data_df
        self.resp_var = resp_var
        self.cont_var = cont_var
        self.cat_var = cat_var
        self.with_interactions = with_interactions

    def fit(self):
        formula = self.resp_var + ' ~ '
        
        if self.cont_var != None:
            formula += self.cont_var
            
            if self.cat_var != None:
                if self.with_interactions:
                    formula += ' * ' + self.cat_var
                else:
                    formula += ' + ' + self.cat_var
        else:
            if self.cat_var != None:
                formula +=  self.cat_var
            else:
                formula += '1'
        
        self.model = ols(formula, data=self.data_df).fit()

        measures_df = pd.concat([
            self.model.summary2().tables[0][[0, 1]].rename(columns={0: 'measure', 1: 'value'}),
            self.model.summary2().tables[0][[2, 3]].rename(columns={2: 'measure', 3: 'value'})])
        measures_df['measure'] = measures_df['measure'].str.replace(':', '')
        measures_df = measures_df.set_index('measure')

        anova_results_df = pd.DataFrame(columns=['Sum of squares', 'Df', 'Mean of squares', 'Root mean of squares'])
        anova_results_df.loc['Model', 'Df'] = int(measures_df.loc['Df Model', 'value'])
        anova_results_df.loc['Residuals', 'Df'] = int(measures_df.loc['Df Residuals', 'value'])
        anova_results_df.loc['Model', 'Sum of squares'] = float(measures_df.loc['Scale', 'value'])
        anova_results_df.loc['Residuals', 'Sum of squares'] = float(measures_df.loc['R-squared', 'value'])
        anova_results_df.loc['Model', 'Mean of squares'] = \
            anova_results_df.loc['Model', 'Sum of squares']/anova_results_df.loc['Model', 'Df']
        anova_results_df.loc['Residuals', 'Mean of squares'] = \
            anova_results_df.loc['Residuals', 'Sum of squares']/anova_results_df.loc['Residuals', 'Df']
        anova_results_df.loc['Model', 'Root mean of squares'] = math.sqrt(
            anova_results_df.loc['Model', 'Mean of squares'])
        anova_results_df.loc['Residuals', 'Root mean of squares'] = math.sqrt(
            anova_results_df.loc['Residuals', 'Mean of squares'])
        anova_results_df.loc['Total'] = anova_results_df.sum()
        anova_results_df.loc['Total', 'Root mean of squares'] = math.sqrt(
            anova_results_df.loc['Total', 'Mean of squares'])
        
        display(Markdown('#### ANOVA analysis'))
        display(anova_results_df)
        display(Markdown('#### Regression parameters'))
        display(self.model.summary2().tables[1])
    
    def check_model(self):
        if self.model == None:
            print('No model is defined, please run fit() on your regression first.')
            return
        
        display(Markdown('#### Checking linearity and identically distributed errors'))
        if self.cont_var != None:
            expl_data = self.data_df[self.cont_var]
            
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=[16, 8])
            
            ax1.plot([min(expl_data), max(expl_data)], [0, 0], color='C7', linestyle='dashed')
            ax1.title.set_text('Residuals vs. explanatory variable "' + self.cont_var + '"')
            ax1.set_ylabel('residuals')
            ax1.set_xlabel(self.cont_var)
            ax1.scatter(expl_data, self.model.resid)
        
        else:
            fig, ax0= plt.subplots(figsize=[8, 8])
        
        predictions = self.model.predict()
        
        ax0.plot([min(predictions), max(predictions)], [0, 0], color='C7', linestyle='dashed')
        ax0.title.set_text('Residuals vs predictions')
        ax0.scatter(self.model.predict(), self.model.resid)
        ax0.set_ylabel('residuals')
        ax0.set_xlabel('predictions')
        
        plt.show()
        
        display(Markdown('#### Checking normality of errors'))
        resid_df = pd.DataFrame(self.model.resid, columns={'residuals'})
        plot_box_hist_and_qq(resid_df, 'residuals')
    
    def plot_predictions(self, alpha=0.05):
        fig, axis = plt.subplots(figsize=[16,8])
        predictions = self.model.predict()
        
        predictions_df = self.model.get_prediction().summary_frame(alpha)
        
        data_df = self.data_df.join(predictions_df)
        
        categories = ['all']
        
        if self.cat_var != None:
            categories = data_df[self.cat_var].unique().tolist()
        
        for category in categories:
            if category == 'all':
                cat_data_df = data_df.copy()
                data_label = 'Data points'
            else:
                cat_data_df = data_df.loc[data_df[self.cat_var] == category].copy()
                data_label = 'Data points for ' + self.cat_var + ' ' + category
            
            if self.cont_var != None:
                x = cat_data_df[self.cont_var]
                vs_var = self.cont_var
            else:
                x = [category]*len(cat_data_df)
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
        predictions_df = self.model.get_prediction(data_df).summary_frame(alpha=0.05)
        
        predictions_df.rename(columns={
            'mean':'prediction', 'mean_se':'se', 'mean_ci_lower':'CI low', 'mean_ci_upper':'CI high', 
            'obs_ci_lower':'PI low', 'obs_ci_upper': 'PI high'}, inplace=True)
        return data_df.join(predictions_df[['prediction', 'PI low', 'PI high']])


