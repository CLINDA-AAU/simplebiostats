# Simple Biostatistics in Python

## Motivation
This package was intended as a set of conveniency functions (mostly) based StatsModels (and SciPy.Stats) since it might be a bit challenging for a new comer to Python to find the tools to do simple biostatistics.

This package presuppose some knowledge of [pandas](https://pandas.pydata.org/) and is intended to work within a [Jupyter](https://jupyter.org/) notebooks.

For advanced users, we definitely advise them to use the full packages instead:
* [StatsModels](https://www.statsmodels.org/)
* [Scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html)

## How to use
For Jupyter notebooks, Download the folder and place it in your notebooks folder.
The folder structure should look like that:

```
project 
│
└───notebooks
    │   notebook1.ipynb
    |   notebook2.ipynb
    |   ...
    │
    └───data
    |   │   data.csv
    |   │   ...
    |
    └───simplebiostats
        │   simplebiostats.py
        │   ...

```

Once in there, you can import the function by inserting `from simplebiostats.simplebiostats import <name of the function>`.

A typical usecase will look like that

```
import pandas
from simplebiostats.simplebiostats import check_normality

data_df = pandas.read_csv('data/data.csv')

check_normality(data_df, 'variable')
```
You can also import `.dta` or `.xslx` files with `pandas.read_stata(...)` and `pandas.read_excel(...)`.

## Functions
In the following `data_df` is a pandas dataframe containing the data each variable being a column.
`variable` is a string containing the name of the variable of interest containted in the `data_df`.

### summarize(data_df)
Provide some basic data about all the variables in the data. This contains the mean, the standard deviation, the inter-quantile ratio (IQR), the standard error, the skewness, the kurtosis, the median and the 2.5%, 25%, 75% and 97.5% percentiles.

### check_normality(data_df, variable)
Function to plot a boxplot, an histogram with fitted normal distribution and a Q-Q plot.

### get_ci(data_df, variable, alpha=0.05)
Function to get the confidence interval of the mean assuming a normal distribution. You can adjust the alpha if needed.

### get_pi(data_df, variable, alpha=0.05)
Same as above but for the confidence interval of the prediction.

### ttest_1samp(data_df, variable, hypothesis=0)
T-test for a variable compared to an hypothesis which is 0 by default.

### ttest_ind_by_group(data_df, group_variable, variable)
T-test for a variable where the data is splitted in 2 groups according to the `group_variable`. The 2 variables are assumed to be independant.

### ftest_std_by_group(data_df, group_variable, variable)
F-test of equal variance between 2 groups as define by the `group_variable`.

### bootstrap_by_group(data_df, group_variable, variable, n_iter)
Bootstrap method to evaluate the distribution of the mean.

### ranksums_by_group(data_df, group_variable, variable)
Non-parametric test of equality between two groups.

### power_calculation_ttest(mean_diff, std, alpha=0.05, power=None, n_participants=None)
To calculate either the power or the number of participants needed based on t-test, according to which variable between `power` and `n_participants` is set.

### plot_scatter_blandalt(input_df, var1, var2)
Plot a scatter plot of `var1` vs `var2` and a Bland-Altman plot for these variable (which is diff vs mean). This is used for paired data to evaluate the "same distribution" assumption.

### ttest_paired(data_df, var1, var2)
T-test but for paired data.

### anova_by_group(data_df, resp_var, group_var)
ANalysis Of VAriance (ANOVA) in the case of multiple means between groups. It includes a Bartlett's test of "same distribution".

### kruskal_by_group(data_df, resp_var, group_var):
Kruskal-Wallis test of "same distribution". Same idea as the Bartlett's test but non-parametric. 

### LinearRegression(data_df, resp_var, cont_var=None, cat_var=None, with_interactions=False)
To perform a linear regression. You need first to create the object before you can use on it the method below. It's thought to be used in very simple case where there is a continuous variable (`cont_var`) and a categorical one(`cat_var`). `resp_var` is the response variable (or the outcome). The `with_interactions` flag tells if you want to include the interaction between the 2 variables or not.

#### fit()
To fit the model. It will return the coefficients of the fit as well as an ANOVA.

#### check_model()
This will plot a scatter plot of the residuals vs the predictions and one for the the residuals vs the continous variable. The goal is to make sure than the residuals are independant of the rest.
This will furthermore plot the figures from the `check_normality` method on the residuals to make sure they are also normally distributed.

#### plot_predictions(alpha=0.05)
This function will plot the data with the prediction intervals along the continous variable.

#### predict(data_df, alpha=0.05)
It will calculate the predictions for an input data set, including confidence intervals.
