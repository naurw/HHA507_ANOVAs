#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 21:22:45 2021

@author: William
"""

import pandas as pd
mentalHealth = pd.read_csv('/Users/William/Desktop/mentalHealthTech.csv')

mentalHealth.info()

# =============================================================================
# This dataset contains the following data:
# 
# Timestamp
# 
# Age
# 
# Gender
# 
# Country
# 
# state: If you live in the United States, which state or territory do you live in?
# 
# self_employed: Are you self-employed?
# 
# family_history: Do you have a family history of mental illness?
# 
# treatment: Have you sought treatment for a mental health condition?
# 
# work_interfere: If you have a mental health condition, do you feel that it interferes with your work?
# 
# no_employees: How many employees does your company or organization have?
# 
# remote_work: Do you work remotely (outside of an office) at least 50% of the time?
# 
# tech_company: Is your employer primarily a tech company/organization?
# 
# benefits: Does your employer provide mental health benefits?
# 
# care_options: Do you know the options for mental health care your employer provides?
# 
# wellness_program: Has your employer ever discussed mental health as part of an employee wellness program?
# 
# seek_help: Does your employer provide resources to learn more about mental health issues and how to seek help?
# 
# anonymity: Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?
# 
# leave: How easy is it for you to take medical leave for a mental health condition?
# 
# mentalhealthconsequence: Do you think that discussing a mental health issue with your employer would have negative consequences?
# 
# physhealthconsequence: Do you think that discussing a physical health issue with your employer would have negative consequences?
# 
# coworkers: Would you be willing to discuss a mental health issue with your coworkers?
# 
# supervisor: Would you be willing to discuss a mental health issue with your direct supervisor(s)?
# 
# mentalhealthinterview: Would you bring up a mental health issue with a potential employer in an interview?
# 
# physhealthinterview: Would you bring up a physical health issue with a potential employer in an interview?
# 
# mentalvsphysical: Do you feel that your employer takes mental health as seriously as physical health?
# 
# obs_consequence: Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?
# 
# comments: Any additional notes or comments
# =============================================================================
"""
Variables of interest: 
IV[1] = 'gender'
IV[2] = 'supervisor'
IV[3] = 'no_employees' 
DV = 'Age'; only int dtype within the dataset that can be interesting
IN most cases AGE should NOT be a dependent variable, but in this case it will be used to test for normal regression
Will use logistic regression on a later date to explore how age will affect the willingness to speak up or a categorical dependent variable 
"""

mentalHealth['Gender'].value_counts()

# =============================================================================
# Transform:
# 1) 'Gender' column into Male and Female for explicit answers 
# 2) 'supervisor' column to only contain Yes and Some of them; want to see willingness to speak up 
# 3) Filter out only United States and those working in the tech industry 
#
# Perform: 
# 4) value_counts() on all three IVs to see for balance or imbalance
# =============================================================================
# Part 1:
mentalHealth['Gender'].replace(('Make', 'Male', 'msle', 'Malr', 'Man', 'male', 'M','m', 'Mail', 'Mal', 'maile', 'male', 'Cis Male', 'Cis Man', 'cis male', 'Male (CIS)'), 'Male', inplace = True)
mentalHealth['Gender'].replace(('Female', 'female', 'F', 'f', 'Woman', 'woman', 'femake', 'femail', 'Femake', 'Cis Female', 'cis-female/femme', 'Female (cis)'), 'Female', inplace = True)
# Group non-explicit answers into a new value called Other 
sexCleaned = ['Male', 'Female']
mentalHealth.loc[~mentalHealth['Gender'].isin(sexCleaned), 'Gender'] = 'Other'
mentalHealth['Gender'].value_counts()

# Part 2 & 3: 
usaTech = mentalHealth[(mentalHealth['tech_company'] == 'Yes') & (mentalHealth['Country'] == 'United States')]
list(usaTech)
usaTech['supervisor'].value_counts()
speakUp = usaTech.loc[usaTech['supervisor'] != 'No']

# Part 4: 
speakUp['supervisor'].value_counts()
speakUp['Gender'].value_counts()
speakUp['no_employees'].value_counts().sort_values()
"""
All IVs seem to be unbalanced. Only exception is number of employees; that seems to be half balanced.  
"""

# Optional for further cleaning/transforming: Creating new column with age ranges 
individualAge = speakUp['Age']
bins = [18, 30, 40, 50, 60, 70, 120]
ageLabels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70+']
speakUp['ageRange'] = pd.cut(speakUp['Age'], bins=bins, labels=ageLabels, right= False)

# =============================================================================
# Assumptions 
# =============================================================================
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, bartlett, shapiro, norm
#!pip install pingouin
#conda install -c conda-forge pingouin
from bioinfokit.analys import stat
import pingouin as pg
import statsmodels.stats.multicomp as mc

# Normality Tests using ANOVA framework 
modelsmf = smf.ols("Age ~ C(Gender) + C(supervisor) + C(no_employees)", data= speakUp).fit()
stats.shapiro(modelsmf.resid)
"""
ShapiroResult(statistic=0.965419590473175, pvalue=1.7726346612789712e-08)

The p-value < 0.05 shows there is significance between individuals working in the tech industry and willingness to talk to their supervisor about mental health, and their age. 
This dataframe only includes responses of either "Yes or Some of Them", which can be generalized as some action with talking to their supervisor. 
"""
model1smf = smf.ols("Age ~ C(Gender) + C(supervisor) + C(no_employees)", data= usaTech).fit()
stats.shapiro(model1smf.resid)
"""
ShapiroResult(statistic=0.4127904176712036, pvalue=1.995981506614983e-40)

The p-value < 0.05 shows there is signifiance between individuals working in the tech industry and talking to their supervisor about mental health, and their age. 
Lower p-value when including ALL of the values within supervisor indicates something is siginificant here.
This dataframe includes responses of "Yes, No, and Some of Them" with talking to their supervisor. 
""" 
model2smf = smf.ols("Age ~ C(Gender)", data= speakUp).fit()
stats.shapiro(model2smf.resid)
"""ShapiroResult(statistic=0.9573780298233032, pvalue=9.21490883598608e-10)"""
model3smf = smf.ols("Age ~ C(supervisor)", data= speakUp).fit()
stats.shapiro(model3smf.resid)
"""ShapiroResult(statistic=0.9562885165214539, pvalue=6.343605085668003e-10)"""
model4smf = smf.ols("Age ~ C(no_employees)", data= speakUp).fit()
stats.shapiro(model4smf.resid)
"""ShapiroResult(statistic=0.9653993844985962, pvalue=1.7586470946184818e-08)"""

speakUp.describe()
"""
Mean is less than the std, indicating that this is generally normally distributed; other variables do not show because they are categoricals or floats
"""

speakUp.skew()
"""
Pandas skew function to test for direction and it indicates a right skew; positive.
Returns 0.520851 
"""

# Graphical Tests for normality 
fig = plt.figure(figsize= (10, 10))
ax = fig.add_subplot(111)
normality_plot, stat = stats.probplot(modelsmf.resid, plot= plt, rvalue= True)
ax.set_title("Probability plot of regression residuals \n with R value")
ax.set
plt.show()
"""
The plot shows R^2 to be 0.9631 meaning there is a strong positive correlation, with high variance and high confidence that it is a good fit within the linear regression model. 
The observations all fall within the general vicinity of the mean, therefore indicating there is significance, and reinforces the previous statistical test. 
THere are several outliers but they are well within the range of the distribution.

Reinforces the previous statistical test results; right skewed
"""

normality_plot, stat = stats.probplot(model1smf.resid, plot= plt, rvalue= True)
ax.set_title("Probability plot of regression residuals \n with R value")
ax.set
plt.show()
"""
The plot shows R^2 to be 0.4029 meaning there is a positive correlation, with variance and confidence that it is a somewhat weak fits within the linear regression model. 
The observations all fall within the general vicinity of the mean, therefore indicating there is significance, and reinforces the previous statistical test. 
There is one noticeable outlier when looking at this chart.
"""

# Box and bar plot visualizations of distributions 
genderAge = sns.boxplot(x='Gender', y='Age', data= speakUp, palette='Set3')
genderAgeBar = sns.barplot(x='Gender', y='Age', data= speakUp, palette='Set1')
supAge = sns.boxplot(x='supervisor', y='Age', data= speakUp, palette='Set3')
supAgeBar = sns.barplot(x='supervisor', y='Age', data= speakUp, palette='Set1')
numAge = sns.boxplot(x='no_employees', y='Age', data= speakUp, palette='Set3')
numAgeBar = sns.barplot(x='no_employees', y='Age', data= speakUp, palette='Set1')
genderAgeSup = sns.boxplot(x= 'Gender', y= 'Age', hue= 'supervisor', data= speakUp, palette= 'Set3')
genderAgeSupBar = sns.barplot(x= 'Gender', y= 'Age', hue= 'supervisor', data= speakUp, palette= 'Set1')
genderAgeNum = sns.boxplot(x= 'Gender', y= 'Age', hue= 'no_employees', data= speakUp, palette= 'Set3')
genderAgeNumBar = sns.barplot(x= 'Gender', y= 'Age', hue= 'no_employees', data= speakUp, palette= 'Set1')
numAgeSup = sns.boxplot(x= 'no_employees', y= 'Age', hue= 'supervisor', data= speakUp, palette= 'Set3')
numAgeSupBar = sns.barplot(x= 'no_employees', y= 'Age', hue= 'supervisor', data= speakUp, palette= 'Set1')
numAgeGender = sns.boxplot(x= 'no_employees', y= 'Age', hue= 'Gender', data= speakUp, palette= 'Set3')
numAgeGenderBar = sns.barplot(x= 'no_employees', y= 'Age', hue= 'Gender', data= speakUp, palette= 'Set1')

"""
Created some boxplots with variations to see what trends there might be. 
Among all the genders, females are likely to voice their concerns about mental health with their supervisors. 
Males are ambivalous with voicing their concerns. And others are more likely to be hesistant with voicing their concerns only to select supervisors. 
There are more males within this sample than the other genders. Males have higher range of deviation with more noticeable outliers.
Individuals grouped under 'other' have the least variation. 
Companies with more than 1000 employees have the highest rate of dicussing with theirsupervisor about mental health. 
"""
# =============================================================================
# ANOVAs
# =============================================================================
# =============================================================================
# Three-way ANOVA Main effect 
# =============================================================================
model = ols('Age ~ C(Gender) + C(supervisor) + C(no_employees)', data=speakUp).fit()
anovaTable = sm.stats.anova_lm(model, typ=3)
anovaTable
"""
                       sum_sq     df           F        PR(>F)
Intercept        26854.854772    1.0  471.630785  1.652817e-70
C(Gender)          700.098386    2.0    6.147640  2.337318e-03
C(supervisor)        4.131184    1.0    0.072553  7.877888e-01
C(no_employees)    726.763475    5.0    2.552716  2.724281e-02
Residual         23744.154956  417.0         NaN           NaN
"""
# =============================================================================
# Three-way ANOVA Interaction effect 
# =============================================================================
model1 = ols('Age ~ C(Gender) + C(supervisor) + C(no_employees) + C(Gender):C(supervisor) + C(Gender):C(no_employees) + C(no_employees):C(supervisor)', data=speakUp).fit()
anovaTableInteractions = sm.stats.anova_lm(model1, typ=3)
anovaTableInteractions
"""
                                     sum_sq     df           F        PR(>F)
Intercept                       6133.495257    1.0  118.401789  2.452449e-24
C(Gender)                        440.879100    2.0    4.255394  1.483354e-02
C(supervisor)                     13.524787    1.0    0.261084  6.096574e-01
C(no_employees)                  401.128767    5.0    1.548688  1.736698e-01
C(Gender):C(supervisor)          122.739641    2.0    1.184691  3.069088e-01
C(Gender):C(no_employees)       1922.283387   10.0    3.710801  1.727440e-04
C(no_employees):C(supervisor)    322.540196    5.0    1.245272  2.870972e-01
Residual                       20772.757084  401.0         NaN           NaN
"""
# =============================================================================
# One-way ANOVA 3 times 
# =============================================================================
model2 = ols('Age ~ C(Gender)', data=speakUp).fit()
anovaTable2 = sm.stats.anova_lm(model2, typ=1)
anovaTable2
"""
              df       sum_sq     mean_sq         F    PR(>F)
C(Gender)    2.0    677.74188  338.870940  5.857495  0.003095
Residual   423.0  24471.62197   57.852534       NaN       NaN
"""
model3 = ols('Age ~ C(supervisor)', data=speakUp).fit()
anovaTable3 = sm.stats.anova_lm(model3, typ=1)
anovaTable3
"""
                  df        sum_sq    mean_sq         F    PR(>F)
C(supervisor)    1.0     29.017021  29.017021  0.489771  0.484413
Residual       424.0  25120.346829  59.246101       NaN       NaN
"""
model4 = ols('Age ~ C(no_employees)', data=speakUp).fit()
anovaTable4 = sm.stats.anova_lm(model4, typ=1)
anovaTable4
"""
                    df        sum_sq     mean_sq         F    PR(>F)
C(no_employees)    5.0    661.164344  132.232869  2.267942  0.047021
Residual         420.0  24488.199505   58.305237       NaN       NaN
"""

"""
According to the ANOVAs performed, there is significant difference between the variables: age, gender, supervisor, and number of employees. 
"""

# =============================================================================
# Pingouin (post-hoc tuskey equivalent: howell test)
# =============================================================================
pgTest = pg.pairwise_gameshowell(data=speakUp, dv='Age', between='Gender').round(3)
pgTest1 = pg.pairwise_gameshowell(data=speakUp, dv='Age', between='supervisor').round(3)
pgTest2 = pg.pairwise_gameshowell(data=speakUp, dv='Age', between='no_employees').round(3)
