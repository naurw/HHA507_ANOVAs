#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 14:26:42 2021

@author: William
"""

import pandas as pd
mentalHealth = pd.read_csv('/Users/William/Desktop/mentalHealthTech.csv')
mentalHealthsample = mentalHealth.sample(100)

mentalHealth.info()
# =============================================================================
# Part A: Transform and clean
# 1) 'Gender' column into Male and Female for explicit answers 
# 2) Filter out only United States and those working in the tech industry 
# 3) 'Age' column to only contain adults >=18 
# 4) 'supervisor' column to only contain Yes and Some of them; want to see willingness to speak up 
# 5) 'no_employees' column renaming values
# 6) Create new column to group ages into age intervals
# 7) Create new column that encodes categorical values for linear regression

# Part B: Perform statistical tests 
# 1) value_counts() on all three IVs to see for balance or imbalance of variables selected
# =============================================================================

# Part A
mentalHealth['Gender'].replace(('Make', 'Male', 'msle', 'Malr', 'Man', 'male', 'M','m', 'Mail', 'Mal', 'maile', 'male', 'Cis Male', 'Cis Man', 'cis male', 'Male (CIS)'), 'Male', inplace = True)
mentalHealth['Gender'].replace(('Female', 'female', 'F', 'f', 'Woman', 'woman', 'femake', 'femail', 'Femake', 'Cis Female', 'cis-female/femme', 'Female (cis)'), 'Female', inplace = True)
# Group non-explicit answers into a new value called Other 
sexCleaned = ['Male', 'Female']
mentalHealth.loc[~mentalHealth['Gender'].isin(sexCleaned), 'Gender'] = 'Other'
mentalHealth['Gender'].value_counts()
mentalHealth['no_employees'].replace('More than 1000', '1000+', inplace = True)
individualAge = mentalHealth['Age']
bins = [18, 30, 40, 50, 60, 70, 120]
ageLabels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70+']
mentalHealth['ageRange'] = pd.cut(mentalHealth['Age'], bins=bins, labels=ageLabels, right= False)
list(mentalHealth)

speakUp = mentalHealth.loc[(mentalHealth['Age'] >= 18) & (mentalHealth['tech_company'] == 'Yes') & (mentalHealth['Country'] == 'United States')]
#speakUp = mentalHealth.loc[(mentalHealth['supervisor'] != 'No') & (mentalHealth['supervisorCode'] != 0) & (mentalHealth['Age'] >= 18) & (mentalHealth['tech_company'] == 'Yes') & (mentalHealth['Country'] == 'United States')]
speakUp.info()
mentalHealth['supervisor'] = mentalHealth['supervisor'].astype('category')
mentalHealth[['supervisor']].dtypes
mentalHealth['supervisorCode'] = mentalHealth['supervisor'].cat.codes
mentalHealth.dtypes
speakUp.dtypes
# Rerun speakUp to include the supervisorCode column into the dataframe 

# Part B: These are some of the things that I thought would be interesting
speakUp['supervisor'].value_counts()
speakUp['Gender'].value_counts()
speakUp['no_employees'].value_counts().sort_index()
speakUp['Age'].value_counts().sort_index()
speakUp['ageRange'].value_counts()
speakUp['supervisorCode'].value_counts()

# =============================================================================
# Since the variables of choice are categorical or object dtypes, we can perform 
# A) logistic regression through binary or multiclass 
# B) linear regression using encoding via cat.codes that was done earlier  <-- another way to code is simply find and replace through data dictionary 
# =============================================================================

# =============================================================================
# ANOVA + Assumption test
# =============================================================================
import scipy.stats as stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
# Normality Tests using ANOVA framework 
modelsmf = smf.ols("supervisorCode ~ C(Gender) + C(ageRange) + C(no_employees)", data= speakUp).fit()
stats.shapiro(modelsmf.resid)

model2smf = smf.ols("supervisorCode ~ C(Gender)", data= speakUp).fit()
stats.shapiro(model2smf.resid)

model3smf = smf.ols("supervisorCode ~ C(ageRange)", data= speakUp).fit()
stats.shapiro(model3smf.resid)

model4smf = smf.ols("supervisorCode ~ C(no_employees)", data= speakUp).fit()
stats.shapiro(model4smf.resid)
"""
All p-values < 0.05 which indicates that it follows normal distribution 
"""
# Probaility plot (P-P plot) 
fig = plt.figure(figsize= (10, 10))
ax = fig.add_subplot(111)
normality_plot, stat = stats.probplot(modelsmf.resid, plot= plt, rvalue= True)
ax.set_title("Probability plot of regression residuals \n with R value")
ax.set
plt.show()

speakUp.describe()
speakUp.skew()


# Series of box plots and bar plots to visualze the distribution; ageRange was chosen over age for simplicity 
import seaborn as sns 

genderCode = sns.boxplot(x='Gender', y='supervisorCode', data= speakUp, palette='Set3')
genderCodeBar = sns.barplot(x='Gender', y='supervisorCode', data= speakUp, palette='Set1')

numCode = sns.boxplot(x='no_employees', y='supervisorCode', data= speakUp, palette='Set3')
numCodeBar = sns.barplot(x='no_employees', y='supervisorCode', data= speakUp, palette='Set1')

ageCode = sns.boxplot(x='ageRange', y='supervisorCode', data= speakUp, palette='Set3')
ageCodeBar = sns.barplot(x='ageRange', y='supervisorCode', data= speakUp, palette='Set1')

genderAgeSup = sns.boxplot(x= 'Gender', y= 'supervisorCode', hue= 'no_employees', data= speakUp, palette= 'Set3')
genderAgeSupBar = sns.barplot(x= 'Gender', y= 'supervisorCode', hue= 'no_employees', data= speakUp, palette= 'Set1')

ageCodeNum = sns.boxplot(x= 'ageRange', y= 'supervisorCode', hue= 'no_employees', data= speakUp, palette= 'Set3')
ageCodeNumBar = sns.barplot(x= 'ageRange', y= 'supervisorCode', hue= 'no_employees', data= speakUp, palette= 'Set1')

ageCodeGender = sns.boxplot(x= 'ageRange', y= 'supervisorCode', hue= 'Gender', data= speakUp, palette= 'Set3')
ageCodeGenderBar = sns.barplot(x= 'ageRange', y= 'supervisorCode', hue= 'Gender', data= speakUp, palette= 'Set1')

numCodeGender = sns.boxplot(x= 'no_employees', y= 'supervisorCode', hue= 'Gender', data= speakUp, palette= 'Set3')
numCodeGenderBar = sns.barplot(x= 'no_employees', y= 'supervisorCode', hue= 'Gender', data= speakUp, palette= 'Set1')

numCodeAge = sns.boxplot(x= 'no_employees', y= 'supervisorCode', hue= 'ageRange', data= speakUp, palette= 'Set3')
numCodeAgeBar = sns.barplot(x= 'no_employees', y= 'supervisorCode', hue= 'ageRange', data= speakUp, palette= 'Set1')

# Factorial ANOVAs + one way ANOVAs
from statsmodels.formula.api import ols
import statsmodels.api as sm

model = ols('supervisorCode ~ C(Gender) + C(ageRange) + C(no_employees)', data=speakUp).fit()
anovaTable = sm.stats.anova_lm(model, typ=3)
anovaTable
"""
                     sum_sq     df          F        PR(>F)
Intercept         33.733395    1.0  48.749268  7.797791e-12
C(Gender)          6.892088    2.0   4.979995  7.163383e-03
C(ageRange)        5.078330    5.0   1.467773  2.104677e-01
C(no_employees)    5.831484    5.0   1.685455  1.360503e-01
Residual         411.726595  595.0        NaN           NaN
"""
model1 = ols('supervisorCode ~ C(Gender) + C(ageRange) + C(no_employees) + C(Gender):C(ageRange) + C(Gender):C(no_employees) + C(ageRange):C(Gender)', data=speakUp).fit()
anovaTableInteractions = sm.stats.anova_lm(model1, typ=3)
anovaTableInteractions
"""
                               sum_sq     df         F    PR(>F)
Intercept                    6.367032    1.0  9.150579  0.002596
C(Gender)                    3.492406    2.0  2.509610  0.082182
C(ageRange)                  3.189447    5.0  0.916763  0.453706
C(no_employees)              0.704360    5.0  0.202459  0.961427
C(Gender):C(ageRange)        5.157032   10.0  0.741159  0.637122
C(Gender):C(no_employees)    7.364330   10.0  1.058388  0.392025
Residual                   403.567746  580.0       NaN       NaN
"""

model2 = ols('supervisorCode ~ C(Gender)', data=speakUp).fit()
anovaTable2 = sm.stats.anova_lm(model2, typ=1)
anovaTable2
"""
              df      sum_sq   mean_sq         F    PR(>F)
C(Gender)    2.0    7.475930  3.737965  5.337511  0.005037
Residual   605.0  423.693478  0.700320       NaN       NaN
"""
#from bioinfokit.analys import stat
#res = stat() 
#res.anova_stat(df=speakUp, res_var='supervisorCode', anova_model='supervisorCode ~ C(Gender)')
#res.anova_summary
#              df      sum_sq   mean_sq         F    PR(>F)
#C(Gender)    2.0    7.475930  3.737965  5.337511  0.005037
#Residual   605.0  423.693478  0.700320       NaN       NaN

model3 = ols('supervisorCode ~ C(ageRange)', data=speakUp).fit()
anovaTable3 = sm.stats.anova_lm(model3, typ=1)
anovaTable3
"""
                df      sum_sq   mean_sq         F    PR(>F)
C(ageRange)    5.0    6.280377  1.256075  1.782672  0.114363
Residual     602.0  424.170908  0.704603       NaN       NaN
"""

model4 = ols('supervisorCode ~ C(no_employees)', data=speakUp).fit()
anovaTable4 = sm.stats.anova_lm(model4, typ=1)
anovaTable4
"""
                    df      sum_sq   mean_sq         F    PR(>F)
C(no_employees)    5.0    5.843782  1.168756  1.654242  0.143734
Residual         602.0  425.325626  0.706521       NaN       NaN
"""

# =============================================================================
# Pingouin (post-hoc tuskey equivalent: howell test)
# =============================================================================
import pingouin as pg
pgTest = pg.pairwise_gameshowell(data=speakUp, dv='supervisorCode', between='Gender').round(3)
pgTest1 = pg.pairwise_gameshowell(data=speakUp, dv='supervisorCode', between='ageRange').round(3)
pgTest2 = pg.pairwise_gameshowell(data=speakUp, dv='supervisorCode', between='no_employees').round(3)

from bioinfokit.analys import stat
res = stat() 
res.anova_stat(df=speakUp, res_var='supervisorCode', anova_model='supervisorCode ~ C(Gender)')
res.anova_summary
