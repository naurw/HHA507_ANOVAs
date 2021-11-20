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
speakUp['supervisorCode'].value_counts()
speakUpSample = speakUp.sample(10)

# Part B: These are some of the things that I thought would be interesting
speakUp['supervisor'].value_counts()
speakUp['Gender'].value_counts()
speakUp['no_employees'].value_counts().sort_index()
speakUp['Age'].value_counts().sort_index()
speakUp['ageRange'].value_counts()
speakUp['supervisorCode'].value_counts()
"""
sample sizes are unbalanced 
"""
# =============================================================================
# Since the variables of choice are categorical or object dtypes, we can perform 
# A) logistic regression through binary or multiclass 
# B) linear regression using encoding via cat.codes that was done earlier  <-- another way to code is simply find and replace through data dictionary 
# =============================================================================

# =============================================================================
# ANOVA + Assumption test
# 1) See the differences between each variable 
# 
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
# Probaility plot (P-P plot) of ALL three variables 
fig = plt.figure(figsize= (10, 10))
ax = fig.add_subplot(111)
normality_plot, stat = stats.probplot(modelsmf.resid, plot= plt, rvalue= True)
ax.set_title("Probability plot of regression residuals \n with R value")
ax.set
plt.show()

# Series of box plots and bar plots to visualze the distribution; ageRange was chosen over age for simplicity and less graphical confusion
import seaborn as sns 

genderCode = sns.boxplot(x='Gender', y='supervisorCode', data= speakUp, palette='Set3')
genderCodeBar = sns.barplot(x='Gender', y='supervisorCode', data= speakUp, palette='Set1', capsize=.2)

numCode = sns.boxplot(x='no_employees', y='supervisorCode', data= speakUp, palette='Set3')
numCodeBar = sns.barplot(x='no_employees', y='supervisorCode', data= speakUp, palette='Set1', capsize=.2)

ageCode = sns.boxplot(x='ageRange', y='supervisorCode', data= speakUp, palette='Set3')
ageCodeBar = sns.barplot(x='ageRange', y='supervisorCode', data= speakUp, palette='Set1', capsize=.2)

genderAgeSup = sns.boxplot(x= 'Gender', y= 'supervisorCode', hue= 'no_employees', data= speakUp, palette= 'Set3')
genderAgeSupBar = sns.barplot(x= 'Gender', y= 'supervisorCode', hue= 'no_employees', data= speakUp, palette= 'Set1', capsize=.2)

ageCodeNum = sns.boxplot(x= 'ageRange', y= 'supervisorCode', hue= 'no_employees', data= speakUp, palette= 'Set3')
ageCodeNumBar = sns.barplot(x= 'ageRange', y= 'supervisorCode', hue= 'no_employees', data= speakUp, palette= 'Set1', capsize=.2)

ageCodeGender = sns.boxplot(x= 'ageRange', y= 'supervisorCode', hue= 'Gender', data= speakUp, palette= 'Set3')
ageCodeGenderBar = sns.barplot(x= 'ageRange', y= 'supervisorCode', hue= 'Gender', data= speakUp, palette= 'Set1', capsize=.2)

numCodeGender = sns.boxplot(x= 'no_employees', y= 'supervisorCode', hue= 'Gender', data= speakUp, palette= 'Set3')
numCodeGenderBar = sns.barplot(x= 'no_employees', y= 'supervisorCode', hue= 'Gender', data= speakUp, palette= 'Set1', capsize=.2)

numCodeAge = sns.boxplot(x= 'no_employees', y= 'supervisorCode', hue= 'ageRange', data= speakUp, palette= 'Set3')
numCodeAgeBar = sns.barplot(x= 'no_employees', y= 'supervisorCode', hue= 'ageRange', data= speakUp, palette= 'Set1', capsize=.2)
"""
Generalizations from barplots 
1) The larger the company, the less likely the employee will discuss about mental health with their supervisor 
2) Men are more likely to discuss about their mental health with their supervisor than female / more men work in the tech industry than females 
3) Older individuals working within tech industry are more likely to talk to their supervisors about their mental health 
4) Dataset is retrospective and dated; can expect shift with more relevant/concurrent data 
"""
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


PR(>F) is < 0.05, indicating there is variation / differences between the IV and DV (main effects)
F value is inverse to p-value; the higher the F value, the more significant the p-value
"""
model1 = ols('supervisorCode ~ C(Gender) + C(ageRange) + C(no_employees) + C(Gender):C(ageRange) + C(Gender):C(no_employees) + C(ageRange):C(no_employees)', data=speakUp).fit()
anovaTableInteractions = sm.stats.anova_lm(model1, typ=3)
anovaTableInteractions
"""
                                 sum_sq     df          F    PR(>F)
Intercept                      9.025915    1.0  13.226554  0.000301
C(Gender)                      5.083961    2.0   3.725012  0.024706
C(ageRange)                    2.594741    5.0   0.760465  0.551309
C(no_employees)                0.566343    5.0   0.165984  0.974980
C(Gender):C(ageRange)          7.176487   10.0   1.051641  0.393740
C(Gender):C(no_employees)     10.432404   10.0   1.528762  0.134247
C(ageRange):C(no_employees)   29.690706   25.0   1.740348  0.040042
Residual                     386.243311  566.0        NaN       NaN

PR(>F) changed with the introduction of new variables (interactions). Only gender and ageRange:no_employees is significant 
"""

model2 = ols('supervisorCode ~ C(Gender)', data=speakUp).fit()
anovaTable2 = sm.stats.anova_lm(model2, typ=1)
anovaTable2
"""
              df      sum_sq   mean_sq         F    PR(>F)
C(Gender)    2.0    7.475930  3.737965  5.337511  0.005037
Residual   605.0  423.693478  0.700320       NaN       NaN


Gender is significantly different compared to supervisorCode because <0.05
"""

model3 = ols('supervisorCode ~ C(ageRange)', data=speakUp).fit()
anovaTable3 = sm.stats.anova_lm(model3, typ=1)
anovaTable3
"""
                df      sum_sq   mean_sq         F    PR(>F)
C(ageRange)    5.0    6.280377  1.256075  1.782672  0.114363
Residual     602.0  424.170908  0.704603       NaN       NaN

Age range is not significantly different compared to supervisorCode because >0.05
"""

model4 = ols('supervisorCode ~ C(no_employees)', data=speakUp).fit()
anovaTable4 = sm.stats.anova_lm(model4, typ=1)
anovaTable4
"""
                    df      sum_sq   mean_sq         F    PR(>F)
C(no_employees)    5.0    5.843782  1.168756  1.654242  0.143734
Residual         602.0  425.325626  0.706521       NaN       NaN

Number of employees is not significantly different compared to supervisorCode because >0.05
"""

# Alternative to using statsmodel that will yield the same results 
from bioinfokit.analys import stat
# =============================================================================
# res = stat() 
# res.anova_stat(df=speakUp, res_var='supervisorCode', anova_model='supervisorCode ~ C(Gender)')
# res.anova_summary
#               df      sum_sq   mean_sq         F    PR(>F)
# (Gender)    2.0    7.475930  3.737965  5.337511  0.005037
# Residual   605.0  423.693478  0.700320       NaN       NaN
# 
# model3 = ols('supervisorCode ~ C(ageRange)', data=speakUp).fit()
# anovaTable3 = sm.stats.anova_lm(model3, typ=1)
# anovaTable3
# =============================================================================
"""
                df      sum_sq   mean_sq         F    PR(>F)
C(ageRange)    5.0    6.280377  1.256075  1.782672  0.114363
Residual     602.0  424.170908  0.704603       NaN       NaN

Age range doesn't have a significant difference compared to supervisorCode; will see if nonaggregated version will show a difference 
"""

model4 = ols('supervisorCode ~ C(no_employees)', data=speakUp).fit()
anovaTable4 = sm.stats.anova_lm(model4, typ=1)
anovaTable4
"""
                    df      sum_sq   mean_sq         F    PR(>F)
C(no_employees)    5.0    5.843782  1.168756  1.654242  0.143734
Residual         602.0  425.325626  0.706521       NaN       NaN

Number of employees doesn't have a significant differnce 
"""

# =============================================================================
# Post Hoc tests: Games-Howell Test and Tukey Test
# =============================================================================
import pingouin as pg
pgTest = pg.pairwise_gameshowell(data=speakUp, dv='supervisorCode', between='Gender').round(3)
pgTest
"""
        A      B  mean(A)  mean(B)   diff     se      T       df   pval  hedges
0  Female   Male    0.913    1.176 -0.263  0.079 -3.350  237.969  0.003  -0.325
1  Female  Other    0.913    1.000 -0.087  0.267 -0.326   10.281  0.900  -0.106
2    Male  Other    1.176    1.000  0.176  0.261  0.674    9.428  0.772   0.215
"""
res = stat()
res.tukey_hsd(df=speakUp, res_var='supervisorCode', xfac_var='Gender', anova_model='supervisorCode ~ C(Gender)')
res.tukey_summary
"""
   group1  group2      Diff     Lower     Upper   q-value   p-value
0    Male  Female  0.230159 -0.307178  0.767495  1.452923  0.558784
1    Male   Other  0.714286 -1.255098  2.683669  1.230278  0.648139
2  Female   Other  0.944444 -1.058569  2.947458  1.599389  0.500000
"""

pgTest1 = pg.pairwise_gameshowell(data=speakUp, dv='supervisorCode', between='ageRange').round(3)
pgTest1
"""
       A      B  mean(A)  mean(B)   diff     se      T       df   pval  hedges
0  18-29  30-39    1.018    1.215 -0.197  0.076 -2.598  460.202  0.073  -0.234
1  18-29  40-49    1.018    1.023 -0.004  0.107 -0.041  161.561  0.900  -0.005
2  18-29  50-59    1.018    1.158 -0.140  0.214 -0.652   20.892  0.900  -0.156
3  18-29  60-69    1.018    1.333 -0.315  0.669 -0.471    2.030  0.900  -0.273
4  30-39  40-49    1.215    1.023  0.192  0.103  1.874  143.250  0.336   0.229
5  30-39  50-59    1.215    1.158  0.057  0.212  0.270   20.121  0.900   0.064
6  30-39  60-69    1.215    1.333 -0.118  0.668 -0.177    2.022  0.900  -0.102
7  40-49  50-59    1.023    1.158 -0.135  0.225 -0.601   25.322  0.900  -0.151
8  40-49  60-69    1.023    1.333 -0.311  0.673 -0.462    2.073  0.900  -0.269
9  50-59  60-69    1.158    1.333 -0.175  0.698 -0.251    2.398  0.900  -0.150
"""
res = stat()
res.tukey_hsd(df=speakUp, res_var='supervisorCode', xfac_var='ageRange', anova_model='supervisorCode ~ C(ageRange)')
res.tukey_summary
"""
  group1 group2      Diff     Lower     Upper   q-value   p-value
0  30-39  40-49  0.067736 -0.583067  0.718539  0.388281  0.900000
1  30-39  18-29  0.231602 -0.463701  0.926905  1.242633  0.792688
2  30-39  50-59  0.053030 -1.101114  1.207174  0.171411  0.900000
3  40-49  18-29  0.163866 -0.622887  0.950619  0.777006  0.900000
4  40-49  50-59  0.014706 -1.196732  1.226144  0.045286  0.900000
5  18-29  50-59  0.178571 -1.057343  1.414486  0.539012  0.900000
"""
pgTest2 = pg.pairwise_gameshowell(data=speakUp, dv='supervisorCode', between='no_employees').round(3)
pgTest2
"""
           A         B  mean(A)  mean(B)  ...      T       df   pval  hedges
0        1-5   100-500    1.176    1.163  ...  0.099  139.326  0.900   0.016
1        1-5     1000+    1.176    0.962  ...  1.703  121.079  0.528   0.246
2        1-5    26-100    1.176    1.109  ...  0.530  126.114  0.900   0.078
3        1-5  500-1000    1.176    1.200  ... -0.130   60.363  0.900  -0.028
4        1-5      6-25    1.176    1.222  ... -0.349  134.268  0.900  -0.052
5    100-500     1000+    1.163    0.962  ...  1.807  176.669  0.465   0.241
6    100-500    26-100    1.163    1.109  ...  0.477  180.716  0.900   0.065
7    100-500  500-1000    1.163    1.200  ... -0.217   51.710  0.900  -0.046
8    100-500      6-25    1.163    1.222  ... -0.507  187.361  0.900  -0.071
9      1000+    26-100    0.962    1.109  ... -1.513  290.848  0.635  -0.175
10     1000+  500-1000    0.962    1.200  ... -1.473   41.677  0.661  -0.292
11     1000+      6-25    0.962    1.222  ... -2.569  265.776  0.109  -0.305
12    26-100  500-1000    1.109    1.200  ... -0.560   43.292  0.900  -0.112
13    26-100      6-25    1.109    1.222  ... -1.094  257.746  0.876  -0.134
14  500-1000      6-25    1.200    1.222  ... -0.134   45.974  0.900  -0.027
"""
res = stat()
res.tukey_hsd(df=speakUp, res_var='supervisorCode', xfac_var='no_employees', anova_model='supervisorCode ~ C(no_employees)')
res.tukey_summary
"""
      group1    group2      Diff     Lower     Upper   q-value   p-value
0     26-100       1-5  0.277778 -0.632952  1.188507  1.268411  0.900000
1     26-100      6-25  0.244444 -0.719381  1.208270  1.054712  0.900000
2     26-100  500-1000  0.194444 -1.156386  1.545275  0.598614  0.900000
3     26-100     1000+  0.230159 -0.640665  1.100983  1.099130  0.900000
4     26-100   100-500  0.444444 -0.519381  1.408270  1.917658  0.726478
5        1-5      6-25  0.033333 -1.013015  1.079682  0.132481  0.900000
6        1-5  500-1000  0.083333 -1.327563  1.494230  0.245627  0.900000
7        1-5     1000+  0.047619 -0.913745  1.008983  0.205989  0.900000
8        1-5   100-500  0.166667 -0.879682  1.213015  0.662406  0.900000
9       6-25  500-1000  0.050000 -1.395738  1.495738  0.143824  0.900000
10      6-25     1000+  0.014286 -0.997520  1.026092  0.058716  0.900000
11      6-25   100-500  0.200000 -0.892875  1.292875  0.761047  0.900000
12  500-1000     1000+  0.035714 -1.349758  1.421187  0.107200  0.900000
13  500-1000   100-500  0.250000 -1.195738  1.695738  0.719122  0.900000
14     1000+   100-500  0.214286 -0.797520  1.226092  0.880741  0.900000
"""