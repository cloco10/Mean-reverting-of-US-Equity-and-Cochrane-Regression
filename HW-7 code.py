#!/usr/bin/env python
# coding: utf-8

# # HW-7
# ### Name: Qihui Du     USCID: 4115-8356-08

# ### Problem 1

# In[364]:


import pandas as pd
import statsmodels.tsa.stattools as ts
import numpy as np
import datetime as dt
import scipy.stats as sps
import itertools
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


# In[365]:


#read dataset from 1948 onward, pre-selected in Excel
df1=pd.read_excel('HW7-SPX Total Return_Long History_UPDATED.xlsx',
              sheet_name='Dataset', header=0,
              index_col=0) 
df1['RF TR']=((df1['3M Govt']/100+1)**(1/12)).cumprod()#annualize 3M Govt return rate
df1


# In[366]:


def func1(year):
    dff = df1.copy()
    lst = list(itertools.repeat(np.nan, 12 *year))
    
    spx_n=dff['SPX TR'][12 * year:].tolist()#create list of SPX TR with different horizons (in months)
    spx_n.extend(lst)
    govt_n=dff['RF TR'][12 * year:].tolist()#create list of 3M Govt with different horizons (in months)
    govt_n.extend(lst)
    
    spx_var = 'SPX_' + str(year)
    govt_var = 'Govt_' + str(year)
    dff[spx_var]=spx_n
    dff[govt_var] = govt_n
    
    dff['result_spx'] = np.log(dff[spx_var]/dff['SPX TR'])#calculating log returns of SPX TR with different horizons
    dff['result_Govt'] = np.log(dff[govt_var]/dff['RF TR'] )#calculating log returns of T-bills with different horizons

    logexret = (dff['result_spx'] - dff['result_Govt']).tolist()#calculating log excess returns with different horizons
    ret = np.nan_to_num(logexret)[:876-12*(year)]
    var=np.nanvar(ret)#calculating variances with different horizons
    return var


# In[367]:


#return all the variances with [1,2,3,5,7,10] horizons
var1=func1(1)
var2=func1(2)
var3=func1(3)
var5=func1(5)
var7=func1(7)
var10=func1(10)
var_df=pd.DataFrame(data=[var1,var2,var3,var5,var7,var10],
                    index=pd.Index(['yr1_var','yr2_var','yr3_var','yr5_var','yr7_var','yr10_var']),
                    columns=['variance'])
var_df


# In[368]:


#calculate VR for each horizon
def func2(x,year):
    VR=var_df['variance'][x-1]/(year*var_df['variance'][0])
    return VR


# In[369]:


VR1=func2(1,1)
VR2=func2(2,2)
VR3=func2(3,3)
VR5=func2(4,5)
VR7=func2(5,7)
VR10=func2(6,10)
VR_df=pd.DataFrame(data=[VR1,VR2,VR3,VR5,VR7,VR10],
                    index=pd.Index(['yr1','yr2','yr3','yr5','yr7','yr10']),
                    columns=['Variance Ratio'])
VR_df


# ### Problem 2

# In[370]:


#calculate Jr_q for each horizon
VR_df['Jr_q']=VR_df['Variance Ratio']-1


# In[371]:


#calculate Z_stat for each horizon
def func4(b,c):
    Z_q=(((876-12*b)*b)**(1/2)/((2*(b-1))**(1/2)))*VR_df['Jr_q'][c]
    return Z_q
Z_q2=func4(2,1)
Z_q3=func4(3,2)
Z_q5=func4(5,3)
Z_q7=func4(7,4)
Z_q10=func4(10,5)
Z=[0,Z_q2,Z_q3,Z_q5,Z_q7,Z_q10]
Z_stat=np.array(Z)
p_value=2*(1-sps.norm.cdf(abs(Z_stat)))
VR_df['Z_stat']=Z_stat
VR_df['p_value']=p_value
VR_df


# ##### Explanation: as can see from the z-stat&p-value, using a significance of 2, the abs of z-stat all less than 2, and p-values are all bigger than 0.05, indicating that the null hypothesis cannot be rejected, therefore the log excess returns are serially uncorrelated and homoscedastic.

# ### Problem 3
# #### a) First Regression

# In[372]:


# calculate log excess return from problem 1
def func5(year):
    dff = df1.copy()
    spx_n=dff['SPX TR'][12 * year:].tolist()#create list of SPX TR with different horizons (in months)
    lst = list(itertools.repeat(np.nan, 12 *year))
    spx_n.extend(lst)
    govt_n=dff['RF TR'][12 * year:].tolist()#create list of 3M Govt with different horizons (in months)
    govt_n.extend(lst)
    
    spx_var = 'SPX_' + str(year)
    govt_var = 'Govt_' + str(year)
    dff[spx_var]=spx_n
    dff[govt_var] = govt_n
    
    dff['result_spx'] = np.log(dff[spx_var]/dff['SPX TR'])#calculating log returns of SPX TR with different horizons
    dff['result_Govt'] = np.log(dff[govt_var]/dff['RF TR'] )#calculating log returns of T-bills with different horizons
    dff['logexret']= (dff['result_spx']-dff['result_Govt'])
    R_t= np.array(dff['logexret'].tolist()).reshape(-1,1)
    R_t=np.nan_to_num(R_t)[:876-12*(year)]
    return R_t
#calculate D/P from the begining of a period
def Div(year):
    dff = df1.copy()
    lst = list(itertools.repeat(np.nan, 12 *year))
    divy_n=dff['Div Yield'][:876-12*(year)].tolist()
    divy_n.extend(lst)
    Div_y = 'Div_' + str(year)
    dff[Div_y]= divy_n
    D=np.array(dff[Div_y].tolist()).reshape(-1,1)
    D=np.nan_to_num(D)[:876-12*(year)]
    return D


# In[373]:


#regression 1
def reg1(year):
    x=Div(year)
    y=func5(year)
    x = sm.add_constant(x)
    model1 = sm.OLS(y, x).fit()
    coef1=model1.params
    t1=model1.tvalues
    R21=model1.rsquared
    OLS1=[coef1[0],coef1[1], t1[0],t1[1], R21]
    return OLS1
ols1=pd.DataFrame(columns=['Intercept','Coef','t-stat-i','t-stat-c','R2'])
for i in [1, 2, 3, 5, 7, 10]:
    ols1.loc[len(ols1)] = reg1(i)
ols1.index=['yr1','yr2','yr3','yr5','yr7','yr10']
ols1


# #### b) Second Regression

# In[374]:


#calculate log real div growth rate
def realdiv(year):
    dff = df1.copy()
    rdiv_n=dff['Real Dividends'][12 * year:].tolist()#create list of Real Div with different horizons (in months)
    lst = list(itertools.repeat(np.nan, 12 *year))
    rdiv_n.extend(lst)
    
    rdiv='RDiv_'+str(year)
    dff[rdiv]=rdiv_n
    
    Rdiv_t=np.array(np.log(((dff[rdiv]/dff['Real Dividends'])**(1/(12*(year)))).tolist())).reshape(-1,1)
    Rdiv = np.nan_to_num(Rdiv_t)[:876-12*(year)]
    return Rdiv               


# In[375]:


def reg2(year):
    x=Div(year)
    y=realdiv(year)
    x = sm.add_constant(x)
    model2 = sm.OLS(y, x).fit()
    coef2=model2.params
    t2=model2.tvalues
    R22=model2.rsquared
    OLS2=[coef2[0],coef2[1], t2[0],t2[1], R22]
    return OLS2
ols2=pd.DataFrame(columns=['intercept','coef','ti','tc','R2'])

for i in [1, 2, 3, 5, 7, 10]:
    ols2.loc[len(ols2)] = reg2(i)
ols2.index=['yr1','yr2','yr3','yr5','yr7','yr10']
ols2


# In[ ]:




