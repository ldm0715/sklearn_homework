#!/usr/bin/env python
# coding: utf-8

# In[1]:


#载入数据
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import matplotlib.pyplot as plt#用于作图

data = pd.read_excel("北京市空气质量数据.xlsx")
data


# In[2]:


from sklearn import linear_model
reg=linear_model.LinearRegression(fit_intercept=True,normalize=True)#建议默认参数


# In[3]:


plt.rcParams['font.sans-serif']='SimHei' #中文支持
plt.rcParams['axes.unicode_minus']=False


# In[4]:


data[(data.AQI == 0)].index.tolist() 


# In[5]:


data = data.drop(data[(data['AQI'] == 0)].index)
data


# In[6]:


data = data.drop(data[(data['PM2.5']>=200) | (data['SO2']>=20)].index)
data


# In[7]:


reg = linear_model.LinearRegression(fit_intercept=True,normalize=False)

plt.figure(figsize=(8,5))

x1 = np.array(data['CO']).reshape(-1,1)
y1 = np.array(data['PM2.5']).reshape(-1,1)

plt.title("CO与PM2.5",size=20)
plt.ylabel("PM2.5",size = 15)
plt.xlabel("CO",size = 15)

LR1 = LinearRegression()
LR1.fit(x1,y1)
y_predict_1 = LR1.predict(x1)

line1 = plt.scatter(x1,y1)
plt.plot(x1,y_predict_1,'red')
plt.show()


# In[8]:


print(LR1.coef_ )
print(LR1.intercept_)


# In[9]:


line1 = "Line1: Y = {:.2f}".format(LR1.intercept_[0])
for i in range (len(LR1.coef_)):
    line1 += " + {:.2f}X{:d}".format(LR1.coef_[0][i],i+1)
print(line1)


# In[10]:


#evaluate the model1
from sklearn.metrics import mean_squared_error,r2_score
mean_squared_error_1 = mean_squared_error(y1,y_predict_1)
r2_score_1 = r2_score(y1,y_predict_1)
print(mean_squared_error_1,r2_score_1)


# In[11]:


reg = linear_model.LinearRegression(fit_intercept=True,normalize=False)

plt.figure(figsize=(8,5))

x2 = data.loc[: ,['CO','SO2']]
y2 = data.loc[: ,['PM2.5']]

plt.title("'CO','SO2'与PM2.5的关系",size=20)
plt.xlabel("Comprehensive factors",size = 15)
plt.ylabel("PM2.5",size = 15)

LR2 = LinearRegression()
LR2.fit(x2,y2)
y_predict_2 = LR2.predict(x2)

plt.scatter(y2,y_predict_2)
plt.plot([y2.min(),y2.max()], [y2.min(),y2.max()], 'red')
plt.show()


# In[17]:


print(LR2.coef_ )
print(LR2.intercept_)


# In[18]:


line2 = "Line2: Y = {:.2f}".format(LR2.intercept_[0])
for i in range (len(LR2.coef_)):
    line2 += " + {:.2f}X{:d}".format(LR2.coef_[0][i],i+1)
print(line2)


# In[13]:


#evaluate the model2
from sklearn.metrics import mean_squared_error,r2_score
mean_squared_error_2 = mean_squared_error(y2,y_predict_2)
r2_score_2 = r2_score(y2,y_predict_2)
print(mean_squared_error_2,r2_score_2)


# In[14]:


reg = linear_model.LinearRegression(fit_intercept=True,normalize=False)

plt.figure(figsize=(8,5))

x3 = data.loc[: ,['CO','SO2','NO2','O3']]
y3 = data.loc[: ,['PM2.5']]

plt.title("多因素与PM2.5",size=20)
plt.xlabel("Comprehensive factors",size = 15)
plt.ylabel("PM2.5",size = 15)


LR3 = LinearRegression()
LR3.fit(x3,y3)
y_predict_3 = LR3.predict(x3)

plt.scatter(y3,y_predict_3)
plt.plot([y3.min(),y3.max()], [y3.min(),y3.max()], 'red')
plt.show()


# In[15]:


line3 = "Line3 : Y = {:.2f}".format(LR3.intercept_[0])
for i in range (len(LR3.coef_)):
    line3 += " + {:.2f}X{:d}".format(LR3.coef_[0][i],i+1)
print(line3)


# In[16]:


#evaluate the model2
from sklearn.metrics import mean_squared_error,r2_score
mean_squared_error_3 = mean_squared_error(y3,y_predict_3)
r2_score_3 = r2_score(y2,y_predict_3)
print(mean_squared_error_3,r2_score_3)


# In[ ]:





# In[ ]:




