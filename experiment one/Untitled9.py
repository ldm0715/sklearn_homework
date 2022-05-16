#!/usr/bin/env python
# coding: utf-8

# In[1]:


#载入数据
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import matplotlib.pyplot as plt#用于作图

data = pd.read_csv("housing_price.csv")
data#查看前几行，默认前五行


# In[2]:


from sklearn import linear_model
reg=linear_model.LinearRegression(fit_intercept=True,normalize=True)#建议默认参数


# In[3]:


plt.rcParams['font.sans-serif']='SimHei' #中文支持
plt.rcParams['axes.unicode_minus']=False


# In[4]:


reg = linear_model.LinearRegression(fit_intercept=True,normalize=False)

plt.figure(figsize=(10,8))

x1 = np.array(data['size']).reshape(-1,1)
y1 = np.array(data['Price']).reshape(-1,1)

plt.title("面积与房价的关系",size=20)
plt.xlabel("Avg. Area Number of Rooms",size = 15)
plt.ylabel("Price",size = 15)

LR1 = LinearRegression()
LR1.fit(x1,y1)
y_predict_1 = LR1.predict(x1)

plt.scatter(x1,y1)
plt.plot([x1.min(),x1.max()],[y1.min(),y1.max()],'red')
plt.show()


# In[5]:


print(LR1.coef_ )
print(LR1.intercept_)


# In[6]:


s1= "LR1: Y = {:.2f}".format(LR1.intercept_[0])
for i in range (len(LR1.coef_)):
    s1 += " + {:.2f} X{:d}".format(LR1.coef_[0][i],i+1)
print(s1)


# In[7]:


x0 = 12
result1 =int(LR1.coef_)  + int(LR1.intercept_)*x0
print(result1)


# In[8]:


#evaluate the model1
from sklearn.metrics import mean_squared_error,r2_score
mean_squared_error_1 = mean_squared_error(y1,y_predict_1)
r2_score_1 = r2_score(y1,y_predict_1)
print(mean_squared_error_1,r2_score_1)


# In[9]:


reg = linear_model.LinearRegression(fit_intercept=True,normalize=False)

plt.figure(figsize=(10,8))

x2 = data.loc[: ,['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Area Population','size']]
y2 = data.loc[: ,['Price']]


plt.title("多因素与价格",size=20)
plt.xlabel("预测值",size = 15)
plt.ylabel("真实值",size = 15)

LR2 = LinearRegression()
LR2.fit(x2,y2)
y_predict_2 = LR2.predict(x2)

plt.scatter(y2,y_predict_2)
plt.plot([y2.min(),y2.max()], [y2.min(),y2.max()], 'red')
plt.show()


# In[10]:


print(LR2.coef_ )
print(LR2.intercept_)


# In[11]:


#evaluate the model2
from sklearn.metrics import mean_squared_error,r2_score
mean_squared_error_2 = mean_squared_error(y2,y_predict_2)
r2_score_2 = r2_score(y2,y_predict_2)
print(mean_squared_error_2,r2_score_2)


# In[12]:


print(LR2.coef_ )
print(LR2.intercept_)


# In[13]:


s2 = "Line2: Y = {:.2f}".format(LR2.intercept_[0])
for i in range (len(LR2.coef_)):
    s2 += " + {:.2f} X{:d}".format(LR2.coef_[0][i],i+1)
print(s2)


# In[14]:


x1 = 3* pow(10,6)
result2 = "{:.1f}e6".format((-813067.29 + 21.63*x1)/pow(10,6))
print(result2)


# In[15]:


x1 = 3* pow(10,6)
result2 = int(LR2.intercept_) + int(LR2.coef_ )*x1
print(result2)


# In[ ]:




