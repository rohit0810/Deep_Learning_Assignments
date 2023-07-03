#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data=pd.read_csv("E:/excel/boston.csv")


# In[4]:


data.head()


# In[4]:


data.shape #optional


# In[5]:


data.isnull().sum()  


# In[6]:


data.dropna(inplace=True)


# In[7]:


data.describe()  #optional


# In[8]:


data.info()  #optional


# In[9]:


import seaborn as sns
sns.histplot(data.PRICE)


# In[10]:


correlation = data.corr()
correlation.loc['PRICE']


# In[11]:


import matplotlib.pyplot as plt
fig,axes = plt.subplots(figsize=(15,12))
sns.heatmap(correlation,square = True,annot = True)


# In[12]:


plt.figure(figsize=(20, 5))

features = ['LSTAT', 'RM', 'PTRATIO']
for i, col in enumerate(features):
    plt.subplot(1, len(features), i+1)
    x = data[col]
    y = data.PRICE
    plt.scatter(x, y, marker='o')
    plt.title("Variation in House prices")
    plt.xlabel(col)
    plt.ylabel('House prices in $1000')
    
plt.show()


# In[3]:


X = data.iloc[:,:-1]
y= data.PRICE


# In[ ]:





# In[14]:


from sklearn.model_selection import train_test_split
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std



# In[15]:


from sklearn.linear_model import LinearRegression


# In[16]:


regressor = LinearRegression()


# In[17]:


regressor.fit(X_train,y_train)


# In[18]:


y_pred = regressor.predict(X_test)


# In[19]:


from sklearn.metrics import mean_squared_error
rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
print(rmse)


# In[20]:


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(r2) #accuracy without deep learning


# In[21]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[22]:


import tensorflow as tf
from keras.layers import Dense, Activation,Dropout
from keras.models import Sequential

model = Sequential()
model.add(Dense(128,activation = 'relu',input_dim =13))
model.add(Dense(64,activation = 'relu'))
model.add(Dense(32,activation = 'relu'))
model.add(Dense(16,activation = 'relu'))
model.add(Dense(1))


# In[23]:


model.compile(optimizer = 'adam',loss ='mean_squared_error',metrics=['mae'])


# In[24]:


history = model.fit(X_train, y_train, epochs=100, validation_split=0.05)


# In[25]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go


# In[27]:


y_pred = model.predict(X_test)
mse_nn, mae_nn = model.evaluate(X_test, y_test)
print('Mean squared error on test data: ', mse_nn)
print('Mean absolute error on test data: ', mae_nn)


# In[28]:


from sklearn.metrics import mean_absolute_error
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
print('Mean squared error on test data: ', mse_lr)
print('Mean absolute error on test data: ', mae_lr)


# In[29]:


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(r2) #accuracy with deep learning


# In[30]:


from sklearn.metrics import mean_squared_error
rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
print(rmse) #optional


# In[31]:


#giving an input
import sklearn
new_data = sklearn.preprocessing.StandardScaler().fit_transform(([[0.1, 10.0,
5.0, 0, 0.4, 6.0, 50, 6.0, 1, 400, 20, 300, 10]]))
prediction = model.predict(new_data)
print("Predicted house price:", prediction) 


# In[ ]:





# In[ ]:





# In[ ]:




