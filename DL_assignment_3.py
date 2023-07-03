#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn import metrics


# In[2]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[3]:


plt.imshow(x_train[0], cmap='gray')
plt.show() 


# In[4]:


print(x_train[0])


# In[5]:


print("X_train shape", x_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", x_test.shape)
print("y_test shape", y_test.shape)


# In[6]:


x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')


# In[7]:


x_test = x_test.astype('float32')
x_train /= 255 # Each image has Intensity from 0 to 255
x_test /= 255


# In[8]:


num_classes = 10
y_train = np.eye(num_classes)[y_train] # Return a 2-D array with ones on the diagonal and zeros elsewhere.
y_test = np.eye(num_classes)[y_test] 


# In[9]:


model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))


# In[10]:


model.add(Dropout(0.2))
model.add(Dense(512, activation='relu')) #returns a sequence of another vectors of dimension 512
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))


# In[11]:


model.compile(loss='categorical_crossentropy', # for a multi-class classification problem
optimizer=RMSprop(),
metrics=['accuracy'])


# In[12]:


batch_size = 128 # batch_size argument is passed to the layer to define a batch size for the inputs.
epochs = 10
history = model.fit(x_train, y_train,
batch_size=batch_size,
epochs=epochs,
verbose=1, # verbose=1 will show you an animated progress bar eg. [==========]
validation_data=(x_test, y_test))


# In[13]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[14]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[15]:


#verification of an image
plt.imshow(x_train[1], cmap='gray')
plt.show() 
input_image = x_train[1].reshape(1, 784)
predictions = model.predict(input_image)
predicted_class = np.argmax(predictions[0])
print("Predicted class:", predicted_class)


# In[ ]:





# In[ ]:





# In[ ]:




