#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import libraries and dataset #
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('UK_foods.csv',index_col='Unnamed: 0')
df.transpose()

plt.figure(figsize=(10,8))
sns.heatmap(df)


# ## Autoencoder

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# Encoder #
encoder = Sequential()
encoder.add(Dense(units=8,activation='relu',input_shape=[17]))
encoder.add(Dense(units=4,activation='relu',input_shape=[8]))
encoder.add(Dense(units=2,activation='relu',input_shape=[4]))

# Decoder #
decoder = Sequential()
decoder.add(Dense(units=4,activation='relu',input_shape=[2]))
decoder.add(Dense(units=8,activation='relu',input_shape=[4]))
decoder.add(Dense(units=17,activation='relu',input_shape=[8]))

# Autoencoder #
autoencoder = Sequential([encoder,decoder])
autoencoder.compile(loss="mse" ,optimizer=SGD(lr=1.5))


# In[ ]:


# Scale the data #
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(df.transpose().values)
scaled_df

# Training #
autoencoder.fit(scaled_df,scaled_df,epochs=15)


# In[ ]:


# Run the data through only encoder #
encoded_2dim = encoder.predict(scaled_df)
encoded_2dim
df.transpose().index

results = pd.DataFrame(data=encoded_2dim,index=df.transpose().index,
                      columns=['C1','C2
                               
results = results.reset_index()
results
sns.scatterplot(x='C1',y='C2',data=results,hue='index')

