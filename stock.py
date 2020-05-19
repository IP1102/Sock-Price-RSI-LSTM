#%%Importing Libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
#Read the data 
df = pd.read_csv('RELIANCE.csv')

#Plot the data 
df.plot(x = 'Date', y = ['Close','Adj Close'])

#Drop Columns
df.drop(columns = ['Volume'], inplace = True)
df.drop(columns = ['Open', 'High', 'Low', 'Close'], inplace = True)
df.plot()

#%%
data = pd.read_csv('RELIANCE.csv')
data = data.set_index('Date')
X = data.iloc[:,0:5].values

#%%Calculation of RSI
df['Diff'] = df['Adj Close'].diff()
df["Up Mov"] = df['Diff'].apply(lambda x: x if x>0 else 0)
df["Down Mov"] = df['Diff'].apply(lambda x: abs(x) if x<0 else 0)
print(df[['Adj Close', 'Diff','Up Mov','Down Mov']])

#Calculation of Average Upward Movement
df['Avg Up'] = " "
df['Avg Up'].loc[0] = df['Up Mov'].loc[0:14].mean()
df['Avg Up'].loc[14] = df['Avg Up'].loc[0]
df['Avg Up'].loc[0] = ""

for i in range(15,len(df.index)):
    df['Avg Up'].loc[i] = (df['Avg Up'].loc[i-1]*13+df['Up Mov'].loc[i])/14
    

#Calculation of Average Downward Movement
df['Avg Down'] = " "
df['Avg Down'].loc[0] = df['Down Mov'].loc[0:14].mean()
df['Avg Down'].loc[14] = df['Avg Down'].loc[0]
df['Avg Down'].loc[0] = ""

for i in range(15,len(df.index)):
    df['Avg Down'].loc[i] = (df['Avg Down'].loc[i-1]*13+df['Down Mov'].loc[i])/14


#RS Calculation
df['RS'] = ""
for i in range(14,len(df.index)):
    df['RS'].loc[i] = (df['Avg Up'].loc[i]/df['Avg Down'].loc[i])
    
#RSI
df['RSI'] = ""
for i in range(14,len(df.index)):
    df['RSI'].loc[i] = 100-100/(df['RS'].loc[i]+1)

#RSI to numeric
df['RSI'] = pd.to_numeric(df['RSI'])

#Visualize
df[['Adj Close','RSI']].plot()

#Copy into the new Data Frame
df2 = df[['Adj Close','RSI']].copy()

#%%Creating the array for LSTM

#Remove NaN
df2.isna().sum()
df2['RSI'].fillna(df['RSI'].mean(), inplace = True)
df2['Adj Close'].fillna(method = 'ffill', inplace = True)

#Get the data in array format
train_set = df2.iloc[:,:].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
train_set = sc.fit_transform(train_set)

#Creating the arrays
x = np.zeros((train_set.shape[0]-60,60,train_set.shape[1]))
y = np.zeros((train_set.shape[0]-60,))

for i in range(train_set.shape[0]-60):
    x[i] = train_set[i:60+i]
    y[i] = train_set[60+i,1]


#%%LSTM
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM,Bidirectional,Dense,Dropout

#Splitting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x, y, test_size=0.25)

model = Sequential()

#Input and 1st Hidden layer
model.add(LSTM(units = 32,return_sequences = True,input_shape = (X_train.shape[1],X_train.shape[2])))
model.add(Dropout(0.1))

#2nd Layer
model.add(LSTM(units=32, return_sequences = True))
model.add(Dropout(0.1))

model.add(LSTM(units=32, return_sequences = True))
model.add(Dropout(0.1))

model.add(LSTM(units=32))
model.add(Dropout(0.1))

model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss= 'mean_squared_error', metrics = ['mean_squared_error'])

history = model.fit(X_train,Y_train,
          epochs = 100,
          batch_size = 10,
          validation_data = (X_test,Y_test),
          callbacks = [tf.keras.callbacks.EarlyStopping(patience=4)])


    
model.predict(X_test)
model.evaluate(X_test,Y_test)

#%%Visualize
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.plot(sc.inverse_transform(Y_test.reshape(1,-1)))
plt.plot(sc.inverse_transform(Y_train.reshape(1,-1)))




