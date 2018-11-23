#!/usr/bin/python

import pandas as pd
 
#print pd.DataFrame({'A' : [10,20], 'B' : [2,21], 'C': [ 1,2] })
pd_sum = pd.DataFrame({'x1' : [ 0.1, 0.2, 0.5, 1, 5, 10, 20],  
                      'x2' :  [ 0.2, 0.5, 0.1, 2, 4, 8,  40],
                      'y' :   [ 0.3, 0.7, 0.6, 3, 9, 18, 60]})
print pd_sum

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

train_x = pd_sum.drop(columns=['y'])
train_y = pd_sum[['y']]

model = Sequential()
n_cols = train_x.shape[1]

model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping_monitor = EarlyStopping(patience=10)
model.fit(train_x, train_y, validation_split=0.5, epochs=1230, callbacks=[early_stopping_monitor])

#test_x=train_x[0:1]
test_x =pd.DataFrame({'x1': [11] , 'x2': [2] })
print "testing"
print test_x
print model.predict(test_x)


