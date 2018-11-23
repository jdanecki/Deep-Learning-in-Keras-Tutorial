#!/usr/bin/python

import pandas as pd
import os

pd_sum = pd.read_csv("data/suma.csv")
print pd_sum.head()

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.utils import plot_model

train_x = pd_sum.drop(columns=['y'])
train_y = pd_sum[['y']]

model = Sequential()
n_cols = train_x.shape[1]

model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping_monitor = EarlyStopping(patience=5)
model.fit(train_x, train_y, validation_split=0.2, epochs=300, callbacks=[early_stopping_monitor])

test_x =pd.DataFrame({'x1': [0.5] , 'x2': [0.2] })
print "testing"
print test_x
print model.predict(test_x)

test_x =pd.DataFrame({'x1': [0.1] , 'x2': [0.9] })
print "testing"
print test_x
print model.predict(test_x)

test_x =pd.DataFrame({'x1': [1] , 'x2': [9] })
print "testing"
print test_x
print model.predict(test_x)

plot_model(model, to_file='model.png', show_shapes=True, rankdir='LR')
