#!/usr/bin/python

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

train_df = pd.read_csv("data/hourly_wages_data.csv")
print train_df.head()
train_X = train_df.drop(columns=['wage_per_hour'])
print "train_X.head()"
print train_X.head()

train_Y = train_df[['wage_per_hour']]
print "train_Y.head()"
print train_Y.head()

model = Sequential()
n_cols = train_X.shape[1]

model.add(Dense(20, activation='relu', input_shape=(n_cols,)))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping_monitor = EarlyStopping(patience=3)
model.fit(train_X, train_Y, validation_split=0.2, epochs=130, callbacks=[early_stopping_monitor])

test_X=train_X[0:1]
print model.predict(test_X)

