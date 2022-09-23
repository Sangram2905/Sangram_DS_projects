# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:27:25 2022

@author: sangramp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template 

df_PV_full = pd.read_csv('Product_sales_visibility.csv')


#######################################################################

#DL Model
dx = df_PV_full.drop(['Sales','yr_establishment'],axis=1).values
dy = df_PV_full['Sales'].values


# train split for test and train of ML
from sklearn.model_selection import train_test_split
dx_train,dx_test,dy_train,dy_test = train_test_split(dx,dy,test_size=0.3,random_state=10)

from sklearn.preprocessing import MinMaxScaler
MinMaxScaler_obj = MinMaxScaler()

#MinMaxScaler_obj.fit(dx_train) # fit will run the std.Dev and min max numbers in data set and fit accordingly the other data

dx_train = MinMaxScaler_obj.fit_transform(dx_train)
dx_test = MinMaxScaler_obj.transform(dx_test)

#Deep learning
from tensorflow.keras.models import Sequential # basci model
from tensorflow.keras.layers import Dense


Sequential_model = Sequential()
Sequential_model.add(Dense(10,activation='relu'))
Sequential_model.add(Dense(20,activation='relu'))
Sequential_model.add(Dense(40,activation='relu'))
Sequential_model.add(Dense(20,activation='relu'))
Sequential_model.add(Dense(10,activation='relu'))
Sequential_model.add(Dense(1))
Sequential_model.compile(optimizer = 'rmsprop',loss='mse')
Sequential_model.fit(x=dx_train,y=dy_train,validation_data=(dx_test,dy_test),batch_size=100,epochs=1000)

#Loss evaluation 
loss = pd.DataFrame(Sequential_model.history.history)
#loss.plot()

#Sequential_model.evaluate(x_test,y_test)

from keras.models import load_model
## for multipal model use in other program we can save the trained model
Sequential_model.save('my_Sequential_model_product_sales_visiblity.h5')

 # to store on local drive
pickle.dump(MinMaxScaler_obj,open('scaling.pkl','wb'))
pickle.dump(Sequential_model,open('psp_Sequential_model.pkl','wb'))


model_final = load_model('my_Sequential_model_product_sales_visiblity.h5')
   
pickled_model = pickle.load(open('psp_Sequential_model.pkl','rb')) # to open the stored model

#y_pred_P = model_final.predict(dx_test) # redict with h5 file

# dy_pred = Sequential_model.predict(dx_test)


# from sklearn.metrics import mean_absolute_error,mean_squared_error

# print(mean_absolute_error(dy_test,dy_pred))
# print((mean_squared_error(dy_test,dy_pred))**0.5)

# print("Error in prediction value",((mean_squared_error(dy_test,dy_pred))**0.5)/df_PV_full['Sales'].mean())

# from sklearn.metrics import explained_variance_score
# print(explained_variance_score(dy_test, dy_pred))

# plt.scatter(dy_test, dy_pred)

## Single Shop Sales target

# single_shop_data = df_PV_full.drop(['Sales','yr_establishment'],axis=1).iloc[107]
# single_shop_data_s = MinMaxScaler_obj.transform(single_shop_data.values.reshape(-1,10))

# Sales_target = (Sequential_model.predict(single_shop_data_s))[0][0]

# actual_sales = df_PV_full.iloc[107][0]
# print(actual_sales)

