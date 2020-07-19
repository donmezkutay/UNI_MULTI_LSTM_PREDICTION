"""This code integrated from Tensorflow's own webpage exercise by KUTAY DÖNMEZ"""
"""Let's Use ERA5 data for Samsun Merkez Between 2017-2018 with 2 years of data"""
"""And Predict One step forward Temperature with Univariate and Multivariate model"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import matplotlib as mpl
import os

#Preparing data
data = xr.open_dataset(r'era5_2017_2018.nc')
data_temp = xr.open_dataset(r'ERA5_2MTEMP_2017_2018.nc')

#equate the variables
evaporation   = data['e']
precipitation = data['tp']
temperature   = data_temp['t2m']

#interpolate and make data 1d with time series dependent only
#41.344167|36.256389| samsun bölge 17030
lat = 41.344167
lon = 36.256389
p = precipitation.interp(latitude = lat, longitude = lon).values * 1000 # mm unit
e = evaporation.interp(latitude = lat, longitude = lon).values * 1000 # mm unit
t = temperature.interp(latitude = lat, longitude = lon).values - 273.15 # Celsius unit

#Prepare the dates
dates = data['time'].values

#now build a pandas dataset
instance = {'Date':dates,
            'Evap':e,
            'Temp':t,
            'Prec':p}
pd_data = pd.DataFrame(data=instance, )
pd_data.index = pd_data['Date']
pd_data = pd_data.drop(columns='Date')

"""This code builded with the very help of the tensorflow's exercise"""
class lstm():
    """Defining class in order to build easy univariate LSTM"""
    def __init__(self):
        self = self
    
    def prepare_data(self, data, target, init_index, finit_index, size_unit_window_history, 
                     size_unit_window_target, step, single_step, univariate):
        """Returns data as prepared according to keras univariate lstm input"""
        self.data = [] #X
        self.labels = [] #Y
        self.finit_index = finit_index
        #decide initial index to be used with respect to history size
        self.init_index = init_index + size_unit_window_history  
        
        #check if finit index is passed, if not decide it as the last index 
        #that is going to predict the target.
        
        if self.finit_index == None:
            self.finit_index = len(data) - size_unit_window_target
        
        #loop to feed the windows with according data and label
        for i in range(self.init_index, self.finit_index):
            self.indices = range(i-size_unit_window_history, i, step)
            
            #chechk if univariate or multivariate
            if univariate == True:
                self.data.append(np.reshape(data[self.indices], (size_unit_window_history, 1)))
            else:
                self.data.append(data[self.indices])

            if single_step == True:
                self.labels.append(target[i+size_unit_window_target])
            else:
                self.labels.append(target[i:i+size_unit_window_target])
        
        return np.array(self.data), np.array(self.labels)
    
    def standardize_data(self, data, TRAIN_SPLIT):
        """Standardizing the data"""
        self.train_mean = data[:TRAIN_SPLIT].mean()
        self.train_std = data[:TRAIN_SPLIT].std()
        self.data = (data - self.train_mean)/ self.train_std
        return self.data
    
    def train_val_to_tfdata(self, x_train, y_train, x_val, y_val,
                            BATCH_SIZE, BUFFER_SIZE):
        """Returns more optimized train and validation Data"""
        self.train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        self.train_data = self.train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
        
        self.val_data   = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        self.val_data   = self.val_data.cache().batch(BATCH_SIZE).repeat()
        
        return self.train_data, self.val_data
        
    def check_single_window_shape(self, train_data):
        """Input Prepared data, and check the single window size"""
        print ('Single window of past history : {}'.format(self.train_data[0].shape))      
    
    def build_lstm_model(self, train_data, units, target_size, single_step):
        """Building LSTM model, checking if single step desired or multistep desired"""
        
        self.target_size = target_size
        if single_step == True:
            self.target_size = 1
        
        model = tf.keras.models.Sequential()
        if single_step == True:
            model.add(tf.keras.layers.LSTM(units, input_shape = train_data.shape[-2:]) )
        elif single_step == False:
            model.add(tf.keras.layers.LSTM(units, return_sequences=True, input_shape = train_data.shape[-2:]) )
        
        if single_step == True:
            model.add(tf.keras.layers.Dense(self.target_size))
        elif single_step == False:
            model.add(tf.keras.layers.Dense(self.target_size))
        
        model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
        
        return model
    
    def fit_lstm_model(self, model, train_data, val_data, evaluation_interval, epochs, val_step):
        """Fit the model to the data"""
        
        self.model = model
        self.history = self.model.fit(train_data, epochs=epochs, steps_per_epoch=evaluation_interval,
                       validation_data = val_data, validation_steps=val_step )
        
        return self.history
    
    def create_time_steps(self, length):
        return list(range(-length, 0))

    def show_plot(self, plot_data, delta, title):
        self.labels = ['History', 'True Future', 'Model Prediction']
        self.marker = ['.-', 'rx', 'go']
        self.time_steps = self.create_time_steps(plot_data[0].shape[0])
        if delta:
            self.future = delta
        else:
            self.future = 0

        plt.title(title)
        for i, x in enumerate(plot_data):
            if i:
                plt.plot(self.future, plot_data[i], self.marker[i], markersize=10,
                   label=self.labels[i])
            else:
                plt.plot(self.time_steps, plot_data[i].flatten(), self.marker[i], label=self.labels[i])
        plt.legend()
        plt.xlim([self.time_steps[0], (self.future+5)*2])
        plt.xlabel('Time-Step')
        return plt
        
        
        
#Building Univariate lstm 
TRAIN_SPLIT = int(17520 * 80 / 100)
temp_dt = pd_data['Temp'].values

#Start model instance
p = lstm()
std_temp = p.standardize_data(data = temp_dt, TRAIN_SPLIT=TRAIN_SPLIT)
univariate_past_history = 20
univariate_future_target = 0

#prepare data for input to model
x_train_uni, y_train_uni = p.prepare_data(std_temp, std_temp, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target, step=1,
                                           single_step=True, univariate=True)
x_val_uni, y_val_uni = p.prepare_data(std_temp, std_temp, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target, step=1,
                                       single_step=True, univariate=True)

#print ('Single window of past history')
#print (x_train_uni[0])
#print ('\n Target temperature to predict')
#print (y_train_uni[0])

#look at the real history
#p.show_plot([x_train_uni[1], y_train_uni[1]], 0, 'Sample Example')

BATCH_SIZE = 256
BUFFER_SIZE = 10000

#numpy data to tf data
tf_train, tf_val = p.train_val_to_tfdata(x_train_uni, y_train_uni,
                                         x_val_uni, y_val_uni, BATCH_SIZE,
                                         BUFFER_SIZE)
#define lstm model
model = p.build_lstm_model(x_train_uni, 32, 1, True)

for x, y in tf_val.take(1):
    print(model.predict(x).shape)
    
EVALUATION_INTERVAL = 200
EPOCHS = 10

#let's fit the model
p.fit_lstm_model(model, tf_train, tf_val, EVALUATION_INTERVAL,
                 EPOCHS, val_step=50)


#buradaki -1 , 256 batchden birisinin trainde tahmin edilmiş y sini göstermekte -1'i değiştirebilirsin
#make prediction using validation data
for x, y in tf_val.take(2):
    plot = p.show_plot([x[10].numpy(), y[10].numpy(),
                    model.predict(x)[10]], 0, 'Simple LSTM model')
    plot.show()
    
#We can train the model by also using numpy arrays instead of TF dataset
model.predict(np.reshape(x_val_uni[0], (1,20,1))) # 3d giriş olmalı numpy ile

#History
#Train for 200 steps, validate for 50 steps
#Epoch 1/10
#200/200 [==============================] - 11s 54ms/step - loss: 0.2227 - val_loss: 0.0876
#Epoch 2/10
#200/200 [==============================] - 5s 27ms/step - loss: 0.0766 - val_loss: 0.0648
#Epoch 3/10
#200/200 [==============================] - 5s 27ms/step - loss: 0.0640 - val_loss: 0.0493
#Epoch 4/10
#200/200 [==============================] - 5s 27ms/step - loss: 0.0577 - val_loss: 0.0468
#Epoch 5/10
#200/200 [==============================] - 5s 27ms/step - loss: 0.0547 - val_loss: 0.0454
#Epoch 6/10
#200/200 [==============================] - 5s 27ms/step - loss: 0.0527 - val_loss: 0.0459
#Epoch 7/10
#200/200 [==============================] - 5s 27ms/step - loss: 0.0518 - val_loss: 0.0482
#Epoch 8/10
#200/200 [==============================] - 5s 27ms/step - loss: 0.0507 - val_loss: 0.0469
#Epoch 9/10
#200/200 [==============================] - 5s 27ms/step - loss: 0.0505 - val_loss: 0.0458
#Epoch 10/10
#200/200 [==============================] - 5s 26ms/step - loss: 0.0499 - val_loss: 0.0433

#Let's do multivariate time series prediction using LSTM
#Building Univariate lstm 
TRAIN_SPLIT = int(17520 * 80 / 100)
multi_data = pd_data.values

#Start instance
p = lstm()
std_temp = p.standardize_data(data = multi_data, TRAIN_SPLIT=TRAIN_SPLIT)
univariate_past_history = 720
univariate_future_target = 72 #predicting future 72. index
x_train_uni, y_train_uni = p.prepare_data(std_temp, std_temp[:,1], 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target, step=6,
                                           single_step=True, univariate=False)
x_val_uni, y_val_uni = p.prepare_data(std_temp, std_temp[:,1], TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target, step=6,
                                       single_step=True, univariate=False)

#print ('Single window of past history')
#print (x_train_uni[0])
#print ('\n Target temperature to predict')
#print (y_train_uni[0])

#look at the real history
#p.show_plot([x_train_uni[1][:,1], y_train_uni[1]], 0, 'Sample Example')

BATCH_SIZE = 256
BUFFER_SIZE = 10000

tf_train, tf_val = p.train_val_to_tfdata(x_train_uni, y_train_uni,
                                         x_val_uni, y_val_uni, BATCH_SIZE,
                                         BUFFER_SIZE)

#build model
#define lstm model
model = p.build_lstm_model(x_train_uni, 32, 1, True)

for x, y in tf_val.take(1):
    print(model.predict(x).shape)
    
EVALUATION_INTERVAL = 200
EPOCHS = 10

#fit the model
p.fit_lstm_model(model, tf_train, tf_val,EVALUATION_INTERVAL,
                 EPOCHS, val_step=50)

#Train for 200 steps, validate for 50 steps
#Epoch 1/10
#200/200 [==============================] - 35s 174ms/step - loss: 0.3674 - val_loss: 0.3238
#Epoch 2/10
#200/200 [==============================] - 31s 156ms/step - loss: 0.3219 - val_loss: 0.3274
#Epoch 3/10
#200/200 [==============================] - 31s 155ms/step - loss: 0.3177 - val_loss: 0.3639
#Epoch 4/10
#200/200 [==============================] - 31s 156ms/step - loss: 0.3097 - val_loss: 0.3452
#Epoch 5/10
#200/200 [==============================] - 30s 152ms/step - loss: 0.2964 - val_loss: 0.3239
#Epoch 6/10
#200/200 [==============================] - 30s 150ms/step - loss: 0.2770 - val_loss: 0.3159
#Epoch 7/10
#200/200 [==============================] - 31s 154ms/step - loss: 0.2710 - val_loss: 0.3635
#Epoch 8/10
#200/200 [==============================] - 31s 157ms/step - loss: 0.2551 - val_loss: 0.2954
#Epoch 9/10
#200/200 [==============================] - 32s 161ms/step - loss: 0.2399 - val_loss: 0.2871
#Epoch 10/10
#200/200 [==============================] - 31s 157ms/step - loss: 0.2338 - val_loss: 0.2798

#now let's predict validation data using multivariate LSTM
for x, y in tf_val.take(2):
    plot = p.show_plot([x[-1][:,1].numpy(), y[-1].numpy(),
                    model.predict(x)[-1]], 0, 'Simple LSTM model')
    plot.show()