import os
from datetime import datetime

import tensorflow as tf
import matplotlib.pyplot as plt

from utils.dataloader import getData
from utils.callbacks import *
from misc.misc import *
from models.lstm import LSTM

tf.keras.backend.clear_session()

MODEL_NAME = "cnn"
EVALUATION_INTERVAL = 202
EPOCHS = 25

BATCH_SIZE = 256
BUFFER_SIZE = 10000

PAST_HISTORY = 20
FUTURE_TARGET = 1
STEPS = 1
MAIN_FILE = ".\\data\\main\\SRRL6.csv"

## features for dataset 1
# FEATURES = ["index", "MN",  "MAX Temp", "Sunshine hour", "MAX Temp",".RH", "mean sea LP", \
#                     "Vapour press", "PM2.5", "Solar Radiation"] ## Always put target class in the end

## features for dataset 2
FEATURES = ["Avg Global Solar Radiation [W/m^2]", " Temp [deg C]",  "Avg Station Pressure [mBar]", "Avg Wind Speed ", " RH [%]"] ## Always put target class in the end

TARGET = " RH [%]"
FEATURES.append(TARGET)
DATE_TIME = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")



data = getData(MAIN_FILE, FEATURES, 
                    PAST_HISTORY, FUTURE_TARGET, STEPS)
x_train, y_train = data.call(single_step = True)

x_val, y_val = data.call(for_validation=True, single_step = True)

print ('Single window of past history : {}'.format(x_train[0].shape))


train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data = tf.data.Dataset.from_tensor_slices((x_val,y_val))
val_data = val_data.batch(BATCH_SIZE).repeat()




logdir = os.path.join(os.getcwd(),"tensorboard_logs")

single_step_model = LSTM(x_train.shape[-2:])
single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mse', metrics=['mse','mae','mape'])




single_step_model.fit(train_data, epochs=EPOCHS,
                          steps_per_epoch=EVALUATION_INTERVAL,
                          validation_data=val_data,
                          validation_steps=202,
                          callbacks=logger(MODEL_NAME,DATE_TIME))


x_test, data_mean, data_std = data.testdata(single_step = True)

log_prediction(single_step_model, x_test, MODEL_NAME, DATE_TIME, data_mean[-1], data_std[-1])


data = {
  "MODEL_NAME":MODEL_NAME,
  "EVALUATION_INTERVAL": EVALUATION_INTERVAL,
  "EPOCHS": EPOCHS,

  "BATCH_SIZE": BATCH_SIZE,
  "BUFFER_SIZE": BUFFER_SIZE,

  "PAST_HISTORY": PAST_HISTORY,
  "FUTURE_TARGET": FUTURE_TARGET,
  "STEPS": STEPS,
  "MAIN_FILE": MAIN_FILE,
  "FEATURES": FEATURES,
  "TARGET": TARGET,
  "FEATURES": FEATURES,
  "DATE_TIME": DATE_TIME 
}

configLogger(data, MODEL_NAME, DATE_TIME)
