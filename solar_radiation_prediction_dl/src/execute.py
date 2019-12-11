import os

import tensorflow as tf
import matplotlib.pyplot as plt

from utils.dataloader import getData
from models.cnnlstm import cnnLSTM

EVALUATION_INTERVAL = 200
EPOCHS = 30

BATCH_SIZE = 16
BUFFER_SIZE = 50

PAST_HISTORY = 20
FUTURE_TARGET = 10
STEPS = 1
MAIN_FILE = ".\\data\\main\\main.csv"
FEATURES = ["Solar Radiation",  "MAX Temp", \
                    "Vapour press", "PM2.5"]
TARGET = "wind speed"



data = getData(MAIN_FILE, FEATURES, TARGET, 
                    PAST_HISTORY, FUTURE_TARGET, STEPS)
x_train, y_train = data.call(single_step = True)

x_val, y_val = data.call(for_validation=True, single_step = True)

print ('Single window of past history : {}'.format(x_train[0].shape))


train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data = tf.data.Dataset.from_tensor_slices((x_val,y_val))
val_data = val_data.batch(BATCH_SIZE).repeat()


# single_step_model = tf.keras.models.Sequential()
# single_step_model.add(tf.keras.layers.LSTM(32,
#                                            input_shape=x_train.shape[-2:]))
# single_step_model.add(tf.keras.layers.Dense(1))



logdir = os.path.join(os.getcwd(),"tensorboard_logs")

single_step_model = cnnLSTM(x_train.shape[-2:])
single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')




def get_callbacks(name):
  return [
    tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
    tf.keras.callbacks.TensorBoard(logdir),
  ]



history = single_step_model.fit(train_data, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data,
                                            validation_steps=50,
                                            callbacks=get_callbacks("logs"))
