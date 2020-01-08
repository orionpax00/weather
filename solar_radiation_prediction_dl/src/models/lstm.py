import json
import tensorflow as tf


class LSTM(tf.keras.Model):
    """
        base cnn lstm model
    """

    def __init__(self, input_shape):
        super(LSTM, self).__init__(name="")
        self.input_shape_ = input_shape

        ##lstm model
        self.lstm1 = tf.keras.layers.LSTM(256,activation='tanh')
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.flatten1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128)
        self.dense2 = tf.keras.layers.Dense(64)
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self,input_tensor):
        
        x = self.lstm1(input_tensor)
        x = self.flatten1(x)
        x = self.dropout1(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
      
        return x


