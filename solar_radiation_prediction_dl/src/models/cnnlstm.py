import json

import tensorflow as tf



class lstmCNN(tf.keras.Model):
    """
        base cnn lstm model
    """

    def __init__(self, input_shape):
        super(lstmCNN, self).__init__(name="")


        ##lstm model
        self.lstm1 = tf.keras.layers.LSTM(input_shape[-2]*input_shape[-1],activation='relu')

        #reshaping layers
        self.reshape1 = tf.keras.layers.Reshape((input_shape[-2] , input_shape[-1]))

        ##cnn model 
        self.conv1 = tf.keras.layers.Conv1D(16, 3 , activation="relu")
        self.maxpool1 = tf.keras.layers.MaxPooling1D(pool_size=2)
        self.flatten1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(1)


    


    def call(self,input_tensor):

        x = self.lstm1(input_tensor)
        x = self.reshape1(x)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.flatten1(x)
        x = self.dense1(x)
        

        return x


